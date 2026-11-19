"""Flash size of the same XGBoost model exported five ways, cross-compiled for Cortex-M4F.

    python -m bench.sizes [zoo model names...]

For every model this generates and compiles (compile only; nothing is linked or uploaded)

  xgb_edge          the xgb-edge export (integer-only, default options, validation = zoo test rows)
  float_baseline    bench/float_baseline.py: float32 table walk, one 8-byte record per node
  m2cgen            m2cgen's if-else C (double arithmetic)
  tl2cgen           TL2cgen's if-else C, default settings
  tl2cgen_quantize  TL2cgen's if-else C with quantize=1

and reports, per variant, .text, .rodata, .data+.bss and total flash (text + rodata + data)
summed over the variant's object files as printed by ``arm-none-eabi-size -A``, plus the
compile time.  A compile that fails, exceeds the timeout or exceeds the memory limit is a
result too and is reported as such.  Results are printed and merged into
``build/bench/sizes.json``; sources and objects stay in ``build/bench/<model>/<variant>/``.

Things to keep in mind when reading the numbers:

* m2cgen computes in ``double``.  A Cortex-M4F FPU is single precision only, so every
  comparison and addition is a call into the compiler's software floating-point library.
  TL2cgen compares in float but adds every leaf as a double, so it needs that library too.
* Only the object files are measured.  Library code a variant needs at link time (soft-float
  helpers such as __aeabi_dadd/__aeabi_dcmplt, libm's exp/expf, memcpy) is NOT included;
  the undefined symbols of each variant are listed so the dependency is visible.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import altgen, float_baseline  # noqa: E402

ROOT = altgen.ROOT
BUILD = altgen.BUILD

DEFAULT_MODELS = ["bin_d2_r5_hist", "mc3_d3_r10_hist", "mc5_d4_r50_hist_missing", "bin_d6_r200_exact_missing",
                  "mc10_digits_d6_r100_hist", "mc10_d10_r30_exact", "mc16_d5_r397_hist"]

#: nRF52840 (Cortex-M4F) target flags; every variant is compiled with exactly these.
TARGET_CFLAGS = ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16", "-Os",
                 "-ffunction-sections", "-fdata-sections"]

VARIANTS = ["xgb_edge", "float_baseline", "m2cgen", "tl2cgen", "tl2cgen_quantize"]

#: Optional extra variants (--tl2cgen-parallel-comp N): the same TL2cgen code split into N
#: translation units, for models whose single predict() function is too big to compile.
SPLIT_VARIANTS = {"tl2cgen_split": "tl2cgen", "tl2cgen_quantize_split": "tl2cgen_quantize"}
ALL_VARIANTS = VARIANTS + list(SPLIT_VARIANTS)

#: Diagnostics only (no effect on generated code): our own generators must be warning-free.
OWN_WARNING_FLAGS = ["-Wall", "-Wextra", "-Werror"]

NOTES = [
    "Sizes are sums over the object files of a variant (arm-none-eabi-size -A): .text* / .rodata* / "
    ".data* + .bss*; flash total = text + rodata + data. Nothing is linked.",
    "m2cgen uses double for every comparison and addition. The Cortex-M4F FPU (FPv4-SP) is single "
    "precision only, so that is software floating point (calls to __aeabi_dcmplt, __aeabi_dadd, ...).",
    "TL2cgen compares in float but writes every leaf as 'result[i] += <double literal>', so each "
    "leaf addition is software double arithmetic too (__aeabi_f2d, __aeabi_dadd, __aeabi_d2f).",
    "Helper library code is NOT included in the object sizes: the soft-float routines, exp/expf "
    "and memcpy that a variant's 'undefined' list names come from libgcc/libm/libc at link time.",
    "TL2cgen objects also contain small unused entry points (postprocess, get_num_feature, ...) "
    "that a linker with --gc-sections would drop; they are counted here (well under 200 bytes).",
]


def _rel(path: str) -> str:
    """Path relative to the repository when it is inside it (as in build/bench/...), else unchanged."""
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


# --------------------------------------------------------------------------- running the compiler

_LIBPROC = None


def _process_memory_bytes(pid: int, rss_kb: int) -> int:
    """Memory one process really occupies, including what the OS has compressed or swapped out.

    Resident size alone is not enough: under memory pressure macOS compresses a big compiler's
    pages, and its RSS shrinks to a fraction of what it holds (measured: 0.9 GB resident for a
    16 GB cc1).  macOS: the physical footprint from proc_pid_rusage(); Linux: VmRSS + VmSwap;
    elsewhere the RSS that ``ps`` reported.
    """
    global _LIBPROC
    if sys.platform == "darwin":
        try:
            if _LIBPROC is None:
                _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
            info = (ctypes.c_uint64 * 12)()          # struct rusage_info_v0: 16-byte uuid + 10 x uint64
            if _LIBPROC.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(0), ctypes.byref(info)) == 0:
                return max(int(info[9]), rss_kb * 1024)   # ri_phys_footprint
        except (OSError, AttributeError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            total = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith(("VmRSS:", "VmSwap:")):
                        total += int(line.split()[1]) * 1024
            return total
        except (OSError, ValueError, IndexError):
            pass
    return rss_kb * 1024


def _group_memory_bytes(pgid: int) -> int:
    """Memory of every process in a process group (compiler driver + cc1 + as)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,rss="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == str(pgid) and parts[0].isdigit() and parts[2].isdigit():
            total += _process_memory_bytes(int(parts[0]), int(parts[2]))
    return total


def run_limited(cmd: List[str], log_path: str, timeout: float, max_mem: Optional[int],
                tmpdir: Optional[str] = None) -> Dict:
    """Run a compile with a wall-clock limit and a memory limit.

    Returns {"outcome": "ok" | "failed" | "timeout" | "memory", "seconds", "peak_mem_bytes", "returncode"}.
    The compiler's stderr goes to ``log_path``.  ``tmpdir`` becomes the compiler's TMPDIR and is
    removed afterwards, so a killed compiler does not leave its (large) temporary files behind.
    """
    t0 = time.time()
    peak = 0
    outcome = None
    env = None
    if tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        env = dict(os.environ, TMPDIR=tmpdir)
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        interval = 0.05
        while True:
            try:
                proc.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                pass
            interval = min(1.0, interval * 2)
            elapsed = time.time() - t0
            mem = _group_memory_bytes(proc.pid) if elapsed > 1.0 else 0
            peak = max(peak, mem)
            if elapsed > timeout:
                outcome = "timeout"
            elif max_mem and mem > max_mem:
                outcome = "memory"
            if outcome:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                break
    if outcome is None:
        outcome = "ok" if proc.returncode == 0 else "failed"
    seconds = time.time() - t0
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return {"outcome": outcome, "seconds": seconds, "peak_mem_bytes": peak, "returncode": proc.returncode}


def summarize_compiler_log(text: str) -> str:
    """One line for a failed compile: the first error, and how often the same message repeats."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    errors = [ln for ln in lines if "rror" in ln]
    if not errors:
        return (lines[-1] if lines else "no compiler output")[:300]
    core = re.sub(r"^\S+?:\d+(:\d+)?: ", "", errors[0])      # drop "file:line: "
    same = sum(1 for ln in errors if ln.endswith(core))
    return core[:300] + (f" ({same} such messages)" if same > 1 else "")


def read_sizes(size_tool: str, obj: str) -> Dict:
    """Section sizes of one object file, grouped the way the report needs them."""
    r = subprocess.run([size_tool, "-A", obj], capture_output=True, text=True, check=True)
    res = {"text": 0, "rodata": 0, "data": 0, "bss": 0, "other": {}}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].isdigit() or parts[0] == "Total":
            continue
        sec, size = parts[0], int(parts[1])
        if sec.startswith(".text"):
            res["text"] += size
        elif sec.startswith(".rodata"):
            res["rodata"] += size
        elif sec.startswith(".data"):
            res["data"] += size
        elif sec.startswith(".bss"):
            res["bss"] += size
        elif size and not sec.startswith((".comment", ".ARM.attributes", ".debug", ".note")):
            res["other"][sec] = res["other"].get(sec, 0) + size   # never silently dropped
    return res


def undefined_symbols(nm_tool: str, objs: List[str]) -> List[str]:
    """External symbols the objects need from elsewhere (library helpers), minus those they define."""
    undefined, defined = set(), set()
    for obj in objs:
        r = subprocess.run([nm_tool, obj], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "U":
                undefined.add(parts[1])
            elif len(parts) == 3 and parts[1] in "TDRBtdrb":
                if parts[1].isupper():
                    defined.add(parts[2])
    return sorted(undefined - defined)


# --------------------------------------------------------------------------- generating the variants

def generate_variant(variant: str, tr, outdir: str, gen_timeout: float, parallel_comp: int = 0) -> Dict:
    """Write the C of one variant into ``outdir``.  Returns {"sources": [...], ...} or {"error": text}."""
    t0 = time.time()
    try:
        if variant in SPLIT_VARIANTS:
            return altgen.generate_subprocess(SPLIT_VARIANTS[variant], tr.model_path, outdir, timeout=gen_timeout,
                                              params={"parallel_comp": parallel_comp})
        if variant in altgen.VARIANTS:
            return altgen.generate_subprocess(variant, tr.model_path, outdir, timeout=gen_timeout)
        os.makedirs(outdir, exist_ok=True)
        notes: List[str] = []
        if variant == "xgb_edge":
            from xgb_edge import ExportOptions, export_model
            exp = export_model(tr.model_path, ExportOptions(name="xgb_model"), validation=tr.X)
            paths = exp.write(outdir)
            notes.append(f"{exp.lplan.bits}-bit leaves (auto), reported model data {exp.report['flash']['data_bytes']} B")
        elif variant == "float_baseline":
            paths = float_baseline.write(tr.model_path, "float_model", outdir)
        else:
            raise ValueError(variant)
        sources = [p for p in paths if p.endswith(".c")]
        return {"variant": variant, "sources": sources, "outdir": outdir, "notes": notes,
                "seconds": round(time.time() - t0, 2), "source_bytes": sum(os.path.getsize(s) for s in sources)}
    except Exception:  # noqa: BLE001 - a failed generation is a result
        return {"variant": variant, "error": "generation failed: " + traceback.format_exc()[-1500:]}


def measure_variant(variant: str, tr, model_dir: str, cc: str, timeout: float, max_mem: Optional[int],
                    parallel_comp: int = 0) -> Dict:
    """Generate, cross-compile and measure one variant of one model."""
    outdir = os.path.join(model_dir, variant)
    row: Dict = {"model": tr.spec.name, "variant": variant, "dir": _rel(outdir),
                 "text": None, "rodata": None, "data": None, "bss": None, "data_bss": None, "flash_total": None,
                 "compile_seconds": None, "measured_at": datetime.datetime.now().isoformat(timespec="seconds")}
    if variant in SPLIT_VARIANTS:
        row["parallel_comp"] = parallel_comp
    gen = generate_variant(variant, tr, outdir, timeout, parallel_comp)
    if "error" in gen:
        row["status"] = gen["error"]
        return row
    if variant in SPLIT_VARIANTS:
        # a run with a larger parallel_comp may have left tu<N>.c files that are not part of this one
        current = {os.path.basename(src) for src in gen["sources"]}
        for fn in os.listdir(outdir):
            if re.fullmatch(r"tu\d+\.(c|o|cc\.log)", fn) and re.sub(r"\.(o|cc\.log)$", ".c", fn) not in current:
                os.remove(os.path.join(outdir, fn))
    row["generate_seconds"] = gen.get("seconds")
    row["source_bytes"] = gen.get("source_bytes")
    row["notes"] = list(gen.get("notes", []))
    extra = list(gen.get("extra_cflags", []))
    if variant in ("xgb_edge", "float_baseline"):
        extra += OWN_WARNING_FLAGS
    row["extra_cflags"] = extra
    size_tool = re.sub(r"gcc(\.exe)?$", "size", cc)
    nm_tool = re.sub(r"gcc(\.exe)?$", "nm", cc)
    objects: List[Dict] = []
    obj_paths: List[str] = []
    totals = {"text": 0, "rodata": 0, "data": 0, "bss": 0}
    other: Dict[str, int] = {}
    seconds = 0.0
    status = "ok"
    for src in gen["sources"]:
        obj = os.path.splitext(src)[0] + ".o"
        log = os.path.splitext(src)[0] + ".cc.log"
        if os.path.exists(obj):
            os.remove(obj)
        cmd = [cc] + TARGET_CFLAGS + extra + ["-c", src, "-o", obj]
        r = run_limited(cmd, log, timeout, max_mem, tmpdir=os.path.join(outdir, "cc_tmp"))
        seconds += r["seconds"]
        if r["outcome"] != "ok" and os.path.exists(obj):
            os.remove(obj)          # a partial object must not be mistaken for a result
        entry = {"source": _rel(src), "object": _rel(obj),
                 "compile_seconds": round(r["seconds"], 2), "peak_mem_mb": round(r["peak_mem_bytes"] / 2 ** 20),
                 "outcome": r["outcome"]}
        objects.append(entry)
        base = os.path.basename(src)
        if r["outcome"] == "timeout":
            status = f"compile timeout: {base} not done after {timeout:.0f} s"
        elif r["outcome"] == "memory":
            status = (f"compile killed: {base} exceeded the {max_mem / 2 ** 30:.0f} GB memory limit after "
                      f"{r['seconds']:.0f} s")
        elif r["outcome"] == "failed":
            with open(log, errors="replace") as f:
                text = f.read()
            status = f"compile failed after {r['seconds']:.0f} s: {base}: {summarize_compiler_log(text)}"
            entry["compiler_log"] = _rel(log)
        if status != "ok":
            break
        with open(log, errors="replace") as f:
            warn = f.read()
        if warn.strip():
            entry["compiler_output"] = warn[-600:]
        sz = read_sizes(size_tool, obj)
        obj_paths.append(obj)
        entry.update({k: sz[k] for k in totals})
        for k in totals:
            totals[k] += sz[k]
        for k, v in sz["other"].items():
            other[k] = other.get(k, 0) + v
    row["objects"] = objects
    row["compile_seconds"] = round(seconds, 2)
    row["status"] = status
    if status == "ok":
        row.update(totals)
        row["data_bss"] = totals["data"] + totals["bss"]
        row["flash_total"] = totals["text"] + totals["rodata"] + totals["data"]
        row["undefined"] = undefined_symbols(nm_tool, obj_paths)
        if other:
            row["other_sections"] = other
    return row


# --------------------------------------------------------------------------- reporting

LABELS = {"xgb_edge": "xgb-edge", "float_baseline": "float32 baseline", "m2cgen": "m2cgen (double)",
          "tl2cgen": "TL2cgen", "tl2cgen_quantize": "TL2cgen quantize=1",
          "tl2cgen_split": "TL2cgen split", "tl2cgen_quantize_split": "TL2cgen q=1 split"}


def _n(v) -> str:
    return "-" if v is None else f"{v:,}"


def format_table(rows: List[Dict], model_info: Dict[str, str]) -> str:
    out: List[str] = []
    head = (f"{'variant':<20}{'.text':>12}{'.rodata':>12}{'.data+.bss':>12}{'flash total':>13}"
            f"{'x xgb-edge':>12}{'compile s':>11}  status")
    by_model: Dict[str, List[Dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, mrows in by_model.items():
        out.append("")
        out.append(f"{model}" + (f"  ({model_info[model]})" if model in model_info else ""))
        out.append(head)
        ref = next((r["flash_total"] for r in mrows if r["variant"] == "xgb_edge" and r["flash_total"]), None)
        for r in mrows:
            ratio = f"{r['flash_total'] / ref:.2f}" if ref and r["flash_total"] is not None else "-"
            secs = "-" if r["compile_seconds"] is None else f"{r['compile_seconds']:.1f}"
            status = r["status"] if r["status"] == "ok" else r["status"].splitlines()[0][:120]
            out.append(f"{LABELS.get(r['variant'], r['variant']):<20}{_n(r['text']):>12}{_n(r['rodata']):>12}"
                       f"{_n(r['data_bss']):>12}{_n(r['flash_total']):>13}{ratio:>12}{secs:>11}  {status}")
    return "\n".join(out)


def format_undefined(rows: List[Dict]) -> str:
    """Which library helpers each variant pulls in (union over the models measured)."""
    per: Dict[str, set] = {}
    for r in rows:
        if r.get("status") == "ok":
            per.setdefault(r["variant"], set()).update(r.get("undefined", []))
    lines = []
    for v in ALL_VARIANTS:
        if v in per:
            lines.append(f"  {LABELS[v]:<20}{', '.join(sorted(per[v])) or '(none)'}")
    return "\n".join(lines)


def _model_info(tr) -> str:
    from xgb_edge import load
    m = load(tr.model_path)
    nodes = m.n_internal + m.n_leaves
    return (f"{m.objective}, {m.n_classes} classes, {m.n_features} features, {len(m.trees)} trees, "
            f"{nodes:,} nodes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="zoo model names (default: %s)" % " ".join(DEFAULT_MODELS))
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset of: " + ", ".join(ALL_VARIANTS) + " (default: the first five)")
    ap.add_argument("--timeout", type=float, default=1200.0,
                    help="seconds allowed for each compile (and each code generation); default 1200")
    ap.add_argument("--max-mem-gb", type=float, default=None,
                    help="kill a compile whose memory footprint exceeds this (default: half of physical RAM; "
                         "0 = no limit)")
    ap.add_argument("--tl2cgen-parallel-comp", type=int, default=0, metavar="N",
                    help="also measure TL2cgen with parallel_comp=N (code split into N translation units) as "
                         "the extra variants tl2cgen_split and tl2cgen_quantize_split; off by default")
    ap.add_argument("--cc", default=None, help="arm-none-eabi-gcc (default: found on PATH or in Arduino15)")
    ap.add_argument("--build-dir", default=BUILD,
                    help="where generated sources and objects go, as <model>/<variant>/ (default: build/bench)")
    ap.add_argument("--json", default=None,
                    help="result file; rows of models measured earlier are kept (default: <build-dir>/sizes.json)")
    ap.add_argument("--show", action="store_true",
                    help="measure nothing: print the table of everything recorded in the result file")
    args = ap.parse_args(argv)

    from harness import zoo
    from xgb_edge.report import find_cross_compiler

    names = args.models or DEFAULT_MODELS
    build_dir = os.path.abspath(args.build_dir)
    json_path = os.path.abspath(args.json or os.path.join(build_dir, "sizes.json"))
    if args.show:
        with open(json_path) as f:
            doc = json.load(f)
        shown = [r for r in doc["results"] if not args.models or r["model"] in args.models]
        print(f"compiler : {doc['compiler']}\nflags    : {' '.join(doc['cflags'])}")
        print(format_table(shown, doc.get("model_info", {})))
        print("\nLibrary helpers each variant references (not included in the sizes above):")
        print(format_undefined(shown))
        print("\nNotes")
        for n in doc.get("notes", []):
            print("  - " + n)
        return 0
    if args.variants is None:
        variants = list(VARIANTS) + (list(SPLIT_VARIANTS) if args.tl2cgen_parallel_comp > 0 else [])
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        for v in variants:
            if v not in ALL_VARIANTS:
                ap.error(f"unknown variant {v!r}")
            if v in SPLIT_VARIANTS and args.tl2cgen_parallel_comp <= 0:
                ap.error(f"variant {v} needs --tl2cgen-parallel-comp N")
    for n in names:
        zoo.by_name(n)
    cc = args.cc or find_cross_compiler()
    if not cc:
        print("no arm-none-eabi-gcc found; pass --cc", file=sys.stderr)
        return 2
    if args.max_mem_gb is None:
        try:
            max_mem: Optional[int] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2
        except (ValueError, OSError):
            max_mem = None
    else:
        max_mem = int(args.max_mem_gb * 2 ** 30) or None
    cc_version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]

    print(f"compiler : {cc_version}")
    print(f"flags    : {' '.join(TARGET_CFLAGS)}")
    print(f"limits   : {args.timeout:.0f} s per compile"
          + (f", {max_mem / 2 ** 30:.0f} GB of compiler memory" if max_mem else ", no memory limit"))
    if any(v in SPLIT_VARIANTS for v in variants):
        print(f"split    : the 'split' variants are TL2cgen with parallel_comp={args.tl2cgen_parallel_comp} "
              "(not its default settings)")
    rows: List[Dict] = []
    info: Dict[str, str] = {}
    try:
        vers = altgen.versions()
    except ImportError as e:
        vers = {"error": str(e)}
    order = {s.name: i for i, s in enumerate(zoo.SPECS)}

    def save() -> int:
        """Merge this run's rows into the result file: they replace older rows of the same model and variant."""
        doc: Dict = {}
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    doc = json.load(f)
            except ValueError:
                doc = {}
        if doc.get("cflags") != TARGET_CFLAGS or doc.get("compiler") != cc_version:
            doc = {}
        from harness.zoo import code_stamp
        for r in rows:
            if r["variant"] == "xgb_edge":
                r["code"] = code_stamp("sizes")   # lets harness/results.py see when the code has changed since
        done = {(r["model"], r["variant"]) for r in rows}
        kept = [r for r in doc.get("results", []) if (r["model"], r["variant"]) not in done]
        merged = sorted(kept + rows, key=lambda r: (order.get(r["model"], 1 << 30), ALL_VARIANTS.index(r["variant"])))
        doc = {"compiler": cc_version, "cc": cc, "cflags": TARGET_CFLAGS, "timeout_seconds": args.timeout,
               "max_memory_bytes": max_mem, "versions": vers, "notes": NOTES,
               "model_info": {**doc.get("model_info", {}), **info}, "results": merged}
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path + ".tmp", "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(json_path + ".tmp", json_path)
        return len(merged)

    t_start = time.time()
    for name in names:
        tr = zoo.train(zoo.by_name(name))
        info[name] = _model_info(tr)
        model_dir = os.path.join(build_dir, name)
        for variant in variants:
            print(f"  [{time.time() - t_start:6.0f} s] {name}: {variant} ...", file=sys.stderr, flush=True)
            rows.append(measure_variant(variant, tr, model_dir, cc, args.timeout, max_mem,
                                        args.tl2cgen_parallel_comp))
            save()      # after every measurement, so an interrupted run keeps what it has

    print(format_table(rows, info))
    print("\nLibrary helpers each variant references (not included in the sizes above):")
    print(format_undefined(rows))
    print("\nNotes")
    for n in NOTES:
        print("  - " + n)
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        print("\nNot measured")
        for r in failed:
            print(f"  - {r['model']} / {LABELS.get(r['variant'], r['variant'])}: " + r["status"].replace("\n", "\n      "))

    n_merged = save()
    print(f"\nwrote {_rel(json_path)} ({n_merged} rows; {len(rows)} from this run); "
          f"sources and objects are in {_rel(build_dir)}/<model>/<variant>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
