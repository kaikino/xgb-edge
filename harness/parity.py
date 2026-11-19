"""Host parity harness: train -> export -> compile -> compare against xgboost.

Run with ``make parity`` or ``python -m harness.parity``.  For every zoo model
and export variant it

  A. checks the tool's float32 reference traversal against xgboost's pred_leaf
     on every test row (this is what "lossless thresholds" is measured against),
  B. exports C, compiles it for the host with -Wall -Wextra -Wpedantic -Werror
     and AddressSanitizer/UBSan,
  C. runs every test row through the compiled C and requires
       - integer margins identical to the tool's own exact integer simulation,
       - |margin - xgboost margin| within the reported bounds,
       - the same class as xgboost on every row the tool did not flag as at
         risk in advance (rows it did flag are counted and reported),
       - no more comparisons than the report's worst case,
  D. cross-compiles the same C for Cortex-M4 with -Wall -Wextra -Werror and
     requires the object's .rodata to equal the reported model data size.

Exit status is non-zero if any requirement fails.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import zoo  # noqa: E402
from xgb_edge import ExportOptions, export_model, load  # noqa: E402
from xgb_edge.quantize import INPUT_KINDS  # noqa: E402
from xgb_edge.refsim import leaf_indices  # noqa: E402
from xgb_edge.report import available_targets, find_cross_compiler, measure  # noqa: E402

BUILD = os.path.join(zoo.ROOT, "build", "parity")

HOST_CFLAGS = ["-std=c99", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow",
               "-Werror", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"]

DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "@NAME@.h"
extern unsigned long @NAME@_comparisons;
#define NF @UNAME@_NUM_FEATURES
#define NM @UNAME@_NUM_MARGINS
int main(int argc, char **argv)
{
    long rows = argc > 1 ? atol(argv[1]) : 0, r;
    static @CTYPE@ x[NF + 1];
    static int32_t out[NM + 3];
    for (r = 0; r < rows; r++) {
        if (NF > 0 && fread(x, sizeof x[0], NF, stdin) != (size_t)NF) return 2;
        out[2] = @NAME@_predict(x, NULL);
        @NAME@_comparisons = 0;
        out[0] = @NAME@_predict(x, out + 3);
        out[1] = (int32_t)@NAME@_comparisons;
        if (fwrite(out, sizeof out[0], NM + 3, stdout) != (size_t)(NM + 3)) return 3;
    }
    return 0;
}
"""


def compile_host(exp, workdir: str, cc: str = "cc", extra: Optional[List[str]] = None) -> str:
    os.makedirs(workdir, exist_ok=True)
    name = exp.generated.name
    exp.write(workdir)
    drv = os.path.join(workdir, "driver.c")
    with open(drv, "w") as f:
        f.write(DRIVER.replace("@NAME@", name).replace("@UNAME@", name.upper())
                .replace("@CTYPE@", INPUT_KINDS[exp.options.input].ctype))
    exe = os.path.join(workdir, "run")
    cmd = [cc] + HOST_CFLAGS + (extra or []) + [f"-D{name.upper()}_COUNT_COMPARISONS", "-I", workdir,
                                                 os.path.join(workdir, name + ".c"), drv, "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("host compile failed:\n" + " ".join(cmd) + "\n" + r.stderr)
    if r.stderr.strip():
        raise RuntimeError("host compile produced diagnostics:\n" + r.stderr)
    return exe


def run_host(exe: str, inputs: np.ndarray, n_margins: int):
    n = inputs.shape[0]
    r = subprocess.run([exe, str(n)], input=inputs.tobytes(), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"host run failed (exit {r.returncode}):\n{r.stderr.decode(errors='replace')[-3000:]}")
    out = np.frombuffer(r.stdout, dtype=np.int32).reshape(n, n_margins + 3)
    return out[:, 0].astype(np.int64), out[:, 3:].astype(np.int64), out[:, 1].astype(np.int64), out[:, 2].astype(np.int64)


def rows_for_input(X: np.ndarray, input_kind: str) -> np.ndarray:
    """Indices of the test rows representable in the given device input type."""
    kind = INPUT_KINDS[input_kind]
    if kind.is_float:
        return np.arange(X.shape[0])
    with np.errstate(invalid="ignore"):
        ok = ~np.isnan(X).any(axis=1) & (X == np.rint(X)).all(axis=1) \
            & (X >= kind.lo).all(axis=1) & (X <= kind.hi).all(axis=1)
    return np.nonzero(ok)[0]


def variant_label(opts: Dict) -> str:
    if not opts:
        return "auto"
    return ",".join(f"{k.replace('min_', '').replace('_bytes', 'B').replace('leaf_', 'l').replace('thresholds', 'thr')}={v}"
                    for k, v in opts.items())


def check_variant(tr: zoo.Trained, model, opts: Dict, workdir: str, cross_cc: Optional[str]) -> Dict:
    """Export one variant and verify it.  Returns a result row; 'failures' lists violated requirements."""
    fails: List[str] = []
    options = ExportOptions(name="m", **opts)
    sel = rows_for_input(tr.X, options.input)
    res: Dict = {"model": tr.spec.name, "variant": variant_label(opts), "rows": int(sel.size), "failures": fails}
    if sel.size == 0:
        fails.append("no test rows are representable in this input type")
        return res
    X = tr.X[sel]
    xgb_class, xgb_margin = tr.xgb_class[sel], tr.xgb_margin[sel].astype(np.float64)

    exp = export_model(model, options, validation=X)
    ev, lp, rep = exp.evaluation, exp.lplan, exp.report
    exe = compile_host(exp, workdir)
    c_class, c_margin, c_cmp, c_class_null = run_host(exe, exp.device_inputs(X), model.n_groups)

    # C == exact integer simulation
    if not np.array_equal(c_margin, ev.integer):
        bad = np.nonzero((c_margin != ev.integer).any(axis=1))[0]
        fails.append(f"C margins differ from the integer simulation on {bad.size} rows (first: row {sel[bad[0]]})")
    if not np.array_equal(c_class, ev.int_class):
        fails.append("C class differs from the integer simulation")
    if not np.array_equal(c_class, c_class_null):
        fails.append("predict(x, NULL) returns a different class than predict(x, margins)")

    scale = 2.0 ** -lp.shift
    sh = ev.shared
    # the tool's own bound: integer vs exact
    q_err = c_margin * scale - sh.exact
    if (q_err > lp.err_hi[None, :]).any() or (q_err < lp.err_lo[None, :]).any():
        fails.append("integer margin error exceeds the reported worst-case bound")
    # xgboost's float32 margins vs exact: inside the float-noise bound
    f_err = np.abs(xgb_margin - sh.exact)
    if (f_err > sh.float_noise).any():
        i = int(np.argmax((f_err - sh.float_noise).max(axis=1)))
        fails.append(f"xgboost margin differs from the exact margin by more than the noise bound "
                     f"(row {sel[i]}: {f_err[i].max():.3g} > {sh.float_noise[i].max():.3g})")
    # C vs xgboost margins: inside the sum of both bounds
    tot = np.abs(c_margin * scale - xgb_margin)
    if (tot > lp.worst_error[None, :] + sh.float_noise).any():
        fails.append("C margin differs from xgboost's margin by more than the reported bounds")

    # classes
    differ = c_class != xgb_class
    unflagged = differ & ~ev.at_risk
    if unflagged.any():
        fails.append(f"{int(unflagged.sum())} rows differ from xgboost without having been flagged "
                     f"(first: row {sel[np.nonzero(unflagged)[0][0]]})")
    # the run-time certainty check: a margin gap above SAFE_GAP must always mean xgboost's class
    from xgb_edge.analysis import integer_gap
    certain = integer_gap(c_margin, model.is_binary) > exp.safe_gap
    if (differ & certain).any():
        fails.append(f"{int((differ & certain).sum())} rows differ from xgboost although their margin gap "
                     f"exceeds SAFE_GAP ({exp.safe_gap})")
    if f"_SAFE_GAP {exp.safe_gap}L" not in exp.generated.header:
        fails.append("SAFE_GAP macro missing from the header")
    res["uncertain"] = int((~certain).sum())
    # where xgboost's own rounding cannot matter, xgboost must agree with exact arithmetic
    if (xgb_class[~sh.float_risk] != sh.ref_class[~sh.float_risk]).any():
        fails.append("xgboost disagrees with exact arithmetic on a row not flagged for float32 noise")

    # comparisons
    cw = rep["comparisons_worst_case"]
    if c_cmp.size and int(c_cmp.max()) > cw["total"]:
        fails.append(f"measured {int(c_cmp.max())} comparisons, more than the reported worst case {cw['total']}")
    floor = sh.path_len + cw["argmax"]
    if (c_cmp < floor).any():
        fails.append("fewer comparisons counted than tree splits visited")

    # cross-compile with -Wall -Wextra -Werror for the nRF52840 (Cortex-M4) and the ESP32s
    # (Xtensa and RISC-V): no warnings, and .rodata equals the reported data size
    if cross_cc:
        for target, (cc, flags) in available_targets().items():
            if target == "avr":
                continue   # 8-bit AVR is checked separately, on models small enough to exist there
            try:
                m = measure(exp.generated.header, exp.generated.source, "m", cc, flags)
                res[f"code_{target}"] = m["code_bytes"]
                res[f"stack_{target}"] = m["stack_bytes"]
                # .rodata must equal the emitted tables, except that a compiler may fold tables of a
                # few elements into code (it unrolls the tiny loops that read them)
                tiny = sum(a["bytes"] for a in rep["flash"]["arrays"] if a["count"] <= 8)
                missing = rep["flash"]["data_bytes"] - m["rodata_bytes"]
                if missing < 0 or missing > tiny:
                    fails.append(f"{target}: reported data size {rep['flash']['data_bytes']} != .rodata {m['rodata_bytes']}")
                elif missing:
                    res["folded_" + target] = missing
                if m["static_ram_bytes"] != 0:
                    fails.append(f"{target}: generated code uses {m['static_ram_bytes']} bytes of static RAM")
            except RuntimeError as e:
                fails.append(f"{target} compile failed or warned: " + str(e)[-400:])

    held = sel < tr.n_heldout
    res.update({
        "leaf_bits": lp.bits, "shift": lp.shift, "layout": f"{exp.packed.hdr_bytes}{exp.packed.code_bytes}"
        f"{exp.packed.off_bytes}{exp.packed.leaf_bytes}{exp.packed.leaf_mode[0]}{exp.tplan.mode[0]}",
        "data_bytes": rep["flash"]["data_bytes"],
        "bytes_per_split": rep["layout"]["bytes_per_split"],
        "at_risk": int(ev.at_risk.sum()), "at_risk_quant": int(ev.quant_risk.sum()),
        "at_risk_float": int(sh.float_risk.sum()),
        "differ": int(differ.sum()), "differ_heldout": int((differ & held).sum()),
        "differ_unflagged": int(unflagged.sum()),
        "worst_error": float(lp.worst_error.max()), "seen_error": float(np.abs(q_err).max()),
        "cmp_worst": cw["total"], "cmp_seen": int(c_cmp.max()) if c_cmp.size else 0,
    })
    return res


def check_synthetic(model_path: str, X: np.ndarray, opts: Dict, workdir: str, cross: bool = False) -> Dict:
    """Run the full set of parity checks on an arbitrary model file and feature rows."""
    from harness.synth import xgb_reference
    X = np.ascontiguousarray(X, dtype=np.float32)
    cls, margin, leaf = xgb_reference(model_path, X)
    model = load(model_path)
    head: List[str] = []
    ref_leaf = leaf_indices(model, X)
    if not np.array_equal(ref_leaf, leaf):
        bad = np.argwhere(ref_leaf != leaf)
        head.append(f"float32 reference traversal differs from xgboost pred_leaf at {bad.shape[0]} "
                    f"(row, tree) pairs, first {tuple(bad[0])}")
    spec = zoo.Spec(os.path.basename(model_path), {}, {}, 0)
    tr = zoo.Trained(spec, workdir, model_path, X, X.shape[0], cls, margin, leaf.astype(np.int32), model.objective)
    r = check_variant(tr, model, opts, workdir, find_cross_compiler() if cross else None)
    r["failures"] = head + r["failures"]
    return r


def check_spec(name: str, only_variant: Optional[int] = None) -> List[Dict]:
    spec = zoo.by_name(name)
    t0 = time.time()
    try:
        tr = zoo.train(spec)
        model = load(tr.model_path)
    except Exception:
        return [{"model": name, "variant": "-", "failures": ["training/loading failed:\n" + traceback.format_exc()]}]
    out: List[Dict] = []
    # A. reference traversal == xgboost on every row and every tree
    ref_leaf = leaf_indices(model, tr.X)
    head: List[str] = []
    if not np.array_equal(ref_leaf, tr.xgb_leaf):
        bad = np.argwhere(ref_leaf != tr.xgb_leaf)
        head.append(f"float32 reference traversal differs from xgboost pred_leaf at {bad.shape[0]} "
                    f"(row, tree) pairs, first {tuple(bad[0])}")
    cross = find_cross_compiler()
    for vi, opts in enumerate(spec.exports):
        if only_variant is not None and vi != only_variant:
            continue
        wd = os.path.join(BUILD, name, f"v{vi}")
        try:
            r = check_variant(tr, model, opts, wd, cross)
        except Exception:
            r = {"model": name, "variant": variant_label(opts), "failures": ["exception:\n" + traceback.format_exc()]}
        r["failures"] = head + r["failures"]
        r["splits"] = model.n_internal
        r["trees"] = len(model.trees)
        r["classes"] = model.n_classes
        r["rows_total"] = int(tr.X.shape[0])
        out.append(r)
    for r in out:
        r["seconds"] = round(time.time() - t0, 1)
    return out


def print_table(rows: List[Dict]) -> None:
    cols = [("model", 28), ("variant", 30), ("K", 3), ("trees", 6), ("splits", 8), ("rows", 6), ("bits", 4),
            ("layout", 7), ("data B", 9), ("B/split", 7), ("risk", 5), ("differ", 6), ("unflagged", 9), ("result", 6)]
    print(" ".join(f"{c:<{w}}" if i < 2 else f"{c:>{w}}" for i, (c, w) in enumerate(cols)))
    for r in rows:
        ok = not r["failures"]
        vals = [r["model"][:28], r["variant"][:30], r.get("classes", ""), r.get("trees", ""), r.get("splits", ""),
                r.get("rows", ""), r.get("leaf_bits", ""), r.get("layout", ""), r.get("data_bytes", ""),
                f"{r['bytes_per_split']:.2f}" if "bytes_per_split" in r else "", r.get("at_risk", ""),
                r.get("differ", ""), r.get("differ_unflagged", ""), "ok" if ok else "FAIL"]
        print(" ".join(f"{str(v):<{w}}" if i < 2 else f"{str(v):>{w}}" for i, (v, (_, w)) in enumerate(zip(vals, cols))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="zoo model names (default: all)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for s in zoo.SPECS:
            print(f"{s.name:<30} {len(s.exports):>2} variants  {s.note}")
        return 0
    names = args.models or [s.name for s in zoo.SPECS]
    for n in names:
        zoo.by_name(n)
    t0 = time.time()
    rows: List[Dict] = []
    if args.jobs > 1 and len(names) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for part in ex.map(check_spec, names):
                rows.extend(part)
    else:
        for n in names:
            rows.extend(check_spec(n))
    print_table(rows)
    failed = [r for r in rows if r["failures"]]
    for r in failed:
        print(f"\nFAIL {r['model']} [{r['variant']}]")
        for f in r["failures"]:
            print("  - " + f.replace("\n", "\n    "))
    total_rows = sum(r.get("rows", 0) for r in rows)
    flagged = sum(r.get("at_risk", 0) for r in rows)
    differ = sum(r.get("differ", 0) for r in rows)
    print(f"\n{len(rows)} exports of {len(names)} models, {total_rows:,} row checks in {time.time() - t0:.0f}s: "
          f"{len(rows) - len(failed)} ok, {len(failed)} failed; "
          f"{flagged} rows flagged at risk in advance, {differ} of those actually differ from xgboost, "
          f"{sum(r.get('differ_unflagged', 0) for r in rows)} unflagged differences")
    targets = [t for t in available_targets() if t != "avr"]
    print(f"each export also compiled with -Wall -Wextra -Werror for: host (clang, + -Wpedantic -Wconversion), "
          f"{', '.join(targets) if targets else 'no cross compilers found'}")
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
