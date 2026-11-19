"""Print the result tables used in the README from the JSON files the harnesses write.

    python -m harness.results            # all tables, as Markdown
    python -m harness.results device     # one of: parity, device, sizes, esp32, oldver
    python -m harness.results --update-readme   # rewrite the tables in README.md
    python -m harness.results --check           # are the results complete, current and what the README shows?

Nothing is measured here; every number comes from
  build/parity/summary.json        (python -m harness.parity)
  build/device/results/*.json      (python -m harness.device ...)
  build/bench/sizes.json           (python -m bench.sizes)
  build/packages/results.json      (python -m harness.build_packages)
  build/oldver/summary.json        (python -m harness.oldver)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Incomplete(Exception):
    """A result file is missing, partial, or was produced by different exporter code."""


#: problems found while building the tables (filled by the table functions)
PROBLEMS: List[str] = []


def _stamp() -> str:
    from harness import zoo
    return zoo.code_stamp()


def _check_fresh(rows: List[Dict], what: str, kind: str = "") -> None:
    from harness import zoo
    now = zoo.code_stamp(kind)
    stale = sorted({str(r.get("code")) for r in rows if r.get("code") != now})
    if stale:
        PROBLEMS.append(f"{what}: results were produced by other code (stamp {', '.join(stale)}, "
                        f"current {now}); re-run it")


def _load(rel: str):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _n(v) -> str:
    return "-" if v is None else f"{v:,}"


def _table(head: List[str], rows: List[List[str]], right: Optional[List[int]] = None) -> str:
    right = right if right is not None else list(range(1, len(head)))
    sep = ["---:" if i in right else "---" for i in range(len(head))]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def parity() -> str:
    rows = _load("build/parity/summary.json")
    if not rows:
        return "(no parity results: run python -m harness.parity)"
    from harness import zoo
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    missing = [s.name for s in zoo.SPECS if s.name not in by]
    short = [s.name for s in zoo.SPECS if s.name in by and len(by[s.name]) != len(s.exports)]
    if missing or short:
        PROBLEMS.append(f"parity: no complete result for {len(missing) + len(short)} zoo models "
                        f"({', '.join((missing + short)[:5])}{', ...' if len(missing + short) > 5 else ''}); "
                        "run python -m harness.parity")
    _check_fresh(rows, "parity")
    out = []
    for m, rs in by.items():
        ok = all(not r["failures"] for r in rs)
        out.append([f"`{m}`", rs[0].get("classes", ""), _n(rs[0].get("trees")), _n(rs[0].get("splits")),
                    len(rs), _n(sum(r.get("rows", 0) for r in rs)), sum(r.get("at_risk", 0) for r in rs),
                    sum(r.get("differ", 0) for r in rs), sum(r.get("differ_unflagged", 0) for r in rs),
                    "pass" if ok else "FAIL"])
    tot = [f"**{len(by)} models**", "", "", "", len(rows), _n(sum(r.get("rows", 0) for r in rows)),
           sum(r.get("at_risk", 0) for r in rows), sum(r.get("differ", 0) for r in rows),
           sum(r.get("differ_unflagged", 0) for r in rows),
           "pass" if all(not r["failures"] for r in rows) else "FAIL"]
    return _table(["model", "classes", "trees", "splits", "exports", "row checks", "flagged in advance",
                   "differ (all flagged)", "differ, not flagged", "result"], out + [tot])


def _device_results() -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "build", "device", "results", "*_Os.json"))):
        with open(path) as f:
            r = json.load(f)
        if r["kind"] == "xgbe" and "__xgbe0" not in os.path.basename(path):
            continue
        out.setdefault(r["model"], {})[r["kind"]] = r
    return out


def device() -> str:
    from harness import zoo
    res = _device_results()
    if not res:
        return "(no device results: run python -m harness.device suite)"
    order = [s.name for s in zoo.SPECS if s.name in res]
    missing = [s.name for s in zoo.SPECS if s.board and not res.get(s.name, {}).get("xgbe")]
    if missing:
        PROBLEMS.append(f"device: no on-board result for {', '.join(missing)}; run python -m harness.device suite")
    _check_fresh([r for m in res.values() for r in m.values()], "device", "device")
    rows = []
    for m in order:
        x, f = res[m].get("xgbe"), res[m].get("float")
        if not x:
            continue
        t = x["tool_report"]
        if f and (f.get("rows") != x["rows"] or f.get("row_set") != x.get("row_set")):
            PROBLEMS.append(f"device: {m}: the float baseline was run on different rows than xgb-edge")
        rows.append([
            f"`{m}`", x.get("input", "f32"), _n(x["rows"]), x["differ_from_xgboost"],
            f"{t['leaf_bits']}", _n(t["total_bytes"]), _n(x["linked_model_bytes"]),
            _n(f["linked_model_bytes"]) if f else "does not fit",
            f"{f['linked_model_bytes'] / x['linked_model_bytes']:.1f}x" if f else "-",
            f"{x['timing']['us_mean']:,.0f}", f"{f['timing']['us_mean']:,.0f}" if f else "-",
            f"{f['timing']['us_mean'] / x['timing']['us_mean']:.2f}x" if f else "-",
            _n(t["comparisons_worst_case"]), f"{x.get('stack_reported', t.get('stack_bytes'))} / {x['stack_measured']}",
            "pass" if x["ok"] else "FAIL"])
    text = _table(["model", "input", "rows sent to the board", "differ from xgboost", "leaf bits",
                   "flash reported by the tool (B)", "flash in the linked image (B)", "float baseline flash (B)",
                   "float / xgb-edge", "xgb-edge time (us)", "float baseline time (us)", "float / xgb-edge time",
                   "worst-case comparisons", "stack reported / measured (B)", "result"], rows, right=list(range(2, 15)))
    # the summary sentences are generated too, so that they cannot drift from the table
    both = [(res[m]["xgbe"], res[m]["float"]) for m in order if res[m].get("xgbe") and res[m].get("float")]
    notes = []
    for label, pick in (("float-input", lambda x: x.get("input", "f32") == "f32"),
                        ("integer-input", lambda x: x.get("input", "f32") != "f32")):
        sel = [(x, f) for x, f in both if pick(x) and x["tool_report"]["comparisons_worst_case"] >= 100]
        if not sel:
            continue
        size = [f["linked_model_bytes"] / x["linked_model_bytes"] for x, f in sel]
        time = [f["timing"]["us_mean"] / x["timing"]["us_mean"] for x, f in sel]
        one = len(sel) == 1
        notes.append(f"For the {len(sel)} {label} model{'' if one else 's'} run both ways (the toy model aside), the "
                     f"float baseline is {min(size):.1f}" + ("" if one else f" to {max(size):.1f}")
                     + f" times larger and takes {min(time):.2f}" + ("" if one else f" to {max(time):.2f}")
                     + " times as long.")
    exact = [m for m in order if res[m].get("float") and res[m]["float"].get("margins_bit_identical_to_xgboost")]
    inexact = [m for m in order if res[m].get("float") and not res[m]["float"].get("margins_bit_identical_to_xgboost")]
    if exact or inexact:
        notes.append(f"The float baseline's margins on the board were bit-identical to XGBoost's for {len(exact)} of "
                     f"{len(exact) + len(inexact)} models"
                     + (f"; not for {', '.join('`' + m + '`' for m in inexact)} (largest difference "
                        f"{max(res[m]['float']['max_margin_diff'] for m in inexact):.1e}), where the baseline folds "
                        "the DART tree weights into its leaves" if inexact else "") + ".")
    partial = [m for m in order if res[m].get("xgbe") and res[m]["xgbe"].get("rows_total", 0) > res[m]["xgbe"]["rows"]]
    if partial:
        x = res[partial[0]]["xgbe"]
        notes.append(f"`{partial[0]}` is exported with integer inputs, so it is run on the {x['rows']:,} of its "
                     f"{x['rows_total']:,} test rows that are integers (the others are non-integer or missing "
                     "values placed around thresholds); the float baseline is given the same rows.")
    return text + "\n\n" + "\n".join("* " + n for n in notes)


def sizes() -> str:
    d = _load("build/bench/sizes.json")
    if not d:
        return "(no size results: run python -m bench.sizes)"
    rows = d["results"]
    _check_fresh([r for r in rows if r["variant"] == "xgb_edge"], "sizes (python -m bench.sizes --variants xgb_edge)",
                 "sizes")
    models: List[str] = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    variants = ["xgb_edge", "float_baseline", "m2cgen", "tl2cgen", "tl2cgen_quantize", "tl2cgen_split",
                "tl2cgen_quantize_split"]
    variants = [v for v in variants if any(r["variant"] == v for r in rows)]
    names = {"xgb_edge": "xgb-edge", "float_baseline": "float32 table walk", "m2cgen": "m2cgen",
             "tl2cgen": "TL2cgen", "tl2cgen_quantize": "TL2cgen quantize=1", "tl2cgen_split": "TL2cgen, split",
             "tl2cgen_quantize_split": "TL2cgen quantize=1, split"}

    def short(status: str) -> str:
        s = status.lower()
        if "branch out of range" in s:
            return "does not assemble"
        if "memory" in s:
            return "compiler out of memory"
        if "timeout" in s or "timed out" in s:
            return "compile timeout"
        return "failed"
    out = []
    for m in models:
        line = [f"`{m}`", d.get("model_info", {}).get(m, "").split(", ")[-1].replace(" nodes", "")]
        base = next((r for r in rows if r["model"] == m and r["variant"] == "xgb_edge"), None)
        for v in variants:
            r = next((r for r in rows if r["model"] == m and r["variant"] == v), None)
            if r is None:
                line.append("-")
            elif r.get("flash_total") is None:
                line.append(short(str(r.get("status", ""))))
            elif v != "xgb_edge" and base and base.get("flash_total"):
                line.append(f"{r['flash_total']:,} ({r['flash_total'] / base['flash_total']:.1f}x)")
            else:
                line.append(f"{r['flash_total']:,}")
        out.append(line)
    text = _table(["model", "XGBoost nodes"] + [names[v] for v in variants], out)
    per_node = []
    for m in models:
        base = next((r for r in rows if r["model"] == m and r["variant"] == "xgb_edge"), None)
        nodes = d.get("model_info", {}).get(m, "").split(", ")[-1].replace(" nodes", "").replace(",", "")
        if base and base.get("flash_total") and nodes.isdigit() and int(nodes) >= 1000:
            per_node.append(base["flash_total"] / int(nodes))
    if per_node:
        text += (f"\n\nxgb-edge needs {min(per_node):.1f} to {max(per_node):.1f} bytes per XGBoost node "
                 "(splits and leaves, code included) on the models with at least 1,000 nodes.")
    return text


def esp32() -> str:
    rows = _load("build/parity/summary.json") or []
    pk = _load("build/packages/results.json") or {}
    if pk:
        _check_fresh([pk], "packages (python -m harness.build_packages)", "packages")
    else:
        PROBLEMS.append("packages: no result file; run python -m harness.build_packages")
    from harness import zoo
    board = [s.name for s in zoo.SPECS if s.board]
    out = []
    for m in board:
        r = next((r for r in rows if r["model"] == m and r["variant"] == "auto"), None)
        if not r:
            continue
        out.append([f"`{m}`", _n(r["data_bytes"]), _n(r.get("code_cortex-m4")), _n(r.get("code_esp32")),
                    _n(r.get("code_esp32c3")), _n(r["data_bytes"] + r["code_esp32"]) if r.get("code_esp32") else "-",
                    _n(r["data_bytes"] + r["code_esp32c3"]) if r.get("code_esp32c3") else "-"])
    text = _table(["model", "model data (B)", "code, Cortex-M4 (B)", "code, ESP32 Xtensa (B)",
                   "code, ESP32-C3 RISC-V (B)", "total on ESP32 (B)", "total on ESP32-C3 (B)"], out)
    if pk:
        lines = []
        for b in pk["platforms"].get("arduino", {}).get("builds", []):
            lines.append([f"Arduino `{b['fqbn']}` ({b['variant']})", b["status"], _n(b.get("program_bytes"))])
        idf = pk["platforms"].get("esp-idf", {})
        if idf:
            lines.append([f"ESP-IDF {idf.get('sdk', {}).get('esp-idf', '')} `{idf.get('target')}`", idf.get("status"),
                          _n(idf.get("app_bin_bytes"))])
        for b in pk["platforms"].get("zephyr", {}).get("builds", []):
            lines.append([f"Zephyr {pk['platforms']['zephyr'].get('sdk', {}).get('zephyr', '')} `{b['board']}`",
                          b["status"], _n(b.get("flash_used_bytes"))])
        text += "\n\n" + _table(["packaged example build", "status", "program size (B)"], lines)
    return text


def oldver() -> str:
    rows = _load("build/oldver/summary.json")
    if not rows:
        return "(no results: run python -m harness.oldver)"
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["model"].split("/")[0], []).append(r)
    _check_fresh(rows, "oldver", "oldver")
    from harness import oldver as _ov
    absent = [v for v in _ov.VERSIONS if v not in by]
    if absent:
        PROBLEMS.append(f"oldver: no result for xgboost {', '.join(absent)}; run python -m harness.oldver")
    out = [[v, len({r["model"] for r in rs}), len(rs), _n(sum(r.get("rows", 0) for r in rs)),
            sum(r.get("differ_unflagged", 0) for r in rs),
            "pass" if all(not r["failures"] for r in rs) else "FAIL"] for v, rs in sorted(by.items())]
    text = _table(["xgboost release that trained and predicted", "models", "exports", "row checks",
                   "differ, not flagged", "result"], out)
    notes = sorted({r["note"] for r in rows if r.get("note")})
    if notes:
        which = sorted({r["model"].split("/")[0] for r in rows if r.get("note")})
        text += ("\n\nEach model is compared with that release's `Booster.predict` and with its "
                 "`inplace_predict` / `XGBClassifier.predict`. One exception, on " + " and ".join(which)
                 + ": " + notes[0] + ".")
    return text


def optlevels() -> str:
    """One model at several optimisation levels, xgb-edge and the float baseline (each really re-flashed)."""
    model = "mc5_d4_r50_hist_missing"
    rows, seen = [], []
    ids = set()
    for opt in ("-Os", "-O2", "-Ofast"):
        tag = opt.replace("-", "_")
        x = _load(f"build/device/results/{model}__xgbe0{tag}.json")
        f = _load(f"build/device/results/{model}__float{tag}.json")
        if not x or not f:
            PROBLEMS.append(f"optlevels: no on-board result for {model} at {opt}; run "
                            f"python -m harness.device run {model} --opt={opt} (and with --kind float)")
            continue
        seen += [x, f]
        ids.update({x.get("build_id"), f.get("build_id")})
        rows.append([f"`{opt}`", _n(x["build"]["program_bytes"]), _n(x["linked_model_bytes"]),
                     f"{x['timing']['us_mean']:,.0f}", _n(f["linked_model_bytes"]), f"{f['timing']['us_mean']:,.0f}",
                     f"{f['timing']['us_mean'] / x['timing']['us_mean']:.2f}x"])
    _check_fresh(seen, "optlevels", "device")
    if seen and len(ids) != len(seen):
        PROBLEMS.append("optlevels: two results come from the same firmware build; re-run them")
    return _table(["compiler flag", "xgb-edge firmware (B)", "xgb-edge model (B)", "xgb-edge time (us)",
                   "float model (B)", "float time (us)", "float / xgb-edge time"], rows)


def flashlimit() -> str:
    d = _load("build/device/results/flash_limit.json")
    if not d:
        PROBLEMS.append("flashlimit: no measurement; run python -m harness.flash_limit 798720 798724")
        return "(no measurement: run python -m harness.flash_limit)"
    rows = [[_n(p["program_bytes"]), "accepted, runs" if p["accepted"] else "rejected by the bootloader"]
            for p in d["probes"]]
    return _table(["firmware size (B)", "upload over DFU"], rows)


TABLES = {"parity": parity, "device": device, "optlevels": optlevels, "flashlimit": flashlimit, "sizes": sizes,
          "esp32": esp32, "oldver": oldver}


def update_readme(path: Optional[str] = None) -> List[str]:
    """Rewrite the tables between the <!-- table:NAME --> markers of the README."""
    import re
    path = path or os.path.join(ROOT, "README.md")
    text = open(path).read()
    done = []
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- table:{name} -->\n).*?(\n<!-- /table:{name} -->)", re.S)
        if pat.search(text):
            body = fn()
            text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
            done.append(name)
    with open(path, "w") as f:
        f.write(text)
    return done


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] in ("--update-readme", "--check"):
        if args[0] == "--check":
            for fn in TABLES.values():
                fn()
        else:
            print("updated README tables: " + ", ".join(update_readme()))
        if args[0] == "--check" and not PROBLEMS:
            # the README must show exactly what the result files say
            import re
            text = open(os.path.join(ROOT, "README.md")).read()
            for name, fn in TABLES.items():
                m = re.search(rf"<!-- table:{name} -->\n(.*?)\n<!-- /table:{name} -->", text, re.S)
                if not m or m.group(1) != fn():
                    PROBLEMS.append(f"README: the '{name}' table differs from the result files; "
                                    "run python -m harness.results --update-readme")
        for msg in dict.fromkeys(PROBLEMS):
            print("PROBLEM: " + msg)
        if not PROBLEMS:
            print("result files are complete, were produced by the current exporter code, and match the README"
                  if args[0] == "--check" else "all result files are complete and current")
        return 1 if PROBLEMS else 0
    names = args or list(TABLES)
    for n in names:
        if n not in TABLES:
            print(f"unknown table {n!r}; choose from {', '.join(TABLES)}", file=sys.stderr)
            return 2
        print(f"### {n}\n\n{TABLES[n]()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
