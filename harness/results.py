"""Print the result tables used in the README from the JSON files the harnesses write.

    python -m harness.results            # all tables, as Markdown
    python -m harness.results device     # one of: parity, device, sizes, esp32, oldver

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
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
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
    rows = []
    for m in order:
        x, f = res[m].get("xgbe"), res[m].get("float")
        if not x:
            continue
        t = x["tool_report"]
        rows.append([
            f"`{m}`", _n(x["rows"]), x["differ_from_xgboost"],
            f"{t['leaf_bits']}", _n(t["total_bytes"]), _n(x["linked_model_bytes"]),
            _n(f["linked_model_bytes"]) if f else "does not fit",
            f"{f['linked_model_bytes'] / x['linked_model_bytes']:.1f}x" if f else "-",
            f"{x['timing']['us_mean']:,.0f}", f"{f['timing']['us_mean']:,.0f}" if f else "-",
            f"{f['timing']['us_mean'] / x['timing']['us_mean']:.2f}x" if f else "-",
            _n(t["comparisons_worst_case"]), f"{x['stack_measured']}",
            "pass" if x["ok"] else "FAIL"])
    return _table(["model", "rows sent to the board", "differ from xgboost", "leaf bits",
                   "flash reported by the tool (B)", "flash in the linked image (B)", "float baseline flash (B)",
                   "float / xgb-edge", "xgb-edge time (us)", "float baseline time (us)", "float / xgb-edge time",
                   "worst-case comparisons", "stack measured (B)", "result"], rows)


def sizes() -> str:
    d = _load("build/bench/sizes.json")
    if not d:
        return "(no size results: run python -m bench.sizes)"
    rows = d["results"]
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
    return _table(["model", "XGBoost nodes"] + [names[v] for v in variants], out)


def esp32() -> str:
    rows = _load("build/parity/summary.json") or []
    pk = _load("build/packages/results.json") or {}
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
    out = [[v, len(rs), _n(sum(r.get("rows", 0) for r in rs)), sum(r.get("differ_unflagged", 0) for r in rs),
            "pass" if all(not r["failures"] for r in rs) else "FAIL"] for v, rs in sorted(by.items())]
    return _table(["xgboost version that trained and predicted", "exports", "row checks", "differ, not flagged",
                   "result"], out)


TABLES = {"parity": parity, "device": device, "sizes": sizes, "esp32": esp32, "oldver": oldver}


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(TABLES)
    for n in names:
        if n not in TABLES:
            print(f"unknown table {n!r}; choose from {', '.join(TABLES)}", file=sys.stderr)
            return 2
        print(f"### {n}\n\n{TABLES[n]()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
