"""Human-readable rendering of the export report, and optional compiler measurement."""
from __future__ import annotations

import glob
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

#: Flags for the reference target used when nothing else is given (Cortex-M4F, e.g. nRF52840).
DEFAULT_CFLAGS = "-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -Os"

#: Interpreter code size assumed when no cross compiler is available to measure it.
CODE_BYTES_ESTIMATE = 300


def find_cross_compiler() -> Optional[str]:
    """Locate arm-none-eabi-gcc on PATH or inside an Arduino installation."""
    exe = shutil.which("arm-none-eabi-gcc")
    if exe:
        return exe
    pats = [
        "~/Library/Arduino15/packages/*/tools/arm-none-eabi-gcc/*/bin/arm-none-eabi-gcc",
        "~/.arduino15/packages/*/tools/arm-none-eabi-gcc/*/bin/arm-none-eabi-gcc",
    ]
    for pat in pats:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if hits:
            return hits[-1]
    return None


def measure(header: str, source: str, name: str, cc: str, cflags: str = DEFAULT_CFLAGS) -> Dict:
    """Compile the generated C and read code/data/stack sizes from the object file."""
    size_tool = re.sub(r"gcc(\.exe)?$", "size", cc)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, name + ".h"), "w") as f:
            f.write(header)
        src = os.path.join(d, name + ".c")
        with open(src, "w") as f:
            f.write(source)
        obj = os.path.join(d, name + ".o")
        cmd = [cc] + shlex.split(cflags) + ["-std=c99", "-Wall", "-Wextra", "-Werror", "-fstack-usage",
                                           "-ffunction-sections", "-fdata-sections", "-c", src, "-o", obj]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
        r = subprocess.run([size_tool, "-A", obj], capture_output=True, text=True, check=True)
        text = rodata = other = 0
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                sec, sz = parts[0], int(parts[1])
                if sec.startswith(".text"):
                    text += sz
                elif sec.startswith(".rodata"):
                    rodata += sz
                elif sec.startswith((".data", ".bss")):
                    other += sz
        stack = None
        su = os.path.join(d, name + ".su")
        if os.path.exists(su):
            for line in open(su):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0].endswith(f"{name}_predict") and parts[1].isdigit():
                    stack = int(parts[1])
    return {"cc": os.path.basename(cc), "cflags": cflags, "code_bytes": text, "rodata_bytes": rodata,
            "static_ram_bytes": other, "stack_bytes": stack}


def add_flash_totals(rep: Dict, measured: Optional[Dict]) -> None:
    fl = rep["flash"]
    if measured:
        fl["code_bytes"] = measured["code_bytes"]
        fl["code_source"] = f"measured with {measured['cc']} {measured['cflags']}"
        fl["rodata_bytes_measured"] = measured["rodata_bytes"]
        rep["ram"]["stack_bytes"] = measured["stack_bytes"]
        rep["ram"]["stack_source"] = "measured (-fstack-usage)"
        rep["ram"]["static_bytes"] = measured["static_ram_bytes"]
    else:
        fl["code_bytes"] = CODE_BYTES_ESTIMATE
        fl["code_source"] = "estimate (no cross compiler found; pass --cc to measure)"
        rep["ram"]["stack_bytes"] = rep["ram"]["rank_array_bytes"] + 48
        rep["ram"]["stack_source"] = "estimate (rank array + ~48 bytes of locals)"
    fl["total_bytes"] = fl["data_bytes"] + fl["code_bytes"]


def _n(v) -> str:
    return f"{v:,}"


def render(rep: Dict) -> str:
    L: List[str] = []
    lay, fl, ram, cmp_, lf = (rep["layout"], rep["flash"], rep["ram"], rep["comparisons_worst_case"],
                             rep["leaves_fixed_point"])
    L.append(f"xgb-edge report for '{rep['name']}'")
    L.append(f"  model      : {rep['objective']}, {rep['classes']} classes, {rep['trees']} trees "
             f"({rep['booster']}, xgboost {rep['xgboost_version'] or '?'}), max depth {rep['max_depth']}")
    L.append(f"               {_n(rep['splits'])} splits + {_n(rep['leaves'])} leaves = "
             f"{_n(rep['xgboost_nodes'])} XGBoost nodes; {rep['features_used']} of {rep['features']} features used")
    if rep["trees_folded_into_bias"]:
        L.append(f"               {rep['trees_folded_into_bias']} single-leaf trees folded into the base margin")
    L.append(f"  input      : {rep['input']}")
    L.append("")
    L.append("Flash")
    for a in fl["arrays"]:
        L.append(f"  {a['name']:<28} {a['type']:>9}[{a['count']}]".ljust(56) + f"{_n(a['bytes']):>10} B  {a['what']}")
    L.append(f"  {'model data':<54}{_n(fl['data_bytes']):>10} B  (exact)")
    if "code_bytes" in fl:
        L.append(f"  {'interpreter code':<54}{_n(fl['code_bytes']):>10} B  ({fl['code_source']})")
        L.append(f"  {'total':<54}{_n(fl['total_bytes']):>10} B")
    enc = ("thresholds stored in the nodes" if lay.get("threshold_encoding") == "direct"
           else f"thresholds as ranks into {lay['cuts']} cuts")
    L.append(f"  layout: {enc}; {lay['header_bytes']}-byte header, {lay['threshold_code_bytes']}-byte threshold code, "
             f"{lay['offset_bytes']}-byte offsets, leaves {lay['leaf_store']} "
             f"({lay['leaf_operand_bytes']}-byte operand"
             + (f", {lay['leaf_table_entries']} table entries x {lay['leaf_table_entry_bytes']} B"
                if lay["leaf_store"] == "table" else "") + ")")
    L.append(f"  {lay['bytes_per_split']:.2f} bytes per split node in the stream; "
             f"{lay['bytes_per_xgboost_node']:.2f} bytes of model data per XGBoost node")
    L.append("")
    L.append("RAM")
    L.append(f"  static (.data/.bss): {ram['static_bytes']} B")
    if "stack_bytes" in ram and ram["stack_bytes"] is not None:
        L.append(f"  stack during predict: {ram['stack_bytes']} B  ({ram['stack_source']}); "
                 f"includes the {ram['rank_array_bytes']}-byte rank array")
    else:
        L.append(f"  stack during predict: rank array of {ram['rank_array_bytes']} B plus a few words of locals")
    L.append("")
    L.append("Worst-case comparisons per inference")
    if lay.get("threshold_encoding") == "direct":
        L.append(f"  feature preparation (NaN tests): {_n(cmp_['feature_ranking'])}")
    else:
        L.append(f"  feature ranking (binary search{' + NaN test' if rep['input'] == 'f32' else ''}): "
                 f"{_n(cmp_['feature_ranking'])}")
    L.append(f"  tree splits (sum of tree depths): {_n(cmp_['tree_splits'])}")
    L.append(f"  class selection: {cmp_['argmax']}")
    L.append(f"  total: {_n(cmp_['total'])}")
    L.append("")
    L.append("Leaf values")
    L.append(f"  {lf['bits']}-bit fixed point, margins = integer / 2^{lf['shift']} (LSB {lf['lsb']:.3g}); "
             f"largest |leaf| {lf['max_abs_leaf']:.6g}")
    L.append(f"  worst-case error on a class margin, over all possible inputs: {lf['worst_case_margin_error']:.3g}")
    L.append("  split thresholds: lossless (every comparison equals XGBoost's float32 x < threshold)")
    v = rep.get("validation")
    L.append("")
    if v:
        L.append(f"Validation ({_n(v['samples'])} samples)")
        L.append(f"  smallest gap between the top two class margins: {v['smallest_margin_gap']:.3g}")
        L.append(f"  samples close enough to a tie for the class to change: {v['at_risk']}")
        L.append(f"    ... within reach of leaf rounding: {v['at_risk_from_leaf_rounding']}")
        L.append(f"    ... within reach of XGBoost's own float32 rounding "
                 f"(bound up to {v['xgboost_float32_noise_bound_max']:.3g}): "
                 f"{v['at_risk_from_xgboost_float32_rounding']}")
        if v["at_risk"]:
            idx = v["at_risk_indices"]
            shown = ", ".join(str(i) for i in idx[:20]) + (", ..." if len(idx) > 20 else "")
            L.append(f"    at-risk sample rows: {shown}")
        L.append(f"  samples whose integer class differs from exact arithmetic: "
                 f"{v['class_differs_from_exact_arithmetic']}"
                 + (f" (rows {v['differs_indices'][:20]})" if v["class_differs_from_exact_arithmetic"] else ""))
        L.append("  every sample not listed as at risk is guaranteed to get XGBoost's class")
        L.append(f"  split comparisons per inference on this data: mean {v['tree_comparisons_mean']:.1f}, "
                 f"max {v['tree_comparisons_max']}")
    else:
        L.append("Validation: none given, so the tie risk on real data was not checked")
    for n in rep.get("notes", []):
        L.append(f"note: {n}")
    return "\n".join(L) + "\n"
