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


#: Cross-compile targets the tool knows how to measure: name -> (compiler, flags).
TARGETS = {
    "cortex-m4": ("arm-none-eabi-gcc", DEFAULT_CFLAGS),                       # nRF52840 and friends
    "esp32": ("xtensa-esp32-elf-gcc", "-mlongcalls -Os"),                     # Xtensa ESP32
    "esp32c3": ("riscv32-esp-elf-gcc", "-march=rv32imc_zicsr_zifencei -mabi=ilp32 -Os"),  # RISC-V ESP32
    "avr": ("avr-gcc", "-mmcu=atmega328p -Os"),                               # 8-bit, 16-bit int
}

_ARDUINO_DIRS = ("~/Library/Arduino15", "~/.arduino15", "~/AppData/Local/Arduino15")


def find_compiler(exe_name: str) -> Optional[str]:
    """Locate a cross compiler on PATH or inside an Arduino installation."""
    exe = shutil.which(exe_name)
    if exe:
        return exe
    for base in _ARDUINO_DIRS:
        hits = sorted(glob.glob(os.path.expanduser(f"{base}/packages/*/tools/*/*/bin/{exe_name}")))
        if hits:
            return hits[-1]
    return None


def find_cross_compiler() -> Optional[str]:
    """Locate arm-none-eabi-gcc (the default measurement target)."""
    return find_compiler("arm-none-eabi-gcc")


def available_targets() -> Dict[str, tuple]:
    """The targets in TARGETS whose compiler is installed: name -> (compiler path, flags)."""
    out = {}
    for name, (exe, flags) in TARGETS.items():
        path = find_compiler(exe)
        if path:
            out[name] = (path, flags)
    return out


def measure(header: str, source: str, name: str, cc: str, cflags: str = DEFAULT_CFLAGS) -> Dict:
    """Compile the generated C and read code/data/stack sizes from the object file."""
    size_tool = re.sub(r"(gcc|g\+\+)(\.exe)?$", "size", cc)
    if size_tool == cc:
        # not a GCC-style name (e.g. "cc" or "clang"): fall back to a binutils-compatible size on PATH
        size_tool = shutil.which("size") or "size"
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, name + ".h"), "w") as f:
            f.write(header)
        src = os.path.join(d, name + ".c")
        with open(src, "w") as f:
            f.write(source)
        obj = os.path.join(d, name + ".o")
        cmd = [cc] + shlex.split(cflags) + ["-std=c99", "-Wall", "-Wextra", "-Werror", "-fstack-usage",
                                           "-ffunction-sections", "-fdata-sections", "-c", src, "-o", obj]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            raise RuntimeError(f"cannot run {cc}: {e}") from None
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
        try:
            r = subprocess.run([size_tool, "-A", obj], capture_output=True, text=True)
        except OSError as e:
            raise RuntimeError(f"cannot run {size_tool}: {e}") from None
        if r.returncode != 0 or "section" not in r.stdout:
            raise RuntimeError(f"{size_tool} -A cannot read the object file (need a binutils-style size tool "
                               f"for this compiler): {(r.stderr or r.stdout).strip()[:200]}")
        text = rodata = other = 0
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                sec, sz = parts[0], int(parts[1])
                # .literal: Xtensa literal pools (code); .srodata: RISC-V small read-only data
                if sec.startswith((".text", ".literal")):
                    text += sz
                elif sec.startswith((".rodata", ".srodata")):
                    rodata += sz
                elif sec.startswith((".data", ".bss", ".sdata", ".sbss")):
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
        fl["code_source"] = (f"measured for {measured['target']} with " if measured.get("target") else
                             "measured with ") + f"{measured['cc']} {measured['cflags']}"
        fl["rodata_bytes_measured"] = measured["rodata_bytes"]
        rep["ram"]["stack_bytes"] = measured["stack_bytes"]
        rep["ram"]["stack_source"] = "measured (-fstack-usage)"
        rep["ram"]["static_bytes"] = measured["static_ram_bytes"]
        if measured.get("target") == "avr":
            # classic AVR copies const data to RAM at start-up unless it is placed in PROGMEM
            rep["ram"]["static_bytes"] += measured["rodata_bytes"]
            rep.setdefault("notes", []).append(
                f"AVR: the {measured['rodata_bytes']:,} bytes of model tables are const data, which avr-gcc "
                "keeps in RAM as well as flash; they are counted as static RAM above. The generated code "
                "does not use PROGMEM.")
        # A compiler may fold a table of a few elements into code (seen for toy models);
        # the measured object is then the truth for the total.
        fl["total_bytes"] = measured["code_bytes"] + measured["rodata_bytes"]
        return
    else:
        fl["code_bytes"] = CODE_BYTES_ESTIMATE
        fl["code_source"] = "estimate: no cross compiler was run (see --target / --cc)"
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
    L.append(f"  {'model data':<54}{_n(fl['data_bytes']):>10} B  (the tables above)")
    if "code_bytes" in fl:
        folded = fl["data_bytes"] - fl.get("rodata_bytes_measured", fl["data_bytes"])
        if folded:
            L.append(f"  {'  tables the compiler folded into code':<54}{_n(-folded):>10} B")
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
    L.append(f"  certainty check for any input: a margin gap above {lf['safe_gap']} "
             f"({lf['safe_gap_margin_units']:.3g}) guarantees XGBoost's class ({rep['name'].upper()}_SAFE_GAP)")
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
        L.append(f"  samples whose margin gap is within the run-time certainty check (SAFE_GAP): {v['within_safe_gap']}")
        L.append(f"  split comparisons per inference on this data: mean {v['tree_comparisons_mean']:.1f}, "
                 f"max {v['tree_comparisons_max']}")
    else:
        L.append("Validation: none given, so the tie risk on real data was not checked")
    for n in rep.get("notes", []):
        L.append(f"note: {n}")
    return "\n".join(L) + "\n"
