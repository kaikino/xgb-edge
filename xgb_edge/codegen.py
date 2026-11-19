"""Emit the C header and source for a packed model.

The generated source is self-contained C99: const tables plus one small
interpreter function specialised to this model's field widths.  It performs
no floating-point operation; float32 inputs are only ever read as bit
patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from . import __version__
from .model import Model, XgbEdgeError
from .pack import Packed
from .quantize import LeafPlan, ThresholdPlan

_UINT = {1: "uint8_t", 2: "uint16_t", 4: "uint32_t"}
_SINT = {1: "int8_t", 2: "int16_t", 4: "int32_t"}


@dataclass
class CArray:
    name: str
    ctype: str
    elem_bytes: int
    count: int
    what: str

    @property
    def bytes(self) -> int:
        return self.elem_bytes * self.count


@dataclass
class Generated:
    name: str
    header: str
    source: str
    arrays: List[CArray] = field(default_factory=list)
    q_bytes: int = 0             # size of the on-stack rank array

    @property
    def data_bytes(self) -> int:
        return sum(a.bytes for a in self.arrays)


def check_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise XgbEdgeError(f"model name {name!r} is not a valid C identifier")
    return name


def _array_width(v: int) -> int:
    return 1 if v < (1 << 8) else 2 if v < (1 << 16) else 4


def _fmt_array(ctype: str, name: str, values, per_line: int = 20) -> str:
    vals = [str(int(v)) for v in values]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    body = ",\n    ".join(lines)
    return f"static const {ctype} {name}[{len(vals)}] = {{\n    {body}\n}};\n"


def _rd(expr: str, width: int) -> str:
    return f"XE_RD{width}({expr})"


_RD_MACROS = {
    1: "#define XE_RD1(p) ((uint32_t)(p)[0])",
    2: "#define XE_RD2(p) ((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8))",
    3: "#define XE_RD3(p) ((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8) | ((uint32_t)(p)[2] << 16))",
    4: ("#define XE_RD4(p) ((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8) | "
        "((uint32_t)(p)[2] << 16) | ((uint32_t)(p)[3] << 24))"),
}


def _slot_writes(tp: ThresholdPlan, qtype: str, missing: bool) -> List[str]:
    """C statements storing used feature ``u``'s rank (or its missing codes) into its slot(s).

    Slot numbering (see ThresholdPlan): the default-right slot of feature u is u
    (for u < NR); its default-left slot is NR + u (u < NB) or NB + u (u >= NR).
    """
    nu, nr, nb = len(tp.used_features), tp.n_right_slots, tp.n_both
    vr = f"(({qtype}){tp.missing_right_code}u)" if missing else f"({qtype})c"
    vl = f"(({qtype})0u)" if missing else f"({qtype})c"
    if nb == 0:
        # every used feature has exactly one slot, at index u
        if nr == nu:
            return [f"q[u] = {vr};"]
        if nr == 0 or not missing:
            return [f"q[u] = {vl};"]
        return [f"q[u] = (u < {nr}u) ? {vr} : {vl};"]
    out = [f"q[u] = {vr};" if nr == nu else f"if (u < {nr}u) q[u] = {vr};"]
    if nb == nr:
        # no right-only features: the default-left slot of every u is NR + u
        out.append(f"q[{nr}u + u] = {vl};" if nb == nu or tp.n_left_only else f"if (u < {nb}u) q[{nr}u + u] = {vl};")
    elif tp.n_left_only == 0:
        out.append(f"if (u < {nb}u) q[{nr}u + u] = {vl};")
    else:
        out.append(f"if (u < {nb}u) q[{nr}u + u] = {vl}; else if (u >= {nr}u) q[{nb}u + u] = {vl};")
    return out


def generate(model: Model, tp: ThresholdPlan, lp: LeafPlan, pk: Packed, name: str,
             summary_lines: List[str] = ()) -> Generated:
    name = check_name(name)
    P = name.upper()
    kind = tp.input_kind
    flat = pk.flat
    n_trees = len(flat.stored_trees)
    n_groups = model.n_groups
    arrays: List[CArray] = []
    defs: List[str] = []

    def add(ctype: str, elem: int, suffix: str, values, what: str, per_line: int = 20) -> str:
        arr_name = f"{name}_{suffix}"
        arrays.append(CArray(arr_name, ctype, elem, len(values), what))
        defs.append(f"/* {what} */\n" + _fmt_array(ctype, arr_name, values, per_line))
        return arr_name

    has_nodes = flat.n_nodes > 0
    n_used = len(tp.used_features)
    n_cuts = tp.n_cuts
    qtype = _UINT[tp.code_bytes]
    q_bytes = tp.n_slots * tp.code_bytes if has_nodes else 0

    hdr_doc = [
        f"{name}: XGBoost {model.objective} model exported by xgb-edge {__version__}.",
        "Integer-only inference; split decisions are bit-exact with XGBoost.",
    ] + list(summary_lines)

    # ------------------------------------------------------------------ header
    in_t = kind.ctype
    if kind.is_float:
        feat_doc = ("features[i] is feature i as an IEEE-754 float32, exactly as given to XGBoost.\n"
                    " * NaN means \"missing\" and follows the model's default directions.")
    else:
        feat_doc = (f"features[i] is feature i as {in_t}. The model must have been trained on these\n"
                    " * same integer values; integer inputs cannot be missing.")
    if model.is_binary:
        ret_doc = "Returns the predicted class, 0 or 1 (1 iff the margin is > 0)."
        m_doc = (f"If margins is not NULL, margins[0] receives the summed margin (log-odds) in\n"
                 f" * fixed point: real value = margins[0] / 2^{P}_MARGIN_SHIFT.")
    else:
        ret_doc = (f"Returns the predicted class 0..{model.n_classes - 1}: the index of the largest margin "
                   "(lowest index on ties).")
        m_doc = (f"If margins is not NULL, margins[k] receives the summed margin of class k in\n"
                 f" * fixed point: real value = margins[k] / 2^{P}_MARGIN_SHIFT.")
    header = f"""/* {name}.h - generated by xgb-edge {__version__}. Do not edit.
 *
 * {(chr(10) + ' * ').join(hdr_doc)}
 */
#ifndef {P}_H
#define {P}_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define {P}_NUM_FEATURES {model.n_features}
#define {P}_NUM_CLASSES {model.n_classes}
#define {P}_NUM_MARGINS {n_groups}
#define {P}_MARGIN_SHIFT {lp.shift}

/* {feat_doc}
 * {m_doc}
 * {ret_doc}
 * Uses no floating-point operations and no static RAM; reentrant.
 */
int {name}_predict(const {in_t} *features, int32_t *margins);

#ifdef __cplusplus
}}
#endif

#endif /* {P}_H */
"""

    # ------------------------------------------------------------------ source
    src: List[str] = []
    src.append(f"/* {name}.c - generated by xgb-edge {__version__}. Do not edit. */\n")
    src.append(f'#include "{name}.h"\n')
    if kind.is_float and has_nodes and n_cuts:
        src.append("#include <string.h>\n")
    src.append(f"""
/* Test hook: define {P}_COUNT_COMPARISONS to count comparisons per inference. */
#ifdef {P}_COUNT_COMPARISONS
unsigned long {name}_comparisons;
#define XE_COUNT() ((void)({name}_comparisons++))
#else
#define XE_COUNT() ((void)0)
#endif
""")

    if not has_nodes:
        # Degenerate model: no split anywhere, the prediction is a constant.
        bias = lp.bias
        best = (1 if bias[0] > 0 else 0) if model.is_binary else int(np.argmax(bias))
        add("int32_t", 4, "bias", bias, "constant margins (the model has no splits)")
        body = f"""
int {name}_predict(const {in_t} *features, int32_t *margins)
{{
    (void)features;
    if (margins) {{
        uint32_t g;
        for (g = 0; g < {n_groups}u; g++) margins[g] = {name}_bias[g];
    }}
    return {best};
}}
"""
        src.extend(defs)
        src.append(body)
        return Generated(name, header, "".join(src), arrays, 0)

    # tables
    feat_expr = end_expr = None
    if n_cuts:
        ends = np.cumsum([c.size for c in tp.cuts])
        if n_used == 1:
            feat_expr, end_expr = f"{tp.used_features[0]}u", f"{int(ends[-1])}u"
        else:
            feat_w = _array_width(max(tp.used_features))
            feat_expr = add(_UINT[feat_w], feat_w, "feat", tp.used_features,
                            "input column of each used feature") + "[u]"
            end_w = _array_width(int(ends[-1]))
            end_expr = add(_UINT[end_w], end_w, "cut_end", ends,
                           "end of each used feature's cuts in the cut table") + "[u]"
        allcuts = np.concatenate(tp.cuts)
        add(kind.cut_ctype, kind.cut_bytes, "cuts", allcuts,
            "sorted split thresholds per feature (as integer keys)", 10 if kind.cut_bytes == 4 else 16)
    if n_groups > 1:
        bias_expr = add("int32_t", 4, "bias", lp.bias, "per-class base margin (fixed point)", 8) + "[g]"
        ge_w = pk.group_end_bytes
        add(_UINT[ge_w], ge_w, "group_end", flat.group_end, "trees are stored grouped by class: end index per class")
    else:
        bias_expr = f"{lp.bias[0]}L"
    advance = ""
    if n_trees > 1:
        ts_w = pk.tree_size_bytes
        ts_name = add(_UINT[ts_w], ts_w, "tree_size", pk.tree_sizes, "byte length of each tree in the node stream")
        advance = f"            tree += {ts_name}[t];\n"
    add("uint8_t", 1, "nodes", pk.nodes, "packed split nodes", 24)
    if pk.leaf_mode == "table":
        add(_SINT[pk.leaf_table_bytes], pk.leaf_table_bytes, "leaves", pk.leaf_table,
            "distinct leaf values (fixed point)", 12)

    used_rd = {pk.leaf_bytes}
    if pk.type_counts[0]:
        used_rd.add(pk.off_bytes)
    if pk.hdr_bytes == 2 or pk.code_bytes == 2:
        used_rd.add(2)
    src.append("\n/* little-endian field readers for the node stream */\n")
    for w in sorted(used_rd):
        src.append(_RD_MACROS[w] + "\n")
    src.append("\n")
    src.extend(d + "\n" for d in defs)

    # leaf operand -> int32 value
    lw = pk.leaf_bytes
    if pk.leaf_mode == "table":
        leaf_expr = f"(int32_t){name}_leaves[{_rd('p', lw)}]"
        leaf_decl = ""
    elif lw == 4:
        leaf_decl = "            uint32_t v;\n"
        leaf_expr = None
    else:
        sign = 1 << (8 * lw - 1)
        leaf_expr = f"(int32_t)({_rd('p', lw)} ^ 0x{sign:X}UL) - 0x{sign:X}L"
        leaf_decl = ""
    if leaf_expr is None:
        leaf_add = ("            v = XE_RD4(p);\n"
                    "            acc += (v & 0x80000000UL) ? -(int32_t)(~v) - 1 : (int32_t)v;\n")
    else:
        leaf_add = f"            acc += {leaf_expr};\n"

    # node decode
    H, C = pk.hdr_bytes, pk.code_bytes
    if H == 1:
        dec_type = "type = (uint32_t)p[0] >> 6;"
        dec_slot = "p[0] & 0x3Fu"
    else:
        dec_type = "type = (uint32_t)p[1] >> 6;"
        dec_slot = "XE_RD2(p) & 0x3FFFu"
    dec_code = f"p[{H}]" if C == 1 else f"XE_RD2(p + {H})"
    ii_branch = ""
    if pk.type_counts[0]:
        ii_branch = f"""                if (type == 0u) {{            /* two subtrees: left one follows */
                    if (!left) p += {_rd('p', pk.off_bytes)};
                    p += {pk.off_bytes};
                }} else """
    else:
        ii_branch = "                "
    walk = f"""            const uint8_t *p = tree;
{leaf_decl}{advance}            for (;;) {{
                uint32_t type;
                int left;
                {dec_type}
                left = (uint32_t)q[{dec_slot}] < (uint32_t){dec_code};
                XE_COUNT();
                p += {H + C};
{ii_branch}if (type == 3u) {{            /* two leaves */
                    if (!left) p += {lw};
                    break;
                }} else if ((type == 1u) == left) {{   /* 1: left child is the leaf, 2: right child */
                    break;
                }} else {{
                    p += {lw};
                }}
            }}
{leaf_add}"""

    # quantizer
    if n_cuts:
        search = f"""            while (lo < hi) {{
                const uint32_t mid = (lo + hi) >> 1;
                XE_COUNT();
                if ({name}_cuts[mid] <= key) lo = mid + 1u; else hi = mid;
            }}
            c = lo - start;"""
        if kind.is_float:
            miss = "\n            ".join(_slot_writes(tp, qtype, True))
            pres = "\n            ".join(_slot_writes(tp, qtype, False))
            quant = f"""    /* 1. rank each used feature among its split thresholds (integer compares on the bits) */
    for (u = 0; u < {n_used}u; u++) {{
        const uint32_t end = {end_expr};
        uint32_t bits, mag;
        memcpy(&bits, &features[{feat_expr}], sizeof bits);
        mag = bits & 0x7FFFFFFFUL;
        XE_COUNT();
        if (mag > 0x7F800000UL) {{              /* NaN: missing value */
            {miss}
        }} else {{
            const int32_t key = (bits >> 31) ? -(int32_t)mag : (int32_t)mag;
            uint32_t lo = start, hi = end, c;
{search}
            {pres}
        }}
        start = end;
    }}
"""
        else:
            pres = "\n            ".join(_slot_writes(tp, qtype, False))
            quant = f"""    /* 1. rank each used feature among its split thresholds */
    for (u = 0; u < {n_used}u; u++) {{
        const uint32_t end = {end_expr};
        {{
            const int32_t key = features[{feat_expr}];
            uint32_t lo = start, hi = end, c;
{search}
            {pres}
        }}
        start = end;
    }}
"""
        quant_decl = "    uint32_t u, start = 0;\n"
    else:
        # integer inputs whose every split is decided by the type's range alone
        quant = f"""    (void)features;
    for (u = 0; u < {tp.n_slots}u; u++) q[u] = 0;
"""
        quant_decl = "    uint32_t u;\n"

    if model.is_binary:
        body = f"""
int {name}_predict(const {in_t} *features, int32_t *margins)
{{
    {qtype} q[{tp.n_slots}];
    const uint8_t *tree = {name}_nodes;
    int32_t acc = {bias_expr};
    uint32_t t;
{quant_decl}
{quant}
    /* 2. walk every tree and sum the leaves */
    for (t = 0; t < {n_trees}u; t++) {{
        {{
{walk}        }}
    }}
    if (margins) margins[0] = acc;
    XE_COUNT();
    return acc > 0;
}}
"""
    else:
        body = f"""
int {name}_predict(const {in_t} *features, int32_t *margins)
{{
    {qtype} q[{tp.n_slots}];
    const uint8_t *tree = {name}_nodes;
    int32_t best_acc = 0;
    int best = 0;
    uint32_t g, t = 0;
{quant_decl}
{quant}
    /* 2. walk every tree (stored grouped by class), sum the leaves, keep the arg max */
    for (g = 0; g < {n_groups}u; g++) {{
        const uint32_t tend = {name}_group_end[g];
        int32_t acc = {bias_expr};
        for (; t < tend; t++) {{
{walk}        }}
        if (margins) margins[g] = acc;
        if (g == 0u) {{
            best_acc = acc;
        }} else {{
            XE_COUNT();
            if (acc > best_acc) {{ best_acc = acc; best = (int)g; }}
        }}
    }}
    return best;
}}
"""
    src.append(body)
    return Generated(name, header, "".join(src), arrays, q_bytes)
