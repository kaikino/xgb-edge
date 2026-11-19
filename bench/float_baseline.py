"""Float32 table-walk baseline: what a careful engineer would write without xgb-edge.

The same interpreter idea as xgb-edge, but with no quantization at all: every
XGBoost node (split or leaf) becomes one 8-byte record

    { float value; uint16_t feature_and_flags; uint16_t right; }

and the nodes of each tree are stored in pre-order, so the left child of a
split is always the next record and only the right child needs an offset.
Leaves are summed in float32, starting from the base margin and in XGBoost's
tree order, so on an IEEE-754 host the margins are bit-identical to XGBoost's
own (``output_margin=True``) for gbtree models.

DART models are the one exception: XGBoost multiplies every leaf by its tree
weight at predict time (as a fused multiply-add on builds that have one),
while this generator folds the weight into the stored leaf (one float32
rounding per leaf), so DART margins can differ in the last few bits.

Every tree in the file is used, as ``Booster.predict()`` does.  For a model
trained with early stopping pass ``xgb_edge.load(path).first_rounds(n)``
instead of the path to get what ``XGBClassifier.predict()`` computes.

Usage::

    python -m bench.float_baseline model.json --name my_model --out-dir out/

writes ``out/my_model.h`` and ``out/my_model.c``.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgb_edge.codegen import check_name  # noqa: E402
from xgb_edge.model import Model, load  # noqa: E402

#: Narrow record: 15-bit feature index + default-left flag, 16-bit right-child offset.
NARROW_MAX_FEATURES = 1 << 15
NARROW_MAX_OFFSET = (1 << 16) - 1


def _uint_type(max_value: int) -> Tuple[str, int]:
    """Smallest unsigned C type that holds ``max_value``, and its size in bytes."""
    if max_value < (1 << 8):
        return "uint8_t", 1
    if max_value < (1 << 16):
        return "uint16_t", 2
    return "uint32_t", 4


def _f32_literals(values: np.ndarray) -> Tuple[List[str], bool]:
    """C float literals that convert back to exactly these float32 values.

    numpy prints a float32 as the shortest decimal that identifies it uniquely,
    so a correctly rounding compiler (GCC, clang) reads back the same bits.
    Returns the literals and whether any of them needs <math.h> (INFINITY).
    """
    values = np.asarray(values, dtype=np.float32)
    out: List[str] = []
    need_math = False
    for v in values:
        if np.isfinite(v):
            out.append(str(v) + "f")
        elif np.isnan(v):
            raise ValueError("NaN cannot be stored in the node table")
        else:
            need_math = True
            out.append("INFINITY" if v > 0 else "-INFINITY")
    # Sanity check of the formatting (not of the compiler): parse the text back.
    finite = np.isfinite(values)
    if finite.any():
        back = np.array([float(s[:-1]) for s, ok in zip(out, finite) if ok], dtype=np.float64).astype(np.float32)
        if not np.array_equal(back.view(np.uint32), values[finite].view(np.uint32)):
            raise AssertionError("float literal does not round-trip")  # pragma: no cover
    return out, need_math


def _layout(model: Model):
    """Flatten the model: trees grouped by class (XGBoost order within a class), nodes in pre-order.

    Returns (value float32[], feature int64[], default_left bool[], right int64[],
    tree_len int64[], group_end int64[], tree_ids int64[]).  ``right`` is the index
    offset of the right child from its parent, 0 for a leaf.
    """
    order_trees = sorted(range(len(model.trees)), key=lambda i: (model.trees[i].group, i))
    values, feats, dlefts, rights, tree_len = [], [], [], [], []
    for ti in order_trees:
        t = model.trees[ti]
        order = t.preorder
        n = order.size
        pos = np.full(t.left.size, -1, dtype=np.int64)
        pos[order] = np.arange(n)
        internal = t.left[order] >= 0
        at = np.nonzero(internal)[0]
        if not np.array_equal(pos[t.left[order[at]]], at + 1):
            raise AssertionError(f"tree {ti}: nodes are not in pre-order")  # pragma: no cover
        right = np.zeros(n, dtype=np.int64)
        right[at] = pos[t.right[order[at]]] - at
        value = t.threshold[order].astype(np.float32)   # threshold at splits, leaf value at leaves
        if t.weight != 1.0:
            # DART: fold the tree weight into the leaves (one float32 rounding per leaf)
            with np.errstate(over="ignore"):
                value[~internal] = (value[~internal] * np.float32(t.weight)).astype(np.float32)
        values.append(value)
        feats.append(np.where(internal, t.feature[order], 0))
        dlefts.append(internal & t.default_left[order])
        rights.append(right)
        tree_len.append(n)
    group_end = np.cumsum(np.bincount([model.trees[i].group for i in order_trees], minlength=model.n_groups))

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    return (cat(values, np.float32), cat(feats, np.int64), cat(dlefts, bool), cat(rights, np.int64),
            np.asarray(tree_len, dtype=np.int64), group_end.astype(np.int64),
            np.asarray(order_trees, dtype=np.int64))


def _int_table(ctype: str, name: str, values, per_line: int = 16) -> str:
    vals = [str(int(v)) for v in values]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f"static const {ctype} {name}[{len(vals)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n"


def generate(model_path, name: str, force_wide: bool = False) -> Tuple[str, str]:
    """Return ``(header_text, source_text)`` of the float32 baseline for an XGBoost JSON model.

    ``model_path`` is the path of a model saved with ``save_model("*.json")`` (an already
    loaded :class:`xgb_edge.model.Model` is accepted too, e.g. one cut to its best
    iteration).  ``force_wide`` is a test hook that selects the 12-byte record even when
    the 8-byte one is enough.
    """
    check_name(name)
    model = model_path if isinstance(model_path, Model) else load(model_path)
    P = name.upper()
    n_groups, n_trees = model.n_groups, len(model.trees)
    value, feat, dleft, right, tree_len, group_end, tree_ids = _layout(model)
    n_nodes = int(value.size)
    n_split = int((right != 0).sum())

    # ---- record field widths: wider than 16 bits only if this model needs it
    max_feat = int(feat.max()) if n_nodes else 0
    max_off = int(right.max()) if n_nodes else 0
    why_wide = []
    if max_feat >= NARROW_MAX_FEATURES:
        why_wide.append(f"it splits on feature {max_feat}, which does not fit in 15 bits")
    if max_off > NARROW_MAX_OFFSET:
        why_wide.append(f"a right-child offset of {max_off} records does not fit in 16 bits")
    if force_wide and not why_wide:
        why_wide.append("the wide layout was requested explicitly (test hook)")
    wide = bool(why_wide)
    field_t = "uint32_t" if wide else "uint16_t"
    rec_bytes = 12 if wide else 8
    flag = "0x80000000u" if wide else "0x8000u"
    mask = "0x7FFFFFFFu" if wide else "0x7FFFu"
    flag_bit = 31 if wide else 15

    base = model.base_margin.astype(np.float32)
    base_lit, need_math = _f32_literals(base)
    val_lit, need_math2 = _f32_literals(value)
    need_math = need_math or need_math2
    ff = feat | (dleft.astype(np.int64) << flag_bit)

    len_t, len_bytes = _uint_type(int(tree_len.max()) if n_trees else 0)
    ge_t, ge_bytes = _uint_type(n_trees)
    multiclass = n_groups > 1
    data_bytes = n_nodes * rec_bytes
    parts = [f"{n_nodes} node records x {rec_bytes} B = {n_nodes * rec_bytes}"]
    if n_trees:
        data_bytes += n_trees * len_bytes
        parts.append(f"tree lengths {n_trees} x {len_bytes} B = {n_trees * len_bytes}")
    if multiclass and n_trees:
        data_bytes += n_groups * (ge_bytes + 4)
        parts.append(f"per-class tree ranges and base margins {n_groups} x {ge_bytes + 4} B = "
                     f"{n_groups * (ge_bytes + 4)}")
    elif multiclass:
        data_bytes += n_groups * 4
        parts.append(f"base margins {n_groups} x 4 B = {n_groups * 4}")

    doc = [
        f"{model.objective}, {model.n_classes} classes, {model.n_features} features, {n_trees} trees"
        + (f" ({model.booster})" if model.booster != "gbtree" else "")
        + f", {n_nodes} nodes ({n_split} splits + {n_nodes - n_split} leaves).",
        "Plain float32 table walk, no quantization: one record per XGBoost node, each tree's",
        "nodes in pre-order (the left child of a split is the next record).",
    ]
    if wide:
        doc.append(f"Records are {rec_bytes} bytes with 32-bit fields instead of the usual 8 bytes, because "
                   + " and ".join(why_wide) + ".")
    else:
        doc.append("Records are 8 bytes: float value, 15-bit feature index + default-left flag, "
                   "16-bit right-child offset.")
    doc.append("Model data: " + "; ".join(parts) + f"; total {data_bytes} bytes of .rodata.")
    if model.booster == "dart":
        doc.append("DART: every leaf is stored already multiplied by its tree weight (rounded to float32),")
        doc.append("so margins can differ from XGBoost's in the last bits; XGBoost multiplies at predict time.")
    doc_text = "\n * ".join(doc)

    if model.is_binary:
        ret_doc = "Returns the predicted class, 0 or 1 (1 iff the margin is > 0)."
        m_doc = "If margins is not NULL, margins[0] receives the raw margin (log-odds for binary:logistic)."
    else:
        ret_doc = (f"Returns the predicted class 0..{model.n_classes - 1}: the index of the largest margin "
                   "(lowest index on ties).")
        m_doc = (f"If margins is not NULL, margins[k] receives the raw margin of class k "
                 f"(k < {P}_NUM_MARGINS).")

    header = f"""/* {name}.h - float32 table-walk baseline, generated by bench/float_baseline.py. Do not edit.
 *
 * {doc_text}
 */
#ifndef {P}_H
#define {P}_H

#ifdef __cplusplus
extern "C" {{
#endif

#define {P}_NUM_FEATURES {model.n_features}
#define {P}_NUM_CLASSES {model.n_classes}
#define {P}_NUM_MARGINS {n_groups}

/* features[i] is feature i as a float, exactly as given to XGBoost; NaN means "missing"
 * and follows the model's default directions.
 * {m_doc}
 * Margins are summed in float32 from the base margin in XGBoost's tree order.
 * {ret_doc}
 * Uses no static RAM and no dynamic memory; reentrant.  Do not build with -ffast-math
 * (the missing-value handling relies on IEEE-754 NaN comparisons).
 */
int {name}_predict(const float *features, float *margins);

#ifdef __cplusplus
}}
#endif

#endif /* {P}_H */
"""

    src: List[str] = []
    src.append(f"/* {name}.c - float32 table-walk baseline, generated by bench/float_baseline.py. Do not edit.\n"
               f" *\n * {doc_text}\n */\n")
    src.append(f'#include "{name}.h"\n\n')
    if need_math:
        src.append("#include <math.h>\n")
    src.append("#include <stdint.h>\n")

    def walk(indent: int) -> str:
        """The tree walk for tree ``t`` rooted at ``root``; adds its leaf to ``acc``."""
        text = f"""const {name}_node_t *n = root;
while (n->right != 0u) {{
    const float x = features[n->feature_and_flags & {mask}];
    /* x < threshold goes left.  A NaN compares false both ways, so testing
     * !(x >= threshold) instead sends a missing value left where the flag asks for it. */
    const int left = (n->feature_and_flags & {flag}) ? !(x >= n->value) : (x < n->value);
    n += left ? 1u : n->right;
}}
acc += n->value;
root += {name}_tree_len[t];
"""
        return "".join(" " * indent + line + "\n" for line in text.splitlines())

    if n_trees:
        src.append(f"""
typedef struct {{
    float value;                 /* split: threshold (go left iff x < value); leaf: leaf value */
    {field_t} feature_and_flags;  /* split: feature index; top bit set: a missing value goes left */
    {field_t} right;              /* split: index of the right child minus this node's index; 0: leaf */
}} {name}_node_t;

/* the tables below rely on the {rec_bytes}-byte record layout */
typedef char {name}_node_size_check[(sizeof({name}_node_t) == {rec_bytes}) ? 1 : -1];

/* all trees, {'grouped by class and in XGBoost order within a class' if multiclass else 'in XGBoost order'} */
static const {name}_node_t {name}_nodes[{n_nodes}] = {{
""")
        lines: List[str] = []
        start = 0
        per_line = 4
        ff_list, right_list = ff.tolist(), right.tolist()
        for k in range(n_trees):
            n = int(tree_len[k])
            t = model.trees[int(tree_ids[k])]
            what = f"tree {t.index}" + (f", class {t.group}" if multiclass else "") + f": {n} nodes"
            lines.append(f"    /* {what} */")
            recs = [f"{{{val_lit[i]}, {ff_list[i]}, {right_list[i]}}}" for i in range(start, start + n)]
            for i in range(0, n, per_line):
                lines.append("    " + ", ".join(recs[i:i + per_line]) + ",")
            start += n
        src.append("\n".join(lines) + "\n};\n\n")
        src.append("/* number of node records in each tree */\n")
        src.append(_int_table(len_t, f"{name}_tree_len", tree_len, 20))

    if multiclass:
        src.append(f"\n/* base margin of each class */\nstatic const float {name}_base[{n_groups}] = {{\n    "
                   + ", ".join(base_lit) + "\n};\n")
        if n_trees:
            src.append("\n/* trees of class g are [group_end[g - 1], group_end[g]) */\n")
            src.append(_int_table(ge_t, f"{name}_group_end", group_end, 16))
            src.append(f"""
int {name}_predict(const float *features, float *margins)
{{
    const {name}_node_t *root = {name}_nodes;
    float best_margin = 0.0f;
    int best = 0;
    uint32_t g, t = 0;
    for (g = 0; g < {n_groups}u; g++) {{
        float acc = {name}_base[g];
        for (; t < {name}_group_end[g]; t++) {{
{walk(12)}        }}
        if (margins) margins[g] = acc;
        if (g == 0u || acc > best_margin) {{
            best_margin = acc;
            best = (int)g;
        }}
    }}
    return best;
}}
""")
        else:
            src.append(f"""
int {name}_predict(const float *features, float *margins)
{{
    int best = 0;
    uint32_t g;
    (void)features;
    for (g = 0; g < {n_groups}u; g++) {{
        if (margins) margins[g] = {name}_base[g];
        if ({name}_base[g] > {name}_base[best]) best = (int)g;
    }}
    return best;
}}
""")
    else:
        if n_trees:
            body = f"""    const {name}_node_t *root = {name}_nodes;
    float acc = {base_lit[0]};   /* base margin */
    uint32_t t;
    for (t = 0; t < {n_trees}u; t++) {{
{walk(8)}    }}
"""
        else:
            body = f"    const float acc = {base_lit[0]};   /* base margin */\n    (void)features;\n"
        src.append(f"""
int {name}_predict(const float *features, float *margins)
{{
{body}    if (margins) margins[0] = acc;
    return acc > 0.0f;
}}
""")
    return header, "".join(src)


def write(model_path, name: str, outdir: str) -> List[str]:
    """Generate the baseline and write ``<outdir>/<name>.h`` and ``.c``; returns both paths."""
    header, source = generate(model_path, name)
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, name + ".h"), os.path.join(outdir, name + ".c")]
    for path, text in zip(paths, (header, source)):
        with open(path, "w") as f:
            f.write(text)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the float32 table-walk baseline C for an XGBoost JSON model.")
    ap.add_argument("model", help="XGBoost model saved with save_model('model.json')")
    ap.add_argument("-n", "--name", default="xgb_float", help="C identifier prefix (default: xgb_float)")
    ap.add_argument("-o", "--out-dir", default=".", help="directory for <name>.h and <name>.c")
    args = ap.parse_args(argv)
    for p in write(args.model, args.name, args.out_dir):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
