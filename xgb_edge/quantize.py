"""Quantization: the part of xgb-edge that has to be exactly right.

Thresholds (lossless)
---------------------
XGBoost sends a row left at a split iff ``x < t`` in float32.  For every
feature we collect the distinct thresholds the model uses on it, sort them
into ``cuts[0] < cuts[1] < ...`` and replace

* each feature value ``x`` by its rank  ``c(x) = #{ j : cuts[j] <= x }``
* each threshold ``t = cuts[j]`` by the code ``j + 1``

Then ``x < cuts[j]``  <=>  ``c(x) <= j``  <=>  ``c(x) < j + 1``: the integer
comparison ``c(x) < code`` gives the same answer as the float comparison for
*every* float32 input, including values equal to a threshold (they are
counted by ``<=`` and therefore go right, exactly like ``x < t`` being false).

On the device the rank is found by a binary search over integer "keys".  For
float32 inputs the key is the IEEE-754 bit pattern mapped to a signed integer
that orders the same way floats do (-0.0 and +0.0 share key 0); NaN is the
missing value.  For integer inputs the key is the integer itself and the cut
is the smallest integer ``n`` whose float32 value is ``>= t``.

That is the "rank" encoding.  When nearly every split has its own threshold
(models trained with tree_method="exact"), a table of cuts costs more than it
saves, so there is a second, "direct" encoding: the node stores the threshold
key itself and is compared with the feature's key (both biased to unsigned so
one unsigned compare works for every type).  For integer inputs this is the
classic picture: an int8/int16 feature is compared with an int8/int16
threshold, which is the smallest integer whose float32 value is >= t.  The
exporter picks whichever encoding makes the model smaller.

Missing values need no flag in the node: a feature gets up to two "slots"
holding its rank.  In the default-right slot a missing value is stored as the
largest code (never ``< code``), in the default-left slot as 0 (always
``< code``, codes start at 1).  A node simply reads the slot matching its own
default direction.

Leaves (fixed point, bounded error)
-----------------------------------
Leaf values become integers ``round(v * 2**shift)``.  All bookkeeping is done
in exact arithmetic so the reported worst-case error is a true bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np

from .model import Model, UnsupportedModelError, XgbEdgeError


@dataclass(frozen=True)
class InputKind:
    name: str
    ctype: str             # C type of one feature value
    lo: Optional[int]      # value range for integer inputs
    hi: Optional[int]
    cut_ctype: str         # C type of one cut
    cut_bytes: int

    @property
    def is_float(self) -> bool:
        return self.lo is None


INPUT_KINDS: Dict[str, InputKind] = {
    "f32": InputKind("f32", "float", None, None, "int32_t", 4),
    "i8": InputKind("i8", "int8_t", -(2 ** 7), 2 ** 7 - 1, "int8_t", 1),
    "u8": InputKind("u8", "uint8_t", 0, 2 ** 8 - 1, "uint8_t", 1),
    "i16": InputKind("i16", "int16_t", -(2 ** 15), 2 ** 15 - 1, "int16_t", 2),
    "u16": InputKind("u16", "uint16_t", 0, 2 ** 16 - 1, "uint16_t", 2),
    "i32": InputKind("i32", "int32_t", -(2 ** 31), 2 ** 31 - 1, "int32_t", 4),
}


def f32_key(values) -> np.ndarray:
    """Map float32 values to int64 keys with the same ordering as the floats.

    ``key(a) < key(b)  <=>  a < b`` for all non-NaN a, b, and key(-0.0) ==
    key(+0.0) == 0.  The device computes the same key with integer operations
    only.  NaN has no key (it is the missing value); callers must filter it.
    """
    v = np.ascontiguousarray(values, dtype=np.float32)
    if np.isnan(v).any():
        raise ValueError("f32_key is undefined for NaN")
    bits = v.view(np.uint32).astype(np.int64)
    mag = bits & 0x7FFFFFFF
    return np.where((bits >> 31) != 0, -mag, mag)


def int_cut(t: float, lo: int, hi: int) -> int:
    """Smallest integer ``n`` in [lo, hi] with ``float32(n) >= t``; ``hi + 1`` if none.

    Then for every integer ``n`` in [lo, hi]:  ``float32(n) < t  <=>  n < int_cut``.
    Uses the real float32 conversion (monotone), so there is no closed-form
    rounding to get wrong.
    """
    t32 = np.float32(t)
    if np.isnan(t32):
        raise ValueError("NaN threshold")

    def not_left(n: int) -> bool:
        return bool(np.float32(n) >= t32)

    if not not_left(hi):
        return hi + 1
    if not_left(lo):
        return lo
    a, b = lo, hi  # not_left(a) is False, not_left(b) is True
    while b - a > 1:
        mid = (a + b) // 2
        if not_left(mid):
            b = mid
        else:
            a = mid
    return b


@dataclass
class ThresholdPlan:
    input_kind: InputKind
    used_features: List[int]       # model feature index of each used feature, in slot order
    n_both: int                    # leading features that need both a default-right and a default-left slot
    n_right_only: int              # then features with only a default-right slot
    n_left_only: int               # then features with only a default-left slot
    cuts: List[np.ndarray]         # per used feature: sorted distinct int64 cut keys
    n_slots: int
    code_bytes: int                # 1 or 2: width of a rank / threshold code
    node_slot: List[np.ndarray]    # per tree: slot of each node (valid for internal nodes)
    node_code: List[np.ndarray]    # per tree: threshold code of each node (internal nodes)
    mode: str = "rank"             # "rank": codes are ranks in the cut tables; "direct": codes are biased keys
    const_slot: Optional[int] = None  # direct mode, integer inputs: slot that always holds 0 (for always-left splits)

    @property
    def key_bias(self) -> int:
        """Direct mode: value added to a key to make it unsigned (stored code = key + bias)."""
        if self.input_kind.is_float:
            return 1 << 31
        return -self.input_kind.lo

    @property
    def n_right_slots(self) -> int:
        return self.n_both + self.n_right_only

    @property
    def missing_right_code(self) -> int:
        return (1 << (8 * self.code_bytes)) - 1

    @property
    def n_cuts(self) -> int:
        return int(sum(c.size for c in self.cuts))

    def right_slot(self, u: int) -> Optional[int]:
        return u if u < self.n_right_slots else None

    def left_slot(self, u: int) -> Optional[int]:
        if u < self.n_both:
            return self.n_right_slots + u
        if u >= self.n_right_slots:
            return self.n_both + u
        return None


def plan_thresholds(model: Model, input_kind: str = "f32", mode: str = "rank") -> ThresholdPlan:
    if input_kind not in INPUT_KINDS:
        raise XgbEdgeError(f"unknown input kind {input_kind!r}; choose one of {', '.join(INPUT_KINDS)}")
    if mode not in ("rank", "direct"):
        raise XgbEdgeError(f"threshold mode must be rank or direct, got {mode!r}")
    kind = INPUT_KINDS[input_kind]
    direct = mode == "direct"

    # Gather every split of every tree.
    t_idx, n_idx, feats, thrs, dls = [], [], [], [], []
    for ti, tree in enumerate(model.trees):
        nodes = tree.internal_nodes
        if nodes.size == 0:
            continue
        t_idx.append(np.full(nodes.size, ti, dtype=np.int64))
        n_idx.append(nodes)
        feats.append(tree.feature[nodes])
        thrs.append(tree.threshold[nodes])
        dls.append(tree.default_left[nodes])
    node_slot = [np.full(t.left.size, -1, dtype=np.int64) for t in model.trees]
    node_code = [np.full(t.left.size, -1, dtype=np.int64) for t in model.trees]
    if not t_idx:
        return ThresholdPlan(kind, [], 0, 0, 0, [], 0, 1, node_slot, node_code, mode)
    t_idx = np.concatenate(t_idx)
    n_idx = np.concatenate(n_idx)
    feats = np.concatenate(feats)
    thrs = np.concatenate(thrs).astype(np.float32)
    dls = np.concatenate(dls)

    codes = np.zeros(feats.size, dtype=np.int64)
    feature_ids = [int(f) for f in np.unique(feats)]
    cuts_by_feature: Dict[int, np.ndarray] = {}
    max_code = 0
    always_left_node = np.zeros(feats.size, dtype=bool)   # direct mode, integer inputs
    for f in feature_ids:
        idx = np.nonzero(feats == f)[0]
        if direct:
            cuts_by_feature[f] = np.zeros(0, dtype=np.int64)
            if kind.is_float:
                # biased key: [0x00800000, 0xFF800000], so 0 and 0xFFFFFFFF are free for missing values
                codes[idx] = f32_key(thrs[idx]) + (1 << 31)
            else:
                uniq, inv = np.unique(thrs[idx], return_inverse=True)
                cut_of = np.array([int_cut(float(t), kind.lo, kind.hi) for t in uniq], dtype=np.int64)
                # go left iff n < cut  <=>  (n - lo) < (cut - lo); cut == hi + 1 does not fit the type
                always_left_node[idx] = (cut_of > kind.hi)[inv]
                codes[idx] = np.where(cut_of > kind.hi, 1, cut_of - kind.lo)[inv]
            continue
        if kind.is_float:
            keys = f32_key(thrs[idx])
            cuts = np.unique(keys)
            codes[idx] = np.searchsorted(cuts, keys) + 1  # threshold cuts[j] -> code j + 1
            top = cuts.size
        else:
            # integer cut for each distinct threshold
            uniq, inv = np.unique(thrs[idx], return_inverse=True)
            cut_of = np.array([int_cut(float(t), kind.lo, kind.hi) for t in uniq], dtype=np.int64)
            real = cut_of[(cut_of > kind.lo) & (cut_of <= kind.hi)]
            cuts = np.unique(real)
            code_of = np.empty(uniq.size, dtype=np.int64)
            always_left = False
            for k, c in enumerate(cut_of):
                if c <= kind.lo:
                    code_of[k] = 0                # no input is < lo: never left
                elif c > kind.hi:
                    code_of[k] = cuts.size + 1    # every input is below it: always left
                    always_left = True
                else:
                    code_of[k] = int(np.searchsorted(cuts, c)) + 1
            codes[idx] = code_of[inv]
            top = cuts.size + (1 if always_left else 0)
        cuts_by_feature[f] = cuts.astype(np.int64)
        max_code = max(max_code, top)

    if direct:
        code_bytes = kind.cut_bytes
    elif max_code <= 0xFF:
        code_bytes = 1
    elif max_code <= 0xFFFF:
        code_bytes = 2
    else:
        worst = max(feature_ids, key=lambda f: cuts_by_feature[f].size)
        raise UnsupportedModelError(
            f"feature {model.feature_label(worst)} has {cuts_by_feature[worst].size} distinct split "
            "thresholds; at most 65535 per feature are supported.")

    # Slots.  Float inputs can be missing (NaN), so a feature needs a slot per
    # default direction that the model actually uses on it.
    if kind.is_float:
        has_dl = {f: bool(dls[feats == f].any()) for f in feature_ids}
        has_dr = {f: bool((~dls[feats == f]).any()) for f in feature_ids}
    else:
        has_dl = {f: False for f in feature_ids}
        has_dr = {f: True for f in feature_ids}
    both = [f for f in feature_ids if has_dl[f] and has_dr[f]]
    r_only = [f for f in feature_ids if has_dr[f] and not has_dl[f]]
    l_only = [f for f in feature_ids if has_dl[f] and not has_dr[f]]
    used = both + r_only + l_only
    n_slots = 2 * len(both) + len(r_only) + len(l_only)
    const_slot = None
    if always_left_node.any():
        const_slot = n_slots   # a slot that always holds 0, compared against code 1
        n_slots += 1
    plan = ThresholdPlan(kind, used, len(both), len(r_only), len(l_only),
                         [cuts_by_feature[f] for f in used], n_slots, code_bytes,
                         node_slot, node_code, mode, const_slot)
    if plan.n_slots > 16384:
        raise UnsupportedModelError(
            f"the model needs {plan.n_slots} feature slots; at most 16384 are supported.")
    u_of = {f: u for u, f in enumerate(used)}
    r_slot = np.array([plan.right_slot(u_of[f]) if plan.right_slot(u_of[f]) is not None else -1
                       for f in feature_ids], dtype=np.int64)
    l_slot = np.array([plan.left_slot(u_of[f]) if plan.left_slot(u_of[f]) is not None else -1
                       for f in feature_ids], dtype=np.int64)
    fpos = np.searchsorted(np.asarray(feature_ids), feats)
    if kind.is_float:
        slots = np.where(dls, l_slot[fpos], r_slot[fpos])
    else:
        slots = r_slot[fpos]
    if const_slot is not None:
        slots = np.where(always_left_node, const_slot, slots)
    assert (slots >= 0).all()
    assert codes.min() >= 0 and codes.max() < (1 << (8 * code_bytes))
    if direct and kind.is_float:
        assert codes.min() > 0 and codes.max() < plan.missing_right_code
    for ti in np.unique(t_idx):
        m = t_idx == ti
        node_slot[ti][n_idx[m]] = slots[m]
        node_code[ti][n_idx[m]] = codes[m]
    return plan


# --------------------------------------------------------------------------- leaves

INT32_MAX = 2 ** 31 - 1
MAX_SHIFT = 62


@dataclass
class LeafPlan:
    bits: int                    # requested maximum signed bit width of a stored leaf
    shift: int                   # fixed-point scale: integer = round(real * 2**shift)
    q: List[np.ndarray]          # per tree: int64[n_nodes], quantized value at leaves
    constant_tree: List[bool]    # per tree: True if it is a single leaf (folded into the bias)
    bias: List[int]              # per group: base margin + folded constant trees, quantized
    err_hi: np.ndarray           # per group: max over all inputs of (integer_margin / 2**shift - exact_margin)
    err_lo: np.ndarray           # per group: min over all inputs of the same
    max_abs_acc: List[int]       # per group: bound on |accumulator| at any point
    max_abs_leaf: float          # largest |leaf value * tree weight| among stored leaves
    max_abs_q: int               # largest |quantized stored leaf|

    @property
    def lsb(self) -> float:
        return math.ldexp(1.0, -self.shift)

    @property
    def worst_error(self) -> np.ndarray:
        """Per group: largest possible |integer margin - exact margin|, in margin units."""
        return np.maximum(np.abs(self.err_hi), np.abs(self.err_lo))


def leaf_products(model: Model) -> List[np.ndarray]:
    """Exact contribution ``value * tree_weight`` of every node, as float64.

    Both factors are float32, so the float64 product is exact.
    """
    out = []
    for t in model.trees:
        p = np.zeros(t.value.size, dtype=np.float64)
        leaves = t.leaf_nodes
        p[leaves] = t.value[leaves].astype(np.float64) * np.float64(t.weight)
        out.append(p)
    return out


def _quantize_at(model: Model, prods: List[np.ndarray], const_real: List[Fraction], shift: int):
    scale = math.ldexp(1.0, shift)
    q, max_q_tree = [], []
    for t, p in zip(model.trees, prods):
        leaves = t.leaf_nodes
        qt = np.zeros(p.size, dtype=np.int64)
        qt[leaves] = np.rint(p[leaves] * scale).astype(np.int64)  # exact: power-of-two scaling, ties to even
        q.append(qt)
        max_q_tree.append(int(np.abs(qt[leaves]).max()))
    bias = [int(round(c * (1 << shift))) for c in const_real]  # Fraction: exact, ties to even
    return q, max_q_tree, bias


def plan_leaves(model: Model, bits: int = 16) -> LeafPlan:
    if not 2 <= bits <= 32:
        raise XgbEdgeError(f"leaf bits must be between 2 and 32, got {bits}")
    prods = leaf_products(model)
    constant = [t.n_internal == 0 for t in model.trees]

    # Exact real-valued constant part of each margin: base margin + single-leaf trees.
    const_real = [Fraction(float(b)) for b in model.base_margin]
    for t, c in zip(model.trees, constant):
        if c:
            const_real[t.group] += Fraction(float(t.value[0])) * Fraction(t.weight)

    stored = [np.abs(p[t.leaf_nodes]).max() for t, p, c in zip(model.trees, prods, constant) if not c]
    max_abs_leaf = float(max(stored)) if stored else 0.0
    limit = (1 << (bits - 1)) - 1

    # Largest shift for which every stored leaf fits in `bits` signed bits.
    if max_abs_leaf > 0.0:
        s = min(MAX_SHIFT, int(math.floor(math.log2(limit / max_abs_leaf))) + 1)
        while s >= 0 and np.rint(max_abs_leaf * math.ldexp(1.0, s)) > limit:
            s -= 1
        if s < 0:
            need = int(np.rint(max_abs_leaf)).bit_length() + 1
            raise UnsupportedModelError(
                f"leaf values reach {max_abs_leaf:g}, which does not fit a {bits}-bit fixed-point "
                f"leaf even with no fractional bits; use --leaf-bits {min(32, max(need, bits + 8))} "
                "or more." if need <= 32 else
                f"leaf values reach {max_abs_leaf:g}, too large for 32-bit fixed point.")
    else:
        s = 16  # no stored leaves (or all zero): only the bias needs a scale

    # ... and for which no accumulator can overflow int32.
    while True:
        q, max_q_tree, bias = _quantize_at(model, prods, const_real, s)
        acc = [abs(b) for b in bias]
        for t, c, mq in zip(model.trees, constant, max_q_tree):
            if not c:
                acc[t.group] += mq
        if max(acc) <= INT32_MAX:
            break
        s -= 1
        if s < 0:
            raise UnsupportedModelError(
                "the summed margins can exceed the int32 range even with no fractional bits "
                f"(worst-case |margin| about {max(acc):g}).")

    # Exact error bounds.  e = q / 2**s - product is exactly representable in float64.
    inv = math.ldexp(1.0, -s)
    hi_terms: List[List[float]] = [[] for _ in range(model.n_groups)]
    lo_terms: List[List[float]] = [[] for _ in range(model.n_groups)]
    for g in range(model.n_groups):
        e = Fraction(bias[g], 1 << s) - const_real[g]
        hi_terms[g].append(float(e))
        lo_terms[g].append(float(e))
    max_abs_q = 0
    for t, p, qt, c in zip(model.trees, prods, q, constant):
        if c:
            continue
        leaves = t.leaf_nodes
        e = qt[leaves] * inv - p[leaves]
        hi_terms[t.group].append(float(e.max()))
        lo_terms[t.group].append(float(e.min()))
        max_abs_q = max(max_abs_q, int(np.abs(qt[leaves]).max()))
    # Every term is exact except the bias term (correctly rounded) and fsum is
    # correctly rounded; pad outward by a few ulps so the bound stays a bound.
    def _outward(terms: List[float], sign: float) -> float:
        total = math.fsum(terms)
        pad = 8.0 * float(np.spacing(max(abs(total), max(abs(x) for x in terms), 1e-300)))
        return total + sign * pad

    err_hi = np.array([_outward(v, +1.0) for v in hi_terms])
    err_lo = np.array([_outward(v, -1.0) for v in lo_terms])

    return LeafPlan(bits=bits, shift=s, q=q, constant_tree=constant, bias=bias,
                    err_hi=err_hi, err_lo=err_lo, max_abs_acc=acc,
                    max_abs_leaf=max_abs_leaf, max_abs_q=max_abs_q)
