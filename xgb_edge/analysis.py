"""Evaluate a model on sample data and decide which predictions are provably safe.

For every sample we compute:

* ``exact``  - the real-valued margin XGBoost approximates in float32:
               base margin + sum of (leaf value * tree weight), as a
               compensated float64 sum (accurate to about 1e-16 relative to
               the sum of magnitudes; ``f64_slack`` bounds what is left and
               is added to every comparison below);
* ``integer`` - the fixed-point margin the generated C computes (exactly).

Two independent things can make the predicted class differ from XGBoost's:

1. our leaf rounding (|integer / 2**shift - exact| is bounded per class by the
   LeafPlan), and
2. XGBoost's own float32 rounding while it sums the leaves (bounded per sample
   below) plus the way its float32 softmax / sigmoid can turn a hair's-breadth
   margin gap into a tie.

A sample is "at risk" when the gap between the winning class and the best
other class is not larger than those bounds.  A sample that is *not* at risk
is guaranteed to get the same class from the generated C as from XGBoost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .model import Model
from .quantize import LeafPlan, leaf_products
from .refsim import as_feature_matrix, tree_leaves

U32 = 2.0 ** -24                 # float32 unit roundoff
DECISION_SLACK = 2.0 ** -21      # margin gap that XGBoost's float32 softmax/sigmoid always resolves


@dataclass
class Shared:
    """Quantization-independent facts about each sample."""

    exact: np.ndarray        # float64 (n, groups): exact margins
    float_noise: np.ndarray  # float64 (n, groups): bound on |XGBoost float32 margin - exact|
    f64_slack: np.ndarray    # float64 (n, groups): bound on the float64 error of ``exact`` itself
    ref_class: np.ndarray    # int64 (n,): class implied by the exact margins
    float_risk: np.ndarray   # bool (n,): XGBoost's own rounding could change its class
    path_len: np.ndarray     # int64 (n,): split comparisons made across all stored trees


@dataclass
class Evaluation:
    shared: Shared
    integer: np.ndarray      # int64 (n, groups): margins the generated C returns
    int_class: np.ndarray    # int64 (n,): class the generated C returns
    quant_risk: np.ndarray   # bool (n,): leaf rounding could change the class

    @property
    def at_risk(self) -> np.ndarray:
        return self.quant_risk | self.shared.float_risk

    @property
    def n(self) -> int:
        return int(self.int_class.size)

    @property
    def differs_from_exact(self) -> np.ndarray:
        return self.int_class != self.shared.ref_class


def _dart_inplace_terms(abs_w: float, abs_base: float, abs_p):
    """Extra rounding of one DART tree on XGBoost's in-place prediction path, in units of u.

    Up to XGBoost 3.0, ``inplace_predict`` (which ``XGBClassifier.predict``
    uses) evaluates a DART tree as ``(fl(base + leaf) - base) * weight``: the
    leaf is first rounded at the magnitude of the base score.  With
    p = leaf * weight that adds at most u * |weight| * (|base| + 2 * |leaf|)
    = u * (|weight| * |base| + 2 * |p|) to the tree's contribution.
    ``Booster.predict(DMatrix)`` does not do this, but a model file does not
    say which path will be used, so every DART model gets the wider bound.
    """
    return 1.001 * (abs_w * abs_base + 2.0 * abs_p)


def _classes(margins: np.ndarray, binary: bool) -> np.ndarray:
    if binary:
        return (margins[:, 0] > 0).astype(np.int64)
    return np.argmax(margins, axis=1).astype(np.int64)  # first maximum, like the C code


def evaluate(model: Model, plans: Sequence[LeafPlan], X, chunk: int = 4096) -> List[Evaluation]:
    """Evaluate ``model`` on ``X`` for one or more leaf plans in a single pass."""
    X = as_feature_matrix(X, model.n_features)
    n, G = X.shape[0], model.n_groups
    prods = leaf_products(model)
    base = model.base_margin.astype(np.float64)
    dart = model.booster == "dart"
    exact = np.zeros((n, G))
    noise = np.zeros((n, G))
    slack64 = np.zeros((n, G))
    path_len = np.zeros(n, dtype=np.int64)
    integer = [np.zeros((n, G), dtype=np.int64) for _ in plans]

    for s in range(0, n, chunk):
        Xc = X[s:s + chunk]
        m = Xc.shape[0]
        prefix = np.zeros((m, G))    # running sum of tree outputs, in model (= XGBoost) order
        comp = np.zeros((m, G))      # Neumaier compensation for that sum
        acc_noise = np.zeros((m, G))
        abs_sum = np.zeros((m, G))
        ints = [np.tile(np.asarray(p.bias, dtype=np.int64), (m, 1)) for p in plans]
        plen = np.zeros(m, dtype=np.int64)
        for j, tree in enumerate(model.trees):
            g = tree.group
            leaf = tree_leaves(tree, Xc)
            v = prods[j][leaf]
            # compensated running sum
            t = prefix[:, g] + v
            big = np.abs(prefix[:, g]) >= np.abs(v)
            comp[:, g] += np.where(big, (prefix[:, g] - t) + v, (v - t) + prefix[:, g])
            prefix[:, g] = t
            cur = prefix[:, g] + comp[:, g]
            # XGBoost adds the trees one by one in float32, starting from the base margin
            # (CPU) or adding it last (GPU); each addition rounds by <= u * |partial sum|,
            # and a DART weight multiplication by <= u * |v|.
            acc_noise[:, g] += np.maximum(np.abs(cur + base[g]), np.abs(cur)) + np.abs(v)
            if dart:
                acc_noise[:, g] += _dart_inplace_terms(abs(tree.weight), abs(base[g]), np.abs(v))
            abs_sum[:, g] += np.abs(v)
            if tree.n_internal:
                plen += tree.node_depth[leaf]
                for k, p in enumerate(plans):
                    ints[k][:, g] += p.q[j][leaf]
        total = prefix + comp + base
        ulp_base = np.array([float(np.spacing(np.float32(abs(b)))) for b in model.base_margin])
        f64 = 1e-15 * (abs_sum + np.abs(base)) + 1e-300   # float64 error of `total` itself
        nz = 1.01 * U32 * (acc_noise + np.abs(total)) \
            + model.base_margin_ulps * ulp_base \
            + f64 + len(model.trees) * 2.0 ** -149
        exact[s:s + m] = total
        noise[s:s + m] = nz
        slack64[s:s + m] = f64
        path_len[s:s + m] = plen
        for k in range(len(plans)):
            integer[k][s:s + m] = ints[k]

    binary = model.is_binary
    ref_class = _classes(exact, binary)
    rows = np.arange(n)
    if binary:
        r, nz = exact[:, 0], noise[:, 0]
        float_safe = np.where(r > 0, r > nz + DECISION_SLACK, r < -nz)
    else:
        win = exact[rows, ref_class]
        gap = win[:, None] - exact                       # >= 0; 0 for the winner itself
        need = noise[rows, ref_class][:, None] + noise + DECISION_SLACK
        ok = gap > need
        ok[rows, ref_class] = True
        float_safe = ok.all(axis=1)
    shared = Shared(exact=exact, float_noise=noise, f64_slack=slack64, ref_class=ref_class,
                    float_risk=~float_safe, path_len=path_len)

    out = []
    for k, p in enumerate(plans):
        if binary:
            r, sl = exact[:, 0], slack64[:, 0]
            quant_safe = np.where(r > 0, r + p.err_lo[0] > sl, r + p.err_hi[0] < -sl)
        else:
            # integer_a - integer_b >= 2**shift * (gap + err_lo[a] - err_hi[b])
            need = p.err_hi[None, :] - p.err_lo[ref_class][:, None] \
                + slack64 + slack64[rows, ref_class][:, None]
            ok = gap > need
            ok[rows, ref_class] = True
            quant_safe = ok.all(axis=1)
        out.append(Evaluation(shared=shared, integer=integer[k],
                              int_class=_classes(integer[k], binary), quant_risk=~quant_safe))
    return out


def static_float_noise(model: Model) -> np.ndarray:
    """Per group: a bound, valid for *every* input, on |XGBoost's float32 margin - exact margin|.

    Same reasoning as the per-sample bound in :func:`evaluate`, but with each
    partial sum replaced by the largest magnitude it can have for any path
    through the trees.  Looser than the per-sample bound, but it needs no data.
    """
    G = model.n_groups
    base = model.base_margin.astype(np.float64)
    prods = leaf_products(model)
    hi = np.zeros(G)
    lo = np.zeros(G)
    acc = np.zeros(G)
    abs_sum = np.zeros(G)
    for t, p in zip(model.trees, prods):
        g = t.group
        leaves = p[t.leaf_nodes]
        pmax, pmin = float(leaves.max()), float(leaves.min())
        hi[g] += pmax
        lo[g] += pmin
        biggest = max(abs(hi[g] + base[g]), abs(lo[g] + base[g]), abs(hi[g]), abs(lo[g]))
        acc[g] += biggest + max(abs(pmax), abs(pmin))
        if model.booster == "dart":
            acc[g] += _dart_inplace_terms(abs(t.weight), abs(base[g]), max(abs(pmax), abs(pmin)))
        abs_sum[g] += max(abs(pmax), abs(pmin))
    total = np.maximum(np.abs(hi + base), np.abs(lo + base))
    ulp_base = np.array([float(np.spacing(np.float32(abs(b)))) for b in model.base_margin])
    return 1.01 * U32 * (acc + total) + model.base_margin_ulps * ulp_base \
        + 1e-15 * (abs_sum + np.abs(base)) + len(model.trees) * 2.0 ** -149


def safe_gap(model: Model, plan: LeafPlan) -> int:
    """Fixed-point margin gap above which the integer class provably equals XGBoost's, for any input.

    If the winning integer margin exceeds every other one by more than this
    (binary: if |margin| exceeds it), then the exact margins are far enough
    apart that neither our leaf rounding nor XGBoost's own float32 rounding
    can change the winner.
    """
    noise = static_float_noise(model)
    if model.is_binary:
        need = max(float(plan.err_hi[0]), -float(plan.err_lo[0]), 0.0) + float(noise[0]) + DECISION_SLACK
    else:
        need = float(plan.err_hi.max() - plan.err_lo.min()) + 2.0 * float(noise.max()) + DECISION_SLACK
    g = math.ceil(need * math.ldexp(1.0, plan.shift) * (1.0 + 1e-9)) + 1
    return int(min(g, 2 ** 31 - 1))


def integer_gap(integer: np.ndarray, binary: bool) -> np.ndarray:
    """What a caller compares with SAFE_GAP: winning margin minus the runner-up (binary: |margin|)."""
    if binary:
        return np.abs(integer[:, 0])
    part = np.partition(integer, -2, axis=1)
    return part[:, -1] - part[:, -2]


def margin_gap(exact: np.ndarray, binary: bool) -> np.ndarray:
    """Distance of each sample from a tie, in margin units."""
    if binary:
        return np.abs(exact[:, 0])
    part = np.partition(exact, -2, axis=1)
    return part[:, -1] - part[:, -2]
