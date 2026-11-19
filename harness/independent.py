"""Checks of the tool's bounds that do not use the tool's own bound code.

harness/parity.py compares the generated C with xgboost using the bounds the
exporter reports.  That alone would also pass for a tool that reported absurdly
large bounds (everything "at risk", SAFE_GAP unreachable).  This module
recomputes, from the model file and xgboost's own leaf indices,

* the worst-case leaf-rounding error per class, in exact rational arithmetic:
  it must equal what the exporter reports;
* a generous ceiling for the float32-noise bound per row and for SAFE_GAP:
  the exporter's figures must not exceed it;
* exact margins for a sample of rows: the exporter's "exact" margins and the
  integer margins of the compiled C must agree with them within the bounds.

Nothing here imports xgb_edge.analysis or xgb_edge.quantize.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List

import numpy as np

U32 = 2.0 ** -24


def _products(model) -> List[np.ndarray]:
    """leaf value * tree weight per node, exactly (float64 holds the product of two float32)."""
    return [t.value.astype(np.float64) * np.float64(t.weight) for t in model.trees]


def leaf_error_bounds(model, q: List[np.ndarray], bias: List[int], shift: int):
    """(lo, hi) per group of integer_margin / 2**shift - exact_margin over all inputs, as Fractions.

    Also verifies every stored leaf is the correctly rounded fixed-point value.
    """
    problems: List[str] = []
    one = Fraction(1, 1 << shift)
    const = [Fraction(float(b)) for b in model.base_margin]
    lo = [Fraction(0)] * model.n_groups
    hi = [Fraction(0)] * model.n_groups
    for ti, t in enumerate(model.trees):
        w = Fraction(float(t.weight))
        leaves = t.leaf_nodes
        if t.n_internal == 0:
            const[t.group] += Fraction(float(t.value[leaves[0]])) * w
            continue
        errs = []
        for n in leaves:
            exact = Fraction(float(t.value[n])) * w
            want = round(exact * (1 << shift))            # ties to even
            got = int(q[ti][n])
            if got != want:
                problems.append(f"tree {ti} leaf {int(n)}: stored {got}, correctly rounded value is {want}")
            errs.append(got * one - exact)
        lo[t.group] += min(errs)
        hi[t.group] += max(errs)
    for g in range(model.n_groups):
        want = round(const[g] * (1 << shift))
        if int(bias[g]) != want:
            problems.append(f"bias of group {g}: stored {int(bias[g])}, correctly rounded value is {want}")
        e = int(bias[g]) * one - const[g]
        lo[g] += e
        hi[g] += e
    return lo, hi, const, problems[:5]


def static_noise_ceiling(model) -> float:
    """A generous ceiling, valid for any input, for XGBoost's float32 summation error on a margin."""
    dart = model.booster == "dart"
    prods = _products(model)
    worst = 0.0
    for g in range(model.n_groups):
        b = abs(float(model.base_margin[g]))
        cum = acc = 0.0
        n = 0
        for t, p in zip(model.trees, prods):
            if t.group != g:
                continue
            m = float(np.abs(p[t.leaf_nodes]).max())
            cum += m
            acc += b * (1.0 + (abs(t.weight) if dart else 0.0)) + cum + m * (3.0 if dart else 1.0)
            n += 1
        ceiling = 1.05 * U32 * (acc + b + cum) + 4.0 * float(np.spacing(np.float32(b))) \
            + 1e-14 * (cum + b) + (len(model.trees) + 1) * 2.0 ** -148
        worst = max(worst, ceiling)
    return worst


def row_noise_ceiling(model, leaf_ids: np.ndarray) -> np.ndarray:
    """Per row and group: a generous ceiling for the float32 noise given the leaves the row reaches."""
    dart = model.booster == "dart"
    prods = _products(model)
    n = leaf_ids.shape[0]
    out = np.zeros((n, model.n_groups))
    for g in range(model.n_groups):
        b = abs(float(model.base_margin[g]))
        cum = np.zeros(n)
        acc = np.zeros(n)
        for j, (t, p) in enumerate(zip(model.trees, prods)):
            if t.group != g:
                continue
            v = np.abs(p[leaf_ids[:, j]])
            cum += v
            acc += b * (1.0 + (abs(t.weight) if dart else 0.0)) + cum + v * (3.0 if dart else 1.0)
        out[:, g] = 1.05 * U32 * (acc + b + cum) + 4.0 * float(np.spacing(np.float32(b))) \
            + 1e-14 * (cum + b) + (len(model.trees) + 1) * 2.0 ** -148
    return out


def exact_margins(model, leaf_ids: np.ndarray, rows: np.ndarray) -> List[List[Fraction]]:
    """Exact margins of the given rows as Fractions, from the leaves xgboost says they reach."""
    out = []
    for i in rows:
        m = [Fraction(float(b)) for b in model.base_margin]
        for j, t in enumerate(model.trees):
            m[t.group] += Fraction(float(t.value[leaf_ids[i, j]])) * Fraction(float(t.weight))
        out.append(m)
    return out


def check(model, exp, leaf_ids: np.ndarray, c_margin: np.ndarray, macros: Dict[str, int],
          sample: int = 24, seed: int = 0) -> List[str]:
    """All independent checks for one export; returns a list of violated requirements."""
    fails: List[str] = []
    lp, ev = exp.lplan, exp.evaluation
    shift = macros["MARGIN_SHIFT"]
    lo, hi, _, problems = leaf_error_bounds(model, lp.q, lp.bias, shift)
    fails += ["independent: " + p for p in problems]

    # 1. the reported worst-case leaf error is the true one (neither smaller nor inflated)
    for g in range(model.n_groups):
        for name, mine, theirs in (("upper", hi[g], float(lp.err_hi[g])), ("lower", lo[g], float(lp.err_lo[g]))):
            tol = 1e-9 * abs(float(mine)) + 64 * float(np.spacing(abs(float(mine)))) + 1e-300
            if abs(float(mine) - theirs) > tol:
                fails.append(f"independent: {name} leaf-error bound of class {g} is {float(mine):.6g} exactly, "
                             f"the exporter reports {theirs:.6g}")
                break

    # 2. the float-noise bound and SAFE_GAP are not inflated beyond a generous ceiling
    ceiling = row_noise_ceiling(model, leaf_ids)
    over = ev.shared.float_noise > ceiling
    if over.any():
        i, g = np.argwhere(over)[0]
        fails.append(f"independent: float-noise bound {ev.shared.float_noise[i, g]:.3g} of row {i} exceeds the "
                     f"ceiling {ceiling[i, g]:.3g}")
    e_max = max(max(abs(float(x)) for x in lo), max(abs(float(x)) for x in hi))
    need = (2.0 * e_max if model.n_groups > 1 else e_max) \
        + (2.0 if model.n_groups > 1 else 1.0) * static_noise_ceiling(model) + 1.01 * 2.0 ** -21
    gap_ceiling = min(2 ** 31 - 1, math.ceil(need * 2.0 ** shift) + 4)
    if macros["SAFE_GAP"] > gap_ceiling:
        fails.append(f"independent: SAFE_GAP {macros['SAFE_GAP']} exceeds the ceiling {gap_ceiling}")
    floor = math.floor((e_max + 2.0 ** -21) * 2.0 ** shift) - 2
    if macros["SAFE_GAP"] < min(floor, 2 ** 31 - 1):
        fails.append(f"independent: SAFE_GAP {macros['SAFE_GAP']} is below what leaf rounding alone needs ({floor})")

    # 3. exact margins of a sample of rows
    n = leaf_ids.shape[0]
    rows = np.random.RandomState(seed).choice(n, size=min(sample, n), replace=False)
    one = Fraction(1, 1 << shift)
    for i, margins in zip(rows, exact_margins(model, leaf_ids, rows)):
        for g, r in enumerate(margins):
            if abs(float(r) - ev.shared.exact[i, g]) > ev.shared.f64_slack[i, g] + 1e-300:
                fails.append(f"independent: the exporter's exact margin of row {i} class {g} is off by "
                             f"{abs(float(r) - ev.shared.exact[i, g]):.3g}")
                return fails
            err = int(c_margin[i, g]) * one - r
            if not lo[g] <= err <= hi[g]:
                fails.append(f"independent: integer margin of row {i} class {g} is outside the exact error bound")
                return fails
    return fails
