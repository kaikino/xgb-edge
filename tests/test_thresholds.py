"""Property-based tests of the lossless threshold quantization.

The dangerous inputs are the ones exactly at a split threshold and the float32
values immediately below and above it.  These tests generate thresholds at
arbitrary float32 values (including zeros, denormals, the largest finite
floats and thresholds that are themselves adjacent floats), build real
XGBoost models around them, and require the compiled C to reach exactly the
leaf that xgboost itself reaches.
"""
import math
import os
import tempfile

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from harness import synth
from harness.parity import check_synthetic, compile_host, run_host
from xgb_edge import ExportOptions, export_model, load
from xgb_edge.quantize import INPUT_KINDS, f32_key, int_cut

F32 = np.float32
FMAX = float(np.finfo(F32).max)
SPECIAL = [0.0, -0.0, 1e-45, -1e-45, 1.1754944e-38, -1.1754944e-38, 1.1754942e-38, FMAX, -FMAX,
           1.0, -1.0, 0.5, 1.5, 2.0 ** 24, 2.0 ** 24 + 2, -(2.0 ** 24), 1e-30, 1e30, 255.0, 256.0, 127.5, -128.5]


def _from_bits(b: int) -> float:
    return float(np.array([b], dtype=np.uint32).view(F32)[0])


finite_f32 = st.one_of(
    st.sampled_from(SPECIAL),
    st.floats(width=32, allow_nan=False, allow_infinity=False),
    st.integers(0, 2 ** 32 - 1).map(_from_bits).filter(lambda v: math.isfinite(v)),
    st.floats(min_value=-300, max_value=300, width=32),
)
any_f32 = st.one_of(finite_f32, st.sampled_from([float("inf"), float("-inf")]))
# harness/mutation.py sets XGBE_NO_SHRINK: a failing example is enough there, minimising it is not needed
_PHASES = [Phase.explicit, Phase.reuse, Phase.generate] if os.environ.get("XGBE_NO_SHRINK") else list(Phase)
LONG = settings(max_examples=40, deadline=None, phases=_PHASES,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


def neighbours(t) -> list:
    t = F32(t)
    with np.errstate(over="ignore"):
        return [t, np.nextafter(t, F32(-np.inf), dtype=F32), np.nextafter(t, F32(np.inf), dtype=F32)]


# --------------------------------------------------------------------------- pure properties

@settings(max_examples=2000, deadline=None)
@given(a=any_f32, b=any_f32)
def test_key_orders_exactly_like_float32(a, b):
    ka, kb = f32_key([a])[0], f32_key([b])[0]
    assert (F32(a) < F32(b)) == (ka < kb)
    assert (F32(a) == F32(b)) == (ka == kb)
    assert -(2 ** 31) < ka < 2 ** 31


@settings(max_examples=500, deadline=None)
@given(t=any_f32)
def test_key_of_adjacent_floats_is_adjacent(t):
    below, above = neighbours(t)[1:]
    k = f32_key([t])[0]
    if np.isfinite(below) and below != F32(t):
        assert f32_key([below])[0] < k
    if np.isfinite(above) and above != F32(t):
        assert f32_key([above])[0] > k


def test_key_of_zeros_and_nan():
    assert f32_key([0.0])[0] == f32_key([-0.0])[0] == 0
    with pytest.raises(ValueError):
        f32_key([np.nan])


@settings(max_examples=300, deadline=None)
@given(thr=st.lists(any_f32, min_size=1, max_size=30), xs=st.lists(any_f32, max_size=30))
def test_rank_comparison_equals_float_comparison(thr, xs):
    """c(x) < code(t)  <=>  x < t, for every threshold and every value incl. the boundary ones."""
    thr = np.array(thr, dtype=F32)
    cuts = np.unique(f32_key(thr))
    probes = list(xs)
    for t in thr:
        probes.extend(neighbours(t))
    probes = np.array([p for p in probes if not np.isnan(p)], dtype=F32)
    rank = np.searchsorted(cuts, f32_key(probes), side="right")      # #{cuts <= x}
    for t in thr:
        code = int(np.searchsorted(cuts, f32_key([t])[0])) + 1
        assert np.array_equal(rank < code, probes < t)


@settings(max_examples=400, deadline=None)
@given(t=st.one_of(any_f32, st.integers(-2 ** 33, 2 ** 33).map(float), st.floats(-70000, 70000)),
       kind=st.sampled_from(["i8", "u8", "i16", "u16", "i32"]), data=st.data())
def test_int_cut_matches_float32_comparison(t, kind, data):
    lo, hi = INPUT_KINDS[kind].lo, INPUT_KINDS[kind].hi
    t = float(F32(t))
    cut = int_cut(t, lo, hi)
    assert lo <= cut <= hi + 1
    cands = {lo, hi, lo + 1, hi - 1, 0 if lo <= 0 else lo}
    for d in range(-3, 4):
        cands.add(cut + d)
        if math.isfinite(t):
            cands.add(int(math.floor(t)) + d)
    cands.update(data.draw(st.lists(st.integers(lo, hi), max_size=5)))
    for n in cands:
        if lo <= n <= hi:
            assert (F32(n) < F32(t)) == (n < cut), (n, t, cut)


def test_int_cut_where_float32_cannot_hold_the_integer():
    # float32(16777217) rounds to 16777216, so 16777217 < 16777217.0f is false and 16777216.0f < t ...
    t = float(F32(16777218.0))
    cut = int_cut(t, -2 ** 31, 2 ** 31 - 1)
    for n in range(16777210, 16777225):
        assert (F32(n) < F32(t)) == (n < cut)
    assert int_cut(float("inf"), -128, 127) == 128
    assert int_cut(float("-inf"), -128, 127) == -128
    assert int_cut(127.5, -128, 127) == 128 and int_cut(127.0, -128, 127) == 127
    assert int_cut(-128.0, -128, 127) == -128 and int_cut(-127.5, -128, 127) == -127


# --------------------------------------------------------------------------- end to end, against xgboost

def _threshold_pools(data, n_features, strategy, max_per_feature=10):
    pools = {}
    for f in range(n_features):
        base = data.draw(st.lists(strategy, min_size=1, max_size=max_per_feature), label=f"thresholds[{f}]")
        vals = [F32(v) for v in base]
        # make some thresholds adjacent floats of other thresholds, and duplicate some
        for v in list(vals[: data.draw(st.integers(0, 2))]):
            vals.extend(n for n in neighbours(v)[1:] if np.isfinite(n))
        pools[f] = np.array(vals, dtype=F32)
    return pools


def _boundary_matrix(rng, pools, n_features, extra_rows=40, with_nan=True):
    values = {f: np.unique(np.array([n for t in pools[f] for n in neighbours(t) if np.isfinite(n)], dtype=F32))
              for f in pools}
    rows = []
    for f in pools:
        for v in values[f]:
            row = np.array([values[g][rng.randint(values[g].size)] for g in range(n_features)], dtype=F32)
            row[f] = v
            rows.append(row)
    for _ in range(extra_rows):
        rows.append(np.array([values[g][rng.randint(values[g].size)] for g in range(n_features)], dtype=F32))
    X = np.array(rows, dtype=F32)
    if with_nan:
        extra = X[rng.randint(0, X.shape[0], size=max(4, X.shape[0] // 4))].copy()
        extra[rng.rand(*extra.shape) < 0.35] = np.nan
        X = np.concatenate([X, extra, np.full((1, n_features), np.nan, dtype=F32)])
    return X


def _leaf_ids_from_c(exp, c_margin):
    """Decode which leaf each (single-tree) group reached from the exact integer margins."""
    model, lp = exp.model, exp.lplan
    out = np.zeros((c_margin.shape[0], len(model.trees)), dtype=np.int64)
    for j, t in enumerate(model.trees):
        q = lp.q[j][t.leaf_nodes]
        assert len(set(q.tolist())) == q.size, "leaf values must be distinct for this test"
        lookup = {int(v) + lp.bias[t.group]: int(n) for v, n in zip(q, t.leaf_nodes)}
        out[:, j] = [lookup[int(v)] for v in c_margin[:, t.group]]
    return out


@LONG
@given(data=st.data())
def test_c_reaches_the_same_leaf_as_xgboost_at_boundaries(data):
    n_features = data.draw(st.integers(1, 5), label="features")
    n_classes = data.draw(st.sampled_from([2, 3, 4]), label="classes")
    groups = 1 if n_classes == 2 else n_classes
    pools = _threshold_pools(data, n_features, finite_f32)
    rng = np.random.RandomState(data.draw(st.integers(0, 2 ** 31 - 1), label="seed"))
    trees = [synth.random_tree(rng, n_features, pools, data.draw(st.integers(1, 30)), group=g,
                               chain=data.draw(st.booleans())) for g in range(groups)]
    synth.number_leaves(trees)
    opts = dict(leaf_bits=32,
                thresholds=data.draw(st.sampled_from(["rank", "direct", "auto"]), label="thresholds"),
                min_code_bytes=data.draw(st.sampled_from([1, 2])),
                min_hdr_bytes=data.draw(st.sampled_from([1, 2])),
                min_off_bytes=data.draw(st.sampled_from([1, 2, 3])),
                leaf_store=data.draw(st.sampled_from(["auto", "inline", "table"])))
    X = _boundary_matrix(rng, pools, n_features)
    with tempfile.TemporaryDirectory() as d:
        path = synth.write_model(os.path.join(d, "m.json"), trees, n_features, n_classes)
        _, _, xgb_leaf = synth.xgb_reference(path, X)
        exp = export_model(path, ExportOptions(name="m", **opts), validation=X)
        exe = compile_host(exp, d)
        _, c_margin, _, _ = run_host(exe, exp.device_inputs(X), exp.model.n_groups)
    assert np.array_equal(_leaf_ids_from_c(exp, c_margin), xgb_leaf)


@LONG
@given(data=st.data())
def test_integer_inputs_reach_the_same_leaf_as_xgboost(data):
    kind = data.draw(st.sampled_from(["i8", "u8", "i16", "u16", "i32"]), label="input")
    lo, hi = INPUT_KINDS[kind].lo, INPUT_KINDS[kind].hi
    n_features = data.draw(st.integers(1, 4), label="features")
    near = st.one_of(
        st.integers(lo - 3, hi + 3).map(float),
        st.integers(lo - 3, hi + 3).map(lambda n: n + 0.5),
        st.floats(min_value=float(F32(lo - 4)), max_value=float(F32(hi + 4)), width=32),
        st.sampled_from([float(lo), float(hi), 0.0, -0.0, 0.5, -0.5, float(hi) + 1, float(lo) - 1, 1e30, -1e30,
                         2.0 ** 24, 2.0 ** 24 + 2, 2.0 ** 31, -(2.0 ** 31)]),
    )
    pools = {f: np.array(data.draw(st.lists(near, min_size=1, max_size=8)), dtype=F32) for f in range(n_features)}
    rng = np.random.RandomState(data.draw(st.integers(0, 2 ** 31 - 1), label="seed"))
    n_classes = data.draw(st.sampled_from([2, 3]))
    groups = 1 if n_classes == 2 else n_classes
    trees = [synth.random_tree(rng, n_features, pools, data.draw(st.integers(1, 20)), group=g) for g in range(groups)]
    synth.number_leaves(trees)
    # integer probes around every threshold, plus the ends of the type's range
    values = {}
    for f in range(n_features):
        v = {lo, hi, lo + 1, hi - 1, min(max(0, lo), hi)}
        for t in pools[f]:
            c = int(np.clip(np.floor(np.float64(t)), lo - 10, hi + 10))
            v.update(range(c - 2, c + 4))
        values[f] = np.array(sorted(x for x in v if lo <= x <= hi), dtype=np.int64)
    rows = []
    for f in range(n_features):
        for v in values[f]:
            row = [values[g][rng.randint(values[g].size)] for g in range(n_features)]
            row[f] = v
            rows.append(row)
    Xi = np.array(rows, dtype=np.int64)
    X = Xi.astype(F32)            # what xgboost sees for these integers
    with tempfile.TemporaryDirectory() as d:
        path = synth.write_model(os.path.join(d, "m.json"), trees, n_features, n_classes)
        _, _, xgb_leaf = synth.xgb_reference(path, X)
        exp = export_model(path, ExportOptions(name="m", input=kind, leaf_bits=32,
                                               thresholds=data.draw(st.sampled_from(["rank", "direct"]), label="thresholds"),
                                               min_code_bytes=data.draw(st.sampled_from([1, 2]))))
        exe = compile_host(exp, d)
        dev = Xi.astype({"i8": np.int8, "u8": np.uint8, "i16": np.int16, "u16": np.uint16, "i32": np.int32}[kind])
        _, c_margin, _, _ = run_host(exe, np.ascontiguousarray(dev), exp.model.n_groups)
    assert np.array_equal(_leaf_ids_from_c(exp, c_margin), xgb_leaf)


@LONG
@given(data=st.data())
def test_forests_with_boundary_inputs_pass_the_full_parity_check(data):
    """Several trees per class, single-leaf trees, unused features, random export options."""
    n_features = data.draw(st.integers(1, 6), label="features")
    n_classes = data.draw(st.sampled_from([2, 3, 5]), label="classes")
    groups = 1 if n_classes == 2 else n_classes
    used = data.draw(st.lists(st.integers(0, n_features - 1), min_size=1, max_size=n_features, unique=True))
    pools = _threshold_pools(data, n_features, finite_f32, max_per_feature=6)
    rng = np.random.RandomState(data.draw(st.integers(0, 2 ** 31 - 1), label="seed"))
    trees = []
    for g in range(groups):
        for _ in range(data.draw(st.integers(1, 6))):
            if data.draw(st.integers(0, 5)) == 0:
                trees.append(synth.SynthTree.single_leaf(float(rng.randn() * 0.3), group=g))
                continue
            t = synth.random_tree(rng, n_features, pools, data.draw(st.integers(1, 14)), group=g,
                                  chain=data.draw(st.booleans()), features=used)
            for leaf in t.leaves:
                t.value[leaf] = float(F32(rng.randn() * 0.4))
            trees.append(t)
    rng.shuffle(trees)
    opts = dict(leaf_bits=data.draw(st.sampled_from([8, 12, 16, 24, 32])),
                thresholds=data.draw(st.sampled_from(["rank", "direct", "auto"]), label="thresholds"),
                leaf_store=data.draw(st.sampled_from(["auto", "inline", "table"])),
                min_code_bytes=data.draw(st.sampled_from([1, 2])),
                min_hdr_bytes=data.draw(st.sampled_from([1, 2])),
                min_off_bytes=data.draw(st.sampled_from([1, 2, 4])),
                min_leaf_bytes=data.draw(st.sampled_from([1, 2, 3, 4])))
    X = _boundary_matrix(rng, pools, n_features)
    base = [float(F32(rng.randn() * 0.2)) for _ in range(groups)] if groups > 1 else [0.5]
    with tempfile.TemporaryDirectory() as d:
        path = synth.write_model(os.path.join(d, "m.json"), trees, n_features, n_classes, base_score=base)
        res = check_synthetic(path, X, opts, d)
    assert res["failures"] == [], res["failures"]


def test_infinite_inputs_match_xgboost_inplace_predict():
    """DMatrix refuses infinite inputs, but Booster.inplace_predict (and so XGBClassifier.predict) accepts
    them.  The generated C must send them where xgboost does: +inf is not below any threshold, -inf is
    below every finite one."""
    import warnings

    import xgboost as xgb
    rng = np.random.RandomState(5)
    pools = {0: F32([-FMAX, -1.0, 0.0, 2.5, FMAX]), 1: F32([-3.0, 1e-30, 7.0]), 2: F32([0.5, FMAX])}
    for n_classes in (2, 3):
        groups = 1 if n_classes == 2 else n_classes
        trees = [synth.random_tree(rng, 3, pools, 14, group=g) for g in range(groups)]
        synth.number_leaves(trees)
        special = [float("inf"), float("-inf"), float("nan"), FMAX, -FMAX, 0.0, 2.5, 1.0]
        X = np.array([[a, b, c] for a in special for b in special for c in special], dtype=F32)
        with tempfile.TemporaryDirectory() as d:
            path = synth.write_model(os.path.join(d, "m.json"), trees, 3, n_classes)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bst = xgb.Booster(model_file=path)
                margin = np.asarray(bst.inplace_predict(X, predict_type="margin")).reshape(X.shape[0], -1)
            for mode in ("rank", "direct"):
                exp = export_model(path, ExportOptions(name="m", leaf_bits=32, thresholds=mode))
                exe = compile_host(exp, os.path.join(d, mode))
                _, c_margin, _, _ = run_host(exe, X, exp.model.n_groups)
                # every group has one tree with distinct, exactly representable leaves: equal margins
                # mean the same leaf was reached
                got = c_margin * 2.0 ** -exp.lplan.shift
                assert np.array_equal(got.astype(F32), margin.astype(F32)), (n_classes, mode)


def test_infinite_inputs_and_thresholds_follow_float_semantics():
    """Infinite *thresholds* cannot be given to xgboost at all; compare with the float32 reference traversal."""
    from xgb_edge.refsim import leaf_indices
    t = synth.SynthTree()
    t.add_leaf(0)
    l, r = t.split(0, 0, float("inf"), False)
    t.split(l, 1, FMAX, True)
    t.split(r, 1, float("-inf"), False)
    synth.number_leaves([t])
    vals = [float("inf"), float("-inf"), FMAX, -FMAX, 0.0, float("nan")]
    X = np.array([[a, b] for a in vals for b in vals], dtype=F32)
    with tempfile.TemporaryDirectory() as d:
        path = synth.write_model(os.path.join(d, "m.json"), [t], 2, 2)
        ref = leaf_indices(load(path), X)
        for mode in ("rank", "direct"):
            exp = export_model(path, ExportOptions(name="m", leaf_bits=32, thresholds=mode))
            exe = compile_host(exp, os.path.join(d, mode))
            _, c_margin, _, _ = run_host(exe, X, 1)
            assert np.array_equal(_leaf_ids_from_c(exp, c_margin), ref), mode
