"""The guarantee terms, one by one: each is needed, and each is tested.

"A row that is not flagged gets XGBoost's class" rests on several separate
ingredients: the leaf-rounding bound, the float32-noise bound, the slack for
XGBoost's float32 softmax/sigmoid, and SAFE_GAP.  For each one there is a model
here on which it decides the outcome, and a check that the harness fails when
the ingredient is removed, or inflated until it is meaningless.  The last part
matters because a tool that flagged everything would otherwise pass.
"""
import numpy as np
import pytest

import xgb_edge.analysis as analysis
import xgb_edge.export as export_mod
from harness import synth
from harness.parity import check_synthetic

F32 = np.float32


def stump(feature, threshold, left, right, group=0):
    t = synth.SynthTree(group=group)
    t.add_leaf(0.0)
    t.split(0, feature, threshold, False, left, right)
    return t


def write(tmp_path, trees, n_features, n_classes, base_score=None, objective=None):
    return synth.write_model(str(tmp_path / "m.json"), trees, n_features, n_classes, objective, base_score)


# --------------------------------------------------------------------------- models where one term decides

@pytest.fixture
def tiny_gap_models(tmp_path):
    """Margins a few 1e-8 apart: XGBoost's float32 softmax / sigmoid turns such gaps into ties."""
    X = F32(np.random.RandomState(7).rand(300, 4))
    (tmp_path / "multi").mkdir()
    (tmp_path / "binary").mkdir()
    multi = write(tmp_path / "multi", [stump(0, 0.5, 1e-8, 3e-8, 0), stump(1, 0.5, 2e-8, 1e-8, 1),
                                       stump(2, 0.5, 2.5e-8, 0.0, 2)], 4, 3, [0.0, 0.0, 0.0])
    binary = write(tmp_path / "binary", [stump(0, 0.5, 1e-8, -1e-8), stump(1, 0.5, 3e-8, 2e-8)], 4, 2, [0.5])
    return X, multi, binary


@pytest.fixture
def float_noise_model(tmp_path):
    """A gbtree model on which XGBoost's own float32 summation changes the class.

    class 0 = 0.5 + 0.5 + 200 leaves of 0.49 ulp(1.0): exactly 1 + 1.17e-5, but float32 rounds
    every tiny leaf away.  class 1 = 0.5 + 0.5 + 2**-20.  Exact arithmetic says class 0,
    XGBoost says class 1.
    """
    tiny = 0.49 * 2.0 ** -23
    trees = [stump(0, 0.5, 0.5, 0.5, 0), stump(0, 0.5, 0.5 + 2.0 ** -20, 0.5 + 2.0 ** -20, 1),
             stump(0, 0.5, -5.0, -5.0, 2)] + [stump(0, 0.5, tiny, tiny, 0) for _ in range(200)]
    path = write(tmp_path, trees, 2, 3, [0.5, 0.5, 0.5])
    return path, F32(np.random.RandomState(0).rand(40, 2))


def _ordinary_model(tmp_path, n_classes=4, seed=3):
    rng = np.random.RandomState(seed)
    pools = {f: F32(rng.randn(8)) for f in range(5)}
    groups = 1 if n_classes == 2 else n_classes
    trees = []
    for g in range(groups):
        for _ in range(12):
            t = synth.random_tree(rng, 5, pools, 10, group=g)
            for leaf in t.leaves:
                t.value[leaf] = float(F32(rng.randn() * 0.3))
            trees.append(t)
    return write(tmp_path, trees, 5, n_classes), F32(rng.randn(400, 5))


# --------------------------------------------------------------------------- the softmax / sigmoid slack

def test_tiny_gaps_are_flagged_because_of_the_decision_slack(tiny_gap_models, tmp_path, monkeypatch):
    X, multi, binary = tiny_gap_models
    for path in (multi, binary):
        res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
        assert res["failures"] == [], res["failures"]
        assert res["at_risk"] == res["rows"]              # every row is a near-tie for XGBoost itself
    # without the slack those rows are no longer flagged, and the harness notices
    monkeypatch.setattr(analysis, "DECISION_SLACK", 0.0)
    broken = [check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w2")) for path in (multi, binary)]
    assert all(r["failures"] for r in broken), "removing DECISION_SLACK went unnoticed"


# --------------------------------------------------------------------------- the float32-noise bound

def test_xgboost_rounding_can_change_the_class_and_is_flagged(float_noise_model, tmp_path):
    path, X = float_noise_model
    for opts in ({"leaf_bits": 32}, {}):
        res = check_synthetic(path, X, opts, str(tmp_path / "w"))
        assert res["failures"] == [], res["failures"]
        assert res["at_risk_float"] == res["rows"] and res["uncertain"] == res["rows"]
        assert res["differ"] == res["rows"]               # XGBoost really does answer differently here


def test_dropping_the_noise_bound_is_noticed(float_noise_model, tmp_path, monkeypatch):
    path, X = float_noise_model
    monkeypatch.setattr(analysis, "static_float_noise", lambda model: np.zeros(model.n_groups))
    res = check_synthetic(path, X, {"leaf_bits": 32}, str(tmp_path / "w"))
    assert any("SAFE_GAP" in f for f in res["failures"]), res["failures"]


def test_dropping_the_per_row_noise_is_noticed(float_noise_model, tmp_path, monkeypatch):
    path, X = float_noise_model
    monkeypatch.setattr(analysis, "U32", 0.0)
    res = check_synthetic(path, X, {"leaf_bits": 32}, str(tmp_path / "w"))
    assert res["failures"], "a zero float-noise bound went unnoticed"


# --------------------------------------------------------------------------- bounds must not be inflated either

def test_ordinary_model_passes(tmp_path):
    for k in (2, 4):
        path, X = _ordinary_model(tmp_path, k)
        res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
        assert res["failures"] == [], res["failures"]


def test_inflated_float_noise_is_noticed(tmp_path, monkeypatch):
    path, X = _ordinary_model(tmp_path)
    monkeypatch.setattr(analysis, "U32", analysis.U32 * 30000)
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
    assert any("ceiling" in f for f in res["failures"]), res["failures"]


def test_unreachable_safe_gap_is_noticed(tmp_path, monkeypatch):
    path, X = _ordinary_model(tmp_path)
    monkeypatch.setattr(export_mod, "safe_gap", lambda model, plan: 2 ** 31 - 1)
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
    assert any("SAFE_GAP" in f and "ceiling" in f for f in res["failures"]), res["failures"]


def test_useless_safe_gap_is_noticed(tmp_path, monkeypatch):
    """SAFE_GAP = 1 on a coarse export: rows beyond it do differ from XGBoost."""
    path, X = _ordinary_model(tmp_path)
    monkeypatch.setattr(export_mod, "safe_gap", lambda model, plan: 1)
    res = check_synthetic(path, X, {"leaf_bits": 8}, str(tmp_path / "w"))
    assert any("SAFE_GAP" in f for f in res["failures"]), res["failures"]


@pytest.mark.parametrize("factor", [2000.0, 0.5, 0.0])
def test_wrong_leaf_error_bound_is_noticed(tmp_path, monkeypatch, factor):
    path, X = _ordinary_model(tmp_path)
    real = export_mod.plan_leaves

    def scaled(model, bits):
        plan = real(model, bits)
        plan.err_hi = plan.err_hi * factor
        plan.err_lo = plan.err_lo * factor
        return plan

    monkeypatch.setattr(export_mod, "plan_leaves", scaled)
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
    assert any("leaf-error bound" in f for f in res["failures"]), res["failures"]


def test_wrongly_rounded_leaf_is_noticed(tmp_path, monkeypatch):
    path, X = _ordinary_model(tmp_path)
    real = export_mod.plan_leaves

    def off_by_one(model, bits):
        plan = real(model, bits)
        t = next(i for i, tree in enumerate(model.trees) if tree.n_internal)
        plan.q[t][model.trees[t].leaf_nodes[0]] += 1
        return plan

    monkeypatch.setattr(export_mod, "plan_leaves", off_by_one)
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
    assert any("correctly rounded" in f for f in res["failures"]), res["failures"]


# --------------------------------------------------------------------------- what the header says

@pytest.mark.parametrize("macro, delta", [("MARGIN_SHIFT", 3), ("NUM_CLASSES", 5), ("SAFE_GAP", 1000)])
def test_wrong_header_macro_is_noticed(tmp_path, monkeypatch, macro, delta):
    import re
    path, X = _ordinary_model(tmp_path)
    real = export_mod.generate

    def wrong(*args, **kwargs):
        gen = real(*args, **kwargs)
        gen.header = re.sub(rf"(#define \w+_{macro} )(-?\d+)", lambda m: m.group(1) + str(int(m.group(2)) + delta),
                            gen.header)
        return gen

    monkeypatch.setattr(export_mod, "generate", wrong)
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
    assert any("header macros" in f for f in res["failures"]), res["failures"]


# --------------------------------------------------------------------------- named features

def test_model_trained_on_named_columns(tmp_path):
    """A model with feature names needs data that carries them; the harness must cope."""
    import warnings

    import xgboost as xgb
    rng = np.random.RandomState(0)
    X = F32(rng.randn(600, 3))
    y = (X[:, 0] + X[:, 2] > 0).astype(int) + (X[:, 1] > 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="multi:softprob", num_class=3, max_depth=3, verbosity=0),
                        xgb.DMatrix(X, label=y, feature_names=["accel_x", "accel_y", "gyro_z"],
                                    feature_types=["float", "float", "float"]), num_boost_round=10)
    path = str(tmp_path / "named.json")
    bst.save_model(path)
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))
    assert res["failures"] == [], res["failures"]
