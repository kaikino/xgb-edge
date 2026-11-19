"""Export-level behaviour: round selection, layout choices, reports and input validation."""
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from harness import synth
from harness.parity import check_synthetic, compile_host, run_host
from harness.zoo import xgb_predict_class
from xgb_edge import ExportOptions, UnsupportedModelError, XgbEdgeError, export_model, load
from xgb_edge.export import to_device_inputs

F32 = np.float32


def _data(n=900, f=8, k=3, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=f, n_informative=min(5, f - 1), n_redundant=0,
                               n_classes=k, n_clusters_per_class=1, random_state=seed)
    return X.astype(F32), y


def _c_classes(exp, X, tmp_path):
    exe = compile_host(exp, str(tmp_path))
    cls, margin, _, _ = run_host(exe, exp.device_inputs(X), exp.model.n_groups)
    return cls, margin


# --------------------------------------------------------------------------- early stopping

@pytest.fixture(scope="module")
def early_stopped(tmp_path_factory):
    import xgboost as xgb
    X, y = _data(1500, 10, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = xgb.XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.5, early_stopping_rounds=3,
                                tree_method="hist", verbosity=0)
        clf.fit(X[:800], y[:800], eval_set=[(X[800:1100], y[800:1100])], verbose=False)
    path = str(tmp_path_factory.mktemp("es") / "es.json")
    clf.save_model(path)
    return clf, path, X[1100:]


def test_early_stopped_model_needs_an_explicit_choice(early_stopped):
    clf, path, X = early_stopped
    model = load(path)
    assert model.best_iteration is not None and model.best_iteration + 1 < model.n_rounds, \
        "fixture should stop early"
    with pytest.raises(XgbEdgeError, match="early stopping.*--rounds best.*--rounds all"):
        export_model(path)


def test_rounds_best_matches_xgbclassifier_predict(early_stopped, tmp_path):
    clf, path, X = early_stopped
    exp = export_model(path, ExportOptions(name="m", rounds="best", leaf_bits=24), validation=X)
    assert len(exp.model.trees) == (exp.model.best_iteration + 1) * 4
    cls, _ = _c_classes(exp, X, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = clf.predict(X)
    safe = ~exp.evaluation.at_risk
    assert np.array_equal(cls[safe], want[safe]) and safe.mean() > 0.9
    assert any("best_iteration" in n for n in exp.report["notes"])


def test_rounds_all_matches_booster_predict(early_stopped, tmp_path):
    import xgboost as xgb
    clf, path, X = early_stopped
    exp = export_model(path, ExportOptions(name="m", rounds="all", leaf_bits=24), validation=X)
    cls, _ = _c_classes(exp, X, tmp_path)
    bst = clf.get_booster()
    want = xgb_predict_class(bst, "multi:softprob", xgb.DMatrix(X))
    safe = ~exp.evaluation.at_risk
    assert np.array_equal(cls[safe], want[safe])
    assert len(exp.model.trees) == exp.model.n_rounds * 4


def test_rounds_number_matches_iteration_range(early_stopped, tmp_path):
    import xgboost as xgb
    clf, path, X = early_stopped
    exp = export_model(path, ExportOptions(name="m", rounds=2, leaf_bits=24), validation=X)
    cls, _ = _c_classes(exp, X, tmp_path)
    p = clf.get_booster().predict(xgb.DMatrix(X), iteration_range=(0, 2))
    safe = ~exp.evaluation.at_risk
    assert np.array_equal(cls[safe], np.argmax(p, axis=1)[safe])
    for bad in (0, 10 ** 6, "sometimes"):
        with pytest.raises(XgbEdgeError, match="rounds"):
            export_model(path, ExportOptions(name="m", rounds=bad))


def test_rounds_best_without_early_stopping_is_an_error(tmp_path):
    path = synth.write_model(str(tmp_path / "m.json"), [synth.SynthTree.single_leaf(0.1)], 2, 2)
    with pytest.raises(XgbEdgeError, match="best_iteration"):
        export_model(path, ExportOptions(rounds="best"))


def test_harness_class_rule_is_xgbclassifier_predict():
    """The harness derives 'xgboost's class' from Booster.predict; it must equal XGBClassifier.predict."""
    import xgboost as xgb
    for k in (2, 5):
        X, y = _data(700, 8, k, seed=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0).fit(X, y)
            obj = "binary:logistic" if k == 2 else "multi:softprob"
            assert np.array_equal(clf.predict(X), xgb_predict_class(clf.get_booster(), obj, xgb.DMatrix(X)))


# --------------------------------------------------------------------------- model shapes

def test_two_class_softprob_is_a_two_margin_model(tmp_path):
    import xgboost as xgb
    X, y = _data(600, 6, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="multi:softprob", num_class=2, max_depth=3, verbosity=0),
                        xgb.DMatrix(X, label=y), num_boost_round=15)
    path = str(tmp_path / "m.json")
    bst.save_model(path)
    assert load(path).n_groups == 2
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path))
    assert res["failures"] == []


def test_model_with_zero_trees(tmp_path):
    import xgboost as xgb
    X, y = _data(300, 5, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="multi:softprob", num_class=3, verbosity=0, base_score=0.5),
                        xgb.DMatrix(X, label=y), num_boost_round=0)
    path = str(tmp_path / "m.json")
    bst.save_model(path)
    exp = export_model(path, ExportOptions(name="m"), validation=X)
    cls, margin = _c_classes(exp, X, tmp_path)
    assert exp.report["flash"]["data_bytes"] == 0 and exp.report["comparisons_worst_case"]["total"] == 0
    assert (cls == cls[0]).all() and (margin == margin[0]).all()


def test_shared_leaf_table_is_chosen_when_it_is_smaller(tmp_path):
    rng = np.random.RandomState(0)
    pools = {f: F32(rng.randn(5)) for f in range(4)}
    trees = []
    for g in range(3):
        for _ in range(40):
            t = synth.random_tree(rng, 4, pools, 7, group=g)
            for leaf in t.leaves:
                t.value[leaf] = float(F32(rng.choice([-0.3, -0.1, 0.05, 0.2])))
            trees.append(t)
    path = synth.write_model(str(tmp_path / "m.json"), trees, 4, 3)
    X = F32(rng.randn(400, 4))
    exp = export_model(path, ExportOptions(name="m", leaf_bits=16), validation=X)
    assert exp.packed.leaf_mode == "table" and exp.packed.leaf_bytes == 1
    assert exp.packed.leaf_table.size <= 4
    forced = export_model(path, ExportOptions(name="m", leaf_bits=16, leaf_store="inline"))
    assert exp.report["flash"]["data_bytes"] < forced.report["flash"]["data_bytes"]
    assert check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path), cross=True)["failures"] == []


def test_deep_chain_tree_and_wide_offsets(tmp_path):
    rng = np.random.RandomState(1)
    pools = {f: F32(rng.randn(40)) for f in range(3)}
    trees = [synth.random_tree(rng, 3, pools, 900, group=g, chain=(g == 0)) for g in range(3)]
    for t in trees:
        for leaf in t.leaves:
            t.value[leaf] = float(F32(rng.randn() * 0.2))
    path = synth.write_model(str(tmp_path / "m.json"), trees, 3, 3)
    model = load(path)
    assert max(t.depth for t in model.trees) > 300
    X = F32(rng.randn(500, 3))
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path), cross=True)
    assert res["failures"] == [] and res["layout"][2] == "2"   # subtrees longer than 255 bytes -> 2-byte offsets


def test_more_than_64_slots_uses_wide_header(tmp_path):
    rng = np.random.RandomState(2)
    n_features = 80
    pools = {f: F32(rng.randn(3)) for f in range(n_features)}
    trees = [synth.random_tree(rng, n_features, pools, 60, group=g) for g in range(3) for _ in range(4)]
    for t in trees:
        for leaf in t.leaves:
            t.value[leaf] = float(F32(rng.randn() * 0.2))
    path = synth.write_model(str(tmp_path / "m.json"), trees, n_features, 3)
    X = F32(rng.randn(300, n_features))
    X[rng.rand(*X.shape) < 0.1] = np.nan
    res = check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path), cross=True)
    assert res["failures"] == [] and res["layout"][0] == "2"


def test_leaves_too_large_for_the_requested_width(tmp_path):
    t = synth.SynthTree()
    t.add_leaf(0)
    t.split(0, 0, 0.5, False, 5000.0, -3.0)
    path = synth.write_model(str(tmp_path / "m.json"), [t], 1, 2)
    with pytest.raises(UnsupportedModelError, match="does not fit a 8-bit"):
        export_model(path, ExportOptions(leaf_bits=8))
    exp = export_model(path, ExportOptions(name="m", leaf_bits="auto"), validation=F32([[0.0], [1.0]]))
    assert exp.lplan.bits >= 16
    cls, margin = _c_classes(exp, F32([[0.0], [1.0]]), tmp_path)
    assert cls.tolist() == [1, 0]
    with pytest.raises(XgbEdgeError, match="leaf bits"):
        export_model(path, ExportOptions(leaf_bits=40))
    with pytest.raises(XgbEdgeError, match="leaf_bits"):
        export_model(path, ExportOptions(leaf_bits="wide"))


def test_huge_margins_still_fit_the_accumulator(tmp_path):
    rng = np.random.RandomState(3)
    trees = []
    for i in range(300):
        t = synth.SynthTree(group=0)
        t.add_leaf(0)
        t.split(0, 0, float(i - 150), bool(i % 2), 900.0, -900.0)
        trees.append(t)
    path = synth.write_model(str(tmp_path / "m.json"), trees, 1, 2)
    X = F32(rng.uniform(-200, 200, size=(400, 1)))
    res = check_synthetic(path, X, {"leaf_bits": 32}, str(tmp_path))
    assert res["failures"] == []
    exp = export_model(path, ExportOptions(leaf_bits=32))
    assert max(exp.lplan.max_abs_acc) <= 2 ** 31 - 1


# --------------------------------------------------------------------------- report and inputs

def test_report_is_consistent_with_the_generated_tables(tmp_path):
    rng = np.random.RandomState(4)
    pools = {f: F32(rng.randn(9)) for f in range(5)}
    trees = [synth.random_tree(rng, 5, pools, 20, group=g) for g in range(4) for _ in range(5)]
    for t in trees:
        for leaf in t.leaves:
            t.value[leaf] = float(F32(rng.randn() * 0.2))
    path = synth.write_model(str(tmp_path / "m.json"), trees, 5, 4)
    X = F32(rng.randn(200, 5))
    exp = export_model(path, ExportOptions(name="mod"), validation=X)
    rep = exp.report
    assert rep["flash"]["data_bytes"] == sum(a["bytes"] for a in rep["flash"]["arrays"])
    assert rep["splits"] == 20 * 20 and rep["leaves"] == 21 * 20
    cw = rep["comparisons_worst_case"]
    assert cw["tree_splits"] == sum(t.depth for t in exp.model.trees) and cw["argmax"] == 3
    assert cw["total"] == cw["feature_ranking"] + cw["tree_splits"] + cw["argmax"]
    assert rep["validation"]["samples"] == 200
    json.dumps(rep)                                         # the report is plain JSON
    for a in exp.generated.arrays:
        assert f"{a.name}[{a.count}]" in exp.generated.source
    assert "float" not in exp.generated.source.replace("const mod_feature_t", "")
    assert "double" not in exp.generated.source


def test_integer_input_data_is_validated(tmp_path):
    t = synth.SynthTree()
    t.add_leaf(0)
    t.split(0, 0, 10.5, False, 0.5, -0.5)
    path = synth.write_model(str(tmp_path / "m.json"), [t], 2, 2)
    with pytest.raises(XgbEdgeError, match="non-integers"):
        export_model(path, ExportOptions(input="i16"), validation=F32([[1.5, 0]]))
    with pytest.raises(XgbEdgeError, match="missing values"):
        export_model(path, ExportOptions(input="i16"), validation=F32([[np.nan, 0]]))
    with pytest.raises(XgbEdgeError, match="outside the range"):
        export_model(path, ExportOptions(input="i8"), validation=F32([[300, 0]]))
    with pytest.raises(XgbEdgeError, match="columns|shape"):
        export_model(path, ExportOptions(), validation=F32([[1, 2, 3]]))
    with pytest.raises(XgbEdgeError, match="unknown input kind"):
        export_model(path, ExportOptions(input="f64"))
    with pytest.raises(XgbEdgeError, match="C identifier"):
        export_model(path, ExportOptions(name="my model"))
    assert to_device_inputs(F32([[3, -4]]), 2, "i8").dtype == np.int8


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "xgb_edge", *args], capture_output=True, text=True)


def test_cli_export_and_report(tmp_path):
    rng = np.random.RandomState(5)
    pools = {f: F32(rng.randn(4)) for f in range(3)}
    trees = [synth.random_tree(rng, 3, pools, 6, group=g) for g in range(3)]
    for t in trees:
        for leaf in t.leaves:
            t.value[leaf] = float(F32(rng.randn() * 0.2))
    path = synth.write_model(str(tmp_path / "m.json"), trees, 3, 3)
    csv = tmp_path / "val.csv"
    csv.write_text("0.1,0.2,0.3\n,1.5,-2\n4,5,6\n")       # an empty cell is a missing value
    out = tmp_path / "out"
    r = _cli("export", path, "-o", str(out), "--name", "clf", "--validation", str(csv),
             "--report-json", str(tmp_path / "rep.json"))
    assert r.returncode == 0, r.stderr
    assert (out / "clf.h").exists() and (out / "clf.c").exists()
    assert "Worst-case comparisons per inference" in r.stdout and "Validation (3 samples)" in r.stdout
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["validation"]["samples"] == 3 and rep["flash"]["total_bytes"] >= rep["flash"]["data_bytes"]
    r2 = _cli("report", path, "--no-measure")
    assert r2.returncode == 0 and "estimate" in r2.stdout and not (tmp_path / "xgb_model.c").exists()
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    r3 = _cli("export", str(bad), "-o", str(out))
    assert r3.returncode == 2 and "xgb-edge: error:" in r3.stderr and "Traceback" not in r3.stderr
    r4 = _cli("export", path, "-o", str(out), "--leaf-bits", "nope")
    assert r4.returncode == 2 and "leaf-bits" in r4.stderr
