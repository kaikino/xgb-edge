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
    assert np.array_equal(cls[safe], want[safe]) and safe.mean() > 0.9
    assert len(exp.model.trees) == exp.model.n_rounds * 4


def test_rounds_number_matches_iteration_range(early_stopped, tmp_path):
    import xgboost as xgb
    clf, path, X = early_stopped
    exp = export_model(path, ExportOptions(name="m", rounds=2, leaf_bits=24), validation=X)
    cls, _ = _c_classes(exp, X, tmp_path)
    p = clf.get_booster().predict(xgb.DMatrix(X), iteration_range=(0, 2))
    safe = ~exp.evaluation.at_risk
    assert np.array_equal(cls[safe], np.argmax(p, axis=1)[safe]) and safe.mean() > 0.9
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


def test_model_with_zero_trees_is_refused(tmp_path):
    import xgboost as xgb
    X, y = _data(300, 5, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="multi:softprob", num_class=3, verbosity=0, base_score=0.5),
                        xgb.DMatrix(X, label=y), num_boost_round=0)
    path = str(tmp_path / "m.json")
    bst.save_model(path)
    with pytest.raises(UnsupportedModelError, match="no trees"):
        export_model(path)


def test_model_without_any_split_is_a_constant(tmp_path):
    trees = [synth.SynthTree.single_leaf(v, group=g) for g, v in ((0, 0.1), (1, 0.3), (2, -0.2), (1, 0.05))]
    path = synth.write_model(str(tmp_path / "m.json"), trees, 4, 3)
    X = F32(np.random.RandomState(0).randn(50, 4))
    exp = export_model(path, ExportOptions(name="m"), validation=X)
    cls, margin = _c_classes(exp, X, tmp_path)
    assert exp.report["flash"]["data_bytes"] == 0 and exp.report["comparisons_worst_case"]["total"] == 0
    assert (cls == 1).all() and (margin == margin[0]).all()
    assert check_synthetic(path, X, {"leaf_bits": 16}, str(tmp_path / "w"))["failures"] == []


# --------------------------------------------------------------------------- binary:logistic base_score

@pytest.mark.parametrize("base_score", [1e-10, 1e-7, 0.9999999, 0.25])
def test_extreme_logistic_base_score_matches_xgboost(tmp_path, base_score):
    """XGBoost >= 3.2 bounds the base score to [1e-6, 1 - 1e-6] before taking the logit."""
    import xgboost as xgb
    X, y = _data(2500, 8, 2, seed=3)
    y = np.where(np.random.RandomState(1).rand(y.size) < 0.3, 1 - y, y)       # noisy labels: margins near 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="binary:logistic", max_depth=3, base_score=base_score, verbosity=0),
                        xgb.DMatrix(X[:1500], label=y[:1500]), num_boost_round=40)
    path = str(tmp_path / "m.json")
    bst.save_model(path)
    res = check_synthetic(path, X[1500:], {"leaf_bits": 24}, str(tmp_path / "w"))
    assert res["failures"] == [], res["failures"]
    exp = export_model(path, ExportOptions(name="m"))
    noted = any("base_score" in n for n in exp.report["notes"])
    assert noted == (not 1e-6 <= base_score <= 1 - 1e-6)


@pytest.mark.parametrize("label", [0, 1])
def test_one_class_training_labels(tmp_path, label):
    """All labels equal: xgboost stores base_score 0 or 1 and still predicts."""
    import xgboost as xgb
    X, _ = _data(400, 5, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="binary:logistic", max_depth=2, verbosity=0),
                        xgb.DMatrix(X, label=np.full(400, label)), num_boost_round=3)
    path = str(tmp_path / "m.json")
    bst.save_model(path)
    res = check_synthetic(path, X[:100], {"leaf_bits": 16}, str(tmp_path / "w"))
    assert res["failures"] == [], res["failures"]


# --------------------------------------------------------------------------- automatic leaf width

def _near_tie_model(tmp_path, spread):
    """Three classes whose margins differ by about ``spread``."""
    rng = np.random.RandomState(0)
    trees = []
    for g in range(3):
        for _ in range(8):
            t = synth.SynthTree(group=g)
            t.add_leaf(0)
            t.split(0, int(rng.randint(3)), float(rng.randn()), False,
                    0.25 + spread * rng.randn(), 0.25 + spread * rng.randn())
            trees.append(t)
    return synth.write_model(str(tmp_path / "m.json"), trees, 3, 3), F32(rng.randn(300, 3))


def test_auto_note_when_no_width_is_enough(tmp_path):
    path, X = _near_tie_model(tmp_path, 3e-9)          # gaps below what even 32-bit leaves resolve
    exp = export_model(path, ExportOptions(name="m"), validation=X)
    note = " ".join(exp.report["notes"])
    ev = exp.evaluation
    still = int((ev.quant_risk & ~ev.shared.float_risk).sum())
    if still:
        assert exp.lplan.bits == 32 and "none of" in note and f"{still} rows are still at risk" in note
    else:
        assert "narrowest" in note
    assert "puts no validation sample at risk" not in note


def test_auto_does_not_pick_a_width_that_changes_classes(tmp_path):
    """When every row is a near-tie for xgboost itself, narrow leaves must not be called harmless."""
    path, X = _near_tie_model(tmp_path, 2e-7)
    auto = export_model(path, ExportOptions(name="m"), validation=X)
    wide = export_model(path, ExportOptions(name="m", leaf_bits=32), validation=X)
    worse = auto.evaluation.differs_from_exact & ~wide.evaluation.differs_from_exact
    assert not worse.any()
    narrow = export_model(path, ExportOptions(name="m", leaf_bits=8), validation=X)
    if (narrow.evaluation.differs_from_exact & ~wide.evaluation.differs_from_exact).any():
        assert auto.lplan.bits > 8


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
    assert max(exp.lplan.max_abs_acc) <= 2 ** 30 - 1      # so that margin differences fit in int32 too


def test_tables_with_identical_contents_are_shared_and_still_referenced(tmp_path):
    """Identical tables are emitted once; every use must follow the surviving name."""
    # 4 classes with one tree each and 4 features with one cut each: the class-end table,
    # the cut-end table and (with equal tree sizes) other small tables can coincide.
    for n_classes, n_feat in ((4, 4), (3, 3), (5, 5), (2, 2)):
        groups = 1 if n_classes == 2 else n_classes
        trees = []
        for g in range(groups):
            t = synth.SynthTree(group=g)
            t.add_leaf(0)
            node = 0
            for f in range(n_feat):
                l, r = t.split(node, f, 0.5 + f, bool(f % 2), 0.1 * (g + 1), -0.1 * (f + 1))
                node = r
            trees.append(t)
        path = synth.write_model(str(tmp_path / f"m{n_classes}.json"), trees, n_feat, n_classes)
        X = F32(np.random.RandomState(0).uniform(-1, 6, size=(200, n_feat)))
        for opts in ({"leaf_bits": 16, "thresholds": "rank"}, {"leaf_bits": 16, "thresholds": "direct"},
                     {"leaf_bits": 16, "thresholds": "rank", "leaf_store": "table"},
                     {"leaf_bits": 8, "thresholds": "rank", "leaf_store": "table"}):
            res = check_synthetic(path, X, opts, str(tmp_path / "w"), cross=True)
            assert res["failures"] == [], (n_classes, opts, res["failures"])
    exp = export_model(path, ExportOptions(name="m", leaf_bits=16, thresholds="rank"))
    names = [a.name for a in exp.generated.arrays]
    assert len(names) == len(set(names))


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
    for std in ("string", "stdint", "math"):
        with pytest.raises(XgbEdgeError, match="standard C header"):
            export_model(path, ExportOptions(name=std))
    assert to_device_inputs(F32([[3, -4]]), 2, "i8").dtype == np.int8
    with pytest.raises(XgbEdgeError, match="numeric"):
        export_model(path, ExportOptions(), validation=np.array([["a", "b"]]))
    with pytest.raises(XgbEdgeError, match="outside the range"):
        export_model(path, ExportOptions(input="i32"), validation=np.array([[2147483648.0, 0.0]]))
    for bad in (16.5, "16.5", True, None):
        with pytest.raises(XgbEdgeError, match="leaf"):
            export_model(path, ExportOptions(leaf_bits=bad))
    with pytest.raises(XgbEdgeError, match="rounds"):
        export_model(path, ExportOptions(rounds=1.5))
    for hook in ("min_leaf_bytes", "min_off_bytes", "min_hdr_bytes", "min_code_bytes"):
        with pytest.raises(XgbEdgeError, match=hook):
            export_model(path, ExportOptions(**{hook: 5}))


def test_awkward_feature_names_do_not_break_the_header(tmp_path):
    import xgboost as xgb
    names = ["accel/*g", "end*/x", "back\\slash\\", "tab\there", "ok", "ünïcode"]
    X, y = _data(300, 6, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="multi:softprob", num_class=3, max_depth=2, verbosity=0),
                        xgb.DMatrix(X, label=y, feature_names=names), num_boost_round=3)
    path = str(tmp_path / "m.json")
    bst.save_model(path)
    exp = export_model(path, ExportOptions(name="m"))
    exp.write(str(tmp_path))
    (tmp_path / "use.c").write_text('#include "m.h"\nint use(const float *x) { return m_predict(x, 0); }\n')
    for cc in (["cc", "-std=c99"], ["c++", "-x", "c++", "-std=c++11"]):
        r = subprocess.run(cc + ["-Wall", "-Wextra", "-Wpedantic", "-Werror", "-c", str(tmp_path / "use.c"), "-o",
                                 str(tmp_path / "use.o")], capture_output=True, text=True)
        assert r.returncode == 0 and not r.stderr, r.stderr


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


def _assert_clean_error(r, *words):
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert r.stderr.startswith("xgb-edge: error:") and "Traceback" not in r.stderr, r.stderr
    for w in words:
        assert w in r.stderr, (w, r.stderr)


def test_cli_reports_bad_paths_and_data_without_tracebacks(tmp_path):
    path = synth.write_model(str(tmp_path / "m.json"), [synth.SynthTree.single_leaf(0.1)], 3, 2)
    t = synth.SynthTree()
    t.add_leaf(0)
    t.split(0, 1, 0.5, False, 0.2, -0.2)
    path = synth.write_model(str(tmp_path / "m.json"), [t], 3, 2)
    out = tmp_path / "out"
    _assert_clean_error(_cli("export", str(tmp_path / "nope.json"), "-o", str(out)), "cannot read the model file")
    _assert_clean_error(_cli("export", str(tmp_path), "-o", str(out)), "cannot read the model file")
    afile = tmp_path / "afile"
    afile.write_text("x")
    _assert_clean_error(_cli("export", path, "-o", str(afile)), "not a directory")
    _assert_clean_error(_cli("export", path, "-o", str(out), "--report-json", str(tmp_path / "no" / "r.json")),
                        "--report-json")
    assert not out.exists(), "nothing may be written when an output location is unusable"
    _assert_clean_error(_cli("report", path, "--leaf-bits", "16.5"), "leaf-bits")
    _assert_clean_error(_cli("report", path, "--rounds", "1.5"), "rounds")
    _assert_clean_error(_cli("export", path, "-o", str(out), "--name", "string"), "standard C header")
    np.save(tmp_path / "text.npy", np.array([["a", "b", "c"]]))
    _assert_clean_error(_cli("report", path, "--validation", str(tmp_path / "text.npy")), "numbers")
    _assert_clean_error(_cli("report", path, "--validation", str(tmp_path / "x.parquet")), "unsupported data file")
    # a CSV: the header line is skipped, an empty or NaN cell is missing, text elsewhere is an error
    good = tmp_path / "good.csv"
    good.write_text("f0,f1,f2\n1,2,3\n,5,NaN\n\n7,8,9\n")
    r = _cli("report", path, "--validation", str(good), "--no-measure")
    assert r.returncode == 0 and "Validation (3 samples)" in r.stdout, r.stderr
    bad = tmp_path / "bad.csv"
    bad.write_text("f0,f1,f2\n1,2,3\n4,oops,6\n")
    _assert_clean_error(_cli("report", path, "--validation", str(bad)), "line 3, column 2", "'oops' is not a number")
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("1,2,3\n4,5\n")
    _assert_clean_error(_cli("report", path, "--validation", str(ragged)), "columns")
    # a compiler we cannot measure with: warn, estimate, still succeed
    r = _cli("report", path, "--cc", "cc", "--cflags=-Os")
    assert r.returncode == 0 and "estimate" in r.stdout and "Traceback" not in r.stderr


def test_avr_report_counts_the_tables_as_ram(tmp_path):
    from xgb_edge.report import available_targets
    if "avr" not in available_targets():
        pytest.skip("avr-gcc not installed")
    rng = np.random.RandomState(0)
    pools = {f: F32(rng.randn(6)) for f in range(4)}
    trees = [synth.random_tree(rng, 4, pools, 12, group=g) for g in range(3)]
    for t in trees:
        for leaf in t.leaves:
            t.value[leaf] = float(F32(rng.randn() * 0.2))
    path = synth.write_model(str(tmp_path / "m.json"), trees, 4, 3)
    rep = tmp_path / "rep.json"
    r = _cli("report", path, "--target", "avr", "--report-json", str(rep))
    assert r.returncode == 0, r.stderr
    d = json.loads(rep.read_text())
    assert d["ram"]["static_bytes"] >= d["flash"]["rodata_bytes_measured"] > 0
    assert "AVR" in r.stdout and "static (.data/.bss): 0 B" not in r.stdout
