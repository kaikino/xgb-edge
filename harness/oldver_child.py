"""Runs under an *older* xgboost: train a few models and record what that version predicts.

Self-contained on purpose (only numpy, scikit-learn and xgboost), because it is
executed with `uv run --with xgboost==<version>` in a throwaway environment.
Usage: oldver_child.py OUTDIR
"""
import json
import os
import sys
import warnings

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

warnings.simplefilter("ignore")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
F32 = np.float32

CASES = {
    "binary_hist": (2, dict(objective="binary:logistic", max_depth=4, tree_method="hist"), 30, 0.0),
    "binary_exact_missing": (2, dict(objective="binary:logistic", max_depth=5, tree_method="exact"), 25, 0.1),
    "multi5_hist_missing": (5, dict(objective="multi:softprob", num_class=5, max_depth=4, tree_method="hist"), 25, 0.1),
    "multi3_softmax_exact": (3, dict(objective="multi:softmax", num_class=3, max_depth=6, tree_method="exact"), 15, 0.0),
    "multi4_dart": (4, dict(objective="multi:softprob", num_class=4, max_depth=3, booster="dart", rate_drop=0.3,
                            skip_drop=0.2), 20, 0.0),
    "multi3_forest": (3, dict(objective="multi:softprob", num_class=3, max_depth=4, num_parallel_tree=3,
                              subsample=0.7, tree_method="hist"), 8, 0.05),
    "binary_basescore": (2, dict(objective="binary:logistic", max_depth=3, base_score=0.8), 20, 0.0),
}

def record(name, path, Xall, obj):
    """Save what this xgboost predicts for Xall on every path a user might take."""
    bst = xgb.Booster(model_file=path)
    dm = xgb.DMatrix(Xall)

    def to_class(p):
        if obj == "multi:softprob":
            return np.argmax(p, axis=1)
        if obj == "multi:softmax":
            return np.asarray(p).astype(np.int64)
        return (np.asarray(p) > 0.5).astype(np.int64)

    cls = to_class(bst.predict(dm))
    margin = bst.predict(dm, output_margin=True).astype(F32).reshape(Xall.shape[0], -1)
    leaf = bst.predict(dm, pred_leaf=True).astype(np.int32).reshape(Xall.shape[0], -1)
    # Booster.inplace_predict is what XGBClassifier.predict uses; it is a separate code path
    margin_ip = np.asarray(bst.inplace_predict(Xall, predict_type="margin")).astype(F32).reshape(Xall.shape[0], -1)
    cls_ip = to_class(np.asarray(bst.inplace_predict(Xall)))
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(path)
        cls_sk = np.asarray(clf.predict(Xall)).astype(np.int64)
    except Exception as e:  # some old releases cannot rebuild the sklearn wrapper from a bare model file
        print(name, ": XGBClassifier.predict not available:", str(e)[:100])
        cls_sk = cls_ip
    np.savez(os.path.join(out, name + ".npz"), X=Xall, cls=cls.astype(np.int64), margin=margin, leaf=leaf,
             objective=obj, margin_ip=margin_ip, cls_ip=cls_ip.astype(np.int64), cls_sk=cls_sk)


for name, (k, params, rounds, missing) in CASES.items():
    X, y = make_classification(n_samples=2600, n_features=10, n_informative=6, n_redundant=0, n_classes=k,
                               n_clusters_per_class=1, random_state=len(name))
    X = X.astype(F32)
    rng = np.random.RandomState(1)
    if missing:
        X[rng.rand(*X.shape) < missing] = np.nan
    Xtr, ytr, Xte = X[:1800], y[:1800], X[1800:]
    params = dict(params, verbosity=0, nthread=2, seed=0)
    bst = xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=rounds)
    path = os.path.join(out, name + ".json")
    bst.save_model(path)
    # rows exactly on, just below and just above split thresholds
    doc = json.load(open(path))
    gb = doc["learner"]["gradient_booster"]
    trees = (gb["gbtree"]["model"] if "gbtree" in gb else gb["model"])["trees"]
    thr = {}
    for t in trees:
        for l, f, c in zip(t["left_children"], t["split_indices"], t["split_conditions"]):
            if l != -1:
                thr.setdefault(f, set()).add(F32(c))
    rows = Xte[rng.randint(0, Xte.shape[0], size=400)].copy()
    feats = sorted(thr)
    for i in range(rows.shape[0]):
        for f in rng.choice(feats, size=rng.randint(1, len(feats) + 1), replace=False):
            t = sorted(thr[f])[rng.randint(len(thr[f]))]
            rows[i, f] = (t, np.nextafter(t, F32(-np.inf)), np.nextafter(t, F32(np.inf)))[rng.randint(3)]
    Xall = np.concatenate([Xte, rows]).astype(F32)
    record(name, path, Xall, params["objective"])


def count_splits(path):
    """Number of split nodes in a saved model."""
    gb = json.load(open(path))["learner"]["gradient_booster"]
    model = gb["gbtree"]["model"] if "gbtree" in gb else gb["model"]
    return sum(l != -1 for t in model["trees"] for l in t["left_children"])


def hand_built_dart():
    """A DART model on which the in-place path of old releases rounds visibly differently.

    class 0: a leaf of 0.5, then 200 trees with leaf 1.49 * 2**-24; class 1: a leaf of 0.5 + 2**-21;
    class 2: a leaf of -5; base_score 0.5, all weights 1.  Exactly, class 0 wins by 1.7e-5, but
    (fl(base + leaf) - base) loses a third of every tiny leaf and adding the rest to 1.0 loses it all.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(60, 2).astype(F32)
    y = np.arange(60) % 3
    bst = xgb.train(dict(objective="multi:softprob", num_class=3, booster="dart", max_depth=1, base_score=0.5,
                         verbosity=0), xgb.DMatrix(X, label=y), 1)
    skel = os.path.join(out, "_skeleton.tmp")
    bst.save_model(skel + ".json")
    doc = json.load(open(skel + ".json"))
    os.remove(skel + ".json")
    gb = doc["learner"]["gradient_booster"]
    model = gb["gbtree"]["model"] if "gbtree" in gb else gb["model"]

    def stump(i, value):
        v = float(F32(value))
        return {"base_weights": [0.0, 0.0, 0.0], "categories": [], "categories_nodes": [],
                "categories_segments": [], "categories_sizes": [], "default_left": [0, 0, 0], "id": i,
                "left_children": [1, -1, -1], "loss_changes": [0.0, 0.0, 0.0], "parents": [2147483647, 0, 0],
                "right_children": [2, -1, -1], "split_conditions": [0.5, v, v], "split_indices": [0, 0, 0],
                "split_type": [0, 0, 0], "sum_hessian": [1.0, 1.0, 1.0],
                "tree_param": {"num_deleted": "0", "num_feature": "2", "num_nodes": "3", "size_leaf_vector": "1"}}

    vals = [(0, 0.5), (1, 0.5 + 2.0 ** -21), (2, -5.0)] + [(0, 1.49 * 2.0 ** -24)] * 200
    model["trees"] = [stump(i, v) for i, (g, v) in enumerate(vals)]
    model["tree_info"] = [g for g, v in vals]
    model["gbtree_model_param"]["num_trees"] = str(len(vals))
    if "iteration_indptr" in model:
        model["iteration_indptr"] = [0, len(vals)]
    weights = [1.0] * len(vals)
    if "weight_drop" in gb:
        gb["weight_drop"] = weights
    else:
        model["weight_drop"] = weights
    doc["learner"]["attributes"] = {}
    path = os.path.join(out, "dart_inplace_rounding.json")
    json.dump(doc, open(path, "w"))
    record("dart_inplace_rounding", path, rng.rand(40, 2).astype(F32), "multi:softprob")


def extreme_base_score():
    """binary:logistic with a base_score outside [1e-6, 1 - 1e-6]: releases differ in how they treat it."""
    for tag, p in (("lo", 1e-7), ("hi", 0.9999999)):
        X, y = make_classification(n_samples=1500, n_features=8, n_informative=5, n_redundant=0, flip_y=0.3,
                                   random_state=7)
        X = X.astype(F32)
        path = os.path.join(out, "binary_base_score_" + tag + ".json")
        # With such a base score the hessians are tiny: the default regularisation would refuse every
        # split.  These settings give trees with real splits and predictions of both classes.
        bst = xgb.train(dict(objective="binary:logistic", max_depth=3, base_score=p, verbosity=0, nthread=2,
                             seed=0, min_child_weight=0, reg_lambda=0, max_delta_step=2, tree_method="exact"),
                        xgb.DMatrix(X[:1000], label=y[:1000]), num_boost_round=40)
        bst.save_model(path)
        splits = count_splits(path)
        assert splits > 50, "the extreme-base-score model should have real splits, has %d" % splits
        record("binary_base_score_" + tag, path, X[1000:], "binary:logistic")


hand_built_dart()
extreme_base_score()
print("xgboost", xgb.__version__, "wrote", len(CASES) + 3, "cases to", out)
