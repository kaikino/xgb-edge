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
    dm = xgb.DMatrix(Xall)
    p = bst.predict(dm)
    obj = params["objective"]
    if obj == "multi:softprob":
        cls = np.argmax(p, axis=1)
    elif obj == "multi:softmax":
        cls = p.astype(np.int64)
    else:
        cls = (p > 0.5).astype(np.int64)
    margin = bst.predict(dm, output_margin=True).astype(F32).reshape(Xall.shape[0], -1)
    leaf = bst.predict(dm, pred_leaf=True).astype(np.int32).reshape(Xall.shape[0], -1)
    np.savez(os.path.join(out, name + ".npz"), X=Xall, cls=cls.astype(np.int64), margin=margin, leaf=leaf,
             objective=obj)
print("xgboost", xgb.__version__, "wrote", len(CASES), "cases to", out)
