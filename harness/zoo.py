"""The model zoo: a spread of trained XGBoost classifiers with test data.

Each entry trains one model (cached under build/zoo/<name>/) and records what
xgboost itself predicts for a set of test rows: the held-out split plus
adversarial rows placed exactly on, just below and just above split
thresholds, plus missing values and extreme magnitudes.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZOO_DIR = os.path.join(ROOT, "build", "zoo")
CACHE_VERSION = 4


@dataclass
class Spec:
    name: str
    data: Dict                     # dataset recipe, see make_data()
    params: Dict                   # xgboost training parameters
    rounds: int
    exports: List[Dict] = field(default_factory=lambda: [{}])  # ExportOptions overrides to test
    board: bool = False            # part of the on-device subset
    int_test: Optional[List[int]] = None  # also generate integer test rows in this range
    note: str = ""

    def key(self) -> str:
        import xgboost
        blob = json.dumps([CACHE_VERSION, xgboost.__version__, self.name, self.data, self.params,
                           self.rounds, self.int_test], sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- data

def make_data(recipe: Dict):
    """Return (X_train, y_train, X_test) as float32 feature matrices (NaN = missing)."""
    from sklearn.datasets import load_digits, make_classification
    r = dict(recipe)
    kind = r.pop("kind", "synth")
    seed = r.pop("seed", 0)
    rng = np.random.RandomState(seed)
    missing = r.pop("missing", 0.0)
    n_test = r.pop("n_test", 1500)
    int_range = r.pop("int_range", None)
    scales = r.pop("scales", None)
    extra = r.pop("extra", None)
    if kind == "digits":
        X, y = load_digits(return_X_y=True)
        reps = r.pop("repeat", 1)
        if reps > 1:  # enlarge with pixel noise so bigger models have something to fit
            Xs, ys = [X], [y]
            for _ in range(reps - 1):
                Xs.append(np.clip(X + rng.randint(-2, 3, size=X.shape), 0, 16))
                ys.append(y)
            X, y = np.concatenate(Xs), np.concatenate(ys)
    elif kind == "synth":
        n_classes = r.pop("classes")
        n_features = r.pop("features")
        n_samples = r.pop("samples", 6000)
        n_inf = r.pop("informative", max(2, min(n_features, int(np.ceil(np.log2(max(n_classes, 2)))) + 4)))
        X, y = make_classification(
            n_samples=n_samples + n_test, n_features=n_features, n_informative=n_inf,
            n_redundant=min(2, max(0, n_features - n_inf)), n_classes=n_classes,
            n_clusters_per_class=1, class_sep=r.pop("sep", 1.0), flip_y=r.pop("flip", 0.02),
            random_state=seed)
    else:
        raise ValueError(kind)
    if r:
        raise ValueError(f"unknown data options {sorted(r)}")
    X = X.astype(np.float64)
    if scales == "wild":
        # wildly different magnitudes per column, including tiny and huge ones
        mags = 10.0 ** rng.uniform(-30, 30, size=X.shape[1])
        mags[: min(3, X.shape[1])] = [1e-38, 1e30, 1.0][: min(3, X.shape[1])]
        X = X * mags
    if extra == "degenerate":
        # constant column, duplicated column, all-missing column, binary column
        n = X.shape[0]
        X = np.column_stack([X, np.full(n, 3.25), X[:, 0], np.full(n, np.nan),
                             (X[:, 1] > np.median(X[:, 1])).astype(float)])
    if int_range is not None:
        lo, hi = int_range
        span = np.nanmax(np.abs(X), axis=0)
        span[span == 0] = 1.0
        X = np.clip(np.rint(X / span * hi), lo, hi)
    if missing:
        X = X.copy()
        X[rng.rand(*X.shape) < missing] = np.nan
    with np.errstate(over="ignore"):
        X = X.astype(np.float32)
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    n_test = min(n_test, X.shape[0] // 3)
    return X[n_test:], y[n_test:], X[:n_test]


def boundary_rows(model, X_base: np.ndarray, rng, n_rows: int, integer: Optional[tuple]) -> np.ndarray:
    """Rows whose features sit exactly on, one float32 below and one above split thresholds."""
    thr: Dict[int, np.ndarray] = {}
    for t in model.trees:
        nodes = t.internal_nodes
        for f in np.unique(t.feature[nodes]):
            v = t.threshold[nodes][t.feature[nodes] == f]
            thr[int(f)] = np.unique(np.concatenate([thr.get(int(f), np.zeros(0, np.float32)), v]))
    feats = sorted(thr)
    if not feats or X_base.shape[0] == 0:
        return np.zeros((0, model.n_features), dtype=np.float32)
    rows = X_base[rng.randint(0, X_base.shape[0], size=n_rows)].copy()
    ninf, pinf = np.float32(-np.inf), np.float32(np.inf)
    for i in range(n_rows):
        k = rng.randint(1, len(feats) + 1) if i % 3 else len(feats)
        for f in rng.choice(feats, size=k, replace=False):
            t = thr[f][rng.randint(thr[f].size)]
            if integer is None:
                choice = rng.randint(3)
                v = (t, np.nextafter(t, ninf, dtype=np.float32), np.nextafter(t, pinf, dtype=np.float32))[choice]
            else:
                c = np.ceil(np.float64(t))
                v = np.clip(c + rng.randint(-2, 3), integer[0], integer[1])
            rows[i, f] = v
    if integer is None:
        special = np.array([0.0, -0.0, np.finfo(np.float32).max, -np.finfo(np.float32).max,
                            np.float32(1e-45), np.float32(-1e-45), np.finfo(np.float32).tiny, 1.0, -1.0],
                           dtype=np.float32)
        m = max(8, n_rows // 5)
        extra = X_base[rng.randint(0, X_base.shape[0], size=m)].copy()
        mask = rng.rand(*extra.shape) < 0.3
        extra[mask] = special[rng.randint(0, special.size, size=int(mask.sum()))]
        nan_rows = X_base[rng.randint(0, X_base.shape[0], size=m)].copy()
        nan_rows[rng.rand(*nan_rows.shape) < 0.3] = np.nan
        all_nan = np.full((1, model.n_features), np.nan, dtype=np.float32)
        all_special = np.tile(special[:, None], (1, model.n_features))
        rows = np.concatenate([rows, extra, nan_rows, all_nan, all_special])
    else:
        lo, hi = integer
        edge = np.array([lo, hi, 0 if lo <= 0 <= hi else lo, lo + 1, hi - 1], dtype=np.float32)
        rows = np.concatenate([rows, np.tile(edge[:, None], (1, model.n_features))])
    return rows.astype(np.float32)


# --------------------------------------------------------------------------- training

def xgb_predict_class(bst, objective: str, dm) -> np.ndarray:
    """The class XGBoost's own predict() reports (as XGBClassifier.predict does it)."""
    p = bst.predict(dm)
    if objective == "multi:softprob":
        return np.argmax(p, axis=1).astype(np.int64)
    if objective == "multi:softmax":
        return p.astype(np.int64)
    return (p > 0.5).astype(np.int64)


@dataclass
class Trained:
    spec: Spec
    dir: str
    model_path: str
    X: np.ndarray            # float32 test rows (held-out + adversarial)
    n_heldout: int           # the first n_heldout rows are the held-out split
    xgb_class: np.ndarray
    xgb_margin: np.ndarray   # float32 (n, groups)
    xgb_leaf: np.ndarray     # int32 (n, trees)
    objective: str


def train(spec: Spec, force: bool = False) -> Trained:
    import xgboost as xgb
    from xgb_edge.model import load
    d = os.path.join(ZOO_DIR, spec.name)
    stamp = os.path.join(d, "key.txt")
    model_path = os.path.join(d, "model.json")
    if not force and os.path.exists(stamp) and open(stamp).read() == spec.key():
        z = np.load(os.path.join(d, "ref.npz"))
        return Trained(spec, d, model_path, z["X"], int(z["n_heldout"]), z["xgb_class"],
                       z["xgb_margin"], z["xgb_leaf"], str(z["objective"]))
    os.makedirs(d, exist_ok=True)
    Xtr, ytr, Xte = make_data(spec.data)
    params = dict(spec.params)
    params.setdefault("nthread", 4)
    params.setdefault("verbosity", 0)
    params.setdefault("seed", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        bst = xgb.train(params, dtrain, num_boost_round=spec.rounds)
    bst.save_model(model_path)
    model = load(model_path)
    rng = np.random.RandomState(1234)
    n_b = min(600, max(100, Xte.shape[0] // 2))
    parts = [Xte, boundary_rows(model, Xte, rng, n_rows=n_b, integer=None)]
    int_test = spec.data.get("int_range") or spec.int_test
    if int_test:
        # integer-valued rows around the thresholds, for the integer-input exports
        base = Xte[~np.isnan(Xte).any(axis=1)]
        base = base if base.shape[0] else np.zeros((1, Xte.shape[1]), dtype=np.float32)
        parts.append(boundary_rows(model, np.rint(base), rng, n_rows=n_b, integer=tuple(int_test)))
    X = np.concatenate(parts).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm = xgb.DMatrix(X)
        cls = xgb_predict_class(bst, model.objective, dm)
        margin = bst.predict(dm, output_margin=True).astype(np.float32).reshape(X.shape[0], -1)
        leaf = bst.predict(dm, pred_leaf=True).astype(np.int32).reshape(X.shape[0], -1)
    np.savez(os.path.join(d, "ref.npz"), X=X, n_heldout=Xte.shape[0], xgb_class=cls,
             xgb_margin=margin, xgb_leaf=leaf, objective=model.objective)
    with open(os.path.join(d, "spec.json"), "w") as f:
        json.dump(asdict(spec), f, indent=1)
    with open(stamp, "w") as f:
        f.write(spec.key())
    return Trained(spec, d, model_path, X, Xte.shape[0], cls, margin, leaf, model.objective)


# --------------------------------------------------------------------------- the zoo

def _mc(k, **kw):
    return dict(objective="multi:softprob", num_class=k, **kw)


def _bin(**kw):
    return dict(objective="binary:logistic", **kw)


WIDTHS = [{"leaf_bits": 8}, {"leaf_bits": 16}, {"leaf_bits": 24}, {"leaf_bits": 32}]
FORCED = [
    {"leaf_bits": 16, "leaf_store": "table"},
    {"leaf_bits": 16, "leaf_store": "inline"},
    {"leaf_bits": 32, "leaf_store": "table"},
    {"leaf_bits": 12, "leaf_store": "table"},
    {"leaf_bits": 16, "min_code_bytes": 2},
    {"leaf_bits": 16, "min_hdr_bytes": 2},
    {"leaf_bits": 16, "min_off_bytes": 2},
    {"leaf_bits": 16, "min_off_bytes": 3, "min_leaf_bytes": 3},
    {"leaf_bits": 24, "min_leaf_bytes": 4, "min_hdr_bytes": 2, "min_code_bytes": 2, "min_off_bytes": 4},
    {"leaf_bits": 8, "leaf_store": "table", "min_leaf_bytes": 2},
]

SPECS: List[Spec] = [
    # --- binary
    Spec("bin_d2_r5_hist", dict(classes=2, features=10, samples=2000), _bin(max_depth=2, tree_method="hist"), 5,
         exports=[{}] + WIDTHS + FORCED, board=True, note="smallest: binary, depth 2, 5 rounds"),
    Spec("bin_d6_r200_exact_missing", dict(classes=2, features=20, missing=0.1, seed=1),
         _bin(max_depth=6, tree_method="exact", eta=0.1), 200, exports=[{}] + WIDTHS, board=True),
    Spec("bin_d4_r60_hinge", dict(classes=2, features=12, seed=2),
         dict(objective="binary:hinge", max_depth=4, tree_method="hist"), 60, exports=[{}, {"leaf_bits": 16}]),
    Spec("bin_d3_r40_basescore", dict(classes=2, features=8, seed=3),
         _bin(max_depth=3, tree_method="hist", base_score=0.9), 40, exports=[{}, {"leaf_bits": 16}]),
    Spec("bin_d10_r40_exact", dict(classes=2, features=15, samples=8000, seed=4, flip=0.1),
         _bin(max_depth=10, tree_method="exact", min_child_weight=0.1), 40, exports=[{}, {"leaf_bits": 16}]),
    # --- few classes
    Spec("mc3_d3_r10_hist", dict(classes=3, features=6, samples=1500, seed=5), _mc(3, max_depth=3, tree_method="hist"), 10,
         exports=[{}] + WIDTHS + FORCED, board=True),
    Spec("mc4_d2_r300_hist", dict(classes=4, features=10, seed=6), _mc(4, max_depth=2, tree_method="hist", eta=0.2), 300,
         exports=[{}] + WIDTHS),
    Spec("mc5_d4_r50_hist_missing", dict(classes=5, features=16, missing=0.15, seed=7),
         _mc(5, max_depth=4, tree_method="hist"), 50, exports=[{}] + WIDTHS + FORCED, board=True),
    Spec("mc5_d7_r40_exact_missing", dict(classes=5, features=12, missing=0.05, seed=8),
         _mc(5, max_depth=7, tree_method="exact"), 40, exports=[{}, {"leaf_bits": 16}, {"leaf_bits": 24}]),
    Spec("mc7_d5_r80_softmax", dict(classes=7, features=14, seed=9),
         dict(objective="multi:softmax", num_class=7, max_depth=5, tree_method="hist"), 80, exports=[{}, {"leaf_bits": 16}]),
    # --- 10+ classes
    Spec("mc10_digits_d6_r100_hist", dict(kind="digits", repeat=4, seed=10),
         _mc(10, max_depth=6, tree_method="hist", eta=0.2), 100,
         exports=[{}] + WIDTHS + [{"input": "u8"}, {"input": "u8", "leaf_bits": 16}, {"input": "i16", "leaf_bits": 16},
                                  {"input": "i32", "leaf_bits": 16}], board=True, int_test=[0, 255]),
    Spec("mc10_d10_r30_exact", dict(classes=10, features=24, samples=12000, seed=11, flip=0.05),
         _mc(10, max_depth=10, tree_method="exact", min_child_weight=0.5), 30, exports=[{}, {"leaf_bits": 16}], board=True),
    Spec("mc12_d8_r120_hist_missing", dict(classes=12, features=30, samples=12000, missing=0.08, seed=12),
         _mc(12, max_depth=8, tree_method="hist", eta=0.15), 120, exports=[{}, {"leaf_bits": 16}]),
    Spec("mc16_d5_r400_hist", dict(classes=16, features=28, samples=12000, seed=13),
         _mc(16, max_depth=5, tree_method="hist", eta=0.1), 400, exports=[{}, {"leaf_bits": 16}, {"leaf_bits": 24}],
         note="several hundred rounds: 6400 trees"),
    # --- structural variants
    Spec("mc4_rf_npt3", dict(classes=4, features=12, seed=14),
         _mc(4, max_depth=5, tree_method="hist", num_parallel_tree=3, subsample=0.7, colsample_bynode=0.7), 25,
         exports=[{}, {"leaf_bits": 16}], note="num_parallel_tree=3 (boosted random forest)"),
    Spec("mc3_dart", dict(classes=3, features=10, seed=15),
         _mc(3, max_depth=4, tree_method="hist", booster="dart", rate_drop=0.2, skip_drop=0.3), 40,
         exports=[{}] + WIDTHS, note="DART: tree weights folded into the leaves"),
    Spec("bin_dart_exact", dict(classes=2, features=10, seed=16),
         _bin(max_depth=3, tree_method="exact", booster="dart", rate_drop=0.1, normalize_type="forest"), 30,
         exports=[{}, {"leaf_bits": 16}]),
    Spec("mc6_lossguide_deep", dict(classes=6, features=14, samples=10000, seed=17, flip=0.08),
         _mc(6, tree_method="hist", grow_policy="lossguide", max_leaves=96, max_depth=0, min_child_weight=0.2), 30,
         exports=[{}, {"leaf_bits": 16}, {"leaf_bits": 16, "leaf_store": "table"}], note="unbalanced trees, unlimited depth"),
    Spec("mc5_maxbin1024", dict(classes=5, features=6, samples=20000, seed=18, flip=0.1),
         _mc(5, max_depth=8, tree_method="hist", max_bin=1024, min_child_weight=0.1), 120,
         exports=[{}, {"leaf_bits": 16}], note=">255 thresholds per feature: 16-bit threshold codes"),
    Spec("mc3_wide_f300", dict(classes=3, features=300, informative=40, samples=5000, seed=19),
         _mc(3, max_depth=6, tree_method="hist", colsample_bynode=0.3), 60,
         exports=[{}, {"leaf_bits": 16}], note=">64 feature slots: 2-byte node header"),
    Spec("mc4_int16", dict(classes=4, features=12, int_range=[-32768, 32767], seed=20),
         _mc(4, max_depth=6, tree_method="hist"), 60,
         exports=[{"input": "i16"}, {"input": "i16", "leaf_bits": 16}, {"input": "i32", "leaf_bits": 16},
                  {"leaf_bits": 16}, {"input": "i16", "leaf_bits": 16, "min_code_bytes": 2}],
         note="integer features, integer-input export"),
    Spec("mc3_int8_exact", dict(classes=3, features=8, int_range=[-128, 127], seed=21),
         _mc(3, max_depth=5, tree_method="exact"), 30,
         exports=[{"input": "i8"}, {"input": "i8", "leaf_bits": 16}, {"input": "i16", "leaf_bits": 16}]),
    Spec("mc5_wild_scales", dict(classes=5, features=12, scales="wild", missing=0.05, seed=22),
         _mc(5, max_depth=5, tree_method="hist"), 40, exports=[{}, {"leaf_bits": 16}],
         note="feature magnitudes from 1e-38 to 1e30"),
    Spec("mc4_degenerate_cols", dict(classes=4, features=8, extra="degenerate", seed=23),
         _mc(4, max_depth=4, tree_method="exact"), 30, exports=[{}, {"leaf_bits": 16}],
         note="constant, duplicated, all-missing and binary columns"),
    Spec("mc3_single_leaf_trees", dict(classes=3, features=6, samples=600, seed=24),
         _mc(3, max_depth=3, tree_method="hist", gamma=4.0), 30, exports=[{}] + WIDTHS,
         note="heavy pruning: many trees are a single leaf"),
    Spec("mc3_all_single_leaf", dict(classes=3, features=5, samples=300, seed=25),
         _mc(3, max_depth=3, tree_method="hist", gamma=1e6), 6, exports=[{}, {"leaf_bits": 16}, {"leaf_bits": 8}],
         note="no split anywhere: constant model"),
    Spec("bin_few_leaf_values", dict(classes=2, features=6, samples=400, int_range=[0, 1], seed=26),
         _bin(max_depth=3, tree_method="exact", eta=1.0, max_delta_step=1), 60,
         exports=[{}, {"leaf_bits": 16}, {"input": "u8"}], note="binary features: few distinct leaves, shared table wins"),
    Spec("mc26_d8_r150_hist", dict(classes=26, features=32, samples=30000, informative=14, seed=27, flip=0.05),
         _mc(26, max_depth=8, tree_method="hist", eta=0.15, min_child_weight=0.5), 150,
         exports=[{}, {"leaf_bits": 16}], note="large: 3900 trees"),
]


def by_name(name: str) -> Spec:
    for s in SPECS:
        if s.name == name:
            return s
    raise KeyError(name)
