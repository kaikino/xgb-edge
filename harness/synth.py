"""Hand-built XGBoost models, for tests that need exact control over thresholds.

``build_json`` turns explicit trees into a complete XGBoost JSON document that
xgboost itself loads (the skeleton comes from a tiny model trained with the
installed xgboost, so the format always matches that version).  That lets a
test place thresholds at chosen float32 values, build single-leaf or very deep
trees, leave features unused, and so on, and still ask xgboost what the right
answer is.
"""
from __future__ import annotations

import copy
import json
import os
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SynthTree:
    """A tree as flat arrays in XGBoost's layout (children of a node are consecutive)."""

    left: List[int] = field(default_factory=list)       # -1 for a leaf
    right: List[int] = field(default_factory=list)
    feature: List[int] = field(default_factory=list)
    value: List[float] = field(default_factory=list)    # threshold for splits, leaf value for leaves
    default_left: List[int] = field(default_factory=list)
    group: int = 0

    def add_leaf(self, value: float) -> int:
        self.left.append(-1)
        self.right.append(-1)
        self.feature.append(0)
        self.value.append(float(np.float32(value)))
        self.default_left.append(0)
        return len(self.left) - 1

    def split(self, node: int, feature: int, threshold: float, default_left: bool,
              left_value: float = 0.0, right_value: float = 0.0):
        """Turn leaf ``node`` into a split; returns the ids of its two new leaf children."""
        l = self.add_leaf(left_value)
        r = self.add_leaf(right_value)
        self.left[node], self.right[node] = l, r
        self.feature[node] = int(feature)
        self.value[node] = float(np.float32(threshold))
        self.default_left[node] = int(bool(default_left))
        return l, r

    @property
    def leaves(self) -> List[int]:
        return [i for i, l in enumerate(self.left) if l < 0]

    @classmethod
    def single_leaf(cls, value: float, group: int = 0) -> "SynthTree":
        t = cls(group=group)
        t.add_leaf(value)
        return t


def random_tree(rng: np.random.RandomState, n_features: int, thresholds: Dict[int, Sequence[float]],
                n_splits: int, group: int = 0, chain: bool = False,
                features: Optional[Sequence[int]] = None) -> SynthTree:
    """Random tree with ``n_splits`` splits whose thresholds are drawn from ``thresholds[feature]``."""
    t = SynthTree(group=group)
    t.add_leaf(0.0)
    feats = list(features) if features is not None else list(range(n_features))
    open_leaves = [0]
    for _ in range(n_splits):
        node = open_leaves.pop(-1 if chain else rng.randint(len(open_leaves)))
        f = feats[rng.randint(len(feats))]
        pool = thresholds[f]
        thr = pool[rng.randint(len(pool))]
        l, r = t.split(node, f, thr, bool(rng.randint(2)))
        if chain:
            # keep extending one side so the tree becomes a deep chain
            open_leaves.extend([l, r] if rng.randint(2) else [r, l])
        else:
            open_leaves.extend([l, r])
    return t


def number_leaves(trees: Sequence[SynthTree], step: float = 2.0 ** -10) -> None:
    """Give every leaf of every tree a distinct value k * step (k = 1, 2, ...), exactly representable."""
    k = 1
    for t in trees:
        for leaf in t.leaves:
            t.value[leaf] = float(np.float32(k * step))
            k += 1


def _fmt(x: float) -> float:
    return float(np.float32(x))


def _tree_json(i: int, t: SynthTree, n_features: int) -> Dict:
    n = len(t.left)
    parents = [2147483647] * n
    for p in range(n):
        if t.left[p] >= 0:
            parents[t.left[p]] = p
            parents[t.right[p]] = p
    return {
        "base_weights": [0.0] * n, "categories": [], "categories_nodes": [], "categories_segments": [],
        "categories_sizes": [], "default_left": list(t.default_left), "id": i,
        "left_children": list(t.left), "loss_changes": [0.0] * n, "parents": parents,
        "right_children": list(t.right), "split_conditions": [_fmt(v) for v in t.value],
        "split_indices": list(t.feature), "split_type": [0] * n, "sum_hessian": [1.0] * n,
        "tree_param": {"num_deleted": "0", "num_feature": str(n_features), "num_nodes": str(n),
                       "size_leaf_vector": "1"},
    }


@lru_cache(maxsize=None)
def _skeleton_text(objective: str, n_classes: int, n_features: int) -> str:
    import tempfile

    import xgboost as xgb
    rng = np.random.RandomState(0)
    X = rng.rand(40, n_features).astype(np.float32)
    y = np.arange(40) % max(n_classes, 2)
    params = {"objective": objective, "max_depth": 1, "verbosity": 0, "base_score": 0.5}
    if objective.startswith("multi:"):
        params["num_class"] = n_classes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=1)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "skeleton.json")
        bst.save_model(p)
        return open(p).read()


def _dump(doc) -> str:
    """Serialise with float32 values written as their shortest round-trip decimal, like xgboost does."""
    def enc(o):
        if isinstance(o, dict):
            return "{" + ",".join(json.dumps(k) + ":" + enc(v) for k, v in o.items()) + "}"
        if isinstance(o, (list, tuple)):
            return "[" + ",".join(enc(v) for v in o) + "]"
        if isinstance(o, bool):
            return "true" if o else "false"
        if isinstance(o, float):
            if o != o:
                return "NaN"
            if o in (float("inf"), float("-inf")):
                return "Infinity" if o > 0 else "-Infinity"
            return np.format_float_scientific(np.float32(o), unique=True, trim="0").replace("e", "E")
        return json.dumps(o)
    return enc(doc)


def build_json(trees: Sequence[SynthTree], n_features: int, n_classes: int = 2,
               objective: Optional[str] = None, base_score: Optional[Sequence[float]] = None) -> str:
    """A complete XGBoost JSON model made of ``trees``.  ``n_classes == 2`` means binary:logistic."""
    if objective is None:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    doc = json.loads(_skeleton_text(objective, n_classes, n_features))
    learner = doc["learner"]
    n_groups = n_classes if objective.startswith("multi:") else 1
    model = learner["gradient_booster"]["model"]
    model["trees"] = [_tree_json(i, t, n_features) for i, t in enumerate(trees)]
    model["tree_info"] = [t.group for t in trees]
    model["gbtree_model_param"]["num_trees"] = str(len(trees))
    if "iteration_indptr" in model:
        model["iteration_indptr"] = [0, len(trees)]
    if base_score is None:
        base_score = [0.5] * n_groups
    old = learner["learner_model_param"]["base_score"]
    vals = ",".join(np.format_float_scientific(np.float32(v), unique=True, trim="0").replace("e", "E")
                    for v in base_score)
    learner["learner_model_param"]["base_score"] = f"[{vals}]" if old.strip().startswith("[") else vals.split(",")[0]
    learner["learner_model_param"]["num_feature"] = str(n_features)
    return _dump(doc)


def write_model(path: str, *args, **kwargs) -> str:
    with open(path, "w") as f:
        f.write(build_json(*args, **kwargs))
    return path


def xgb_reference(path: str, X: np.ndarray):
    """What xgboost itself says for ``X``: (class, margins (n, groups), leaf ids (n, trees))."""
    import xgboost as xgb

    from xgb_edge.model import load
    from harness.zoo import xgb_predict_class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.Booster(model_file=path)
        # a model trained on named columns only accepts data carrying the same names
        dm = xgb.DMatrix(np.asarray(X, dtype=np.float32), feature_names=bst.feature_names,
                         feature_types=bst.feature_types)
        objective = load(path).objective
        cls = xgb_predict_class(bst, objective, dm)
        margin = bst.predict(dm, output_margin=True).astype(np.float32).reshape(X.shape[0], -1)
        leaf = bst.predict(dm, pred_leaf=True).astype(np.int64).reshape(X.shape[0], -1)
    return cls, margin, leaf
