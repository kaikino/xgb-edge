"""Reference evaluation of a model in XGBoost's own float32 semantics.

This is the ground truth the quantized model is compared against inside the
tool (the test-suite additionally checks this module against xgboost itself).
A feature value goes left iff ``x < threshold`` as float32; NaN is "missing"
and follows the node's default direction.
"""
from __future__ import annotations

import numpy as np

from .model import Model, Tree, XgbEdgeError


def as_feature_matrix(X, n_features: int) -> np.ndarray:
    """Coerce input to the float32 matrix XGBoost would see."""
    X = np.asarray(X)
    if X.dtype == np.bool_:
        X = X.astype(np.float32)
    if not (np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, np.integer)):
        raise XgbEdgeError(f"feature data must be numeric, got an array of type {X.dtype}")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise XgbEdgeError(f"feature data must be 2-D (rows x features), got shape {X.shape}")
    if X.shape[1] != n_features:
        raise XgbEdgeError(
            f"feature data has {X.shape[1]} columns but the model expects {n_features}")
    if X.dtype != np.float32:
        with np.errstate(over="ignore", invalid="ignore"):
            X = X.astype(np.float32)
    return np.ascontiguousarray(X)


def tree_leaves(tree: Tree, X: np.ndarray) -> np.ndarray:
    """Leaf node id reached by every row of float32 matrix ``X``."""
    n = X.shape[0]
    node = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    left, right = tree.left, tree.right
    for _ in range(tree.depth):
        internal = left[node] >= 0
        if not internal.any():
            break
        x = X[rows, np.where(internal, tree.feature[node], 0)]
        go_left = np.where(np.isnan(x), tree.default_left[node], x < tree.threshold[node])
        nxt = np.where(go_left, left[node], right[node])
        node = np.where(internal, nxt, node)
    return node


def leaf_indices(model: Model, X: np.ndarray) -> np.ndarray:
    """(rows, trees) matrix of leaf node ids, comparable to xgboost's pred_leaf."""
    X = as_feature_matrix(X, model.n_features)
    out = np.zeros((X.shape[0], len(model.trees)), dtype=np.int32)
    for j, t in enumerate(model.trees):
        out[:, j] = tree_leaves(t, X)
    return out
