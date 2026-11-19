"""Load an XGBoost JSON model into an exact, validated in-memory form.

Everything the exporter needs is read here and checked here.  Any model
feature that the generated C cannot reproduce exactly is rejected with an
error that names the feature; nothing is silently dropped.

Numbers are kept bit-exact: XGBoost stores split thresholds and leaf values
as float32, and writes them to JSON as the shortest decimal that round-trips
to that float32.  We round the decimal text to float32 the same way XGBoost
does when it loads the file (round-to-nearest-even from the exact decimal
value, not via an intermediate double).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import List, Optional

import numpy as np


class XgbEdgeError(ValueError):
    """Base class for every error this tool raises on purpose."""


class UnsupportedModelError(XgbEdgeError):
    """A valid XGBoost model that uses something xgb-edge cannot export exactly."""


class ModelFormatError(XgbEdgeError):
    """The file is not a well-formed XGBoost JSON model."""


#: objective -> (number of classes is taken from num_class?, description)
SUPPORTED_OBJECTIVES = ("multi:softprob", "multi:softmax", "binary:logistic", "binary:hinge")

_TWO25 = 33554432.0
_F32_MIN_NORMAL = 2.0 ** -125  # below this, float32 spacing is not 24-bit relative


class _Midpoint(float):
    """A JSON number whose double value might sit exactly between two float32 values.

    Rounding such a double to float32 can differ from rounding the original
    decimal text, so the text is kept and resolved exactly.
    """

    __slots__ = ("text",)


def _nextafter32(x: np.float32, up: bool) -> np.float32:
    with np.errstate(over="ignore"):
        return np.nextafter(x, np.float32(np.inf if up else -np.inf), dtype=np.float32)


def round_decimal_to_f32(text: str) -> np.float32:
    """Correctly round a decimal string to float32 (ties to even), exactly."""
    d = float(text)
    if math.isnan(d) or math.isinf(d):
        return np.float32(d)
    exact = Fraction(Decimal(text))
    with np.errstate(over="ignore"):
        c = np.float32(d)
    if np.isinf(c):
        # Beyond float32 range: IEEE rounds to inf only above the overflow threshold.
        fmax = np.finfo(np.float32).max
        limit = Fraction(float(fmax)) + Fraction(2) ** 103  # FLT_MAX + half ulp
        if abs(exact) >= limit:
            return c
        return np.float32(math.copysign(float(fmax), d))
    cands = {float(_nextafter32(c, False)), float(c), float(_nextafter32(c, True))}
    cands = [v for v in cands if not math.isinf(v)]
    best = None
    for v in sorted(cands):
        dist = abs(Fraction(v) - exact)
        if best is None or dist < best[0]:
            best = (dist, [v])
        elif dist == best[0]:
            best[1].append(v)
    winners = best[1]
    if len(winners) == 1:
        return np.float32(winners[0])
    # exact tie: pick the one with an even mantissa
    for v in winners:
        if (int(np.float32(v).view(np.uint32)) & 1) == 0:
            return np.float32(v)
    raise AssertionError("no even candidate in tie")  # pragma: no cover


class _FloatHook:
    """json parse_float hook that flags doubles which need exact float32 rounding."""

    def __init__(self) -> None:
        self.flagged = False

    def __call__(self, text: str) -> float:
        d = float(text)
        if d == 0.0 or d != d or d in (math.inf, -math.inf):
            return d
        a = abs(d)
        if a < _F32_MIN_NORMAL:
            suspicious = True
        else:
            m, _ = math.frexp(a)
            x = m * _TWO25
            # a float32 midpoint has exactly 25 significant bits, the last one set
            suspicious = x == math.floor(x) and (int(x) & 1) == 1
        if suspicious:
            v = _Midpoint(d)
            v.text = text
            self.flagged = True
            return v
        return d


def _to_f32(values, hook: Optional[_FloatHook], what: str) -> np.ndarray:
    try:
        arr64 = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ModelFormatError(f"{what}: expected a list of numbers ({e})") from None
    if arr64.ndim != 1:
        raise ModelFormatError(f"{what}: expected a flat list of numbers")
    with np.errstate(over="ignore"):
        out = arr64.astype(np.float32)
    if hook is not None and hook.flagged:
        for i, v in enumerate(values):
            if type(v) is _Midpoint:
                out[i] = round_decimal_to_f32(v.text)
    return out


def _to_int(values, what: str) -> np.ndarray:
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError) as e:
        raise ModelFormatError(f"{what}: expected a list of integers ({e})") from None
    if arr.ndim != 1:
        raise ModelFormatError(f"{what}: expected a flat list")
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.dtype == np.bool_:
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        if (np.issubdtype(arr.dtype, np.floating) and np.all(np.isfinite(arr)) and np.all(np.abs(arr) < 2.0 ** 53)
                and np.all(arr == np.floor(arr))):
            return arr.astype(np.int64)
        raise ModelFormatError(f"{what}: expected integers, got {arr.dtype}")
    if arr.dtype == np.uint64 and arr.size and arr.max() > np.iinfo(np.int64).max:
        raise ModelFormatError(f"{what}: integer out of range")
    return arr.astype(np.int64)


@dataclass
class Tree:
    """One regression tree, exactly as XGBoost evaluates it."""

    index: int                 # position in the model file
    group: int                 # which margin this tree adds to
    weight: float              # DART tree weight as an exact float32 value (1.0 otherwise)
    left: np.ndarray           # int64[n]; -1 marks a leaf
    right: np.ndarray          # int64[n]
    feature: np.ndarray        # int64[n]; split feature of internal nodes
    threshold: np.ndarray      # float32[n]; go left iff x < threshold
    default_left: np.ndarray   # bool[n]; where a missing value goes
    value: np.ndarray          # float32[n]; leaf value at leaves
    preorder: np.ndarray       # reachable node ids, parents before children
    node_depth: np.ndarray     # int64[n]; depth of each reachable node (root = 0)

    @property
    def is_leaf(self) -> np.ndarray:
        return self.left < 0

    @property
    def internal_nodes(self) -> np.ndarray:
        return self.preorder[self.left[self.preorder] >= 0]

    @property
    def leaf_nodes(self) -> np.ndarray:
        return self.preorder[self.left[self.preorder] < 0]

    @property
    def n_internal(self) -> int:
        return int(self.internal_nodes.size)

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_nodes.size)

    @property
    def depth(self) -> int:
        """Largest number of split comparisons on any root-to-leaf path."""
        return int(self.node_depth[self.leaf_nodes].max())


@dataclass
class Model:
    objective: str
    n_features: int
    n_groups: int              # number of margins the model produces (1 for binary)
    n_classes: int             # number of distinct predicted classes
    base_margin: np.ndarray    # float32[n_groups], already in margin space
    base_margin_ulps: int      # how many float32 ulps our base margin may differ from XGBoost's
    trees: List[Tree]
    booster: str               # "gbtree" or "dart"
    num_parallel_tree: int
    feature_names: List[str]
    version: tuple
    round_ptr: List[int] = None            # trees of boosting round r are [round_ptr[r], round_ptr[r + 1])
    best_iteration: Optional[int] = None   # set by early stopping; XGBClassifier.predict stops there
    notes: List[str] = None                # things the user should know about how the file was interpreted

    @property
    def n_rounds(self) -> int:
        return len(self.round_ptr) - 1

    def first_rounds(self, n: int) -> "Model":
        """The model made of the first ``n`` boosting rounds only."""
        if not 0 <= n <= self.n_rounds:
            raise XgbEdgeError(f"cannot keep {n} rounds: the model has {self.n_rounds}")
        import dataclasses
        return dataclasses.replace(self, trees=self.trees[: self.round_ptr[n]],
                                   round_ptr=self.round_ptr[: n + 1])

    @property
    def is_binary(self) -> bool:
        return self.n_groups == 1

    @property
    def n_internal(self) -> int:
        return sum(t.n_internal for t in self.trees)

    @property
    def n_leaves(self) -> int:
        return sum(t.n_leaves for t in self.trees)

    def feature_label(self, f: int) -> str:
        if f < len(self.feature_names) and self.feature_names[f]:
            return f"{f} ({self.feature_names[f]!r})"
        return str(f)


def _get(obj, key, what):
    if not isinstance(obj, dict) or key not in obj:
        raise ModelFormatError(f"missing '{key}' in {what}; is this an XGBoost JSON model saved with save_model('*.json')?")
    return obj[key]


def _int_param(obj, key, what, default=None) -> int:
    if key not in obj:
        if default is not None:
            return default
        raise ModelFormatError(f"missing '{key}' in {what}")
    try:
        return int(obj[key])
    except (TypeError, ValueError):
        raise ModelFormatError(f"{what}.{key} is not an integer: {obj[key]!r}") from None


def _parse_base_score(text, n_groups: int) -> np.ndarray:
    if isinstance(text, (int, float)):
        items = [repr(float(text))]
    elif isinstance(text, str):
        s = text.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        items = [p.strip() for p in s.split(",") if p.strip()]
    else:
        raise ModelFormatError(f"learner_model_param.base_score has unexpected type {type(text).__name__}")
    if not items:
        raise ModelFormatError("learner_model_param.base_score is empty")
    try:
        vals = np.array([round_decimal_to_f32(p) for p in items], dtype=np.float32)
    except Exception:
        raise ModelFormatError(f"cannot parse base_score {text!r}") from None
    if vals.size == 1 and n_groups > 1:
        vals = np.repeat(vals, n_groups)
    if vals.size != n_groups:
        raise ModelFormatError(
            f"base_score has {vals.size} values but the model has {n_groups} output groups")
    if not np.all(np.isfinite(vals)):
        raise ModelFormatError(f"base_score is not finite: {text!r}")
    return vals


def _build_tree(index: int, jt: dict, group: int, weight: float, n_features: int,
                hook: Optional[_FloatHook], feature_names: List[str]) -> Tree:
    what = f"tree {index}"
    tp = jt.get("tree_param", {})
    slv = _int_param(tp, "size_leaf_vector", what + ".tree_param", default=1)
    if slv > 1:
        raise UnsupportedModelError(
            f"{what} has vector leaves (size_leaf_vector={slv}): multi-output trees "
            "(multi_strategy='multi_output_tree') are not supported. Train with the default "
            "multi_strategy='one_output_per_tree'.")
    left = _to_int(_get(jt, "left_children", what), what + ".left_children")
    right = _to_int(_get(jt, "right_children", what), what + ".right_children")
    feature = _to_int(_get(jt, "split_indices", what), what + ".split_indices")
    cond = _to_f32(_get(jt, "split_conditions", what), hook, what + ".split_conditions")
    dleft = _to_int(_get(jt, "default_left", what), what + ".default_left") != 0
    n = left.size
    for name, arr in (("right_children", right), ("split_indices", feature),
                      ("split_conditions", cond), ("default_left", dleft)):
        if arr.size != n:
            raise ModelFormatError(f"{what}: {name} has {arr.size} entries, left_children has {n}")
    if n == 0:
        raise ModelFormatError(f"{what} has no nodes")
    split_type = jt.get("split_type")
    if split_type is not None and len(split_type) not in (0, n):
        raise ModelFormatError(f"{what}: split_type has {len(split_type)} entries, expected {n}")
    stype = _to_int(split_type, what + ".split_type") if split_type else np.zeros(n, dtype=np.int64)

    # Walk the tree from the root; only reachable nodes matter (and are validated).
    preorder = []
    depth = np.zeros(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    while stack:
        nid = stack.pop()
        if seen[nid]:
            raise ModelFormatError(f"{what}: node {nid} is reachable twice (not a tree)")
        seen[nid] = True
        preorder.append(nid)
        l, r = int(left[nid]), int(right[nid])
        if l == -1:
            if r != -1:
                raise ModelFormatError(f"{what}: node {nid} has a right child but no left child")
            if not np.isfinite(cond[nid]):
                raise ModelFormatError(f"{what}: leaf {nid} has a non-finite value {cond[nid]}")
            continue
        if not (0 <= l < n and 0 <= r < n):
            raise ModelFormatError(f"{what}: node {nid} has out-of-range children ({l}, {r})")
        if r != l + 1:
            # XGBoost itself computes the right child as left + 1 when predicting.
            raise ModelFormatError(
                f"{what}: node {nid} has children ({l}, {r}); XGBoost requires right == left + 1")
        f = int(feature[nid])
        if stype[nid] != 0:
            fl = f"{f} ({feature_names[f]!r})" if f < len(feature_names) and feature_names[f] else str(f)
            raise UnsupportedModelError(
                f"{what}, node {nid} is a categorical split on feature {fl}: categorical splits are "
                "not supported. Re-train with the feature one-hot or ordinal encoded "
                "(enable_categorical=False).")
        if not 0 <= f < n_features:
            raise ModelFormatError(
                f"{what}: node {nid} splits on feature {f}, but the model has {n_features} features")
        if np.isnan(cond[nid]):
            raise ModelFormatError(f"{what}: node {nid} has a NaN split threshold")
        depth[l] = depth[r] = depth[nid] + 1
        stack.append(r)
        stack.append(l)
    return Tree(index=index, group=group, weight=weight, left=left, right=right,
                feature=feature, threshold=cond, default_left=dleft, value=cond,
                preorder=np.asarray(preorder, dtype=np.int64), node_depth=depth)


def from_json(doc: dict, hook: Optional[_FloatHook] = None) -> Model:
    """Build a :class:`Model` from a parsed XGBoost JSON document.

    Raises :class:`ModelFormatError` for anything that is not a well-formed
    XGBoost model, including documents whose structure is wrong in ways the
    individual checks do not anticipate.
    """
    try:
        return _from_json(doc, hook)
    except XgbEdgeError:
        raise
    except (TypeError, AttributeError, KeyError, IndexError, ValueError, OverflowError, RecursionError) as e:
        raise ModelFormatError(
            f"this is not a well-formed XGBoost JSON model ({type(e).__name__}: {e})") from None


#: XGBoost >= 3.2 bounds a binary:logistic base_score to [1e-6, 1 - 1e-6] before taking the logit.
LOGISTIC_CLAMP_SINCE = (3, 2, 0)
_KRT_EPS = np.float32(1e-6)


def _logistic_base_margin(p: np.float32, version: tuple, notes: List[str]) -> np.float32:
    """XGBoost's float32 logit of a binary:logistic base_score, as the saving release computes it."""
    one = np.float32(1.0)
    lo, hi = _KRT_EPS, np.float32(one - _KRT_EPS)

    def logit(x: np.float32) -> np.float32:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return np.float32(-np.log(one / x - one, dtype=np.float32))   # Logit(x) = -logf(1.0f / x - 1.0f)

    inside = bool(lo <= p <= hi)
    if inside:
        return logit(p)
    ver = tuple(int(v) for v in version[:3]) if len(version) >= 2 else None
    clamped = logit(np.float32(min(max(p, lo), hi)))
    if ver is None:
        raise UnsupportedModelError(
            f"base_score {p:g} is outside [1e-6, 1 - 1e-6] and the file does not say which XGBoost release "
            "saved it. Releases from 3.2 on bound the base score to that range before converting it to a "
            "margin, earlier ones do not, so the model's margins are ambiguous. Re-save the model with the "
            "xgboost you predict with.")
    if ver >= LOGISTIC_CLAMP_SINCE:
        if not (0.0 <= float(p) <= 1.0):   # XGBoost itself refuses such a model
            raise ModelFormatError(f"base_score {p:g} is outside [0, 1] for binary:logistic")
        notes.append(
            f"base_score {p:g} is outside [1e-6, 1 - 1e-6]; XGBoost {'.'.join(map(str, ver))} (3.2 and later) "
            f"bounds it to that range, so the base margin is {float(clamped):.6g}. Releases before 3.2 would "
            "use the unbounded logit instead.")
        return clamped
    if not (0.0 < float(p) < 1.0):
        raise ModelFormatError(f"base_score {p:g} is outside (0, 1) for binary:logistic")
    raw = logit(p)
    if not np.isfinite(raw):
        raise ModelFormatError(f"base_score {p:g} has no finite logit in float32")
    notes.append(
        f"base_score {p:g} is outside [1e-6, 1 - 1e-6]; the file was saved by XGBoost "
        f"{'.'.join(map(str, ver))}, which uses its logit {float(raw):.6g} as the base margin. XGBoost 3.2 and "
        f"later would bound it first and use {float(clamped):.6g}: if you predict with such a release, "
        "re-save the model with it before exporting.")
    return raw


def _from_json(doc: dict, hook: Optional[_FloatHook] = None) -> Model:
    if not isinstance(doc, dict):
        raise ModelFormatError("the model file does not hold a JSON object")
    notes: List[str] = []
    learner = _get(doc, "learner", "the model file")
    version = tuple(doc.get("version") or ())
    lmp = _get(learner, "learner_model_param", "learner")
    gb = _get(learner, "gradient_booster", "learner")
    booster = _get(gb, "name", "gradient_booster")

    if booster == "gblinear":
        raise UnsupportedModelError(
            "booster 'gblinear' is a linear model, not a tree ensemble; xgb-edge only exports "
            "tree boosters (gbtree, dart).")
    if booster == "dart":
        # XGBoost < 3.3 layout: {"name": "dart", "gbtree": {"model": ...}, "weight_drop": [...]}
        inner = _get(gb, "gbtree", "gradient_booster (dart)")
        gmodel = _get(inner, "model", "gradient_booster.gbtree")
        weight_drop = _get(gb, "weight_drop", "gradient_booster (dart)")
    elif booster == "gbtree":
        gmodel = _get(gb, "model", "gradient_booster")
        weight_drop = gmodel.get("weight_drop")  # XGBoost >= 3.3 stores DART weights here
        if weight_drop is not None:
            booster = "dart"
    else:
        raise UnsupportedModelError(f"unknown booster {booster!r}; expected gbtree or dart")

    objective = _get(_get(learner, "objective", "learner"), "name", "learner.objective")
    if objective not in SUPPORTED_OBJECTIVES:
        hint = ""
        if objective == "binary:logitraw":
            hint = (" (binary:logitraw has no defined class decision: XGBoost's own predict() "
                    "thresholds the raw margin at 0.5, not 0; train with binary:logistic instead)")
        raise UnsupportedModelError(
            f"objective {objective!r} is not supported{hint}. xgb-edge exports classifiers trained "
            f"with one of: {', '.join(SUPPORTED_OBJECTIVES)}.")

    n_features = _int_param(lmp, "num_feature", "learner_model_param")
    num_class = _int_param(lmp, "num_class", "learner_model_param", default=0)
    num_target = _int_param(lmp, "num_target", "learner_model_param", default=1)
    if num_target > 1:
        raise UnsupportedModelError(
            f"the model has num_target={num_target}: multi-target / multi-label models are not supported.")
    if objective.startswith("multi:"):
        if num_class == 1:
            raise UnsupportedModelError(
                f"objective {objective} with num_class=1: a one-class model predicts class 0 for every "
                "input; there is nothing to export.")
        if num_class < 2:
            raise ModelFormatError(f"objective {objective} with num_class={num_class}")
        n_groups, n_classes = num_class, num_class
    else:
        if num_class > 1:
            raise ModelFormatError(f"objective {objective} with num_class={num_class}")
        n_groups, n_classes = 1, 2
    if n_features < 0:
        raise ModelFormatError(f"num_feature={n_features}")

    feature_names = [str(x) for x in (learner.get("feature_names") or [])]

    base_score = _parse_base_score(_get(lmp, "base_score", "learner_model_param"), n_groups)
    ulps = 0
    if objective == "binary:logistic":
        base_margin = np.array([_logistic_base_margin(base_score[0], version, notes)], dtype=np.float32)
        ulps = 2  # libm logf vs numpy log may differ in the last place
    else:
        base_margin = base_score.astype(np.float32)

    gparam = gmodel.get("gbtree_model_param", {})
    num_parallel_tree = _int_param(gparam, "num_parallel_tree", "gbtree_model_param", default=1)
    jtrees = _get(gmodel, "trees", "gradient_booster.model")
    if not isinstance(jtrees, list):
        raise ModelFormatError("gradient_booster.model.trees is not a list")
    if not jtrees:
        # An unfitted model is also where XGBoost treats base_score specially (as a placeholder).
        raise UnsupportedModelError(
            "the model has no trees (it was saved before any boosting round): there is nothing to export.")
    tree_info = _to_int(_get(gmodel, "tree_info", "gradient_booster.model"), "tree_info")
    if "num_trees" in gparam and _int_param(gparam, "num_trees", "gbtree_model_param") != len(jtrees):
        raise ModelFormatError(
            f"gbtree_model_param.num_trees={gparam['num_trees']} but the file holds {len(jtrees)} trees")
    if tree_info.size != len(jtrees):
        raise ModelFormatError(f"tree_info has {tree_info.size} entries for {len(jtrees)} trees")
    if tree_info.size and (tree_info.min() < 0 or tree_info.max() >= n_groups):
        raise ModelFormatError(
            f"tree_info refers to output group {int(tree_info.max())}, but the model has {n_groups}")

    if weight_drop is not None:
        weights = _to_f32(weight_drop, hook, "weight_drop")
        if weights.size != len(jtrees):
            raise ModelFormatError(
                f"DART weight_drop has {weights.size} entries for {len(jtrees)} trees")
        if not np.all(np.isfinite(weights)):
            raise ModelFormatError("DART weight_drop contains non-finite values")
    else:
        weights = np.ones(len(jtrees), dtype=np.float32)

    trees = []
    for i, jt in enumerate(jtrees):
        if not isinstance(jt, dict):
            raise ModelFormatError(f"tree {i} is not a JSON object")
        trees.append(_build_tree(i, jt, int(tree_info[i]), float(weights[i]), n_features,
                                 hook, feature_names))

    # Boosting rounds: needed only to honour early stopping (see ExportOptions.rounds).
    n_trees = len(trees)
    indptr = gmodel.get("iteration_indptr")
    if indptr is not None:
        round_ptr = [int(v) for v in _to_int(indptr, "iteration_indptr")]
        ok = (len(round_ptr) >= 1 and round_ptr[0] == 0 and round_ptr[-1] == n_trees
              and all(a <= b for a, b in zip(round_ptr, round_ptr[1:])))
        if not ok:
            raise ModelFormatError(f"iteration_indptr {round_ptr[:6]}... is inconsistent with {n_trees} trees")
    else:
        per_round = n_groups * max(num_parallel_tree, 1)
        if n_trees % per_round == 0:
            round_ptr = list(range(0, n_trees + 1, per_round))
        else:
            round_ptr = [0, n_trees]
    best_iteration = None
    attrs = learner.get("attributes") or {}
    if isinstance(attrs, dict) and attrs.get("best_iteration") is not None:
        try:
            best_iteration = int(attrs["best_iteration"])
        except (TypeError, ValueError):
            raise ModelFormatError(f"attribute best_iteration is not an integer: {attrs['best_iteration']!r}") from None

    return Model(objective=objective, n_features=n_features, n_groups=n_groups,
                 n_classes=n_classes, base_margin=base_margin, base_margin_ulps=ulps,
                 trees=trees, booster=booster, num_parallel_tree=num_parallel_tree,
                 feature_names=feature_names, version=version, round_ptr=round_ptr,
                 best_iteration=best_iteration, notes=notes)


def load(path) -> Model:
    """Load an XGBoost model saved with ``booster.save_model("model.json")``."""
    try:
        with open(path, "rb") as f:
            head = f.read(1)
            f.seek(0)
            raw = f.read()
    except OSError as e:
        raise XgbEdgeError(f"cannot read the model file {path}: {e.strerror or e}") from None
    if head != b"{":
        raise ModelFormatError(
            f"{path} is not XGBoost JSON (it does not start with '{{'). If it is a UBJSON or legacy "
            "binary model, load it in xgboost and re-save with booster.save_model('model.json').")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ModelFormatError(
            f"{path} is not XGBoost JSON (not UTF-8 text). If it is a UBJSON model, re-save it "
            "with booster.save_model('model.json').") from None
    hook = _FloatHook()
    try:
        doc = json.loads(text, parse_float=hook)
    except json.JSONDecodeError as e:
        raise ModelFormatError(f"{path} is not valid JSON: {e}") from None
    except RecursionError:
        raise ModelFormatError(f"{path} is not valid JSON: nested too deeply") from None
    except ValueError as e:   # e.g. an integer literal with thousands of digits
        raise ModelFormatError(f"{path} is not a usable JSON model: {e}") from None
    return from_json(doc, hook)
