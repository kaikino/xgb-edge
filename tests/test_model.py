"""Model loading: exact float32 parsing, accepted variants, and explicit rejection of the rest."""
import copy
import json
import math
import os
import warnings
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from harness import synth
from xgb_edge import ModelFormatError, UnsupportedModelError, XgbEdgeError, load
from xgb_edge.model import _FloatHook, _to_f32, from_json, round_decimal_to_f32

F32 = np.float32
getcontext().prec = 200


# --------------------------------------------------------------------------- decimal -> float32

def reference_round(text: str) -> np.float32:
    """Independent correctly-rounded decimal -> float32 (pure rational arithmetic)."""
    v = Fraction(Decimal(text))
    if v == 0:
        return F32(-0.0) if text.strip().startswith("-") else F32(0.0)
    sign = -1 if v < 0 else 1
    a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1                      # 2**e <= a < 2**(e+1)
    e = max(e, -126)                # subnormals share the exponent of the smallest normal
    ulp = Fraction(2) ** (e - 23)
    m = round(a / ulp)              # Fraction rounding is half-to-even
    val = m * ulp
    if val >= Fraction(2) ** 128:
        return F32(sign * math.inf)
    return F32(sign * float(val))   # exact: val is a float32


def mid_text(lo: np.float32) -> Decimal:
    hi = np.nextafter(lo, F32(np.inf), dtype=F32)
    return (Decimal(float(lo)) + Decimal(float(hi))) / 2


@settings(max_examples=3000, deadline=None)
@given(bits=st.integers(0, 2 ** 32 - 1))
def test_shortest_repr_round_trips(bits):
    x = np.array([bits], dtype=np.uint32).view(F32)[0]
    if not np.isfinite(x):
        return
    text = np.format_float_scientific(x, unique=True).replace("e", "E")
    got = round_decimal_to_f32(text)
    assert got.view(np.uint32) == x.view(np.uint32) or (x == 0 and got == 0)


@settings(max_examples=1500, deadline=None)
@given(digits=st.text("0123456789", min_size=1, max_size=40), exp=st.integers(-50, 40), neg=st.booleans())
def test_arbitrary_decimals_round_correctly(digits, exp, neg):
    text = ("-" if neg else "") + digits[0] + "." + digits[1:] + "0E" + str(exp)
    with np.errstate(over="ignore"):
        want = reference_round(text)
    got = round_decimal_to_f32(text)
    assert got.view(np.uint32) == want.view(np.uint32), (text, got, want)


@settings(max_examples=800, deadline=None)
@given(bits=st.integers(1, 0x7F7FFFFE), nudge=st.sampled_from([0, 1, -1]), neg=st.booleans())
def test_values_at_and_next_to_float32_midpoints(bits, nudge, neg):
    """The double-rounding trap: a decimal a hair off a float32 midpoint parses to the midpoint as a double."""
    lo = np.array([bits], dtype=np.uint32).view(F32)[0]
    hi = np.nextafter(lo, F32(np.inf), dtype=F32)
    mid = mid_text(lo)
    val = mid * (1 + Decimal(nudge) * Decimal(10) ** -60)
    text = format(-val if neg else val, "E")
    got = round_decimal_to_f32(text)
    if nudge == 0:
        want = lo if (bits & 1) == 0 else hi          # tie: even mantissa
    else:
        want = hi if nudge > 0 else lo
    want = -want if neg else want
    assert got.view(np.uint32) == F32(want).view(np.uint32), (text, got, want)
    # and through the JSON hook + array conversion used by the loader
    hook = _FloatHook()
    arr = _to_f32(json.loads(f"[{text}, 1.5, {text}]", parse_float=hook), hook, "test")
    assert arr[0].view(np.uint32) == F32(want).view(np.uint32) and arr[2] == arr[0] and arr[1] == 1.5


def test_overflow_and_tiny_values():
    assert round_decimal_to_f32("3.4028235E38") == np.finfo(F32).max
    assert round_decimal_to_f32("3.40282356E38") == np.finfo(F32).max       # below the rounding boundary
    assert np.isinf(round_decimal_to_f32("3.4028236E38"))                    # at/after FLT_MAX + half ulp
    tiny = F32(1e-45)                                                         # smallest subnormal, 2**-149
    assert round_decimal_to_f32("1E-45") == tiny and round_decimal_to_f32("7.1E-46") == tiny
    assert round_decimal_to_f32("7E-46") == 0.0                               # below half of it
    half = Decimal(2) ** -150
    assert round_decimal_to_f32(format(half, "E")) == 0.0                     # exact tie -> even (0)
    assert round_decimal_to_f32(format(half * (1 + Decimal(10) ** -70), "E")) == tiny
    assert round_decimal_to_f32(format(half * 3, "E")) == F32(2.8e-45)        # tie between 1 and 2 ulps -> even (2)
    assert round_decimal_to_f32("-0E0") == 0.0


# --------------------------------------------------------------------------- real models

def _train(params, n_classes=3, n_features=6, rounds=3, X=None, y=None, **dm):
    import xgboost as xgb
    rng = np.random.RandomState(0)
    if X is None:
        X = rng.randn(300, n_features).astype(F32)
    if y is None:
        y = rng.randint(0, n_classes, size=X.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xgb.train(dict(verbosity=0, **params), xgb.DMatrix(X, label=y, **dm), num_boost_round=rounds)


def _save(bst, tmp_path, name="m.json"):
    p = str(tmp_path / name)
    bst.save_model(p)
    return p


def test_thresholds_and_leaves_are_bit_exact(tmp_path):
    """Our decimal parsing gives the very float32 values xgboost holds (checked via the UBJSON dump)."""
    import xgboost as xgb
    bst = _train(dict(objective="multi:softprob", num_class=3, max_depth=6), rounds=20)
    model = load(_save(bst, tmp_path))
    raw = json.loads(bst.save_raw("json").decode())
    ubj = str(tmp_path / "m.ubj")
    bst.save_model(ubj)
    # decode split_conditions from UBJSON: typed float32 arrays, big endian, in tree order
    import struct
    blob = open(ubj, "rb").read()
    for t, jt in zip(model.trees, raw["learner"]["gradient_booster"]["model"]["trees"]):
        n = len(jt["split_conditions"])
        key = b"split_conditions[$d#"
        at = blob.index(key) + len(key)
        cnt_type = blob[at:at + 1]
        size = {b"U": 1, b"i": 1, b"I": 2, b"l": 4, b"L": 8}[cnt_type]
        cnt = int.from_bytes(blob[at + 1:at + 1 + size], "big")
        assert cnt == n
        vals = np.frombuffer(blob, dtype=">f4", count=n, offset=at + 1 + size).astype(F32)
        assert np.array_equal(vals.view(np.uint32), t.threshold.view(np.uint32))
        blob = blob[at + 1 + size + 4 * n:]


def test_binary_base_score_becomes_a_logit(tmp_path):
    bst = _train(dict(objective="binary:logistic", base_score=0.25), n_classes=2)
    m = load(_save(bst, tmp_path))
    assert m.n_groups == 1 and m.n_classes == 2
    assert abs(float(m.base_margin[0]) - math.log(0.25 / 0.75)) < 1e-6
    assert m.base_margin_ulps > 0


def test_scalar_base_score_of_old_versions_is_expanded(tmp_path):
    bst = _train(dict(objective="multi:softprob", num_class=3))
    doc = json.loads(open(_save(bst, tmp_path)).read())
    doc["learner"]["learner_model_param"]["base_score"] = "5E-1"
    m = from_json(doc)
    assert m.base_margin.tolist() == [0.5, 0.5, 0.5]


def test_dart_old_and_new_layout(tmp_path):
    bst = _train(dict(objective="multi:softprob", num_class=3, booster="dart", rate_drop=0.5), rounds=6)
    doc = json.loads(open(_save(bst, tmp_path)).read())
    m = from_json(copy.deepcopy(doc))
    assert m.booster == "dart" and any(t.weight != 1.0 for t in m.trees)
    gb = doc["learner"]["gradient_booster"]
    if gb["name"] == "gbtree":      # rewrite into the pre-3.3 layout
        weights = gb["model"].pop("weight_drop")
        doc["learner"]["gradient_booster"] = {"name": "dart", "gbtree": {"model": gb["model"], "name": "gbtree"},
                                              "weight_drop": weights}
    m2 = from_json(doc)
    assert [t.weight for t in m2.trees] == [t.weight for t in m.trees]
    bad = copy.deepcopy(doc)
    bad["learner"]["gradient_booster"]["weight_drop"] = bad["learner"]["gradient_booster"]["weight_drop"][:-1]
    with pytest.raises(ModelFormatError, match="weight_drop"):
        from_json(bad)


def test_num_parallel_tree_is_accepted(tmp_path):
    bst = _train(dict(objective="multi:softprob", num_class=3, num_parallel_tree=4, subsample=0.5), rounds=2)
    m = load(_save(bst, tmp_path))
    assert m.num_parallel_tree == 4 and len(m.trees) == 3 * 4 * 2


# --------------------------------------------------------------------------- rejections

def test_categorical_split_is_rejected(tmp_path):
    import pandas as pd  # noqa: F401
    import xgboost as xgb
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"num": rng.randn(400), "color": pd.Categorical(rng.choice(list("abcdef"), 400))})
    y = (df["color"].cat.codes % 3).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(objective="multi:softprob", num_class=3, tree_method="hist", verbosity=0,
                             max_cat_to_onehot=1),
                        xgb.DMatrix(df, label=y, enable_categorical=True), num_boost_round=3)
    with pytest.raises(UnsupportedModelError, match="categorical split.*color"):
        load(_save(bst, tmp_path))


def test_multi_output_tree_is_rejected(tmp_path):
    bst = _train(dict(objective="multi:softprob", num_class=3, tree_method="hist",
                      multi_strategy="multi_output_tree"))
    with pytest.raises(UnsupportedModelError, match="vector leaves|multi-output"):
        load(_save(bst, tmp_path))


def test_gblinear_is_rejected(tmp_path):
    bst = _train(dict(objective="multi:softprob", num_class=3, booster="gblinear"))
    with pytest.raises(UnsupportedModelError, match="gblinear"):
        load(_save(bst, tmp_path))


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logitraw", "rank:pairwise", "count:poisson"])
def test_non_classifier_objectives_are_rejected(tmp_path, objective):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4).astype(F32)
    y = rng.randint(0, 2, size=200)
    dm = {"group": [100, 100]} if objective.startswith("rank") else {}
    bst = _train(dict(objective=objective), X=X, y=y, **dm)
    with pytest.raises(UnsupportedModelError, match="objective"):
        load(_save(bst, tmp_path))


def test_multilabel_is_rejected(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4).astype(F32)
    y = rng.randint(0, 2, size=(200, 3))
    bst = _train(dict(objective="binary:logistic"), X=X, y=y)
    with pytest.raises(UnsupportedModelError, match="num_target"):
        load(_save(bst, tmp_path))


def test_ubjson_and_garbage_files_are_rejected(tmp_path):
    bst = _train(dict(objective="multi:softprob", num_class=3))
    ubj = str(tmp_path / "m.ubj")
    bst.save_model(ubj)
    with pytest.raises(ModelFormatError, match="UBJSON|not XGBoost JSON"):
        load(ubj)
    p = tmp_path / "bad.json"
    p.write_text('{"learner": ')
    with pytest.raises(ModelFormatError, match="not valid JSON"):
        load(str(p))
    p.write_text('{"hello": 1}')
    with pytest.raises(ModelFormatError, match="learner"):
        load(str(p))
    p.write_bytes(b"\x00\x01binf")
    with pytest.raises(ModelFormatError):
        load(str(p))


def _doc(n_classes=3):
    t = synth.SynthTree()
    t.add_leaf(0)
    l, r = t.split(0, 1, 0.5, True, 0.1, -0.1)
    t.split(l, 0, -1.0, False, 0.2, 0.3)
    return json.loads(synth.build_json([t], 3, n_classes))


def _tree(doc):
    return doc["learner"]["gradient_booster"]["model"]["trees"][0]


@pytest.mark.parametrize("mutate, exc, pattern", [
    (lambda d: _tree(d)["split_type"].__setitem__(0, 1), UnsupportedModelError, "categorical"),
    (lambda d: _tree(d)["tree_param"].__setitem__("size_leaf_vector", "3"), UnsupportedModelError, "vector leaves"),
    (lambda d: _tree(d)["split_indices"].__setitem__(0, 9), ModelFormatError, "feature 9"),
    (lambda d: _tree(d)["split_conditions"].__setitem__(0, float("nan")), ModelFormatError, "NaN split threshold"),
    (lambda d: _tree(d)["split_conditions"].__setitem__(2, float("inf")), ModelFormatError, "non-finite"),
    (lambda d: _tree(d)["left_children"].__setitem__(1, 0), ModelFormatError, "right == left|reachable twice|children"),
    (lambda d: _tree(d)["right_children"].__setitem__(0, 4), ModelFormatError, "right == left"),
    (lambda d: _tree(d)["left_children"].__setitem__(0, 40), ModelFormatError, "out-of-range"),
    (lambda d: _tree(d)["default_left"].pop(), ModelFormatError, "entries"),
    (lambda d: _tree(d).pop("split_conditions"), ModelFormatError, "split_conditions"),
    (lambda d: d["learner"]["gradient_booster"]["model"]["tree_info"].__setitem__(0, 7), ModelFormatError, "output group"),
    (lambda d: d["learner"]["gradient_booster"]["model"]["tree_info"].append(0), ModelFormatError, "tree_info"),
    (lambda d: d["learner"]["learner_model_param"].__setitem__("base_score", "[1,2]"), ModelFormatError, "base_score"),
    (lambda d: d["learner"]["learner_model_param"].__setitem__("base_score", "[NaN,0,0]"), ModelFormatError, "base_score"),
    (lambda d: d["learner"]["learner_model_param"].__setitem__("num_target", "2"), UnsupportedModelError, "num_target"),
    (lambda d: d["learner"]["gradient_booster"].__setitem__("name", "gblinear"), UnsupportedModelError, "gblinear"),
    (lambda d: d["learner"]["gradient_booster"].__setitem__("name", "mystery"), UnsupportedModelError, "unknown booster"),
    (lambda d: d["learner"]["objective"].__setitem__("name", "reg:squarederror"), UnsupportedModelError, "objective"),
    (lambda d: d["learner"].pop("objective"), ModelFormatError, "objective"),
])
def test_malformed_or_unsupported_documents(mutate, exc, pattern):
    doc = _doc()
    mutate(doc)
    with pytest.raises(exc, match=pattern):
        from_json(doc)
    assert issubclass(exc, XgbEdgeError)


def test_unreachable_garbage_nodes_are_ignored():
    doc = _doc()
    t = _tree(doc)
    # append a node nobody points to, with nonsense in it
    for key, val in (("left_children", -1), ("right_children", -1), ("split_indices", 999), ("split_conditions", 1e9),
                     ("default_left", 0), ("split_type", 0), ("parents", 2147483647), ("base_weights", 0.0),
                     ("loss_changes", 0.0), ("sum_hessian", 0.0)):
        t[key].append(val)
    t["tree_param"]["num_nodes"] = str(len(t["left_children"]))
    m = from_json(doc)
    assert m.trees[0].n_leaves == 3 and m.trees[0].n_internal == 2


def test_binary_base_score_out_of_range():
    doc = _doc(n_classes=2)
    doc["learner"]["learner_model_param"]["base_score"] = "[1.5E0]"
    with pytest.raises(ModelFormatError, match="outside"):
        from_json(doc)


# --------------------------------------------------------------------------- binary:logistic base score by release

def _binary_doc(base_score: str, version):
    doc = _doc(n_classes=2)
    doc["learner"]["learner_model_param"]["base_score"] = base_score
    if version is None:
        doc.pop("version", None)
    else:
        doc["version"] = list(version)
    return doc


def test_logistic_base_score_is_bounded_only_by_releases_that_do_so():
    clamp_lo = -math.log(1 / 1e-6 - 1)                       # logit(1e-6) = -13.8155
    new = from_json(_binary_doc("[1E-7]", (3, 4, 1)))
    assert abs(float(new.base_margin[0]) - clamp_lo) < 1e-4 and any("3.2" in n for n in new.notes)
    edge = from_json(_binary_doc("[1E-7]", (3, 2, 0)))
    assert abs(float(edge.base_margin[0]) - clamp_lo) < 1e-4
    for old_version in ((3, 1, 3), (2, 1, 4), (1, 7, 6)):
        old = from_json(_binary_doc("[1E-7]", old_version))
        assert abs(float(old.base_margin[0]) - math.log(1e-7 / (1 - 1e-7))) < 1e-4       # -16.118
        assert any("re-save" in n for n in old.notes)
    # inside the range every release agrees and nothing needs saying
    for version in ((3, 4, 1), (2, 1, 4), None):
        m = from_json(_binary_doc("[2.5E-1]", version))
        assert abs(float(m.base_margin[0]) - math.log(0.25 / 0.75)) < 1e-6 and not m.notes
    # 0 and 1 (all labels equal) exist only where the bound exists
    assert abs(float(from_json(_binary_doc("[0E0]", (3, 4, 1))).base_margin[0]) - clamp_lo) < 1e-4
    assert float(from_json(_binary_doc("[1E0]", (3, 4, 1))).base_margin[0]) > 13.0
    with pytest.raises(ModelFormatError, match="outside"):
        from_json(_binary_doc("[0E0]", (2, 1, 4)))
    with pytest.raises(ModelFormatError, match="finite logit"):
        from_json(_binary_doc("[1E-40]", (2, 1, 4)))
    assert np.isfinite(from_json(_binary_doc("[1E-40]", (3, 4, 1))).base_margin[0])
    with pytest.raises(UnsupportedModelError, match="does not say which XGBoost release"):
        from_json(_binary_doc("[1E-7]", None))


def test_untrained_and_one_class_models_are_refused():
    doc = _doc()
    model = doc["learner"]["gradient_booster"]["model"]
    model["trees"], model["tree_info"] = [], []
    model["gbtree_model_param"]["num_trees"] = "0"
    model["iteration_indptr"] = [0]
    with pytest.raises(UnsupportedModelError, match="no trees"):
        from_json(doc)
    doc = _doc()
    doc["learner"]["learner_model_param"]["num_class"] = "1"
    with pytest.raises(UnsupportedModelError, match="num_class=1"):
        from_json(doc)


@pytest.mark.parametrize("mutate", [
    lambda d: d["learner"]["gradient_booster"]["model"]["trees"].__setitem__(0, [1, 2, 3]),
    lambda d: d["learner"]["gradient_booster"]["model"].__setitem__("trees", {"a": 1}),
    lambda d: _tree(d)["tree_param"].__setitem__("size_leaf_vector", float("inf")),
    lambda d: _tree(d).__setitem__("left_children", "abc"),
    lambda d: _tree(d).__setitem__("split_conditions", [[1.0], [2.0]]),
    lambda d: _tree(d).__setitem__("default_left", None),
    lambda d: d["learner"].__setitem__("learner_model_param", [1, 2]),
    lambda d: d["learner"].__setitem__("gradient_booster", "gbtree"),
    lambda d: d["learner"]["learner_model_param"].__setitem__("num_feature", None),
    lambda d: d["learner"]["learner_model_param"].__setitem__("base_score", {"x": 1}),
    lambda d: d["learner"].__setitem__("attributes", {"best_iteration": [1]}),
    lambda d: d["learner"]["gradient_booster"]["model"].__setitem__("iteration_indptr", "x"),
    lambda d: d.__setitem__("version", "three"),
])
def test_structurally_wrong_documents_give_a_format_error(mutate):
    doc = _doc()
    mutate(doc)
    try:
        from_json(doc)
    except XgbEdgeError:
        pass          # refused cleanly; (a mutation that happens to be harmless may also load)


def _mutations(doc):
    """Every JSON path in the document, for the structural fuzz below."""
    paths = []

    def walk(node, path):
        paths.append(path)
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + (k,))
        elif isinstance(node, list):
            for i, v in enumerate(node[:6]):
                walk(v, path + (i,))
    walk(doc, ())
    return paths


_JUNK = [None, True, -1, 0, 10 ** 30, 1.5, float("nan"), float("inf"), "", "x", "1E400", [], {}, [[1]], {"a": [1]},
         [None], ["x", 1]]


@settings(max_examples=600, deadline=None)
@given(data=st.data())
def test_loader_never_fails_with_anything_but_its_own_errors(data):
    """Replace or delete arbitrary parts of a valid model: the loader must load it or raise XgbEdgeError."""
    doc = _doc(n_classes=data.draw(st.sampled_from([2, 3])))
    paths = [p for p in _mutations(doc) if p]
    for _ in range(data.draw(st.integers(1, 3))):
        path = data.draw(st.sampled_from(paths))
        node = doc
        try:
            for key in path[:-1]:
                node = node[key]
            if data.draw(st.booleans()) and isinstance(node, (dict, list)):
                del node[path[-1]]
            else:
                node[path[-1]] = data.draw(st.sampled_from(_JUNK))
        except (KeyError, IndexError, TypeError):
            continue                      # an earlier mutation removed this path
    try:
        model = from_json(doc)
    except XgbEdgeError:
        return
    # it loaded: then it must also export without a foreign exception
    from xgb_edge import ExportOptions, export_model
    try:
        export_model(model, ExportOptions(name="m"))
    except XgbEdgeError:
        pass
