"""The export pipeline: model -> quantization plans -> packed stream -> C + report."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from .analysis import Evaluation, evaluate, integer_gap, margin_gap, safe_gap
from .codegen import Generated, check_name, generate
from .model import Model, UnsupportedModelError, XgbEdgeError, load
from .pack import TYPE_NAMES, Packed, pack
from .quantize import INPUT_KINDS, LeafPlan, ThresholdPlan, plan_leaves, plan_thresholds

AUTO_LEAF_BITS = (8, 16, 24, 32)
DEFAULT_LEAF_BITS = 16


@dataclass
class ExportOptions:
    name: str = "xgb_model"
    input: str = "f32"                    # f32, i8, u8, i16, u16, i32
    leaf_bits: Union[str, int] = "auto"   # "auto" or 2..32
    leaf_store: str = "auto"              # auto, inline, table
    rounds: Union[str, int] = "auto"      # auto, all, best, or a number of boosting rounds
    thresholds: str = "auto"              # auto, rank (cut tables + ranks), direct (threshold keys in the nodes)
    # Test hooks: force wider fields than the model needs, to exercise every code path.
    min_code_bytes: int = 1
    min_hdr_bytes: int = 1
    min_off_bytes: int = 1
    min_leaf_bytes: int = 1


@dataclass
class Export:
    model: Model
    options: ExportOptions
    tplan: ThresholdPlan
    lplan: LeafPlan
    packed: Packed
    generated: Generated
    evaluation: Optional[Evaluation]
    report: Dict = field(default_factory=dict)
    safe_gap: int = 0      # fixed-point margin gap that guarantees XGBoost's class (the <NAME>_SAFE_GAP macro)

    def write(self, outdir) -> List[str]:
        os.makedirs(outdir, exist_ok=True)
        h = os.path.join(outdir, self.generated.name + ".h")
        c = os.path.join(outdir, self.generated.name + ".c")
        with open(h, "w") as f:
            f.write(self.generated.header)
        with open(c, "w") as f:
            f.write(self.generated.source)
        return [h, c]

    def device_inputs(self, X) -> np.ndarray:
        """Convert a float feature matrix to the array type the generated C takes."""
        return to_device_inputs(X, self.model.n_features, self.options.input)


def to_device_inputs(X, n_features: int, input_kind: str) -> np.ndarray:
    kind = INPUT_KINDS[input_kind]
    X = np.asarray(X)
    if X.dtype == np.bool_:
        X = X.astype(np.uint8)
    if not (np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, np.integer)):
        raise XgbEdgeError(f"feature data must be numeric, got an array of type {X.dtype}")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise XgbEdgeError(f"feature data must have shape (rows, {n_features}), got {X.shape}")
    if kind.is_float:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.ascontiguousarray(X.astype(np.float32))
    if np.issubdtype(X.dtype, np.floating):
        if np.isnan(X).any():
            raise XgbEdgeError(
                f"the data contains missing values (NaN), which --input {kind.name} cannot represent")
        if not np.all(X == np.rint(X)):
            raise XgbEdgeError(
                f"--input {kind.name} requires integer-valued features, but the data contains non-integers")
    if X.size:
        # compare as Python numbers: float32(2**31 - 1) is 2**31, which would hide an overflow
        lo_v, hi_v = X.min().item(), X.max().item()
        if lo_v < kind.lo or hi_v > kind.hi:
            raise XgbEdgeError(
                f"the data has values in [{lo_v:g}, {hi_v:g}], outside the range of {kind.ctype} "
                f"[{kind.lo}, {kind.hi}]")
    dtype = {"i8": np.int8, "u8": np.uint8, "i16": np.int16, "u16": np.uint16, "i32": np.int32}[kind.name]
    return np.ascontiguousarray(X.astype(dtype))


def _whole_number(value, what: str) -> int:
    """``value`` as an int; rejects 16.5, "abc", True and friends instead of truncating them."""
    if isinstance(value, bool):
        raise XgbEdgeError(f"{what}, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise XgbEdgeError(f"{what}, got {value!r}")


def select_rounds(model: Model, rounds: Union[str, int], notes: List[str]) -> Model:
    """Decide how many boosting rounds to export.

    A model trained with early stopping carries more trees than its best
    iteration.  XGBClassifier.predict() silently stops at the best iteration,
    while Booster.predict() uses every tree, so "the same predictions as
    xgboost" is ambiguous and the caller has to say which one is meant.
    """
    best, total = model.best_iteration, model.n_rounds
    if rounds == "auto":
        if best is not None and best + 1 < total:
            raise XgbEdgeError(
                f"the model was trained with early stopping: it holds {total} boosting rounds but its "
                f"best_iteration is {best}. XGBClassifier.predict() uses only the first {best + 1} rounds, "
                f"Booster.predict() uses all {total}. Choose with --rounds best (first {best + 1} rounds, "
                f"matches XGBClassifier.predict) or --rounds all (matches Booster.predict).")
        return model
    if rounds == "all":
        if best is not None and best + 1 < total:
            notes.append(f"exporting all {total} rounds although best_iteration is {best}: this matches "
                         "Booster.predict(), not XGBClassifier.predict()")
        return model
    if rounds == "best":
        if best is None:
            raise XgbEdgeError("--rounds best: the model has no best_iteration attribute "
                               "(it was not trained with early stopping)")
        if not 0 <= best < total:
            raise XgbEdgeError(f"best_iteration {best} is out of range for a model with {total} rounds")
        if best + 1 < total:
            notes.append(f"exporting the first {best + 1} of {total} rounds (best_iteration {best}), "
                         "as XGBClassifier.predict() does")
        return model.first_rounds(best + 1)
    n = _whole_number(rounds, "rounds must be auto, all, best or a whole number")
    if not 1 <= n <= total:
        raise XgbEdgeError(f"--rounds {n} is out of range: the model has {total} boosting rounds")
    if n < total:
        notes.append(f"exporting the first {n} of {total} rounds, like predict(iteration_range=(0, {n}))")
    return model.first_rounds(n)


def _bit_length(n: int) -> int:
    return int(n).bit_length()


def export_model(model: Union[Model, str, os.PathLike], options: Optional[ExportOptions] = None,
                 validation=None) -> Export:
    """Quantize ``model`` and generate C.  ``validation`` is an optional feature matrix."""
    opt = options or ExportOptions()
    check_name(opt.name)
    if not isinstance(model, Model):
        model = load(model)

    for hook_name, allowed in (("min_hdr_bytes", (1, 2)), ("min_off_bytes", (1, 2, 3, 4)),
                               ("min_leaf_bytes", (1, 2, 3, 4)), ("min_code_bytes", (1, 2))):
        if getattr(opt, hook_name) not in allowed:
            raise XgbEdgeError(f"{hook_name} must be one of {allowed}, got {getattr(opt, hook_name)!r}")
    notes: List[str] = list(model.notes or [])
    model = select_rounds(model, opt.rounds, notes)

    if opt.thresholds not in ("auto", "rank", "direct"):
        raise XgbEdgeError(f"thresholds must be auto, rank or direct, got {opt.thresholds!r}")
    if opt.input not in INPUT_KINDS:
        raise XgbEdgeError(f"unknown input kind {opt.input!r}; choose one of {', '.join(INPUT_KINDS)}")

    Xv = None
    if validation is not None:
        Xv = np.asarray(validation)
        to_device_inputs(Xv, model.n_features, opt.input)  # validates range / integrality

    evaluation = None
    if opt.leaf_bits == "auto":
        plans = []
        for b in AUTO_LEAF_BITS:
            try:
                plans.append(plan_leaves(model, b))
            except UnsupportedModelError:
                continue
        if not plans:
            raise UnsupportedModelError("no fixed-point leaf width up to 32 bits can hold this model's leaves")
        if Xv is not None and Xv.shape[0] > 0:
            evals = evaluate(model, plans, Xv)
            widest = evals[-1]
            chosen = None
            for i, ev in enumerate(evals):
                # The narrowest width at which (a) leaf rounding endangers no row that XGBoost's own
                # float32 rounding does not already endanger, and (b) no row gets a class it would
                # not also get at the widest width.
                only_ours = ev.quant_risk & ~ev.shared.float_risk
                worse = ev.differs_from_exact & ~widest.differs_from_exact
                if not only_ours.any() and not worse.any():
                    chosen = i
                    break
            tried = [p.bits for p in plans]
            if chosen is None:
                chosen = len(plans) - 1
                left = int((evals[chosen].quant_risk & ~evals[chosen].shared.float_risk).sum())
                notes.append(f"leaf width: none of {tried} bits keeps every validation row out of reach of leaf "
                             f"rounding; using {plans[chosen].bits} bits, at which {left} rows are still at risk "
                             "from leaf rounding")
            else:
                noisy = int(evals[chosen].shared.float_risk.sum())
                notes.append(f"leaf width chosen automatically: {plans[chosen].bits} bits is the narrowest of "
                             f"{tried} at which leaf rounding puts no validation row at risk"
                             + (f" that XGBoost's own float32 rounding does not already put at risk ({noisy} rows "
                                "are that close to a tie at any width)" if noisy else ""))
            lplan, evaluation = plans[chosen], evals[chosen]
        else:
            lplan = next((p for p in plans if p.bits >= DEFAULT_LEAF_BITS), plans[-1])
            notes.append(f"no validation data given: using the default {lplan.bits}-bit leaves; "
                         "pass validation data to check the tie risk and pick the width automatically")
    else:
        bits = _whole_number(opt.leaf_bits, "leaf_bits must be 'auto' or a whole number")
        lplan = plan_leaves(model, bits)
        if Xv is not None and Xv.shape[0] > 0:
            evaluation = evaluate(model, [lplan], Xv)[0]

    # Threshold encoding: build the candidates and keep the smaller model.
    best = None
    for mode in (("rank", "direct") if opt.thresholds == "auto" else (opt.thresholds,)):
        try:
            tp = plan_thresholds(model, opt.input, mode)
        except UnsupportedModelError:
            if opt.thresholds == "auto" and mode == "rank":
                continue   # e.g. more thresholds on one feature than a rank can count: direct still works
            raise
        if mode == "rank":
            tp.code_bytes = max(tp.code_bytes, opt.min_code_bytes)
        pk = pack(model, tp, lplan, leaf_store=opt.leaf_store, min_hdr_bytes=opt.min_hdr_bytes,
                  min_off_bytes=opt.min_off_bytes, min_leaf_bytes=opt.min_leaf_bytes)
        size = generate(model, tp, lplan, pk, opt.name, sizes_only=True).data_bytes
        if best is None or size <= best[0]:   # on a tie prefer direct: no table search at run time
            best = (size, tp, pk)
    _, tplan, packed = best
    summary = [
        f"{len(model.trees)} trees, {model.n_internal} splits, {model.n_leaves} leaves, "
        f"{model.n_features} features ({len(tplan.used_features)} used), {model.n_classes} classes.",
        f"Leaves: {lplan.bits}-bit fixed point, margins scaled by 2^{lplan.shift}.",
    ]
    gap = safe_gap(model, lplan)
    gen = generate(model, tplan, lplan, packed, opt.name, summary, safe_gap=gap)
    exp = Export(model=model, options=opt, tplan=tplan, lplan=lplan, packed=packed,
                 generated=gen, evaluation=evaluation, safe_gap=gap)
    exp.report = build_report(exp, notes)
    return exp


def build_report(exp: Export, notes: List[str]) -> Dict:
    model, tp, lp, pk, gen = exp.model, exp.tplan, exp.lplan, exp.packed, exp.generated
    stored = pk.flat.stored_trees
    n_split = pk.flat.n_nodes
    n_leaf_refs = int(pk.flat.leaf_refs.size)
    cmp_quant = 0
    if n_split and tp.mode == "direct":
        cmp_quant = len(tp.used_features) if tp.input_kind.is_float else 0   # one NaN test per float feature
    elif n_split and tp.n_cuts:
        for cuts in tp.cuts:
            cmp_quant += _bit_length(cuts.size) + (1 if tp.input_kind.is_float else 0)
    cmp_tree = int(sum(model.trees[t].depth for t in stored))
    if n_split == 0:
        cmp_arg = 0
    else:
        cmp_arg = 1 if model.is_binary else model.n_groups - 1
    data_bytes = gen.data_bytes
    rep: Dict = {
        "name": gen.name,
        "objective": model.objective,
        "booster": model.booster,
        "xgboost_version": ".".join(str(v) for v in model.version),
        "input": tp.input_kind.name,
        "trees": len(model.trees),
        "trees_stored": len(stored),
        "trees_folded_into_bias": len(model.trees) - len(stored),
        "num_parallel_tree": model.num_parallel_tree,
        "classes": model.n_classes,
        "features": model.n_features,
        "features_used": len(tp.used_features),
        "splits": model.n_internal,
        "leaves": model.n_leaves,
        "xgboost_nodes": model.n_internal + model.n_leaves,
        "max_depth": max((t.depth for t in model.trees), default=0),
        "layout": {
            "threshold_encoding": tp.mode,
            "header_bytes": pk.hdr_bytes,
            "threshold_code_bytes": pk.code_bytes,
            "offset_bytes": pk.off_bytes,
            "leaf_store": pk.leaf_mode,
            "leaf_operand_bytes": pk.leaf_bytes,
            "leaf_table_entries": 0 if pk.leaf_table is None else int(pk.leaf_table.size),
            "leaf_table_entry_bytes": pk.leaf_table_bytes,
            "slots": tp.n_slots,
            "cuts": tp.n_cuts,
            "node_types": dict(zip(TYPE_NAMES, pk.type_counts)),
            "bytes_per_split": (len(pk.nodes) / n_split) if n_split else 0.0,
            "bytes_per_xgboost_node": (data_bytes / max(1, model.n_internal + model.n_leaves)),
        },
        "flash": {
            "arrays": [{"name": a.name, "type": a.ctype, "count": a.count, "bytes": a.bytes, "what": a.what}
                       for a in gen.arrays],
            "data_bytes": data_bytes,
        },
        "ram": {
            "static_bytes": 0,
            "rank_array_bytes": gen.q_bytes,
        },
        "comparisons_worst_case": {
            "feature_ranking": cmp_quant,
            "tree_splits": cmp_tree,
            "argmax": cmp_arg,
            "total": cmp_quant + cmp_tree + cmp_arg,
        },
        "leaves_fixed_point": {
            "bits": lp.bits,
            "shift": lp.shift,
            "lsb": lp.lsb,
            "max_abs_leaf": lp.max_abs_leaf,
            "worst_case_margin_error_per_class": [float(x) for x in lp.worst_error],
            "worst_case_margin_error": float(lp.worst_error.max()),
            "error_range_per_class": [[float(a), float(b)] for a, b in zip(lp.err_lo, lp.err_hi)],
            "max_abs_accumulator": int(max(lp.max_abs_acc)),
            "safe_gap": int(exp.safe_gap),
            "safe_gap_margin_units": exp.safe_gap * lp.lsb,
        },
        "notes": list(notes),
    }
    if tp.input_kind.is_float is False:
        rep["notes"].append(
            f"integer inputs ({tp.input_kind.ctype}): features must be the exact integers the model "
            "was trained on; missing values cannot be represented")
    ev = exp.evaluation
    if ev is not None:
        sh = ev.shared
        gaps = margin_gap(sh.exact, model.is_binary)
        risk_idx = np.nonzero(ev.at_risk)[0]
        rep["validation"] = {
            "samples": ev.n,
            "at_risk": int(ev.at_risk.sum()),
            "at_risk_from_leaf_rounding": int(ev.quant_risk.sum()),
            "at_risk_from_xgboost_float32_rounding": int(sh.float_risk.sum()),
            "at_risk_indices": [int(i) for i in risk_idx],
            "class_differs_from_exact_arithmetic": int(ev.differs_from_exact.sum()),
            "differs_indices": [int(i) for i in np.nonzero(ev.differs_from_exact)[0]],
            "smallest_margin_gap": float(gaps.min()) if gaps.size else None,
            "within_safe_gap": int((integer_gap(ev.integer, model.is_binary) <= exp.safe_gap).sum()),
            "xgboost_float32_noise_bound_max": float(sh.float_noise.max()) if sh.float_noise.size else 0.0,
            "tree_comparisons_mean": float(sh.path_len.mean()) if ev.n else 0.0,
            "tree_comparisons_max": int(sh.path_len.max()) if ev.n else 0,
        }
    return rep
