"""Generate C for an XGBoost JSON model with m2cgen and with TL2cgen, and check it on the host.

The same trained model (same trees, thresholds and leaves) goes into each tool:

``m2cgen`` (0.10.0, last release 2022)
    Does not accept XGBoost >= 3.1 models out of the box: ``m2cgen.export_to_c`` on an
    ``XGBClassifier`` that loaded the JSON fails with ``TypeError: unsupported operand
    type(s) for /: 'float' and 'list'`` (binary: ``base_score`` is now a list) or
    ``TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'`` (multiclass:
    ``num_parallel_tree`` is not restored by ``load_model``).  What is done here instead:

    * the JSON is loaded into a real ``xgboost.XGBClassifier``; m2cgen reads the trees
      from its ``get_booster().get_dump(dump_format="json")`` exactly as it always does;
    * ``num_parallel_tree`` and (binary) the scalar ``base_score`` are set on the sklearn
      wrapper from the model file (this does not touch the booster);
    * for multiclass models a three-line subclass of m2cgen's own
      ``XGBoostTreeModelAssembler`` adds the per-class base margin of XGBoost >= 3.1
      where stock m2cgen adds one scalar ``base_score`` to every class.

    The code generator (``m2cgen.interpreters.CInterpreter``) is used unmodified.

``TL2cgen`` (1.0.0) with ``treelite`` (4.8.1)
    ``treelite.frontend.load_xgboost_model`` reads the XGBoost 3.4 JSON as is, including
    the per-class ``base_score`` vector and DART weights; nothing is patched.  TL2cgen
    bundles an older Treelite (4.1.2) and prints a "model originated from a newer Treelite
    version" warning when it receives the model; the host check below is what shows that
    the generated code is nevertheless the same model.

What the host check (:func:`verify_host`, ``python -m bench.altgen verify``) shows about the
generated code, as opposed to about this wrapper:

* m2cgen prints each float32 threshold as its shortest decimal (``0.4134592``) and compares
  in double.  That decimal is not the float32 value, so an input exactly equal to a threshold
  can take the other branch than in XGBoost.  Missing values (NaN) do follow XGBoost's default
  direction: m2cgen orders each test so that NaN falls into the right branch.
* TL2cgen compares in float32 against thresholds written with enough digits, and is exact as
  long as the caller marks missing features with ``data[i].missing = -1``; a NaN passed as
  ``fvalue`` is not recognised and always takes the else branch.
* TL2cgen with ``quantize=1`` writes its threshold table (``quantize.c``) with 8 significant
  digits, which does not identify every float32: a few thresholds are off by one ulp, and
  inputs exactly on them take the other branch.
* Both tools sum the leaves differently from XGBoost (m2cgen in double, TL2cgen by adding a
  double literal to a float and adding the base score last), so margins agree only to
  rounding, and binary base margins are computed in double (XGBoost: float32).

Functions: :func:`generate_m2cgen`, :func:`generate_tl2cgen`, :func:`generate` (by variant
name) and :func:`verify_host`.  Command line::

    python -m bench.altgen gen VARIANT MODEL.json OUTDIR      # one variant, prints a result line
    python -m bench.altgen verify [zoo model names...]        # generate + host class check
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgb_edge.model import load, round_decimal_to_f32  # noqa: E402

from bench import hostrun  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build", "bench")

#: variant name -> short description
VARIANTS = {
    "m2cgen": "m2cgen if-else C, double arithmetic",
    "tl2cgen": "TL2cgen if-else C, default settings (float32 thresholds)",
    "tl2cgen_quantize": "TL2cgen if-else C with quantize=1 (integer threshold indices)",
}


_RESULT_MARK = "ALTGEN_RESULT "


class AltGenError(RuntimeError):
    """A generator cannot represent this model (the message says why)."""


def versions() -> Dict[str, str]:
    import m2cgen
    import tl2cgen
    import treelite
    import xgboost
    return {"m2cgen": m2cgen.__version__, "tl2cgen": tl2cgen.__version__,
            "treelite": treelite.__version__, "xgboost": xgboost.__version__}


def _base_scores(model_path: str) -> List[float]:
    """learner_model_param.base_score as exact float32 values (probability space for logistic)."""
    with open(model_path) as f:
        text = json.load(f)["learner"]["learner_model_param"]["base_score"]
    text = str(text).strip()
    if text.startswith("["):
        text = text[1:-1]
    return [float(round_decimal_to_f32(p.strip())) for p in text.split(",") if p.strip()]


def _run_deep(fn):
    """Run ``fn()`` with a large recursion limit on a thread with a large stack.

    m2cgen assembles and interprets the model recursively; the depth grows with the number
    of trees per class, and its own documentation says to raise the recursion limit.
    """
    box: Dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    old_limit = sys.getrecursionlimit()
    old_stack = threading.stack_size(512 * 1024 * 1024)
    sys.setrecursionlimit(max(old_limit, 1_000_000))
    try:
        t = threading.Thread(target=target)
        t.start()
        t.join()
    finally:
        sys.setrecursionlimit(old_limit)
        threading.stack_size(old_stack)
    if "error" in box:
        raise box["error"]
    return box["value"]


# --------------------------------------------------------------------------- m2cgen

def generate_m2cgen(model_path: str, outdir: str, function_name: str = "score") -> Dict:
    """Write ``<outdir>/model.c``: m2cgen's C for the model.

    The generated entry point is ``void score(double *input, double *output)``; ``output``
    receives the class probabilities (2 for a binary model).  All arithmetic is ``double``.
    """
    import xgboost as xgb
    from m2cgen import interpreters
    from m2cgen.assemblers.boosting import XGBoostTreeModelAssembler, _split_estimator_params_by_classes

    t0 = time.time()
    model = load(model_path)
    notes: List[str] = []
    if model.booster == "dart":
        raise AltGenError("m2cgen reads the trees from XGBoost's text dump, which does not contain the DART "
                          "tree weights; the generated code would be a different model")
    base = _base_scores(model_path)
    npt = model.num_parallel_tree
    groups = [t.group for t in model.trees]
    if groups != [(i // npt) % model.n_groups for i in range(len(groups))]:
        raise AltGenError("m2cgen assumes trees are stored round-robin by class; this model's tree_info is not")

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    expect_groups = 1 if clf.n_classes_ == 2 else clf.n_classes_
    if expect_groups != model.n_groups:
        raise AltGenError(f"m2cgen would build {expect_groups} outputs for {clf.n_classes_} classes, "
                          f"but the model has {model.n_groups} margins")

    if model.is_binary:
        if model.objective == "binary:hinge":
            if base[0] != 0.0:
                raise AltGenError("m2cgen has no binary:hinge support (it would apply logit() to the base score)")
            notes.append("binary:hinge: m2cgen treats the model as logistic and outputs sigmoid(margin); "
                         "output[1] > 0.5 is still margin > 0")
        # m2cgen's own binary path: base margin = -log(1 / base_score - 1), evaluated in double
        clf.set_params(num_parallel_tree=npt, base_score=base[0])
        assembler = XGBoostTreeModelAssembler(clf)
        notes.append("base_score taken from the model file and set on the XGBClassifier wrapper; "
                     "m2cgen converts it to a margin in double precision (XGBoost does it in float32)")
    else:
        class PerClassBaseAssembler(XGBoostTreeModelAssembler):
            """Stock assembler, except that class i gets base margin ``base[i]`` (XGBoost >= 3.1)."""

            def _assemble_multi_class_output(self, estimator_params):
                splits = _split_estimator_params_by_classes(
                    estimator_params, self._output_size, self.multiclass_params_seq_len)
                exprs = [self._assemble_single_output(e, base_score=base[i], split_idx=i)
                         for i, e in enumerate(splits)]
                return self._multi_class_convert_output(exprs)

        clf.set_params(num_parallel_tree=npt, base_score=0.0)
        assembler = PerClassBaseAssembler(clf)
        notes.append("per-class base_score vector applied through a subclass of m2cgen's "
                     "XGBoostTreeModelAssembler (stock m2cgen supports only one scalar base_score)")

    interpreter = interpreters.CInterpreter(function_name=function_name)
    code = _run_deep(lambda: interpreter.interpret(assembler.assemble()))

    os.makedirs(outdir, exist_ok=True)
    src = os.path.join(outdir, "model.c")
    with open(src, "w") as f:
        f.write(code)
    extra_cflags: List[str] = []
    if "exp(" in code and "#include <math.h>" not in code:
        # m2cgen 0.10.0 bug: the multiclass (softmax) output calls exp() without including <math.h>.
        # clang rejects the implicit declaration, gcc 9 only warns.  The file is left as generated;
        # whoever compiles it must add this flag.
        extra_cflags = ["-include", "math.h"]
        notes.append("m2cgen omits #include <math.h> although its softmax helper calls exp(); "
                     "compile with '-include math.h'")
    return {"variant": "m2cgen", "sources": [src], "outdir": outdir, "entry": function_name,
            "n_outputs": model.n_classes, "seconds": round(time.time() - t0, 2),
            "source_bytes": os.path.getsize(src), "notes": notes, "extra_cflags": extra_cflags}


# --------------------------------------------------------------------------- TL2cgen

@contextlib.contextmanager
def _capture_fd2():
    """Collect what native code writes to stderr (Treelite/TL2cgen log lines)."""
    buf = io.StringIO()
    sys.stderr.flush()
    saved = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)
    chunks: List[bytes] = []

    def pump():
        while True:
            b = os.read(r, 65536)
            if not b:
                break
            chunks.append(b)

    th = threading.Thread(target=pump)
    th.start()
    try:
        yield buf
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        th.join()
        os.close(r)
        buf.write(b"".join(chunks).decode(errors="replace"))


def generate_tl2cgen(model_path: str, outdir: str, quantize: bool = False,
                     params: Optional[Dict] = None) -> Dict:
    """Write TL2cgen's C for the model into ``outdir`` (``header.h``, ``main.c``, ...).

    ``quantize=True`` passes ``quantize=1``; everything else is TL2cgen's default, which
    puts all trees into one ``predict()`` function in ``main.c``.  The entry point is
    ``void predict(union Entry *data, int pred_margin, float *result)``; ``result`` must be
    zeroed by the caller, and a missing feature is ``data[i].missing = -1``.
    """
    import tl2cgen
    import treelite

    t0 = time.time()
    p = dict(params or {})
    if quantize:
        p["quantize"] = 1
    model = load(model_path)
    os.makedirs(outdir, exist_ok=True)
    with _capture_fd2() as log:
        tl_model = treelite.frontend.load_xgboost_model(model_path)
        tl2cgen.generate_c_code(tl_model, outdir, params=p)
    notes: List[str] = []
    text = log.getvalue()
    if "newer Treelite version" in text:
        notes.append("TL2cgen warned that the model came from a newer Treelite than the one it bundles "
                     "(some functionalities may be unavailable)")
    if tl_model.num_tree != len(model.trees) or tl_model.num_feature != model.n_features:
        raise AltGenError(f"treelite loaded {tl_model.num_tree} trees / {tl_model.num_feature} features, "
                          f"the model has {len(model.trees)} / {model.n_features}")
    # recipe.json is TL2cgen's own list of what it wrote; anything else in outdir is not ours
    with open(os.path.join(outdir, "recipe.json")) as f:
        sources = [os.path.join(outdir, src["name"] + ".c") for src in json.load(f)["sources"]]
    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise AltGenError(f"TL2cgen's recipe.json lists files it did not write: {missing}")
    return {"variant": "tl2cgen_quantize" if quantize else "tl2cgen", "sources": sources, "outdir": outdir,
            "entry": "predict", "n_outputs": model.n_groups, "params": p,
            "seconds": round(time.time() - t0, 2),
            "source_bytes": sum(os.path.getsize(s) for s in sources), "notes": notes,
            "log": text[-2000:]}


def generate(variant: str, model_path: str, outdir: str, params: Optional[Dict] = None) -> Dict:
    """Generate one variant by name.  ``params`` are extra TL2cgen compiler parameters
    (e.g. ``{"parallel_comp": 32}``); the default is each tool's default settings."""
    if variant == "m2cgen":
        if params:
            raise ValueError("m2cgen takes no parameters")
        return generate_m2cgen(model_path, outdir)
    if variant == "tl2cgen":
        return generate_tl2cgen(model_path, outdir, params=params)
    if variant == "tl2cgen_quantize":
        return generate_tl2cgen(model_path, outdir, quantize=True, params=params)
    raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")


def generate_subprocess(variant: str, model_path: str, outdir: str, timeout: Optional[float] = None,
                        params: Optional[Dict] = None) -> Dict:
    """Run :func:`generate` in a child process so that a crash or a hang cannot take the caller down.

    Returns the generator's result dict, or ``{"error": text}``.
    """
    cmd = [sys.executable, "-m", "bench.altgen", "gen", variant, model_path, outdir]
    for k, v in (params or {}).items():
        cmd += ["--param", f"{k}={v}"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"variant": variant, "error": f"generation timed out after {timeout:.0f} s"}
    if r.returncode != 0:
        tail = (r.stderr.strip() or r.stdout.strip())[-1500:]
        return {"variant": variant, "error": f"generation failed (exit {r.returncode}): {tail}"}
    # the tools log to stdout too, so the result line carries a marker
    for line in reversed(r.stdout.splitlines()):
        if line.startswith(_RESULT_MARK):
            try:
                return json.loads(line[len(_RESULT_MARK):])
            except ValueError:
                break
    return {"variant": variant, "error": "generator printed no result: " + r.stdout[-500:]}


# --------------------------------------------------------------------------- host check

_M2CGEN_DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
void @ENTRY@(double *input, double *output);
#define NF @NF@
#define NO @NO@
int main(int argc, char **argv)
{
    long rows = argc > 1 ? atol(argv[1]) : 0, r;
    static float x[NF + 1];
    static double in[NF + 1], out[NO];
    int i;
    for (r = 0; r < rows; r++) {
        if (NF > 0 && fread(x, sizeof x[0], NF, stdin) != (size_t)NF) return 2;
        for (i = 0; i < NF; i++) in[i] = (double)x[i];   /* exact; NaN stays NaN */
        @ENTRY@(in, out);
        if (fwrite(out, sizeof out[0], NO, stdout) != (size_t)NO) return 3;
    }
    return 0;
}
"""

_TL2CGEN_DRIVER = r"""
#include "header.h"
#define NF @NF@
#define NO (N_TARGET * MAX_N_CLASS)
#define NAN_IS_MISSING @NAN_IS_MISSING@
int main(int argc, char **argv)
{
    long rows = argc > 1 ? atol(argv[1]) : 0, r;
    static float x[NF + 1];
    static union Entry data[NF + 1];
    static float out[NO];
    int i;
    if (get_num_feature() != NF) return 4;
    for (r = 0; r < rows; r++) {
        if (NF > 0 && fread(x, sizeof x[0], NF, stdin) != (size_t)NF) return 2;
        for (i = 0; i < NF; i++) {
            if (NAN_IS_MISSING && x[i] != x[i]) data[i].missing = -1;   /* TL2cgen's missing value */
            else data[i].fvalue = x[i];
        }
        for (i = 0; i < NO; i++) out[i] = 0.0f;           /* predict() only adds to result[] */
        predict(data, 1 /* pred_margin */, out);
        if (fwrite(out, sizeof out[0], NO, stdout) != (size_t)NO) return 3;
    }
    return 0;
}
"""


def _mismatch_summary(cls: np.ndarray, tr) -> Dict:
    bad = cls != tr.xgb_class
    nan_row = np.isnan(tr.X).any(axis=1)
    held = np.arange(tr.X.shape[0]) < tr.n_heldout
    return {
        "rows": int(bad.size),
        "mismatches": int(bad.sum()),
        "rows_heldout": int(held.sum()),
        "mismatches_heldout": int((bad & held).sum()),
        "rows_adversarial": int((~held).sum()),
        "mismatches_adversarial": int((bad & ~held).sum()),
        "rows_with_nan": int(nan_row.sum()),
        "mismatches_rows_with_nan": int((bad & nan_row).sum()),
        "first_mismatch_rows": [int(i) for i in np.nonzero(bad)[0][:10]],
    }


def verify_host(gen: Dict, tr, workdir: Optional[str] = None, cflags=("-O0", "-w"),
                timeout: Optional[float] = None, nan_as_missing: bool = True) -> Dict:
    """Compile a generated variant for the host and compare its class with ``tr.xgb_class`` on every row.

    ``gen`` is the dict returned by a generator; ``tr`` a ``harness.zoo.Trained``.  Classes are
    derived the way a user of each tool would: arg max of ``score()``'s probabilities for m2cgen
    (``output[1] > 0.5`` for binary), arg max of the raw margins for TL2cgen (``margin > 0`` for
    binary).  Returns mismatch counts (see :func:`_mismatch_summary`).

    For TL2cgen a NaN feature is passed as ``missing = -1``, as its API requires;
    ``nan_as_missing=False`` passes the NaN as a plain float instead, to show what happens to a
    caller who does not know that.
    """
    workdir = workdir or gen["outdir"]
    nf = tr.X.shape[1]
    variant = gen["variant"]
    binary = tr.xgb_margin.shape[1] == 1
    drv = os.path.join(workdir, "host_driver.c")
    exe = os.path.join(workdir, "host_run")
    t0 = time.time()
    if variant == "m2cgen":
        n_out = int(gen["n_outputs"])
        text = _M2CGEN_DRIVER.replace("@ENTRY@", gen["entry"]).replace("@NF@", str(nf)).replace("@NO@", str(n_out))
        out_dtype = np.float64
    else:
        n_out = int(gen["n_outputs"])
        text = _TL2CGEN_DRIVER.replace("@NF@", str(nf)).replace("@NAN_IS_MISSING@", "1" if nan_as_missing else "0")
        out_dtype = np.float32
    with open(drv, "w") as f:
        f.write(text)
    sources = list(gen["sources"]) + [drv]
    hostrun.compile_host(sources, exe, list(cflags) + list(gen.get("extra_cflags", [])),
                         include_dirs=[gen["outdir"]], timeout=timeout)
    t_compile = time.time() - t0
    out = hostrun.run_rows(exe, tr.X, out_dtype, n_out, timeout=timeout)
    res: Dict = {"variant": variant, "host_compile_seconds": round(t_compile, 2)}
    if variant == "m2cgen":
        cls = (out[:, 1] > 0.5).astype(np.int64) if binary else np.argmax(out, axis=1).astype(np.int64)
        res["class_rule"] = "output[1] > 0.5" if binary else "arg max of the probabilities"
    else:
        cls = (out[:, 0] > 0).astype(np.int64) if binary else np.argmax(out, axis=1).astype(np.int64)
        res["class_rule"] = "margin > 0" if binary else "arg max of the margins"
        diff = np.abs(out.astype(np.float64) - tr.xgb_margin.astype(np.float64))
        res["max_abs_margin_diff"] = float(diff.max()) if diff.size else 0.0
        res["rows_margin_not_bit_identical"] = int(
            (np.ascontiguousarray(out).view(np.uint32) != tr.xgb_margin.view(np.uint32)).any(axis=1).sum())
        # far more than float rounding can explain: some tree reached a different leaf
        res["rows_margin_off_by_more_than_1e-3"] = int((diff > 1e-3).any(axis=1).sum())
    res.update(_mismatch_summary(cls, tr))
    return res


# --------------------------------------------------------------------------- command line

DEFAULT_VERIFY_MODELS = ["bin_d2_r5_hist", "mc3_d3_r10_hist", "mc5_d4_r50_hist_missing",
                         "mc10_digits_d6_r100_hist"]


def _cmd_verify(names: List[str], gen_timeout: float, timeout: float) -> int:
    from harness import zoo
    rows: List[Dict] = []
    for name in names:
        tr = zoo.train(zoo.by_name(name))
        for variant in VARIANTS:
            outdir = os.path.join(BUILD, "verify", name, variant)
            row: Dict = {"model": name, "variant": variant}
            gen = generate_subprocess(variant, tr.model_path, outdir, timeout=gen_timeout)
            if "error" in gen:
                row["status"] = gen["error"]
            else:
                row["generate_seconds"] = gen["seconds"]
                row["notes"] = gen["notes"]
                try:
                    row.update(verify_host(gen, tr, timeout=timeout))
                    row["status"] = "ok"
                except subprocess.TimeoutExpired:
                    row["status"] = f"host compile/run timed out after {timeout:.0f} s"
                except Exception as e:  # noqa: BLE001 - recorded as the result
                    row["status"] = "host check failed: " + str(e)[-800:]
            rows.append(row)
            if row["status"] == "ok":
                extra = ""
                if "max_abs_margin_diff" in row:
                    extra = (f"; max |margin - xgboost| {row['max_abs_margin_diff']:.2g}, "
                             f"{row['rows_margin_off_by_more_than_1e-3']} rows off by > 1e-3")
                print(f"{name:<28} {variant:<17} rows {row['rows']:5d}  class mismatches {row['mismatches']:4d} "
                      f"(held-out {row['mismatches_heldout']}/{row['rows_heldout']}, adversarial "
                      f"{row['mismatches_adversarial']}/{row['rows_adversarial']}; in rows with NaN "
                      f"{row['mismatches_rows_with_nan']}/{row['rows_with_nan']}){extra}", flush=True)
            else:
                print(f"{name:<28} {variant:<17} {row['status']}", flush=True)
    os.makedirs(BUILD, exist_ok=True)
    path = os.path.join(BUILD, "altgen_verify.json")
    kept: List[Dict] = []
    if os.path.exists(path):      # keep the rows of models checked in earlier runs
        try:
            with open(path) as f:
                kept = [r for r in json.load(f).get("results", []) if r.get("model") not in names]
        except ValueError:
            kept = []
    with open(path, "w") as f:
        json.dump({"versions": versions(), "results": kept + rows}, f, indent=1)
    print(f"\nwrote {path} ({len(kept) + len(rows)} rows; {len(rows)} from this run)")
    print("Classes: m2cgen = arg max of score() probabilities; TL2cgen = arg max of predict(pred_margin=1).")
    print("Adversarial rows sit exactly on / one float32 next to split thresholds, or contain NaN or extremes.")
    print("A margin off by > 1e-3 means some tree reached a different leaf than in XGBoost (not rounding).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="m2cgen / TL2cgen code generation for XGBoost JSON models.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate one variant; prints 'ALTGEN_RESULT {json}'")
    g.add_argument("variant", choices=sorted(VARIANTS))
    g.add_argument("model")
    g.add_argument("outdir")
    g.add_argument("--param", action="append", default=[], metavar="KEY=INT",
                   help="extra TL2cgen compiler parameter, e.g. parallel_comp=32 (repeatable)")
    v = sub.add_parser("verify", help="generate every variant for zoo models and compare classes on the host")
    v.add_argument("models", nargs="*", default=DEFAULT_VERIFY_MODELS)
    v.add_argument("--gen-timeout", type=float, default=1200.0)
    v.add_argument("--timeout", type=float, default=1200.0, help="host compile / run timeout in seconds")
    args = ap.parse_args(argv)
    if args.cmd == "gen":
        try:
            params = {k: int(v) for k, v in (p.split("=", 1) for p in args.param)}
            res = generate(args.variant, args.model, args.outdir, params or None)
        except AltGenError as e:
            print("unsupported: " + str(e), file=sys.stderr)
            return 3
        except Exception:  # noqa: BLE001 - the caller records the text
            traceback.print_exc()
            return 1
        sys.stdout.flush()
        print("\n" + _RESULT_MARK + json.dumps(res), flush=True)
        return 0
    return _cmd_verify(args.models, args.gen_timeout, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
