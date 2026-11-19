"""Check models saved by older xgboost releases.

    python -m harness.oldver [versions...]

For each version it runs harness/oldver_child.py inside a throwaway
environment (`uv run --with xgboost==<version>`), which trains several models
and records that version's own predictions; the models are then exported and
put through the same parity checks as the zoo, with that old version as the
reference.  Requires `uv` and network access for the first run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import zoo  # noqa: E402
from harness.parity import check_variant, print_table  # noqa: E402
from xgb_edge import load  # noqa: E402
from xgb_edge.refsim import leaf_indices  # noqa: E402

VERSIONS = ["1.6.2", "1.7.6", "2.0.3", "2.1.4", "3.0.5"]
OUT = os.path.join(zoo.ROOT, "build", "oldver")
CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oldver_child.py")


def _child_hash() -> str:
    import hashlib
    with open(CHILD, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def make_reference(version: str) -> str:
    d = os.path.join(OUT, version)
    done = os.path.join(d, "done")
    if os.path.exists(done) and open(done).read().split("\n")[0] == _child_hash():
        return d
    cmd = ["uv", "run", "--no-project", "--python", "3.11", "--with", f"xgboost=={version}",
           "--with", "numpy<2", "--with", "scikit-learn<1.6", "--with", "scipy<1.14",
           "--with", "setuptools<70",  # xgboost 1.6 imports pkg_resources
           "python", CHILD, d]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"xgboost {version}: could not produce reference models:\n{(r.stdout + r.stderr)[-2500:]}")
    open(done, "w").write(_child_hash() + "\n" + r.stdout.strip().splitlines()[-1])
    return d


def check_version(version: str) -> List[Dict]:
    rows: List[Dict] = []
    try:
        d = make_reference(version)
    except RuntimeError as e:
        return [{"model": f"xgboost {version}", "variant": "-", "failures": [str(e)]}]
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        case = fn[:-5]
        path = os.path.join(d, fn)
        z = np.load(os.path.join(d, case + ".npz"))
        model = load(path)
        head = []
        if not np.array_equal(leaf_indices(model, z["X"]), z["leaf"]):
            head.append("float32 reference traversal differs from that version's pred_leaf")
        spec = zoo.Spec(f"{version}/{case}", {}, {}, 0)
        tr = zoo.Trained(spec, d, path, z["X"], z["X"].shape[0], z["cls"], z["margin"], z["leaf"], str(z["objective"]))
        # the in-place path, with the class XGBClassifier.predict reported
        ref_cls, quirk = z["cls_sk"], None
        if not np.array_equal(z["cls_sk"], z["cls_ip"]):
            if str(z["objective"]) == "multi:softmax" and set(np.unique(z["cls_sk"]).tolist()) <= {0, 1}:
                # XGBoost 1.x: XGBClassifier.predict thresholds the class *indices* a natively trained
                # multi:softmax booster returns at 0.5, so it only ever answers 0 or 1.  That is a quirk
                # of that wrapper, not a prediction of the model; use the booster's own answer.
                ref_cls = z["cls_ip"]
                quirk = ("XGBClassifier.predict of this release answers only 0/1 for a natively trained "
                         "multi:softmax model; compared with Booster.inplace_predict instead")
            else:
                head.append("XGBClassifier.predict differs from the class implied by inplace_predict")
        tr.xgb_class_inplace, tr.xgb_margin_inplace = ref_cls, z["margin_ip"]
        for opts in ({}, {"leaf_bits": 16}):
            wd = os.path.join(OUT, "work", version, case, "auto" if not opts else "b16")
            r = check_variant(tr, model, opts, wd, None)
            r["failures"] = head + r["failures"]
            r.update(splits=model.n_internal, trees=len(model.trees), classes=model.n_classes)
            if quirk:
                r["note"] = quirk
            rows.append(r)
    return rows


def main(argv=None) -> int:
    versions = (argv if argv is not None else sys.argv[1:]) or VERSIONS
    rows: List[Dict] = []
    for v in versions:
        rows.extend(check_version(v))
    print_table(rows)
    failed = [r for r in rows if r["failures"]]
    for r in failed:
        print(f"\nFAIL {r['model']} [{r['variant']}]")
        for f in r["failures"]:
            print("  - " + f.replace("\n", "\n    "))
    for note in sorted({f"{r['model']}: {r['note']}" for r in rows if r.get("note")}):
        print("note: " + note)
    print(f"\n{len(rows)} exports from xgboost {', '.join(versions)}: {len(rows) - len(failed)} ok, {len(failed)} failed")
    # merge by version, so that checking one release does not drop the others from the summary
    path = os.path.join(OUT, "summary.json")
    stamp = zoo.code_stamp()
    for r in rows:
        r["code"] = stamp
    old = []
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except ValueError:
            old = []
    keep = [r for r in old if r["model"].split("/")[0].replace("xgboost ", "") not in versions]
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(keep + rows, key=lambda r: [int(x) if x.isdigit() else 0 for x in
                                                      r["model"].split("/")[0].replace("xgboost ", "").split(".")]),
                  f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
