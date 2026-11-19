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


def make_reference(version: str) -> str:
    d = os.path.join(OUT, version)
    if os.path.exists(os.path.join(d, "done")):
        return d
    cmd = ["uv", "run", "--no-project", "--python", "3.11", "--with", f"xgboost=={version}",
           "--with", "numpy<2", "--with", "scikit-learn<1.6", "--with", "scipy<1.14",
           "--with", "setuptools<70",  # xgboost 1.6 imports pkg_resources
           "python", CHILD, d]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"xgboost {version}: could not produce reference models:\n{(r.stdout + r.stderr)[-2500:]}")
    open(os.path.join(d, "done"), "w").write(r.stdout.strip().splitlines()[-1])
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
        for opts in ({}, {"leaf_bits": 16}):
            wd = os.path.join(OUT, "work", version, case, "auto" if not opts else "b16")
            r = check_variant(tr, model, opts, wd, None)
            r["failures"] = head + r["failures"]
            r.update(splits=model.n_internal, trees=len(model.trees), classes=model.n_classes)
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
    print(f"\n{len(rows)} exports from xgboost {', '.join(versions)}: {len(rows) - len(failed)} ok, {len(failed)} failed")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
