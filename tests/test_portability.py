"""The generated C must be clean for every compiler we can find, not only the host's.

The host harness already compiles every zoo export for Cortex-M4 and both
ESP32 architectures.  Here small models in every layout variant are also
compiled for an 8-bit AVR (16-bit int: catches integer-width assumptions) and
as C++ (people paste generated C into .cpp files).
"""
import os
import subprocess

import numpy as np
import pytest

from harness import synth
from xgb_edge import ExportOptions, export_model
from xgb_edge.report import available_targets

F32 = np.float32

VARIANTS = [
    dict(),
    dict(thresholds="rank"),
    dict(thresholds="direct"),
    dict(thresholds="rank", min_code_bytes=2, min_hdr_bytes=2, min_off_bytes=2),
    dict(leaf_bits=8), dict(leaf_bits=24), dict(leaf_bits=32),
    dict(leaf_bits=16, leaf_store="table"), dict(leaf_bits=32, leaf_store="table"),
    dict(input="i8", thresholds="rank"), dict(input="i8", thresholds="direct"),
    dict(input="u8", thresholds="direct"), dict(input="i16", thresholds="rank"),
    dict(input="i16", thresholds="direct"), dict(input="u16", thresholds="direct"),
    dict(input="i32", thresholds="rank"), dict(input="i32", thresholds="direct"),
]


def _model(tmp_path, n_classes, seed=0, all_const=False):
    rng = np.random.RandomState(seed)
    pools = {f: F32(np.round(rng.randn(6) * 40)) + F32(0.5) for f in range(5)}
    pools[0] = F32([-200.5, 0.0, 127.0, 300.5, -129.0])     # outside int8: always-left / always-right splits
    groups = 1 if n_classes == 2 else n_classes
    trees = []
    for g in range(groups):
        for k in range(3):
            if all_const or (k == 2 and g == 0):
                trees.append(synth.SynthTree.single_leaf(float(F32(rng.randn() * 0.1)), group=g))
                continue
            t = synth.random_tree(rng, 5, pools, 9, group=g, chain=(k == 1))
            for leaf in t.leaves:
                t.value[leaf] = float(F32(rng.randn() * 0.3))
            trees.append(t)
    return synth.write_model(str(tmp_path / f"m{n_classes}.json"), trees, 5, n_classes)


def _compile(cmd, src_dir, name="m"):
    r = subprocess.run(cmd + ["-c", os.path.join(src_dir, name + ".c"), "-o", os.path.join(src_dir, name + ".o")],
                       capture_output=True, text=True)
    assert r.returncode == 0 and not r.stderr.strip(), " ".join(cmd) + "\n" + r.stderr


@pytest.mark.parametrize("n_classes", [2, 4])
def test_every_layout_compiles_for_every_available_target(tmp_path, n_classes):
    targets = available_targets()
    if not targets:
        pytest.skip("no cross compilers installed")
    path = _model(tmp_path, n_classes)
    seen = set()
    for i, opts in enumerate(VARIANTS):
        exp = export_model(path, ExportOptions(name="m", **opts))
        d = tmp_path / f"v{i}"
        exp.write(str(d))
        seen.add((exp.tplan.mode, exp.packed.leaf_mode, exp.options.input))
        for target, (cc, flags) in targets.items():
            _compile([cc] + flags.split() + ["-std=c99", "-Wall", "-Wextra", "-Werror"], str(d))
    assert {m for m, _, _ in seen} == {"rank", "direct"} and {s for _, s, _ in seen} == {"inline", "table"}


def test_constant_model_compiles_everywhere(tmp_path):
    path = _model(tmp_path, 3, all_const=True)
    exp = export_model(path, ExportOptions(name="m"))
    exp.write(str(tmp_path))
    for target, (cc, flags) in available_targets().items():
        _compile([cc] + flags.split() + ["-std=c99", "-Wall", "-Wextra", "-Werror"], str(tmp_path))


@pytest.mark.parametrize("std", ["-std=c99", "-std=c11", "-std=gnu17"])
def test_strict_host_warnings(tmp_path, std):
    path = _model(tmp_path, 4)
    for i, opts in enumerate(VARIANTS):
        d = tmp_path / f"v{i}"
        export_model(path, ExportOptions(name="m", **opts)).write(str(d))
        _compile(["cc", std, "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow", "-Wcast-qual",
                  "-Wstrict-prototypes", "-Wmissing-prototypes", "-Wundef", "-Werror"], str(d))


def test_generated_c_is_also_valid_cxx(tmp_path):
    path = _model(tmp_path, 4)
    for i, opts in enumerate(VARIANTS):
        d = tmp_path / f"v{i}"
        export_model(path, ExportOptions(name="m", **opts)).write(str(d))
        r = subprocess.run(["c++", "-x", "c++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-c",
                            str(d / "m.c"), "-o", str(d / "m.o")], capture_output=True, text=True)
        assert r.returncode == 0 and not r.stderr.strip(), r.stderr
