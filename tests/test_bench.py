"""Tests for the comparison-benchmark code in bench/ (small zoo models only).

The float32 baseline must compile without warnings and reproduce xgboost's margins bit for
bit; the m2cgen / TL2cgen wrappers and the size measurement get a smoke test each.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import float_baseline, hostrun  # noqa: E402
from harness import zoo  # noqa: E402
from xgb_edge import load  # noqa: E402
from xgb_edge.report import find_cross_compiler  # noqa: E402

#: the flags the baseline is required to be clean under
HOST_CFLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror"]
STRICT_CFLAGS = HOST_CFLAGS + ["-O2", "-Wpedantic", "-Wconversion", "-Wshadow"]

BINARY = "bin_d2_r5_hist"
MISSING = "mc5_d4_r50_hist_missing"
GBTREE_MODELS = [BINARY, "mc3_d3_r10_hist", MISSING, "mc3_single_leaf_trees", "mc3_all_single_leaf"]


def trained(name):
    return zoo.train(zoo.by_name(name))


def build_and_run(tr, workdir, cflags=HOST_CFLAGS, **gen_kwargs):
    """Generate the baseline for a zoo model, compile it and run every test row through it."""
    model = load(tr.model_path)
    header, source = float_baseline.generate(tr.model_path, "m", **gen_kwargs)
    exe, diagnostics = hostrun.build_predict(str(workdir), "m", header, source, cflags)
    assert diagnostics.strip() == "", "compiler diagnostics:\n" + diagnostics
    cls, cls_null, margins = hostrun.run_predict(exe, tr.X, model.n_groups)
    return model, header, source, cls, cls_null, margins


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("name", GBTREE_MODELS)
def test_baseline_reproduces_xgboost_bit_for_bit(name, tmp_path):
    tr = trained(name)
    model, _, _, cls, cls_null, margins = build_and_run(tr, tmp_path)
    assert margins.shape == tr.xgb_margin.shape
    differing = np.nonzero((bits(margins) != bits(tr.xgb_margin)).any(axis=1))[0]
    assert differing.size == 0, f"margins differ from xgboost on rows {differing[:10]}"
    assert np.array_equal(cls, tr.xgb_class)
    assert np.array_equal(cls, cls_null), "predict(x, NULL) must return the same class"


def test_binary_model_is_covered():
    tr = trained(BINARY)
    model = load(tr.model_path)
    assert model.is_binary and tr.xgb_margin.shape[1] == 1
    assert set(np.unique(tr.xgb_class)) == {0, 1}


def test_missing_values_are_covered():
    """The 'missing' model really exercises NaN inputs and both default directions."""
    tr = trained(MISSING)
    model = load(tr.model_path)
    assert np.isnan(tr.X).any(axis=1).sum() > 100
    dl = np.concatenate([t.default_left[t.internal_nodes] for t in model.trees])
    assert dl.any() and not dl.all()
    _, source = float_baseline.generate(tr.model_path, "m")
    flags = [int(m) for m in re.findall(r"\{[^{},]+, (\d+), [1-9]\d*\}", source)]   # split records
    assert len(flags) == model.n_internal
    assert sum(f >= 0x8000 for f in flags) == int(dl.sum())


@pytest.mark.parametrize("name", [BINARY, MISSING])
def test_baseline_is_clean_under_strict_warnings_and_optimisation(name, tmp_path):
    tr = trained(name)
    _, _, _, cls, _, margins = build_and_run(tr, tmp_path, cflags=STRICT_CFLAGS)
    assert np.array_equal(bits(margins), bits(tr.xgb_margin))
    assert np.array_equal(cls, tr.xgb_class)


def test_baseline_c_api_and_record_layout():
    tr = trained(MISSING)
    model = load(tr.model_path)
    header, source = float_baseline.generate(tr.model_path, "my_model")
    assert f"#define MY_MODEL_NUM_FEATURES {model.n_features}\n" in header
    assert f"#define MY_MODEL_NUM_CLASSES {model.n_classes}\n" in header
    assert f"#define MY_MODEL_NUM_MARGINS {model.n_groups}\n" in header
    assert "int my_model_predict(const float *features, float *margins);" in header
    # one 8-byte record per XGBoost node: float + two uint16_t
    n_nodes = model.n_internal + model.n_leaves
    assert f"static const my_model_node_t my_model_nodes[{n_nodes}]" in source
    struct = source[source.index("typedef struct"):source.index("} my_model_node_t;")]
    assert re.findall(r"^\s*(\w+) \w+;", struct, flags=re.M) == ["float", "uint16_t", "uint16_t"]
    assert "sizeof(my_model_node_t) == 8" in source
    assert "malloc" not in source and "static float" not in source

    hb, _ = float_baseline.generate(trained(BINARY).model_path, "b")
    assert "#define B_NUM_CLASSES 2\n" in hb and "#define B_NUM_MARGINS 1\n" in hb


def test_wide_records_give_the_same_answers(tmp_path):
    """The 12-byte layout (for models with > 32768 features or huge trees) is the same interpreter."""
    tr = trained(MISSING)
    _, header, source, cls, _, margins = build_and_run(tr, tmp_path, force_wide=True)
    assert "sizeof(m_node_t) == 12" in source and "32-bit fields" in header
    assert np.array_equal(bits(margins), bits(tr.xgb_margin))
    assert np.array_equal(cls, tr.xgb_class)


def test_dart_weights_are_folded_into_the_leaves(tmp_path):
    """DART is the documented exception to bit-identical margins: same classes, margins within rounding."""
    tr = trained("bin_dart_exact")
    model, header, _, cls, _, margins = build_and_run(tr, tmp_path)
    assert model.booster == "dart" and "DART" in header
    assert np.array_equal(cls, tr.xgb_class)
    assert np.abs(margins - tr.xgb_margin).max() < 1e-5


def test_name_must_be_a_c_identifier():
    with pytest.raises(ValueError):
        float_baseline.generate(trained(BINARY).model_path, "not a name")


def test_cli_writes_header_and_source(tmp_path):
    tr = trained(BINARY)
    r = subprocess.run([sys.executable, "-m", "bench.float_baseline", tr.model_path, "--name", "cli_model",
                        "--out-dir", str(tmp_path)], capture_output=True, text=True, cwd=zoo.ROOT)
    assert r.returncode == 0, r.stderr
    header, source = float_baseline.generate(tr.model_path, "cli_model")
    assert (tmp_path / "cli_model.h").read_text() == header
    assert (tmp_path / "cli_model.c").read_text() == source


# --------------------------------------------------------------------------- Cortex-M4 (compile only)

needs_cross = pytest.mark.skipif(find_cross_compiler() is None, reason="no arm-none-eabi-gcc")


@needs_cross
@pytest.mark.parametrize("name", [BINARY, MISSING])
def test_baseline_on_cortex_m4_has_no_ram_and_no_library_calls(name, tmp_path):
    from bench import sizes
    tr = trained(name)
    model = load(tr.model_path)
    src = float_baseline.write(tr.model_path, "m", str(tmp_path))[1]
    obj = str(tmp_path / "m.o")
    cc = find_cross_compiler()
    r = subprocess.run([cc] + sizes.TARGET_CFLAGS + ["-std=c99", "-Wall", "-Wextra", "-Werror", "-c", src, "-o", obj],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr.strip() == "", r.stderr
    sz = sizes.read_sizes(cc[:-3] + "size", obj)
    assert sz["data"] == 0 and sz["bss"] == 0 and sz["other"] == {}
    documented = int(re.search(r"total (\d+) bytes of \.rodata", open(src).read()).group(1))
    assert sz["rodata"] == documented
    assert sz["rodata"] >= 8 * (model.n_internal + model.n_leaves)
    assert 0 < sz["text"] < 400
    assert sizes.undefined_symbols(cc[:-3] + "nm", [obj]) == []     # hardware float only, no helpers

    # The node table the cross compiler produced is the model, byte for byte: 8-byte little-endian
    # records {float32 value, uint16 feature | default-left << 15, uint16 right offset}.
    value, feat, dleft, right, tree_len, _, _ = float_baseline._layout(model)
    expected = np.zeros(value.size, dtype=[("value", "<f4"), ("feature_and_flags", "<u2"), ("right", "<u2")])
    expected["value"] = value
    expected["feature_and_flags"] = feat | (dleft.astype(np.int64) << 15)
    expected["right"] = right
    assert expected.itemsize == 8 and value.size == model.n_internal + model.n_leaves
    dump = str(tmp_path / "nodes.bin")
    subprocess.run([cc[:-3] + "objcopy", "-O", "binary", "-j", ".rodata.m_nodes", obj, dump], check=True)
    assert open(dump, "rb").read() == expected.tobytes()
    assert int(tree_len.sum()) == value.size and (right[np.cumsum(tree_len) - 1] == 0).all()


@needs_cross
def test_sizes_cli_measures_every_variant(tmp_path):
    pytest.importorskip("m2cgen")
    pytest.importorskip("tl2cgen")
    out = tmp_path / "sizes.json"
    r = subprocess.run([sys.executable, "-m", "bench.sizes", BINARY, "--build-dir", str(tmp_path)],
                       capture_output=True, text=True, cwd=zoo.ROOT)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / BINARY / "float_baseline" / "float_model.o").exists()
    assert "software floating point" in r.stdout and "not included" in r.stdout.lower()
    rows = {row["variant"]: row for row in json.loads(out.read_text())["results"]}
    assert list(rows) == ["xgb_edge", "float_baseline", "m2cgen", "tl2cgen", "tl2cgen_quantize"]
    for row in rows.values():
        assert row["status"] == "ok", row["status"]
        assert row["flash_total"] == row["text"] + row["rodata"] + row["data"] > 0
        assert row["data_bss"] == row["data"] + row["bss"]
    assert rows["float_baseline"]["undefined"] == [] and rows["xgb_edge"]["undefined"] == []
    assert "__aeabi_dadd" in rows["m2cgen"]["undefined"]            # double arithmetic in software


def test_compile_watchdog_reports_timeout_memory_and_failure(tmp_path):
    """A compile that hangs, eats memory or fails must come back as a result, not as an exception."""
    from bench import sizes
    log = str(tmp_path / "log")
    hog = "import time; x = bytearray(400 * 2**20); x[::4096] = b'x' * len(x[::4096]); time.sleep(60)"
    r = sizes.run_limited([sys.executable, "-c", "import time; time.sleep(60)"], log, timeout=1.5, max_mem=None)
    assert r["outcome"] == "timeout" and r["seconds"] < 10
    r = sizes.run_limited(["/bin/sh", "-c", f'{sys.executable} -c "{hog}"; true'], log, timeout=30, max_mem=150 * 2 ** 20)
    assert r["outcome"] == "memory" and r["seconds"] < 20 and r["peak_mem_bytes"] > 150 * 2 ** 20
    r = sizes.run_limited([sys.executable, "-c", "import sys; sys.exit('x.c:1:1: error: nope')"], log, 30, None)
    assert r["outcome"] == "failed"
    assert sizes.summarize_compiler_log(open(log).read()) == "error: nope"
    assert sizes.run_limited([sys.executable, "-c", "pass"], log, 30, None)["outcome"] == "ok"
    many = "/tmp/cc1.s:10: Error: conditional branch out of range\n" * 3
    assert sizes.summarize_compiler_log(many) == "Error: conditional branch out of range (3 such messages)"


# --------------------------------------------------------------------------- m2cgen / TL2cgen

def test_tl2cgen_code_predicts_xgboost_classes(tmp_path):
    pytest.importorskip("tl2cgen")
    pytest.importorskip("treelite")
    from bench import altgen
    tr = trained(MISSING)
    gen = altgen.generate("tl2cgen", tr.model_path, str(tmp_path / "tl2cgen"))
    assert [os.path.basename(s) for s in gen["sources"]] == ["main.c"]
    res = altgen.verify_host(gen, tr)
    assert res["rows"] == tr.X.shape[0] and res["mismatches"] == 0, res
    assert res["max_abs_margin_diff"] < 1e-4 and res["rows_margin_off_by_more_than_1e-3"] == 0

    # quantize=1 writes its threshold table with 8 significant digits, which does not identify
    # every float32, so rows on or next to a threshold may take the other branch.  Ordinary rows
    # must still agree.
    gen = altgen.generate("tl2cgen_quantize", tr.model_path, str(tmp_path / "tl2cgen_quantize"))
    assert sorted(os.path.basename(s) for s in gen["sources"]) == ["main.c", "quantize.c"]
    res = altgen.verify_host(gen, tr)
    assert res["mismatches_heldout"] == 0, res
    assert res["mismatches"] <= 0.01 * res["rows"], res


def test_m2cgen_code_predicts_xgboost_classes_away_from_thresholds(tmp_path):
    pytest.importorskip("m2cgen")
    from bench import altgen
    for name in (BINARY, "mc3_d3_r10_hist"):
        tr = trained(name)
        gen = altgen.generate("m2cgen", tr.model_path, str(tmp_path / name))
        res = altgen.verify_host(gen, tr)
        # m2cgen prints float32 thresholds as short decimals and compares in double, so rows that
        # sit exactly on a threshold may go the other way; ordinary (held-out) rows must agree.
        assert res["mismatches_heldout"] == 0, res
        assert res["mismatches_rows_with_nan"] == 0, res
        assert res["mismatches"] < 0.05 * res["rows"], res


def test_m2cgen_refuses_dart(tmp_path):
    pytest.importorskip("m2cgen")
    from bench import altgen
    with pytest.raises(altgen.AltGenError, match="DART"):
        altgen.generate("m2cgen", trained("bin_dart_exact").model_path, str(tmp_path))
