"""Platform packages (xgb_edge.package): layout, contents, and the examples compiled and run on the host.

The real platform builds (arduino-cli, west, idf.py) are done by
``python -m harness.build_packages``; these tests are the fast part.  The
example programs are compiled here against tiny stand-ins for the platform
headers (Arduino.h, zephyr/kernel.h, freertos/task.h), with warnings as errors,
and their output is compared with the exporter's own integer simulation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from xgb_edge import ExportOptions, XgbEdgeError, export_model
from xgb_edge.analysis import evaluate
from xgb_edge.cli import main as cli_main
from xgb_edge.package import (KINDS, camel_case, f32_literal, int_literal, kconfig_symbol, make_sample,
                              package_files, write_package)
from xgb_edge.report import find_cross_compiler

CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
needs_cc = pytest.mark.skipif(not (CC and CXX), reason="no host C/C++ compiler")

C_WARN = ["-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow", "-Werror"]

EXPECTED_FILES = {
    "arduino": ["library.properties", "keywords.txt", "src/{n}.h", "src/{n}.c",
                "examples/{N}Predict/{N}Predict.ino"],
    "zephyr": ["zephyr/module.yml", "CMakeLists.txt", "Kconfig", "include/{n}.h", "src/{n}.c",
               "samples/predict/CMakeLists.txt", "samples/predict/prj.conf", "samples/predict/src/main.c"],
    "esp-idf": ["CMakeLists.txt", "idf_component.yml", "include/{n}.h", "{n}.c",
                "examples/predict/CMakeLists.txt", "examples/predict/sdkconfig.defaults",
                "examples/predict/main/CMakeLists.txt", "examples/predict/main/predict_main.c"],
}
EXAMPLE = {"arduino": "examples/{N}Predict/{N}Predict.ino", "zephyr": "samples/predict/src/main.c",
           "esp-idf": "examples/predict/main/predict_main.c"}
HEADER = {"arduino": "src/{n}.h", "zephyr": "include/{n}.h", "esp-idf": "include/{n}.h"}
SOURCE = {"arduino": "src/{n}.c", "zephyr": "src/{n}.c", "esp-idf": "{n}.c"}


def rel(template: str, name: str) -> str:
    return template.format(n=name, N=camel_case(name))


# --------------------------------------------------------------------------- models

def _train(tmp, fname, X, y, params, rounds):
    import xgboost as xgb
    path = os.path.join(tmp, fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = xgb.train(dict(params, verbosity=0, nthread=1, seed=0), xgb.DMatrix(X, label=y), num_boost_round=rounds)
    bst.save_model(path)
    return path


@pytest.fixture(scope="session")
def cases(tmp_path_factory):
    """name -> (Export, sample argument or None): float and integer inputs, binary and multiclass."""
    tmp = str(tmp_path_factory.mktemp("models"))
    rng = np.random.RandomState(7)

    Xf = rng.normal(size=(400, 6)).astype(np.float32)
    yf = (Xf[:, 0] + Xf[:, 1] * Xf[:, 2] > 0).astype(int) + (Xf[:, 3] > 0.5).astype(int)
    Xf[rng.rand(*Xf.shape) < 0.1] = np.nan
    Xf[0, 1] = np.nan                       # the sample row has a missing value ...
    Xf[0, 2] = np.float32(-0.0)             # ... and a negative zero
    mc = _train(tmp, "mc.json", Xf, yf, dict(objective="multi:softprob", num_class=3, max_depth=3), 8)

    yb = (Xf[:, 0] > 0).astype(int)
    bn = _train(tmp, "bin.json", Xf, yb, dict(objective="binary:logistic", max_depth=2), 5)

    Xi = rng.randint(-300, 301, size=(400, 5)).astype(np.float32)
    yi = (Xi[:, 0] > 20).astype(int) + (Xi[:, 1] + Xi[:, 2] > 0).astype(int)
    it = _train(tmp, "int.json", Xi, yi, dict(objective="multi:softprob", num_class=3, max_depth=3), 6)

    return {
        "mc_f32": (export_model(mc, ExportOptions(name="demo_model"), validation=Xf), Xf, mc, Xf),
        "bin_zeros": (export_model(bn, ExportOptions(name="m")), None, bn, None),
        "mc_i16": (export_model(it, ExportOptions(name="Gesture_v2", input="i16"), validation=Xi), Xi, it, Xi),
    }


CASES = ["mc_f32", "bin_zeros", "mc_i16"]
ALL = [(c, k) for c in CASES for k in KINDS]


# --------------------------------------------------------------------------- layout and contents

@pytest.mark.parametrize("case,kind", ALL)
def test_layout_and_sources(cases, tmp_path, case, kind):
    exp, sample = cases[case][:2]
    name = exp.generated.name
    out = tmp_path / "pkg"
    paths = write_package(exp, kind, out, sample=sample)

    on_disk = sorted(os.path.relpath(os.path.join(d, f), out).replace(os.sep, "/")
                     for d, _, fs in os.walk(out) for f in fs)
    assert on_disk == sorted(rel(t, name) for t in EXPECTED_FILES[kind])
    assert sorted(os.path.relpath(p, out).replace(os.sep, "/") for p in paths) == on_disk

    # the packaged model is byte for byte what Export.write() produces
    bare = tmp_path / "bare"
    exp.write(bare)
    assert (out / rel(HEADER[kind], name)).read_bytes() == (bare / f"{name}.h").read_bytes()
    assert (out / rel(SOURCE[kind], name)).read_bytes() == (bare / f"{name}.c").read_bytes()

    example = (out / rel(EXAMPLE[kind], name)).read_text()
    assert f"{name}_predict(sample, margins)" in example
    assert f"{name}.h" in example
    assert f"static const {name}_feature_t sample[{name.upper()}_NUM_FEATURES]" in example
    assert f"int32_t margins[{name.upper()}_NUM_MARGINS]" in example
    assert not re.search(r"@[A-Z_]+@", "".join(text for _, text in package_files(exp, kind, sample)))


def test_arduino_metadata(cases, tmp_path):
    exp, sample = cases["mc_f32"][:2]
    write_package(exp, "arduino", tmp_path, sample=sample)
    props = dict(line.split("=", 1) for line in (tmp_path / "library.properties").read_text().splitlines())
    assert props["name"] == "demo_model"
    assert re.fullmatch(r"\d+\.\d+\.\d+", props["version"])
    assert props["includes"] == "demo_model.h"
    assert props["architectures"] == "*"
    assert all("\n" not in v and v == v.strip() for v in props.values())
    # arduino-cli requires the sketch file to be named after its folder
    assert (tmp_path / "examples" / "DemoModelPredict" / "DemoModelPredict.ino").is_file()
    sketch = (tmp_path / "examples" / "DemoModelPredict" / "DemoModelPredict.ino").read_text()
    assert "#include <demo_model.h>" in sketch
    assert "void setup()" in sketch and "void loop()" in sketch
    # TinyUSB cores (Seeed/Adafruit nRF52) need this header for Serial, others must not see it
    assert re.search(r"#if defined\(USE_TINYUSB\)\n#include <Adafruit_TinyUSB.h>[^\n]*\n#endif", sketch)
    assert "demo_model_predict\tKEYWORD2" in (tmp_path / "keywords.txt").read_text()


def test_zephyr_metadata(cases, tmp_path):
    exp, sample = cases["mc_f32"][:2]
    write_package(exp, "zephyr", tmp_path, sample=sample)
    sym = kconfig_symbol("demo_model")
    assert sym == "XGB_EDGE_DEMO_MODEL"
    module = (tmp_path / "zephyr" / "module.yml").read_text()
    assert "name: demo_model\n" in module and "  cmake: .\n" in module and "  kconfig: Kconfig\n" in module
    kconfig = (tmp_path / "Kconfig").read_text()
    assert re.search(rf"^config {sym}\n\tbool ", kconfig, re.M)
    cmake = (tmp_path / "CMakeLists.txt").read_text()
    assert f"if(CONFIG_{sym} " in cmake
    assert "zephyr_library_sources(src/demo_model.c)" in cmake
    assert "zephyr_include_directories(include)" in cmake
    conf = (tmp_path / "samples" / "predict" / "prj.conf").read_text()
    assert f"CONFIG_{sym}=y\n" in conf
    stack = int(re.search(r"^CONFIG_MAIN_STACK_SIZE=(\d+)$", conf, re.M).group(1))
    assert stack >= exp.generated.q_bytes + 1024
    sample_cmake = (tmp_path / "samples" / "predict" / "CMakeLists.txt").read_text()
    assert sample_cmake.index("ZEPHYR_EXTRA_MODULES") < sample_cmake.index("find_package(Zephyr")
    assert "target_sources(app PRIVATE src/main.c)" in sample_cmake
    main_c = (tmp_path / "samples" / "predict" / "src" / "main.c").read_text()
    assert "#include <demo_model.h>" in main_c and "printk(" in main_c


def test_esp_idf_metadata(cases, tmp_path):
    exp, sample = cases["mc_f32"][:2]
    write_package(exp, "esp-idf", tmp_path, sample=sample)
    cmake = (tmp_path / "CMakeLists.txt").read_text()
    assert re.search(r'idf_component_register\(SRCS "demo_model.c"\s+INCLUDE_DIRS "include"\)', cmake)
    proj = (tmp_path / "examples" / "predict" / "CMakeLists.txt").read_text()
    assert proj.index("EXTRA_COMPONENT_DIRS") < proj.index("tools/cmake/project.cmake") < proj.index("project(")
    main_cmake = (tmp_path / "examples" / "predict" / "main" / "CMakeLists.txt").read_text()
    assert 'SRCS "predict_main.c"' in main_cmake and "PRIV_REQUIRES" in main_cmake
    main_c = (tmp_path / "examples" / "predict" / "main" / "predict_main.c").read_text()
    assert '#include "demo_model.h"' in main_c and "void app_main(void)" in main_c
    defaults = (tmp_path / "examples" / "predict" / "sdkconfig.defaults").read_text()
    stack = int(re.search(r"^CONFIG_ESP_MAIN_TASK_STACK_SIZE=(\d+)$", defaults, re.M).group(1))
    assert stack >= max(3584, exp.generated.q_bytes + 2048)
    manifest = (tmp_path / "idf_component.yml").read_text()
    assert re.search(r'^version: "\d+\.\d+\.\d+"$', manifest, re.M)


def test_stack_settings_grow_with_the_rank_array(cases):
    """A model with many feature slots needs more stack than the platform defaults give."""
    import copy
    exp = cases["mc_f32"][0]
    big = copy.copy(exp)
    big.generated = copy.copy(exp.generated)
    big.generated.q_bytes = 6000
    conf = dict(package_files(big, "zephyr"))["samples/predict/prj.conf"]
    assert int(re.search(r"CONFIG_MAIN_STACK_SIZE=(\d+)", conf).group(1)) >= 6000 + 1024
    defaults = dict(package_files(big, "esp-idf"))["examples/predict/sdkconfig.defaults"]
    assert int(re.search(r"CONFIG_ESP_MAIN_TASK_STACK_SIZE=(\d+)", defaults).group(1)) >= 6000 + 2048


# --------------------------------------------------------------------------- the sample row

def test_sample_is_first_validation_row(cases):
    exp, X = cases["mc_f32"][:2]
    smp = make_sample(exp, X)
    assert smp.values.dtype == np.float32 and smp.values.shape == (6,)
    assert np.array_equal(smp.values.view(np.uint32), X[0].view(np.uint32))      # bit for bit, NaN included
    assert "first row of the validation data" in smp.source
    assert smp.expected_class == int(exp.evaluation.int_class[0])
    assert smp.expected_margins == [int(v) for v in exp.evaluation.integer[0]]
    # a single row and a matrix are both accepted
    assert make_sample(exp, X[0]).expected_margins == smp.expected_margins
    main_c = dict(package_files(exp, "zephyr", X))["samples/predict/src/main.c"]
    assert "NAN" in main_c and "#include <math.h>" in main_c and "-0.0f" in main_c


def test_sample_defaults_to_zeros(cases):
    exp = cases["bin_zeros"][0]
    assert exp.evaluation is None
    smp = make_sample(exp, None)
    assert not smp.values.any() and "all zeros" in smp.source
    ev = evaluate(exp.model, [exp.lplan], np.zeros((1, exp.model.n_features), np.float32))[0]
    assert smp.expected_margins == [int(ev.integer[0, 0])] and smp.expected_class == int(ev.int_class[0])
    main_c = dict(package_files(exp, "esp-idf"))["examples/predict/main/predict_main.c"]
    assert "<math.h>" not in main_c and "0.0f" in main_c
    # a matrix without rows also means zeros
    assert not make_sample(exp, np.zeros((0, exp.model.n_features))).values.any()


def test_integer_input_sample(cases):
    exp, X = cases["mc_i16"][:2]
    smp = make_sample(exp, X)
    assert smp.values.dtype == np.int16
    assert smp.values.tolist() == [int(v) for v in X[0]]
    sketch = dict(package_files(exp, "arduino", X))["examples/GestureV2Predict/GestureV2Predict.ino"]
    block = sketch.split("static const Gesture_v2_feature_t sample[GESTURE_V2_NUM_FEATURES] = {")[1].split("};")[0]
    assert [int(t) for t in block.replace("\n", " ").split(",")] == smp.values.tolist()
    assert "NAN" not in sketch and "math.h" not in sketch and "0f" not in block
    with pytest.raises(XgbEdgeError, match="missing"):
        make_sample(exp, np.array([1, 2, np.nan, 4, 5]))
    with pytest.raises(XgbEdgeError, match="non-integers"):
        make_sample(exp, np.array([1, 2, 3.5, 4, 5]))
    with pytest.raises(XgbEdgeError, match="outside the range"):
        make_sample(exp, np.array([1, 2, 40000, 4, 5]))


def test_bad_arguments(cases, tmp_path):
    exp = cases["mc_f32"][0]
    with pytest.raises(XgbEdgeError, match="unknown package kind"):
        write_package(exp, "platformio", tmp_path)
    with pytest.raises(XgbEdgeError, match="expects 6 features"):
        write_package(exp, "arduino", tmp_path, sample=np.zeros(5))
    with pytest.raises(XgbEdgeError, match="one feature row or a matrix"):
        write_package(exp, "arduino", tmp_path, sample=np.zeros((2, 2, 6)))
    assert not any(tmp_path.iterdir())          # nothing is written when the arguments are wrong


def test_names():
    assert camel_case("demo_model") == "DemoModel"
    assert camel_case("xgb_model") == "XgbModel"
    assert camel_case("_m2__x") == "M2X"
    assert camel_case("MyModel") == "MyModel"
    assert camel_case("__") == "Model"
    assert int_literal(-2 ** 31) == "(-2147483647 - 1)"
    assert int_literal(2 ** 31 - 1) == "2147483647" and int_literal(-128) == "-128"
    assert f32_literal(np.nan) == "NAN"
    assert f32_literal(np.inf) == "INFINITY" and f32_literal(-np.inf) == "-INFINITY"
    assert f32_literal(1.0) == "1.0f" and f32_literal(-0.0) == "-0.0f" and f32_literal(0.1) == "0.1f"


@needs_cc
def test_float_literals_are_bit_exact(tmp_path):
    """Every literal must compile back to exactly the float32 it was made from."""
    rng = np.random.RandomState(3)
    special = np.array([0.0, -0.0, 1.0, -1.0, 0.1, 1e-45, -1e-45, 1.17549435e-38, 3.4028235e38, -3.4028235e38,
                        16777216.0, 16777217.0, 1e9, 1e-4, 9.9999e-5, 123456.79, np.inf, -np.inf, np.nan],
                       dtype=np.float32)
    bits = rng.randint(0, 2 ** 32, size=2000, dtype=np.uint64).astype(np.uint32)
    vals = np.concatenate([special, bits.view(np.float32), rng.normal(size=500).astype(np.float32)])
    lits = [f32_literal(v) for v in vals]
    src = tmp_path / "lits.c"
    src.write_text(
        "#include <math.h>\n#include <stdio.h>\n#include <string.h>\n#include <stdint.h>\n"
        "static const float v[] = {\n" + ",\n".join(lits) + "\n};\n"
        "int main(void) { size_t i; for (i = 0; i < sizeof v / sizeof v[0]; i++) {"
        " uint32_t b; memcpy(&b, &v[i], 4); printf(\"%lu\\n\", (unsigned long)b); } return 0; }\n")
    exe = tmp_path / "lits"
    subprocess.run([CC, "-std=c99", "-Wall", "-Wextra", "-Werror", str(src), "-o", str(exe)], check=True)
    got = np.array(subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split(),
                   dtype=np.uint64).astype(np.uint32)
    want = vals.view(np.uint32)
    nan = np.isnan(vals)
    assert np.array_equal(got[~nan], want[~nan])
    assert np.isnan(got[nan].view(np.float32)).all() and nan.sum() > 1


# --------------------------------------------------------------------------- compile and run the examples

ARDUINO_STUB = r"""
#ifndef ARDUINO_STUB_H
#define ARDUINO_STUB_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
struct StubSerial {
  void begin(unsigned long) {}
  explicit operator bool() const { return true; }
  void print(const char *s) { fputs(s, stdout); }
  void print(int v) { printf("%d", v); }
  void print(long v) { printf("%ld", v); }
  void println(const char *s) { puts(s); }
};
extern StubSerial Serial;
unsigned long millis();
void delay(unsigned long ms);
#endif
"""
ARDUINO_MAIN = r"""
#include "Arduino.h"
StubSerial Serial;
unsigned long millis() { return 0; }
void delay(unsigned long) {}
void setup();
void loop();
int main() { setup(); loop(); return 0; }
"""
ZEPHYR_KERNEL_STUB = r"""
#include <stdio.h>
#include <stdlib.h>
#define K_SECONDS(s) (s)
static inline void k_sleep(int seconds) { (void)seconds; exit(0); }
"""
ZEPHYR_PRINTK_STUB = "#include <stdio.h>\n#define printk printf\n"
FREERTOS_STUB = "#define pdMS_TO_TICKS(ms) (ms)\n"
FREERTOS_TASK_STUB = r"""
#include <stdlib.h>
static inline void vTaskDelay(unsigned ticks) { (void)ticks; exit(0); }
"""
IDF_MAIN = "void app_main(void);\nint main(void) { app_main(); return 0; }\n"


def _stubs(tmp_path):
    d = tmp_path / "stubs"
    for path, text in (("Arduino.h", ARDUINO_STUB), ("arduino_main.cpp", ARDUINO_MAIN),
                       ("zephyr/kernel.h", ZEPHYR_KERNEL_STUB), ("zephyr/sys/printk.h", ZEPHYR_PRINTK_STUB),
                       ("freertos/FreeRTOS.h", FREERTOS_STUB), ("freertos/task.h", FREERTOS_TASK_STUB),
                       ("idf_main.c", IDF_MAIN)):
        (d / path).parent.mkdir(parents=True, exist_ok=True)
        (d / path).write_text(text)
    return d


def _compile(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    assert r.returncode == 0 and not r.stderr.strip(), " ".join(str(c) for c in cmd) + "\n" + r.stderr


def _example_commands(kind, pkg, name, stubs, cc, cxx, objdir):
    """(compile commands, objects to link, link driver) for the example of one package."""
    inc = pkg / os.path.dirname(rel(HEADER[kind], name))
    model_o, ex_o, main_o = objdir / "model.o", objdir / "example.o", objdir / "main.o"
    c = [cc, "-std=c99"] + C_WARN + ["-I", inc, "-I", stubs, "-c"]
    cmds = [c + [pkg / rel(SOURCE[kind], name), "-o", model_o]]
    example = pkg / rel(EXAMPLE[kind], name)
    if kind == "arduino":
        cxxc = [cxx, "-std=c++11"] + C_WARN + ["-I", inc, "-I", stubs, "-c"]
        cmds.append(cxxc + ["-x", "c++", "-include", "Arduino.h", example, "-o", ex_o])
        cmds.append(cxxc + [stubs / "arduino_main.cpp", "-o", main_o])
        return cmds, [model_o, ex_o, main_o], cxx
    cmds.append(c + [example, "-o", ex_o])
    if kind == "esp-idf":
        cmds.append(c + [stubs / "idf_main.c", "-o", main_o])
        return cmds, [model_o, ex_o, main_o], cc
    return cmds, [model_o, ex_o], cc


LINE = re.compile(r"^(\w+): class (-?\d+), margins \[([-\d, ]+)\] / 2\^(\d+), self-check (OK|FAILED)$")


@needs_cc
@pytest.mark.parametrize("case,kind", ALL)
def test_example_runs_on_host(cases, tmp_path, case, kind):
    """The example compiles without a warning and prints what the integer simulation predicts."""
    exp, sample = cases[case][:2]
    name = exp.generated.name
    pkg = tmp_path / "pkg"
    write_package(exp, kind, pkg, sample=sample)
    cmds, objs, driver = _example_commands(kind, pkg, name, _stubs(tmp_path), CC, CXX, tmp_path)
    for cmd in cmds:
        _compile(cmd)
    exe = tmp_path / "example"
    _compile([driver] + objs + ["-o", exe])
    out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0
    m = LINE.match(out.stdout.strip())
    assert m, out.stdout

    n = exp.model.n_features
    row = np.zeros((1, n), np.float32) if sample is None else np.asarray(sample)[:1]
    ev = evaluate(exp.model, [exp.lplan], row)[0]
    assert m.group(1) == name
    assert int(m.group(2)) == int(ev.int_class[0])
    assert [int(t) for t in m.group(3).split(",")] == [int(v) for v in ev.integer[0]]
    assert int(m.group(4)) == exp.lplan.shift
    assert m.group(5) == "OK"
    # the same line is quoted in the comment at the top of the example
    assert out.stdout.strip() in (pkg / rel(EXAMPLE[kind], name)).read_text()


@needs_cc
def test_self_check_detects_a_wrong_result(cases, tmp_path):
    """'self-check OK' is not printed unconditionally: a changed expectation makes it FAILED."""
    exp, sample = cases["mc_f32"][:2]
    pkg = tmp_path / "pkg"
    write_package(exp, "esp-idf", pkg, sample=sample)
    main_c = pkg / "examples" / "predict" / "main" / "predict_main.c"
    first = make_sample(exp, sample).expected_margins[0]
    text = main_c.read_text()
    assert text.count(f"\n    {first},") == 1
    main_c.write_text(text.replace(f"\n    {first},", f"\n    {first + 1},"))
    cmds, objs, driver = _example_commands("esp-idf", pkg, "demo_model", _stubs(tmp_path), CC, CXX, tmp_path)
    for cmd in cmds:
        _compile(cmd)
    _compile([driver] + objs + ["-o", tmp_path / "example"])
    out = subprocess.run([str(tmp_path / "example")], capture_output=True, text=True, timeout=30).stdout
    assert out.strip().endswith("self-check FAILED")


CROSS = find_cross_compiler()


@pytest.mark.skipif(not CROSS, reason="no arm-none-eabi-gcc")
@pytest.mark.parametrize("case,kind", ALL)
def test_example_cross_compiles_without_warnings(cases, tmp_path, case, kind):
    """On a 32-bit ARM target int32_t is long, which is where a wrong printf format would show."""
    exp, sample = cases[case][:2]
    pkg = tmp_path / "pkg"
    write_package(exp, kind, pkg, sample=sample)
    cc = [CROSS, "-mcpu=cortex-m4", "-mthumb"]
    cxx = [re.sub(r"gcc(\.exe)?$", "g++", CROSS), "-mcpu=cortex-m4", "-mthumb"]
    cmds, _, _ = _example_commands(kind, pkg, exp.generated.name, _stubs(tmp_path), "CC", "CXX", tmp_path)
    for cmd in cmds:
        _compile((cc if cmd[0] == "CC" else cxx) + cmd[1:])


# --------------------------------------------------------------------------- command line

@pytest.mark.parametrize("kind", KINDS)
def test_cli_package(cases, tmp_path, kind, capsys):
    _, _, model_path, X = cases["mc_f32"]
    val = tmp_path / "val.npy"
    np.save(val, X)
    out = tmp_path / "out" / "mymodel"
    rc = cli_main(["export", model_path, "-o", str(out), "--name", "mymodel", "--package", kind,
                   "--validation", str(val), "--no-measure"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"wrote {kind} package" in printed
    on_disk = sorted(os.path.relpath(os.path.join(d, f), out).replace(os.sep, "/")
                     for d, _, fs in os.walk(out) for f in fs)
    assert on_disk == sorted(rel(t, "mymodel") for t in EXPECTED_FILES[kind])     # and no bare .h/.c besides
    exp = export_model(model_path, ExportOptions(name="mymodel"), validation=X)
    assert dict(package_files(exp, kind, X))[rel(EXAMPLE[kind], "mymodel")] == \
        (out / rel(EXAMPLE[kind], "mymodel")).read_text()


def test_cli_without_package_is_unchanged(cases, tmp_path, capsys):
    model_path = cases["bin_zeros"][2]
    rc = cli_main(["export", model_path, "-o", str(tmp_path), "--name", "m", "--no-measure"])
    assert rc == 0
    assert sorted(os.listdir(tmp_path)) == ["m.c", "m.h"]
    assert "wrote " + os.path.join(str(tmp_path), "m.h") in capsys.readouterr().out


def test_cli_report_has_no_package_option(cases):
    with pytest.raises(SystemExit):
        cli_main(["report", cases["bin_zeros"][2], "--package", "arduino"])
