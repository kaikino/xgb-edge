"""Platform packages: an Export wrapped as an Arduino library, a Zephyr module or an ESP-IDF component.

``write_package(export, kind, outdir, sample=...)`` writes a ready-to-use
package into ``outdir``.  The model header and source in it are byte for byte
what ``Export.write`` produces; the package adds the platform's build files and
one example program.  The example reads nothing from hardware: it runs the
model on a built-in input row, prints the class and the margins, and compares
them with what xgb-edge computed for that row at export time.

Layouts (``<name>`` is the model name, ``<Name>`` its CamelCase form)::

    arduino   library.properties  keywords.txt  src/<name>.h  src/<name>.c
              examples/<Name>Predict/<Name>Predict.ino
    zephyr    zephyr/module.yml  CMakeLists.txt  Kconfig  include/<name>.h  src/<name>.c
              samples/predict/{CMakeLists.txt, prj.conf, src/main.c}
    esp-idf   CMakeLists.txt  idf_component.yml  include/<name>.h  <name>.c
              examples/predict/{CMakeLists.txt, sdkconfig.defaults, main/CMakeLists.txt, main/predict_main.c}
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import __version__
from .analysis import evaluate
from .export import Export, to_device_inputs
from .model import XgbEdgeError
from .quantize import INPUT_KINDS

KINDS = ("arduino", "zephyr", "esp-idf")

#: Version written into library.properties / idf_component.yml (both require a semantic version).
PACKAGE_VERSION = "1.0.0"


# --------------------------------------------------------------------------- the built-in sample

@dataclass
class Sample:
    """The input row built into the examples, and what the generated C returns for it."""

    values: np.ndarray            # 1-D, already in the C feature type (float32 or an integer type)
    source: str                   # where the row came from, for the comment in the example
    expected_class: int
    expected_margins: List[int]   # fixed point, one per margin


def make_sample(export: Export, sample=None) -> Sample:
    """Turn ``sample`` (a feature row, a matrix whose first row is used, or None for zeros) into a Sample.

    The row is in the model's feature units, exactly like validation data: float
    values with NaN for "missing", or the integers an integer-input export takes.
    """
    n = export.model.n_features
    row = None
    if sample is not None:
        arr = np.asarray(sample)
        if arr.ndim == 2:
            row = arr[0] if arr.shape[0] else None
        elif arr.ndim == 1:
            row = arr
        else:
            raise XgbEdgeError(f"sample must be one feature row or a matrix of rows, got shape {arr.shape}")
    if row is None:
        row = np.zeros(n, dtype=np.float32)
        if sample is None:
            source = "all zeros, because no sample row was given at export"
        else:
            source = "all zeros, because the data given at export had no rows"
    elif export.evaluation is not None:
        source = "the first row of the validation data given at export"
    else:
        source = "the sample row given at export"
    if row.shape[0] != n:
        raise XgbEdgeError(f"sample has {row.shape[0]} values but the model expects {n} features")
    values = to_device_inputs(row, n, export.options.input)[0]   # also checks range and integrality
    ev = evaluate(export.model, [export.lplan], row.reshape(1, n))[0]
    return Sample(values=values, source=source, expected_class=int(ev.int_class[0]),
                  expected_margins=[int(m) for m in ev.integer[0]])


def f32_literal(v) -> str:
    """A C expression of type float whose value is exactly the float32 ``v`` (NaN -> ``NAN``)."""
    v = np.float32(v)
    if np.isnan(v):
        return "NAN"
    if np.isinf(v):
        return "INFINITY" if v > 0 else "-INFINITY"
    a = abs(float(v))
    # shortest decimal that reads back as the same float32
    if a == 0.0 or 1e-4 <= a < 1e9:
        s = np.format_float_positional(v, unique=True, trim="0")
    else:
        s = np.format_float_scientific(v, unique=True, trim="0")
    return s + "f"


def int_literal(v) -> str:
    """A C integer constant expression for ``v`` that has type int or wider without a suffix."""
    v = int(v)
    if v == -(2 ** 31):
        return "(-2147483647 - 1)"   # the plain literal would be "minus (a constant too big for int32)"
    return str(v)


def sample_literals(values: np.ndarray) -> List[str]:
    """C initializer for each element of a device-typed sample row."""
    if np.issubdtype(values.dtype, np.floating):
        return [f32_literal(v) for v in values]
    return [int_literal(v) for v in values]


def _init_list(literals: List[str], indent: str, width: int = 96) -> str:
    """Comma-separated initializers wrapped to ``width`` columns."""
    lines, cur = [], ""
    for i, lit in enumerate(literals):
        piece = lit + ("," if i + 1 < len(literals) else "")
        if cur and len(indent) + len(cur) + 1 + len(piece) > width:
            lines.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}" if cur else piece
    if cur:
        lines.append(cur)
    return "\n".join(indent + ln for ln in lines)


# --------------------------------------------------------------------------- shared text

def camel_case(name: str) -> str:
    """``demo_model`` -> ``DemoModel``: the Arduino sketch naming style."""
    out = "".join(p[:1].upper() + p[1:] for p in name.split("_") if p)
    return (out or "Model")[:48]


def kconfig_symbol(name: str) -> str:
    """The Kconfig symbol (without ``CONFIG_``) that enables the Zephyr module."""
    return "XGB_EDGE_" + name.upper()


def _fill(template: str, values: Dict[str, str]) -> str:
    """Replace every ``@KEY@`` of ``template`` (an unknown key raises KeyError)."""
    return re.sub(r"@([A-Z_]+)@", lambda m: values[m.group(1)], template)


def _values(export: Export, smp: Sample, indent: str) -> Dict[str, str]:
    """Placeholder values shared by every template."""
    model, gen = export.model, export.generated
    name = gen.name
    kind = INPUT_KINDS[export.options.input]
    shift = export.lplan.shift
    margins = ", ".join(str(m) for m in smp.expected_margins)
    lits = sample_literals(smp.values)
    n_feat = model.n_features
    plural = "" if n_feat == 1 else "s"
    if model.is_binary:
        what = (f"margins[0] is the raw score (margin) of class 1 in fixed point:\n"
                f" *   real value = margins[0] / 2^{name.upper()}_MARGIN_SHIFT   (here: / 2^{shift})\n"
                f" * The predicted class is 1 when margins[0] > 0, else 0.")
    else:
        what = (f"margins[k] is the raw score (margin) of class k in fixed point:\n"
                f" *   real value = margins[k] / 2^{name.upper()}_MARGIN_SHIFT   (here: / 2^{shift})\n"
                f" * The predicted class is the index of the largest margin.")
    if kind.is_float:
        feat = "float32, NaN (NAN) = missing value"
    else:
        feat = f"{kind.ctype}; the integers the model was trained on"
    return {
        "NAME": name,
        "UNAME": name.upper(),
        "CAMEL": camel_case(name),
        "KSYM": kconfig_symbol(name),
        "VERSION": __version__,
        "PKGVERSION": PACKAGE_VERSION,
        "OBJECTIVE": model.objective,
        "NCLASSES": str(model.n_classes),
        "FEATURES_WORD": f"{n_feat} feature{plural}",
        "FEATURE_KIND": feat,
        "DATA_BYTES": str(gen.data_bytes),
        "STACK_BYTES": str(predict_stack_bytes(export)),
        "MARGIN_DOC": what,
        "SAMPLE_SOURCE": smp.source,
        # a zero-length array is not valid C: a model without features still gets one (ignored) element
        "SAMPLE_LEN": f"{name.upper()}_NUM_FEATURES" if n_feat else "1",
        "SAMPLE_INIT": _init_list(lits if lits else ["0"], indent),
        "EXPECTED_CLASS": str(smp.expected_class),
        "EXPECTED_MARGINS": _init_list([str(m) for m in smp.expected_margins], indent),
        "EXPECTED_LINE": f"{name}: class {smp.expected_class}, margins [{margins}] / 2^{shift}, self-check OK",
        "MATH_INCLUDE": "#include <math.h>      /* NAN, INFINITY */\n" if _needs_math(smp.values) else "",
    }


def _needs_math(values: np.ndarray) -> bool:
    return bool(np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all())


def predict_stack_bytes(export: Export) -> int:
    """Stack need of one predict() call, rounded up: the rank array plus locals and saved registers.

    The frames measured with -fstack-usage for the zoo models (GCC for Cortex-M4,
    Xtensa and RISC-V, -Os/-Og/-Ofast) are the rank array plus at most 72 bytes.
    """
    return int(export.generated.q_bytes) + 96


def _round_up(n: int, step: int) -> int:
    return (n + step - 1) // step * step


#: The data every example carries: the input row and the result xgb-edge expects for it.
_SAMPLE_DATA = """\
/* The input: one row of @FEATURES_WORD@ (@FEATURE_KIND@).
 * This row is @SAMPLE_SOURCE@. */
static const @NAME@_feature_t sample[@SAMPLE_LEN@] = {
@SAMPLE_INIT@
};

/* What xgb-edge computed for this row at export time. The generated C is
 * integer-only, so every target must reproduce these numbers exactly. */
static const int expected_class = @EXPECTED_CLASS@;
static const int32_t expected_margins[@UNAME@_NUM_MARGINS] = {
@EXPECTED_MARGINS@
};
"""

_EXAMPLE_DOC = """\
 * Runs the XGBoost model '@NAME@' on one built-in input row and prints
 * the predicted class and the margins. Nothing is read from hardware.
 * Model: @OBJECTIVE@, @NCLASSES@ classes, @FEATURES_WORD@. Generated by xgb-edge @VERSION@.
 *
 * It prints this line every two seconds:
 *
 *   @EXPECTED_LINE@
 *
 * @MARGIN_DOC@
 * "self-check OK" means that class and margins equal what xgb-edge computed
 * for this row when the model was exported.
 *
 * One @NAME@_predict() call needs about @STACK_BYTES@ bytes of stack and no
 * other RAM."""


# --------------------------------------------------------------------------- Arduino

_ARDUINO_PROPERTIES = """\
name=@NAME@
version=@PKGVERSION@
author=generated by xgb-edge @VERSION@
maintainer=generated by xgb-edge @VERSION@
sentence=XGBoost classifier '@NAME@' as integer-only C, exported by xgb-edge.
paragraph=@OBJECTIVE@ model with @NCLASSES@ classes and @FEATURES_WORD@. Include @NAME@.h and call @NAME@_predict(); see the @CAMEL@Predict example.
category=Data Processing
url=
architectures=*
includes=@NAME@.h
"""

_ARDUINO_KEYWORDS = """\
@NAME@_feature_t\tKEYWORD1
@NAME@_predict\tKEYWORD2
@UNAME@_NUM_FEATURES\tLITERAL1
@UNAME@_NUM_CLASSES\tLITERAL1
@UNAME@_NUM_MARGINS\tLITERAL1
@UNAME@_MARGIN_SHIFT\tLITERAL1
"""

_ARDUINO_SKETCH = """\
/*
 * @CAMEL@Predict
 *
""" + _EXAMPLE_DOC + """
 *
 * Open the serial monitor at 115200 baud to see the output.
 */
#include <@NAME@.h>
@MATH_INCLUDE@
#if defined(USE_TINYUSB)
#include <Adafruit_TinyUSB.h>  // Serial is a TinyUSB port on this core (Adafruit / Seeed nRF52, ...)
#endif

""" + _SAMPLE_DATA + """
void setup() {
  Serial.begin(115200);
  // Give a native USB serial port up to 3 s to be opened; a UART port is ready at once.
  while (!Serial && millis() < 3000) {
    delay(10);
  }
}

void loop() {
  int32_t margins[@UNAME@_NUM_MARGINS];
  const int predicted = @NAME@_predict(sample, margins);
  bool ok = (predicted == expected_class);

  Serial.print("@NAME@: class ");
  Serial.print(predicted);
  Serial.print(", margins [");
  for (int k = 0; k < @UNAME@_NUM_MARGINS; k++) {
    if (margins[k] != expected_margins[k]) {
      ok = false;
    }
    if (k > 0) {
      Serial.print(", ");
    }
    Serial.print(margins[k]);
  }
  Serial.print("] / 2^");
  Serial.print(@UNAME@_MARGIN_SHIFT);
  Serial.println(ok ? ", self-check OK" : ", self-check FAILED");

  delay(2000);
}
"""


def _arduino_files(export: Export, v: Dict[str, str]) -> List[Tuple[str, str]]:
    name, camel = v["NAME"], v["CAMEL"]
    gen = export.generated
    return [
        ("library.properties", _fill(_ARDUINO_PROPERTIES, v)),
        ("keywords.txt", _fill(_ARDUINO_KEYWORDS, v)),
        (f"src/{name}.h", gen.header),
        (f"src/{name}.c", gen.source),
        (f"examples/{camel}Predict/{camel}Predict.ino", _fill(_ARDUINO_SKETCH, v)),
    ]


# --------------------------------------------------------------------------- Zephyr

_ZEPHYR_MODULE_YML = """\
# Zephyr module for the XGBoost model '@NAME@' (generated by xgb-edge @VERSION@).
name: @NAME@
build:
  cmake: .
  kconfig: Kconfig
"""

_ZEPHYR_KCONFIG = """\
# XGBoost model '@NAME@' (generated by xgb-edge @VERSION@).

config @KSYM@
\tbool "XGBoost model '@NAME@' (xgb-edge, integer-only C)"
\thelp
\t  Build @NAME@_predict(): an XGBoost @OBJECTIVE@ model with
\t  @NCLASSES@ classes and @FEATURES_WORD@, exported by xgb-edge.
\t  Adds @DATA_BYTES@ bytes of read-only model data and a few hundred
\t  bytes of code. Uses no static RAM and no floating-point operations;
\t  one call needs about @STACK_BYTES@ bytes of stack.
\t  Include <@NAME@.h> to use it.
"""

_ZEPHYR_CMAKE = """\
# Zephyr module for the XGBoost model '@NAME@' (generated by xgb-edge @VERSION@).
# Enable it with CONFIG_@KSYM@=y, then #include <@NAME@.h>.

if(CONFIG_@KSYM@ AND NOT TARGET xgb_edge_@NAME@)
  zephyr_include_directories(include)
  zephyr_library_named(xgb_edge_@NAME@)
  zephyr_library_sources(src/@NAME@.c)
endif()
"""

_ZEPHYR_SAMPLE_CMAKE = """\
cmake_minimum_required(VERSION 3.20.0)

# This sample sits inside the module it demonstrates (two directories up) and
# registers that directory as a Zephyr module itself, so a plain
#   west build -b <board> <this directory>
# works. In your own application do the same with the path of your copy of the
# module, or pass -DZEPHYR_EXTRA_MODULES=<path> on the west build command line.
get_filename_component(XGB_EDGE_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(NOT XGB_EDGE_MODULE_DIR IN_LIST ZEPHYR_EXTRA_MODULES)
  list(APPEND ZEPHYR_EXTRA_MODULES "${XGB_EDGE_MODULE_DIR}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(@NAME@_predict)

target_sources(app PRIVATE src/main.c)
"""

_ZEPHYR_PRJ_CONF = """\
# Build the model (the symbol comes from the module's Kconfig).
CONFIG_@KSYM@=y

# One @NAME@_predict() call needs about @STACK_BYTES@ bytes of stack; the rest is for printk.
CONFIG_MAIN_STACK_SIZE=@MAIN_STACK@
"""

_ZEPHYR_MAIN = """\
/*
""" + _EXAMPLE_DOC + """
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
@MATH_INCLUDE@
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <@NAME@.h>

""" + _SAMPLE_DATA + """
int main(void)
{
\tfor (;;) {
\t\tint32_t margins[@UNAME@_NUM_MARGINS];
\t\tconst int predicted = @NAME@_predict(sample, margins);
\t\tbool ok = (predicted == expected_class);
\t\tint k;

\t\tprintk("@NAME@: class %d, margins [", predicted);
\t\tfor (k = 0; k < @UNAME@_NUM_MARGINS; k++) {
\t\t\tif (margins[k] != expected_margins[k]) {
\t\t\t\tok = false;
\t\t\t}
\t\t\tprintk("%s%" PRId32, k > 0 ? ", " : "", margins[k]);
\t\t}
\t\tprintk("] / 2^%d, self-check %s\\n", @UNAME@_MARGIN_SHIFT, ok ? "OK" : "FAILED");

\t\tk_sleep(K_SECONDS(2));
\t}
}
"""


def _zephyr_files(export: Export, v: Dict[str, str]) -> List[Tuple[str, str]]:
    name = v["NAME"]
    gen = export.generated
    v = dict(v, MAIN_STACK=str(max(2048, _round_up(predict_stack_bytes(export) + 1536, 256))))
    return [
        ("zephyr/module.yml", _fill(_ZEPHYR_MODULE_YML, v)),
        ("CMakeLists.txt", _fill(_ZEPHYR_CMAKE, v)),
        ("Kconfig", _fill(_ZEPHYR_KCONFIG, v)),
        (f"include/{name}.h", gen.header),
        (f"src/{name}.c", gen.source),
        ("samples/predict/CMakeLists.txt", _fill(_ZEPHYR_SAMPLE_CMAKE, v)),
        ("samples/predict/prj.conf", _fill(_ZEPHYR_PRJ_CONF, v)),
        ("samples/predict/src/main.c", _fill(_ZEPHYR_MAIN, v)),
    ]


# --------------------------------------------------------------------------- ESP-IDF

_IDF_COMPONENT_CMAKE = """\
# ESP-IDF component for the XGBoost model '@NAME@' (generated by xgb-edge @VERSION@).
# The component is named after the directory this file is in.
idf_component_register(SRCS "@NAME@.c"
                       INCLUDE_DIRS "include")
"""

_IDF_COMPONENT_YML = """\
version: "@PKGVERSION@"
description: "XGBoost classifier '@NAME@' as integer-only C, exported by xgb-edge @VERSION@"
"""

_IDF_EXAMPLE_CMAKE = """\
cmake_minimum_required(VERSION 3.16)

# This example sits inside the component it demonstrates (two directories up).
# In your own project, copy that directory to components/@NAME@/ instead; then
# this line is not needed.
get_filename_component(XGB_EDGE_COMPONENT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(EXTRA_COMPONENT_DIRS "${XGB_EDGE_COMPONENT_DIR}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
# Build only the components this example needs (ESP-IDF 5.5 and later; ignored before).
idf_build_set_property(MINIMAL_BUILD ON)
project(@NAME@_predict)
"""

_IDF_EXAMPLE_MAIN_CMAKE = """\
# The model component is named after its directory, three levels up from here.
# In your own project this is simply: PRIV_REQUIRES @NAME@
get_filename_component(xgb_edge_component_dir "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
get_filename_component(xgb_edge_component "${xgb_edge_component_dir}" NAME)

idf_component_register(SRCS "predict_main.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES ${xgb_edge_component})
"""

_IDF_SDKCONFIG_DEFAULTS = """\
# One @NAME@_predict() call needs about @STACK_BYTES@ bytes of stack; the rest is for printf.
CONFIG_ESP_MAIN_TASK_STACK_SIZE=@MAIN_STACK@
"""

_IDF_MAIN = """\
/*
""" + _EXAMPLE_DOC + """
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
@MATH_INCLUDE@
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "@NAME@.h"

""" + _SAMPLE_DATA + """
void app_main(void)
{
    for (;;) {
        int32_t margins[@UNAME@_NUM_MARGINS];
        const int predicted = @NAME@_predict(sample, margins);
        bool ok = (predicted == expected_class);
        int k;

        printf("@NAME@: class %d, margins [", predicted);
        for (k = 0; k < @UNAME@_NUM_MARGINS; k++) {
            if (margins[k] != expected_margins[k]) {
                ok = false;
            }
            printf("%s%" PRId32, k > 0 ? ", " : "", margins[k]);
        }
        printf("] / 2^%d, self-check %s\\n", @UNAME@_MARGIN_SHIFT, ok ? "OK" : "FAILED");

        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
"""


def _esp_idf_files(export: Export, v: Dict[str, str]) -> List[Tuple[str, str]]:
    name = v["NAME"]
    gen = export.generated
    # 3584 is ESP-IDF's default main task stack
    v = dict(v, MAIN_STACK=str(max(3584, _round_up(predict_stack_bytes(export) + 3072, 512))))
    return [
        ("CMakeLists.txt", _fill(_IDF_COMPONENT_CMAKE, v)),
        ("idf_component.yml", _fill(_IDF_COMPONENT_YML, v)),
        (f"include/{name}.h", gen.header),
        (f"{name}.c", gen.source),
        ("examples/predict/CMakeLists.txt", _fill(_IDF_EXAMPLE_CMAKE, v)),
        ("examples/predict/sdkconfig.defaults", _fill(_IDF_SDKCONFIG_DEFAULTS, v)),
        ("examples/predict/main/CMakeLists.txt", _fill(_IDF_EXAMPLE_MAIN_CMAKE, v)),
        ("examples/predict/main/predict_main.c", _fill(_IDF_MAIN, v)),
    ]


# --------------------------------------------------------------------------- entry point

_BUILDERS = {"arduino": (_arduino_files, "  "), "zephyr": (_zephyr_files, "\t"), "esp-idf": (_esp_idf_files, "    ")}


def package_files(export: Export, kind: str, sample=None) -> List[Tuple[str, str]]:
    """The package as ``(relative path with forward slashes, text)`` pairs, without writing anything."""
    if kind not in _BUILDERS:
        raise XgbEdgeError(f"unknown package kind {kind!r}: choose one of {', '.join(KINDS)}")
    build, indent = _BUILDERS[kind]
    smp = make_sample(export, sample)
    return build(export, _values(export, smp, indent))


def write_package(export: Export, kind: str, outdir, sample=None) -> List[str]:
    """Write ``export`` as a package for ``kind`` ('arduino', 'zephyr' or 'esp-idf') into ``outdir``.

    ``outdir`` becomes the package itself: the Arduino library folder, the
    Zephyr module or the ESP-IDF component (which ESP-IDF names after the
    folder, so call the folder after the model).  ``sample`` is the input row
    built into the example: a feature row, or a matrix whose first row is
    used (pass the validation data), in the same units as validation data.
    Without it the example runs on zeros.  Returns the paths written.
    """
    files = package_files(export, kind, sample)
    paths: List[str] = []
    for rel, text in files:
        path = os.path.join(os.fspath(outdir), *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        paths.append(path)
    return paths
