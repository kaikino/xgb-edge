# Using an exported model on Arduino, Zephyr / nRF Connect SDK and ESP-IDF

xgb-edge turns a trained XGBoost classifier (`model.json`) into two small C
files, `<name>.h` and `<name>.c`, that run on a microcontroller with integer
arithmetic only. `--package` wraps those two files as a ready-to-use Arduino
library, Zephyr module or ESP-IDF component, each with one example program.

This page shows how to make each package, where to put it, and the few lines
of code that call the model. The last section says exactly what was built to
check these instructions, and with which tool versions.

## 1. Export a package

```sh
xgb-edge export model.json --name mymodel --validation X.npy \
    --package arduino -o mymodel
```

* `--package` is `arduino`, `zephyr` or `esp-idf`. The output directory (`-o`)
  becomes the package itself, so name it after the model. With `--package`
  the bare `mymodel.h` / `mymodel.c` are **not** also written to the top of
  the output directory; they are inside the package (same bytes).
* `--name` is the prefix of every C identifier and file (`mymodel_predict`,
  `mymodel.h`). It must be a valid C identifier.
* `--validation` is a feature matrix (`.npy`, `.npz` or `.csv`, one row per
  sample, NaN or an empty cell for a missing value). It is optional but
  recommended: xgb-edge uses it to pick the leaf width and to report rows that
  are close to a tie, and its **first row becomes the built-in input of the
  example program**. Without it the example runs on a row of zeros.
* If `xgb-edge` is not on your PATH, use `python -m xgb_edge export ...`.

From Python the same thing is:

```python
from xgb_edge import ExportOptions, export_model
from xgb_edge.package import write_package

exp = export_model("model.json", ExportOptions(name="mymodel"), validation=X)
write_package(exp, "arduino", "mymodel", sample=X)   # sample: a row, or a matrix whose first row is used
```

## 2. The API (the same on every platform)

```c
#include "mymodel.h"

mymodel_feature_t features[MYMODEL_NUM_FEATURES];   /* float by default; fill in feature order */
int32_t margins[MYMODEL_NUM_MARGINS];

int cls = mymodel_predict(features, margins);        /* margins may be NULL */
```

* `features[i]` is feature `i` exactly as it was given to XGBoost for
  training, as a `float`. `NAN` (from `<math.h>`) means "missing". If the
  model was exported with `--input i8|u8|i16|u16|i32`, `mymodel_feature_t` is
  that integer type instead and there is no missing value.
* The return value is the predicted class, `0 .. MYMODEL_NUM_CLASSES-1`.
* `margins[k]` is the model's raw score for class `k` (what XGBoost returns
  with `output_margin=True`, up to the leaf rounding error stated in the
  export report) in fixed point:

  ```
  real margin = margins[k] / 2^MYMODEL_MARGIN_SHIFT
  ```

  in C: `ldexpf((float)margins[k], -MYMODEL_MARGIN_SHIFT)`. The class is the
  index of the largest margin (the lowest index on a tie). A binary model has
  `MYMODEL_NUM_MARGINS == 1`: `margins[0]` is the score of class 1, and the
  class is 1 when it is greater than 0. If you need probabilities, apply
  softmax (multi-class) or the logistic function (`binary:logistic`) to the
  real margins yourself; `mymodel_predict` itself never uses floating point.
* The model is `const` data plus one small function: no heap, no static RAM,
  reentrant. One call needs a little stack (a rank array of one, two or four bytes
  per feature slot plus locals); the exact numbers are in the report that
  `xgb-edge export` prints, and each example says how much its model needs.

Every example program prints one line every two seconds, for instance

```
mymodel: class 1, margins [-110264, 33593, -44637, -12703, -4387] / 2^15, self-check OK
```

`self-check OK` means the class and the margins computed on the device are
identical to what xgb-edge computed for the same row at export time (the
expected values are compiled into the example).

## 3. Arduino

1. Export with `--package arduino` straight into your sketchbook's
   `libraries` folder (File > Preferences shows the sketchbook location; the
   default is `~/Documents/Arduino` on macOS and Windows, `~/Arduino` on
   Linux):

   ```sh
   xgb-edge export model.json --name mymodel --validation X.npy \
       --package arduino -o ~/Documents/Arduino/libraries/mymodel
   ```

   The folder holds `library.properties`, `keywords.txt`, `src/mymodel.h`,
   `src/mymodel.c` and `examples/MymodelPredict/MymodelPredict.ino`.
2. Restart the Arduino IDE, open File > Examples > mymodel > MymodelPredict,
   upload, and open the serial monitor at 115200 baud. With arduino-cli the
   library does not have to be installed:

   ```sh
   arduino-cli compile --fqbn <your board> --library path/to/mymodel \
       path/to/mymodel/examples/MymodelPredict
   ```

3. In your own sketch:

   ```cpp
   #include <mymodel.h>

   void loop() {
     mymodel_feature_t x[MYMODEL_NUM_FEATURES] = { /* your features */ };
     int32_t margins[MYMODEL_NUM_MARGINS];
     int cls = mymodel_predict(x, margins);
     Serial.println(cls);
   }
   ```

Notes

* Boards whose `Serial` is a TinyUSB port (Seeed and Adafruit nRF52 cores,
  among others) need `#include <Adafruit_TinyUSB.h>` in the sketch, or
  `Serial` does not link. The example includes it under
  `#if defined(USE_TINYUSB)`; do the same in your sketch.
* The Seeed nRF52 core runs a program called `python` after linking. If
  `arduino-cli compile` fails at the very end with a Python error, put a
  working `python` first on PATH.
* The Seeed nRF52 core compiles with `-Ofast`. Add
  `--build-property compiler.optimization_flag=-Os` for a smaller program.

## 4. Zephyr and nRF Connect SDK

1. Export with `--package zephyr`, for example into your application:

   ```sh
   xgb-edge export model.json --name mymodel --validation X.npy \
       --package zephyr -o my_app/modules/mymodel
   ```

   The folder is a Zephyr module: `zephyr/module.yml`, `CMakeLists.txt`,
   `Kconfig`, `include/mymodel.h`, `src/mymodel.c`, and a sample application
   in `samples/predict/`.
2. Tell the build about the module. Zephyr does not scan an application's
   `modules/` folder by itself, so add one line to the application's
   `CMakeLists.txt` **before** `find_package(Zephyr ...)`:

   ```cmake
   list(APPEND ZEPHYR_EXTRA_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/modules/mymodel)
   find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
   ```

   or leave `CMakeLists.txt` alone and pass the absolute path when building:
   `west build -b <board> my_app -- -DZEPHYR_EXTRA_MODULES=/abs/path/to/mymodel`.
   (Zephyr has documented this variable as `EXTRA_ZEPHYR_MODULES` since 3.4;
   Zephyr 4.5.0 accepts both spellings, and the sample uses the older one.)
3. Enable the model in `prj.conf`. The symbol is `XGB_EDGE_` followed by the
   model name in upper case:

   ```
   CONFIG_XGB_EDGE_MYMODEL=y
   ```

4. Call it:

   ```c
   #include <mymodel.h>

   mymodel_feature_t x[MYMODEL_NUM_FEATURES] = { /* your features */ };
   int32_t margins[MYMODEL_NUM_MARGINS];
   int cls = mymodel_predict(x, margins);
   ```

   Make sure the calling thread has the stack the model needs (the sample's
   `prj.conf` sets `CONFIG_MAIN_STACK_SIZE` accordingly).

To build the sample that ships in the module (it registers the module two
directories above it by itself):

```sh
west build -b xiao_ble/nrf52840/sense path/to/mymodel/samples/predict
```

nRF Connect SDK uses the same module mechanism, so the steps are the same
there. That has **not** been built with nRF Connect SDK itself, only with
upstream Zephyr (see section 6).

## 5. ESP-IDF

1. Export with `--package esp-idf` into your project's `components` folder.
   ESP-IDF names a component after its folder, so call the folder after the
   model:

   ```sh
   xgb-edge export model.json --name mymodel --validation X.npy \
       --package esp-idf -o my_project/components/mymodel
   ```

   The folder holds `CMakeLists.txt` (`idf_component_register`),
   `idf_component.yml`, `include/mymodel.h`, `mymodel.c`, and an example
   project in `examples/predict/`.
2. Let the component that calls the model depend on it, for example in
   `main/CMakeLists.txt`:

   ```cmake
   idf_component_register(SRCS "main.c"
                          INCLUDE_DIRS "."
                          PRIV_REQUIRES mymodel)
   ```

3. Call it:

   ```c
   #include "mymodel.h"

   mymodel_feature_t x[MYMODEL_NUM_FEATURES] = { /* your features */ };
   int32_t margins[MYMODEL_NUM_MARGINS];
   int cls = mymodel_predict(x, margins);
   ```

To build the example that ships in the component (it adds the component two
directories above it through `EXTRA_COMPONENT_DIRS`):

```sh
cd path/to/mymodel/examples/predict
idf.py set-target esp32
idf.py build
```

## 6. What was actually built

Everything below was produced by `python -m harness.build_packages` on
2026-11-19 (macOS, Darwin 25.6.0, Apple Silicon); the numbers are copied from
its `build/packages/results.json`. The model is a 5-class, 16-feature
XGBoost model with 250 trees (zoo model `mc5_d4_r50_hist_missing`, exported
as `demo_model`, 22,040 bytes of model data), and each build is the example
that ships in the package.

**Every result in this table is "compiled, not run".** The one exception is
recorded separately below it: the Arduino example was also uploaded to a real
XIAO nRF52840 and its output read back.

| Platform | Target | Built with | Result (compiled, not run) |
|---|---|---|---|
| Arduino | Seeed XIAO nRF52840 Sense Plus, `Seeeduino:nrf52:xiaonRF52840SensePlus` | arduino-cli 1.5.1, core Seeeduino:nrf52 1.1.13, arm-none-eabi-gcc 9.2.1 | program 65,612 B with the core's default `-Ofast`, 58,720 B with `-Os`; global variables 7,144 B |
| Arduino | ESP32 Dev Module, `esp32:esp32:esp32` | arduino-cli 1.5.1, core esp32:esp32 3.3.7, xtensa-esp-elf-gcc 14.2.0 | program 302,284 B; global variables 22,068 B |
| Arduino | ESP32-C3 Dev Module (RISC-V), `esp32:esp32:esp32c3` | arduino-cli 1.5.1, core esp32:esp32 3.3.7, riscv32-esp-elf-gcc 14.2.0 | program 296,102 B; global variables 13,152 B |
| ESP-IDF | `esp32` | ESP-IDF v6.1.1, xtensa-esp-elf-gcc 15.2.0 | application binary 148,064 B (ESP-IDF's default `-Og`) |
| Zephyr | `xiao_ble/nrf52840/sense` | Zephyr 4.5.0, Zephyr SDK 1.0.1 (arm-zephyr-eabi-gcc 14.3.0), west 1.5.0 | FLASH 69,416 B, RAM 15,864 B |
| Zephyr | `nrf52840dk/nrf52840` | same | FLASH 43,388 B, RAM 5,632 B |
| nRF Connect SDK | - | not installed | **not built** |

Those sizes are whole programs (core, RTOS, USB or UART console). The model
itself is the same 22,040 bytes of read-only data everywhere plus 320 to 470
bytes of code, depending on the compiler.

Warnings. Each build has `-Wall -Wextra` in effect for the generated model
and the example, and no build reported a warning in either. The platforms
switch some `-Wextra` warnings off again, so the harness names
`-Wunused-parameter` and `-Wsign-compare` explicitly (the exact flags of each
build are in `results.json`). `python -m harness.build_packages
--inject-warning` checks that this would be noticed: it plants an unused
variable in the example and a `-Wtype-limits`, a `-Wsign-compare` and a
`-Wunused-parameter` warning in the model source, and all seven builds
reported all four. (ESP-IDF compiles with `-Werror`; there the planted
`-Wtype-limits` warning stops the build unless the self-test adds
`-Wno-error=type-limits`, which it does to see the other three as well.)

Also built by hand, once, with the same SDKs: a separate Zephyr application
with the module in `modules/` registered by the `list(APPEND
ZEPHYR_EXTRA_MODULES ...)` line, the same application with
`-DZEPHYR_EXTRA_MODULES=` and with `-DEXTRA_ZEPHYR_MODULES=` instead (and,
as expected, a Kconfig error when the module is not registered at all); and
a separate ESP-IDF project with the component in `components/` and
`PRIV_REQUIRES`.

Run on hardware: `python -m harness.device example --board A` exports the
Arduino package for the same model, builds its example with `-Os`, uploads it
to a Seeed XIAO nRF52840 Sense Plus and reads the serial port. The board
printed

```
demo_model: class 1, margins [-110264, 33593, -44637, -12703, -4387] / 2^15, self-check OK
```

which is the class and the margins xgb-edge computed at export time. The
Zephyr and ESP-IDF examples and the ESP32 Arduino builds were **not** verified
on hardware. (The Zephyr sample was flashed to a XIAO once; it booted and
enumerated on USB, but macOS created no serial port for it, so its output
could not be read.)

Not done: no ESP32 or Zephyr image was flashed; nothing was built with nRF Connect SDK, with
the Arduino IDE (only arduino-cli), for AVR boards, or with an ESP-IDF
release other than v6.1.1.

What does run is `pytest tests/test_package.py`: it compiles each example on
the host against stand-ins for the platform headers, with warnings as
errors, runs it, and requires the printed class and margins to equal
xgb-edge's own integer simulation.

### Repeating the builds in this repository

```sh
.venv/bin/python -m harness.build_packages     # all of the above; add --only arduino|esp-idf|zephyr
```

The first run installs ESP-IDF v6.1.1 and Zephyr 4.5.0 with its SDK under
`.sdk/` (about 5 GB) and leaves the packages in `build/packages/<platform>/demo_model`.
By hand, from the repository root:

```sh
# Arduino (the PATH prefix gives the Seeed core a working "python")
env PATH="$PWD/.venv/bin:$PATH" arduino-cli compile --warnings all \
    --fqbn Seeeduino:nrf52:xiaonRF52840SensePlus \
    --library build/packages/arduino/demo_model \
    build/packages/arduino/demo_model/examples/DemoModelPredict
# likewise with --fqbn esp32:esp32:esp32 and --fqbn esp32:esp32:esp32c3
# (with --warnings all the esp32c3 build prints "missing initializer" warnings from
#  the ESP32 core's own headers, .../hal/gpspi_flash_ll.h and spimem_flash_ll.h;
#  none of them is in the model or the example)

# Zephyr
env ZEPHYR_BASE="$PWD/.sdk/zephyrproject/zephyr" \
    ZEPHYR_SDK_INSTALL_DIR="$PWD/.sdk/zephyr-sdk-1.0.1" \
    .sdk/zephyr-venv/bin/west build -p always -b xiao_ble/nrf52840/sense \
    -d build/packages/by-hand/zephyr build/packages/zephyr/demo_model/samples/predict

# ESP-IDF (in a subshell, because export.sh changes the environment)
(
  export IDF_TOOLS_PATH="$PWD/.sdk/espressif" IDF_SKIP_CHECK_SUBMODULES=1
  export PATH="$PWD/.venv/bin:$PATH"     # the Python that install.sh was run with
  . .sdk/esp-idf/export.sh
  idf.py -C build/packages/esp-idf/demo_model/examples/predict \
      -B build/packages/by-hand/esp-idf \
      -D SDKCONFIG="$PWD/build/packages/by-hand/esp-idf/sdkconfig" \
      -D IDF_TARGET=esp32 build
)
```

Two things are particular to the ESP-IDF copy under `.sdk/`. To keep the
download small it is a sparse checkout (`components/` and `tools/` only) with
just the three submodules the example needs (tlsf, mbedtls, micro-ecc), so
`IDF_SKIP_CHECK_SUBMODULES=1` must be set (ESP-IDF would otherwise clone all
the others), and it can only build projects that set
`idf_build_set_property(MINIMAL_BUILD ON)` as the example does. A normal
ESP-IDF installation needs neither. `idf.py` also writes a
`dependencies.lock` file (and, without `-D SDKCONFIG=...`, an `sdkconfig`)
into the example folder; both are its own bookkeeping.
