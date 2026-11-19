# xgb-edge

xgb-edge turns a trained XGBoost classifier into two small C files that run on
a microcontroller using integer arithmetic only.

It exists for two reasons, and both are checked by the tests in this
repository rather than assumed:

1. **The output is small.** A multiclass XGBoost model builds one tree per
   class per round, so even a modest model has tens of thousands of nodes. On
   a 1 MB microcontroller the limit is flash, not the lack of an FPU.
   xgb-edge stores a model in about 3 to 4.5 bytes per XGBoost node. On the
   models below (beyond a toy one) that is roughly 2 to 3 times smaller than
   a hand-written float32 table walk and 4 to 9 times smaller than the
   if-else C that m2cgen and TL2cgen generate, which for the larger models
   could not be compiled for the target at all.
2. **The predictions are the model's predictions.** Every split comparison
   gives exactly the result of XGBoost's float32 `x < threshold`, for every
   possible input. Leaf values are rounded to fixed point; the tool reports
   the worst-case effect of that rounding and tells you, before anything is
   flashed, which of your validation rows are close enough to a tie for the
   class to change. On the 30 test models here the generated C predicted
   XGBoost's class on every row that the tool had not flagged in advance.

The numbers in this README say where they were measured: on a real nRF52840,
on the host, or only cross-compiled. See [What was verified, and where](#what-was-verified-and-where).

## Quick start

Install (Python 3.9 or later; the exporter itself needs only numpy):

```sh
pip install .
```

Train and save the model as JSON, and save some validation features:

```python
import numpy as np, xgboost as xgb

clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, tree_method="hist")
clf.fit(X_train, y_train)              # labels 0..K-1; NaN in X means "missing"
clf.save_model("model.json")           # the JSON format, not .ubj
np.save("X_val.npy", X_val.astype(np.float32))
```

Export:

```sh
xgb-edge export model.json --name gesture --validation X_val.npy -o out/
```

This writes `out/gesture.h` and `out/gesture.c` and prints the report shown
[below](#the-report). Add the two files to your firmware and call:

```c
#include "gesture.h"

float features[GESTURE_NUM_FEATURES];   /* same order and values as the training columns */
int32_t margins[GESTURE_NUM_MARGINS];

int cls = gesture_predict(features, margins);        /* margins may be NULL */
```

Optionally, check how clear the decision was. If the winner beats every
other class by more than `GESTURE_SAFE_GAP`, the class is guaranteed to be
the one XGBoost predicts for this input:

```c
int certain = 1;
for (int k = 0; k < GESTURE_NUM_MARGINS; k++) {
    if (k != cls && margins[cls] - margins[k] <= GESTURE_SAFE_GAP) certain = 0;
}
/* binary model: certain = margins[0] > NAME_SAFE_GAP || margins[0] < -NAME_SAFE_GAP */
```

* The return value is the class index, `0 .. NUM_CLASSES-1`.
* `margins[k]` is the raw score of class `k` (XGBoost's `output_margin=True`)
  in fixed point: real value = `margins[k] / 2^GESTURE_MARGIN_SHIFT`. A binary
  model has one margin; the class is 1 when it is greater than 0.
* A feature that is missing is passed as `NAN`.
* The code uses no floating-point operation (float inputs are only read as
  bit patterns), no heap and no static RAM, and it is reentrant.

To get a ready-made Arduino library, Zephyr module or ESP-IDF component with
an example program, add `--package arduino|zephyr|esp-idf`; see
[docs/integration.md](docs/integration.md).

### Options worth knowing

| option | meaning |
|---|---|
| `--validation FILE` | `.npy`, `.npz` or `.csv` feature matrix (NaN or empty cell = missing). A CSV header line is skipped: either text, or the `0,1,2,...` that pandas writes for unnamed columns; any other text is an error. Used to choose the leaf width and to report rows close to a tie. Recommended. |
| `--leaf-bits auto\|N` | Width of the fixed-point leaf values. `auto` (default) picks the narrowest of 8, 16, 24, 32 bits at which leaf rounding puts no validation row at risk (beyond rows that XGBoost's own float32 rounding already puts at risk); without validation data it uses 16. |
| `--input f32\|i8\|u8\|i16\|u16\|i32` | Type of the feature array. With an integer type the model is evaluated on raw integer features (for example sensor counts) with integer thresholds and no float anywhere. The model must have been trained on those same integer values, and integer inputs cannot be missing. |
| `--rounds all\|best\|N` | Required for a model trained with early stopping, where `XGBClassifier.predict` uses fewer trees than `Booster.predict`; see [Early stopping](#early-stopping). |
| `--target cortex-m4\|esp32\|esp32c3\|avr` | Which installed cross compiler measures code size and stack for the report (default `cortex-m4`). Other compilers: `--cc PATH --cflags="..."` (with the equals sign). |
| `--package arduino\|zephyr\|esp-idf` | Write a platform package instead of bare files. |

`xgb-edge report model.json ...` prints the same report without writing C.

## How it works

### Thresholds are lossless

XGBoost sends a row left at a split when `x < t`, both float32. For each
feature, xgb-edge collects the distinct thresholds the model uses on it and
sorts them into cuts `c0 < c1 < ...`. It then replaces

* a feature value `x` by its rank `r(x)` = the number of cuts `<= x`, and
* a threshold equal to cut `cj` by the code `j + 1`.

`x < cj` holds exactly when at most `j` cuts are `<= x`, that is when
`r(x) < j + 1`. So the integer comparison `r(x) < code` gives the same answer
as the float comparison for every float32 `x`. A value equal to a threshold
is counted by the `<=` and goes right, exactly as `x < t` being false does.
This direction is the part that is easy to get wrong, so it is what the
property tests hammer (see [Tests](#tests)).

On the device the rank is found by binary search over the cuts using integer
comparisons only: a float32 is mapped to an integer key that orders the same
way floats do (negative values mirrored, `-0.0` and `+0.0` sharing a key), and
NaN is recognised from its bits. With hist-trained models a feature has at
most 255 cuts, so a rank is one byte.

When nearly every split has its own threshold (models trained with
`tree_method="exact"`), a cut table costs more than it saves. For those the
exporter stores the threshold key in the node instead and compares it with
the feature's key directly. It builds both encodings and keeps the smaller
one. For integer inputs this second encoding is the familiar picture: an
`int16_t` feature compared with an `int16_t` threshold, where the threshold is
the smallest integer whose float32 value is `>= t` (found by search with real
float32 conversion, not by a rounding formula).

### Missing values cost nothing per node

Each split has a default direction for missing values. Instead of a flag and
a test in every node, a feature gets up to two slots holding its rank: in the
"default right" slot a missing value is stored as the largest code (never
`< code`), in the "default left" slot as 0 (always `< code`). A node simply
reads the slot that matches its own default direction.

### Leaves are fixed point, with a stated error

Each leaf value `v` is stored as `round(v * 2^shift)`, with `shift` chosen per
model so that the largest leaf uses the full width and the 32-bit
accumulators cannot overflow. The exporter does this in exact arithmetic and
reports:

* the **worst-case error on each class margin, over all possible inputs** (the
  sum over trees of the largest rounding error of any leaf in the tree);
* for your validation rows, how many are **at risk**: the gap between the best
  and second-best class is within reach of that rounding, or within reach of
  XGBoost's own float32 rounding when it sums the leaves. Every row that is
  not at risk is guaranteed to get XGBoost's class;
* `<NAME>_SAFE_GAP`, a constant in the header: if the winning margin beats
  every other margin by more than this, the class is guaranteed to be
  XGBoost's **for any input**, not only for validation rows. Firmware can use
  it to tell a certain prediction from a near-tie.

With `--leaf-bits auto` the exporter picks the narrowest leaf width for which
no validation row is at risk beyond XGBoost's own noise. Wider leaves cost
one byte per leaf per step; `--leaf-bits 16` is the usual size/accuracy sweet
spot if you accept a few flagged rows.

### The packed model

Only split nodes are stored; leaves live inside their parent. Each tree's
split nodes are in pre-order, so a child that is itself a split needs no
pointer when it is the next node:

```
header  1 byte: node type (2 bits) | feature slot (6 bits)      2 bytes if the model has more than 64 slots
code    1 byte: threshold rank                                  2 or 4 bytes for wide ranks / direct thresholds
then, by type
  both children are splits : 1-byte length of the left subtree (the right child follows it)
  one child is a leaf      : that leaf; the other child follows
  both children are leaves : left leaf, right leaf
```

A leaf is its fixed-point value (1 to 4 bytes) or, when the model has few
distinct leaf values, an index into a shared table of them; the exporter
keeps whichever makes the model smaller. Field widths grow only when a model
needs it. Trees are stored grouped by class, so the interpreter needs no
per-tree class number: it walks the trees of class 0, then class 1, and so
on, and keeps the running best. The whole run-time is one function of about
300 bytes; prediction is the arg max of the integer margins, with no softmax.
The only library call it can make is `memcpy`, which float-input models use
to read a float's bits; compilers normally inline it (the RISC-V ESP32-C3
build at `-Os` is one that calls the C library instead).

How many bytes a split costs depends on the field widths the model needs.
The 5-class example below uses 4.59 bytes per split node (its leaves
included, since they live in their parent node); with its cut tables and
everything else that is 3.2 bytes per XGBoost node. `make parity` prints the
figure for every test model and layout.

### The report

Printed by `xgb-edge export` before anything is compiled into firmware
(a 5-class model, 250 trees of depth 4):

```
Flash
  gesture_feat                   uint8_t[16]                    16 B  input column of each used feature
  gesture_cut_end               uint16_t[16]                    32 B  end of each used feature's cuts in the cut table
  gesture_cuts                   int32_t[1659]               6,636 B  sorted split thresholds per feature (as integer keys)
  gesture_bias                   int32_t[5]                     20 B  per-class base margin (fixed point)
  gesture_group_end              uint8_t[5]                      5 B  trees are stored grouped by class: end index per class
  gesture_tree_size              uint8_t[250]                  250 B  byte length of each tree in the node stream
  gesture_nodes                  uint8_t[15081]             15,081 B  packed split nodes
  model data                                                22,040 B  (the tables above)
  interpreter code                                             320 B  (measured for cortex-m4 with arm-none-eabi-gcc ... -Os)
  total                                                     22,360 B

RAM
  static (.data/.bss): 0 B
  stack during predict: 80 B  (measured (-fstack-usage)); includes the 32-byte rank array

Worst-case comparisons per inference
  feature ranking (binary search + NaN test): 131
  tree splits (sum of tree depths): 1,000
  class selection: 4
  total: 1,135

Leaf values
  16-bit fixed point, margins = integer / 2^15 (LSB 3.05e-05); largest |leaf| 0.654488
  worst-case error on a class margin, over all possible inputs: 0.000699
  split thresholds: lossless (every comparison equals XGBoost's float32 x < threshold)
  certainty check for any input: a margin gap above 48 (0.00146) guarantees XGBoost's class (GESTURE_SAFE_GAP)

Validation (2,350 samples)
  smallest gap between the top two class margins: 0.00238
  samples close enough to a tie for the class to change: 0
  ...
```

The table sizes are exactly what the generated file declares. The code size
and the stack are measured by compiling that file with the target's compiler
if it is installed (`--target`; otherwise the report says it is an estimate),
and the total is then the size of the compiled object. For toy models a
compiler can fold a table of a few entries into code; the report shows that
when it happens. On the nRF52840 the reported total was equal, to the byte,
to the model's symbols in the linked firmware for every model tested. The
stack figure is the frame of the predict function; the harness measures the
stack depth on the board and fails if it exceeds the figure (with `-Os` the
two were equal on every model).

## Results

### On a real nRF52840, against a float32 baseline

Hardware: a Seeed XIAO nRF52840 Sense Plus (Cortex-M4F, 64 MHz), Arduino core
`Seeeduino:nrf52` 1.1.13, `arm-none-eabi-gcc` 9.2.1, `-Os`. The test firmware
receives every test row over USB serial, runs the model and returns class,
margins and the inference time measured on the board with a 16 MHz hardware
timer (best of 3 runs per row, mean over rows shown). For each model the
xgb-edge firmware and the float-baseline firmware are flashed to the same
board one after the other and are sent the same rows. (During development
the two builds ran side by side on two boards of this type, and swapping the
boards changed the times by less than 0.2%; that cross-check is not part of
the results below.)

The float baseline (`bench/float_baseline.py`) is what a careful engineer
might write without this tool: the same table-walk idea with float32
thresholds and leaves, 8 bytes per node, using the board's FPU.

<!-- table:device -->
| model | rows sent to the board | differ from xgboost | leaf bits | flash reported by the tool (B) | flash in the linked image (B) | float baseline flash (B) | float / xgb-edge | xgb-edge time (us) | float baseline time (us) | float / xgb-edge time | worst-case comparisons | stack reported / measured (B) | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bin_d2_r5_hist` | 1,991 | 0 | 16 | 327 | 327 | 421 | 1.3x | 8 | 8 | 0.98x | 17 | 48 / 48 | pass |
| `bin_d6_r200_exact_missing` | 2,350 | 0 | 16 | 39,934 | 39,934 | 85,984 | 2.2x | 466 | 602 | 1.29x | 1,221 | 184 / 184 | pass |
| `mc3_d3_r10_hist` | 1,710 | 0 | 16 | 1,924 | 1,924 | 3,801 | 2.0x | 53 | 62 | 1.17x | 128 | 56 / 56 | pass |
| `mc5_d4_r50_hist_missing` | 2,350 | 0 | 16 | 22,360 | 22,360 | 55,007 | 2.5x | 454 | 639 | 1.41x | 1,135 | 80 / 80 | pass |
| `mc10_digits_d6_r100_hist` | 2,955 | 0 | 24 | 56,737 | 56,737 | 157,680 | 2.8x | 1,529 | 1,867 | 1.22x | 4,146 | 112 / 112 | pass |
| `mc10_d10_r30_exact` | 2,350 | 0 | 24 | 420,136 | 420,136 | does not fit | - | 1,140 | - | - | 3,033 | 144 / 144 | pass |
| `mc16_d5_r397_hist` | 2,350 | 0 | 24 | 759,929 | 759,929 | does not fit | - | 13,651 | - | - | 32,027 | 80 / 80 | pass |
| `mc3_dart` | 2,350 | 0 | 16 | 10,942 | 10,942 | 29,427 | 2.7x | 224 | 294 | 1.31x | 553 | 64 / 64 | pass |
| `mc4_int16` | 2,145 | 0 | 16 | 41,293 | 41,293 | 130,416 | 3.2x | 578 | 866 | 1.50x | 1,539 | 64 / 64 | pass |
<!-- /table:device -->

* "rows sent to the board" are the test rows of the model: the held-out split
  plus rows placed exactly on, just below and just above split thresholds,
  rows with missing values and rows with extreme magnitudes.
* "does not fit": the firmware would exceed what the board's bootloader
  accepts (see below).
* `mc16_d5_r397_hist` is sized to fill the board; its whole firmware is just
  under the bootloader's limit.
* There is no power measurement equipment here, so no energy figures are
  given. Inference time and the worst-case comparison count are what was
  measured.

The same 5-class model at other optimisation levels (the Arduino core's
default is `-Ofast`); each line is a separate build, flashed and run:

<!-- table:optlevels -->
(run python -m harness.results --update-readme)
<!-- /table:optlevels -->

The board's bootloader accepts less than the 811,008 bytes the Arduino core
reports as the maximum program size. `python -m harness.flash_limit` builds
firmware of an exact size and tries to upload it:

<!-- table:flashlimit -->
(run python -m harness.results --update-readme)
<!-- /table:flashlimit -->

### Flash size against m2cgen and TL2cgen

Each generator's output for the same model file, compiled for Cortex-M4F
(`arm-none-eabi-gcc 9.2.1 -mcpu=cortex-m4 -mthumb -mfloat-abi=hard
-mfpu=fpv4-sp-d16 -Os`). Sizes are `.text + .rodata + .data` of the object
files; nothing was linked or run, and library code the if-else generators
need at link time (software double arithmetic, `exp`) is not counted.
m2cgen 0.10.0, TL2cgen 1.0.0 with Treelite 4.8.1.

<!-- table:sizes -->
| model | XGBoost nodes | xgb-edge | float32 table walk | m2cgen | TL2cgen | TL2cgen quantize=1 | TL2cgen, split | TL2cgen quantize=1, split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bin_d2_r5_hist` | 35 | 327 | 421 (1.3x) | 960 (2.9x) | 1,064 (3.3x) | 1,090 (3.3x) | 1,140 (3.5x) | 1,202 (3.7x) |
| `bin_d6_r200_exact_missing` | 10,706 | 39,934 | 85,984 (2.2x) | 229,436 (5.7x) | 272,632 (6.8x) | 192,780 (4.8x) | 242,548 (6.1x) | 166,768 (4.2x) |
| `mc3_d3_r10_hist` | 444 | 1,924 | 3,801 (2.0x) | 9,382 (4.9x) | 11,186 (5.8x) | 8,736 (4.5x) | 11,462 (6.0x) | 8,640 (4.5x) |
| `mc5_d4_r50_hist_missing` | 6,816 | 22,360 | 55,007 (2.5x) | 145,366 (6.5x) | 174,602 (7.8x) | 120,186 (5.4x) | 159,154 (7.1x) | 107,382 (4.8x) |
| `mc10_digits_d6_r100_hist` | 19,552 | 56,737 | 157,680 (2.8x) | 346,770 (6.1x) | 518,858 (9.1x) | 391,602 (6.9x) | 459,942 (8.1x) | 336,270 (5.9x) |
| `mc10_d10_r30_exact` | 95,696 | 420,136 | 766,436 (1.8x) | does not assemble | does not assemble | does not assemble | 2,874,938 (6.8x) | 2,119,158 (5.0x) |
| `mc16_d5_r397_hist` | 266,322 | 759,929 | 2,137,228 (2.8x) | compiler out of memory | does not assemble | does not assemble | 6,803,106 (9.0x) | 4,728,674 (6.2x) |

xgb-edge needs 2.9 to 4.4 bytes per XGBoost node (splits and leaves, code included) on the models with at least 1,000 nodes.
<!-- /table:sizes -->

* m2cgen and TL2cgen put each model into one or a few very large C functions.
  For the larger models `arm-none-eabi-gcc` could not build them: the
  assembler stops with "conditional branch out of range", or the compiler ran
  past a 12 GB memory limit. "split" is TL2cgen's `parallel_comp=32`, which
  spreads the trees over 32 files.
* m2cgen computes in `double`, which is software floating point on this FPU.
  m2cgen 0.10.0 needed a small subclass to accept XGBoost 3.x per-class base
  scores; the trees are unchanged (details in `bench/altgen.py`).
* On the host the m2cgen and TL2cgen builds were also compared with xgboost
  (`python -m bench.altgen verify`). TL2cgen's code gave xgboost's class on
  every test row. m2cgen's differed on up to 30 rows per model, none of them
  in the held-out data: all were among the adversarial rows (values on or one
  float32 away from a split threshold, missing values, extreme magnitudes).

### ESP32: compiled, not run

No ESP32 was available. Every export in the test suite is compiled with
`-Wall -Wextra -Werror` for ESP32 (Xtensa, `xtensa-esp32-elf-gcc` 14.2.0) and
ESP32-C3 (RISC-V, `riscv32-esp-elf-gcc` 14.2.0), and the packaged example was
built with the Arduino ESP32 core and with ESP-IDF. **None of this was run on
ESP32 hardware.** The model data is the same bytes on every target; only the
interpreter code differs.

<!-- table:esp32 -->
| model | model data (B) | code, Cortex-M4 (B) | code, ESP32 Xtensa (B) | code, ESP32-C3 RISC-V (B) | total on ESP32 (B) | total on ESP32-C3 (B) |
|---|---:|---:|---:|---:|---:|---:|
| `bin_d2_r5_hist` | 131 | 196 | 271 | 316 | 402 | 447 |
| `bin_d6_r200_exact_missing` | 39,726 | 208 | 288 | 328 | 40,014 | 40,054 |
| `mc3_d3_r10_hist` | 1,612 | 312 | 358 | 418 | 1,970 | 2,030 |
| `mc5_d4_r50_hist_missing` | 22,040 | 320 | 374 | 434 | 22,414 | 22,474 |
| `mc10_digits_d6_r100_hist` | 56,409 | 328 | 387 | 432 | 56,796 | 56,841 |
| `mc10_d10_r30_exact` | 419,852 | 284 | 375 | 414 | 420,227 | 420,266 |
| `mc16_d5_r397_hist` | 759,609 | 320 | 378 | 430 | 759,987 | 760,039 |
| `mc3_dart` | 10,638 | 304 | 362 | 416 | 11,000 | 11,054 |

| packaged example build | status | program size (B) |
|---|---:|---:|
| Arduino `Seeeduino:nrf52:xiaonRF52840SensePlus` (core default) | built | 65,612 |
| Arduino `Seeeduino:nrf52:xiaonRF52840SensePlus` (-Os) | built | 58,720 |
| Arduino `esp32:esp32:esp32` (core default) | built | 302,284 |
| Arduino `esp32:esp32:esp32c3` (core default) | built | 296,102 |
| ESP-IDF v6.1.1 `esp32` | built | 148,064 |
| Zephyr v4.5.0 `xiao_ble/nrf52840/sense` | built | 69,416 |
| Zephyr v4.5.0 `nrf52840dk/nrf52840` | built | 43,388 |
<!-- /table:esp32 -->

### Host parity

`make parity` trains the model zoo, exports each model in several variants,
compiles the C for the host (clang, `-Wall -Wextra -Wpedantic -Wconversion
-Werror`, AddressSanitizer and UBSan) and compares every test row with
xgboost.

<!-- table:parity -->
| model | classes | trees | splits | exports | row checks | flagged in advance | differ (all flagged) | differ, not flagged | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bin_d2_r5_hist` | 2 | 5 | 15 | 18 | 35,838 | 52 | 8 | 0 | pass |
| `bin_d6_r200_exact_missing` | 2 | 200 | 5,253 | 7 | 16,450 | 28 | 0 | 0 | pass |
| `bin_d4_r60_hinge` | 2 | 60 | 704 | 2 | 4,700 | 0 | 0 | 0 | pass |
| `bin_d3_r40_basescore` | 2 | 40 | 251 | 2 | 4,700 | 0 | 0 | 0 | pass |
| `bin_d10_r40_exact` | 2 | 40 | 6,681 | 3 | 7,050 | 2 | 0 | 0 | pass |
| `mc3_d3_r10_hist` | 3 | 30 | 207 | 18 | 30,780 | 71 | 6 | 0 | pass |
| `mc4_d2_r300_hist` | 4 | 1,200 | 3,476 | 5 | 11,750 | 296 | 2 | 0 | pass |
| `mc5_d4_r50_hist_missing` | 5 | 250 | 3,283 | 18 | 42,300 | 665 | 14 | 0 | pass |
| `mc5_d7_r40_exact_missing` | 5 | 200 | 9,929 | 4 | 9,400 | 0 | 0 | 0 | pass |
| `mc7_d5_r80_softmax` | 7 | 560 | 11,599 | 2 | 4,700 | 1 | 0 | 0 | pass |
| `mc10_digits_d6_r100_hist` | 10 | 1,000 | 9,276 | 15 | 37,569 | 593 | 25 | 0 | pass |
| `mc10_d10_r30_exact` | 10 | 300 | 47,698 | 3 | 7,050 | 6 | 0 | 0 | pass |
| `mc12_d8_r120_hist_missing` | 12 | 1,440 | 93,342 | 3 | 7,050 | 2 | 0 | 0 | pass |
| `mc16_d5_r400_hist` | 16 | 6,400 | 130,925 | 3 | 7,050 | 12 | 0 | 0 | pass |
| `mc16_d5_r397_hist` | 16 | 6,352 | 129,985 | 2 | 4,700 | 12 | 0 | 0 | pass |
| `mc6_d6_r40_approx_missing` | 6 | 240 | 9,849 | 3 | 7,050 | 0 | 0 | 0 | pass |
| `mc4_rf_npt3` | 4 | 300 | 7,311 | 2 | 4,700 | 1 | 1 | 0 | pass |
| `mc3_dart` | 3 | 120 | 1,758 | 5 | 11,750 | 65 | 4 | 0 | pass |
| `bin_dart_exact` | 2 | 30 | 208 | 2 | 4,700 | 0 | 0 | 0 | pass |
| `mc6_lossguide_deep` | 6 | 180 | 17,100 | 4 | 9,400 | 3 | 0 | 0 | pass |
| `mc5_maxbin1024` | 5 | 600 | 64,546 | 3 | 7,050 | 0 | 0 | 0 | pass |
| `mc3_wide_f300` | 3 | 180 | 7,585 | 3 | 7,050 | 0 | 0 | 0 | pass |
| `mc4_int16` | 4 | 240 | 8,002 | 7 | 15,825 | 0 | 0 | 0 | pass |
| `mc3_int8_exact` | 3 | 90 | 2,008 | 5 | 10,750 | 0 | 0 | 0 | pass |
| `mc5_wild_scales` | 5 | 200 | 4,634 | 3 | 7,050 | 0 | 0 | 0 | pass |
| `mc4_degenerate_cols` | 4 | 120 | 1,592 | 3 | 7,050 | 0 | 0 | 0 | pass |
| `mc3_single_leaf_trees` | 3 | 90 | 215 | 5 | 6,000 | 10 | 0 | 0 | pass |
| `mc3_all_single_leaf` | 3 | 18 | 0 | 3 | 1,800 | 0 | 0 | 0 | pass |
| `bin_few_leaf_values` | 2 | 60 | 20 | 5 | 6,182 | 0 | 0 | 0 | pass |
| `mc26_d8_r150_hist` | 26 | 3,900 | 348,823 | 2 | 4,700 | 15 | 0 | 0 | pass |
| **30 models** |  |  |  | 160 | 342,144 | 1834 | 60 | 0 | pass |
<!-- /table:parity -->

"Flagged in advance" counts rows the tool reported as at risk before the C
was run; most of those come from the deliberately coarse 8-bit and 12-bit
exports. "Differ, not flagged" is the number that has to be zero.

Models saved by older XGBoost releases were checked the same way, with that
release doing both the training and the reference prediction (ordinary
trained models plus a hand-built DART model and extreme base scores chosen to
expose differences between releases):

<!-- table:oldver -->
| xgboost release that trained and predicted | models | exports | row checks | differ, not flagged | result |
|---|---:|---:|---:|---:|---:|
| 1.6.2 | 10 | 20 | 18,880 | 0 | pass |
| 1.7.6 | 10 | 20 | 18,880 | 0 | pass |
| 2.0.3 | 10 | 20 | 18,880 | 0 | pass |
| 2.1.4 | 10 | 20 | 18,880 | 0 | pass |
| 3.0.5 | 10 | 20 | 18,880 | 0 | pass |

Each model is compared with that release's `Booster.predict` and with its `inplace_predict` / `XGBClassifier.predict`. One exception, on 1.6.2 and 1.7.6: XGBClassifier.predict of this release answers only 0/1 for a natively trained multi:softmax model; compared with Booster.inplace_predict instead.
<!-- /table:oldver -->

## What the exporter accepts

| model feature | handling |
|---|---|
| `multi:softprob`, `multi:softmax` (any number of classes), `binary:logistic`, `binary:hinge` | supported |
| `tree_method` hist, exact, approx; depth-wise and loss-guide growth; any depth | supported (the interpreter is iterative, no recursion) |
| missing values and per-split default directions | supported for float inputs (NaN = missing) |
| `base_score` | supported, both the scalar form and the per-class vector of XGBoost 3.1+; for `binary:logistic` it is converted to a margin as the XGBoost release that saved the file does (releases from 3.2 on bound it to [1e-6, 1 - 1e-6] first; the report says so when that matters) |
| `num_parallel_tree > 1` (boosted random forests) | supported |
| DART booster | supported: each tree's weight is folded into its leaves before rounding. Both the old (`"name": "dart"`) and the 3.3+ JSON layout are read |
| single-leaf trees | folded into the base margin |
| a model with no trees at all (saved before any boosting round), `num_class=1` | **rejected** |
| early stopping | you must choose with `--rounds`, see below |
| categorical splits | **rejected** with an error naming the tree, node and feature |
| multi-output trees (`multi_strategy="multi_output_tree"`), multi-label / multi-target | **rejected** |
| `gblinear` booster | **rejected** |
| other objectives (regression, ranking, `binary:logitraw`, ...) | **rejected** |
| UBJSON (`.ubj`) or legacy binary model files | **rejected**; re-save with `save_model("model.json")` |
| malformed files (wrong array lengths, cycles, out-of-range features, NaN thresholds) | **rejected** |

Nothing is silently approximated: if the exporter cannot reproduce a model
exactly up to the stated leaf rounding, it stops with an error that says why.

"XGBoost's class" means the argmax of the probabilities (probability above
0.5 for a binary model) that `Booster.predict` on a `DMatrix` and
`Booster.inplace_predict` return; `XGBClassifier.predict` is the latter. The
test harness checks both paths; they can differ in the last bits of the
margins, most visibly for DART models on XGBoost 3.0 and earlier, and the
reported noise bound covers both. Two wrapper quirks are not the model's
doing: for a two-class model trained with `multi:softprob`,
`XGBClassifier.predict` returns a one-hot matrix rather than class numbers,
and in XGBoost 1.x it answers only 0 or 1 for a natively trained
`multi:softmax` model. Model files written by XGBoost 1.6, 1.7, 2.0, 2.1,
3.0, 3.1, 3.2 and 3.4 are tested.

### Early stopping

A model trained with early stopping holds more trees than its best iteration.
`XGBClassifier.predict` silently stops at the best iteration; `Booster.predict`
uses every tree. Because "the same predictions as xgboost" is ambiguous
there, the exporter refuses to guess and asks for `--rounds best` (what
`XGBClassifier.predict` computes) or `--rounds all` (what `Booster.predict`
computes).

## Limitations

* **Classification only, and only the class.** The C returns the arg max and
  the integer margins. It does not compute probabilities; apply softmax or
  the logistic function to the margins yourself if you need them.
* **Leaf rounding is real.** Margins differ from XGBoost's by up to the
  reported bound, and an input that lies within that bound of a tie between
  two classes can get the other class. The tool flags such rows in your
  validation data and gives you `SAFE_GAP` for run time, but it cannot make a
  near-tie certain. At 8 or 12 bits many rows are near-ties by this standard.
* **`--leaf-bits auto` judges by your validation data.** "No row at risk" is a
  statement about those rows. Unseen inputs can still be near-ties; use
  `SAFE_GAP` if that matters.
* **Float inputs must be float32 and missing must be NaN.** If you trained
  with another `missing=` value, map it to NaN before calling. Infinite
  inputs are treated as larger or smaller than every finite value, which is
  what `inplace_predict` does too (a `DMatrix` refuses them).
* **The reference is the XGBoost release that saved the model file.** Where
  releases differ (how a `binary:logistic` base score outside
  [1e-6, 1 - 1e-6] becomes a margin changed in 3.2), the export follows the
  file's release and the report notes it. If you predict with a different
  release, re-save the model with that one first.
* **Integer inputs** must be the exact integers the model was trained on (no
  scaling in between) and cannot be missing.
* **Per-row `base_margin` is not supported**, nor is predicting with a subset
  of trees at run time.
* **Flash is the limit.** The largest model that fit the XIAO nRF52840 here
  has 6,352 trees and 266,000 XGBoost nodes, and the board's bootloader
  accepts less than the Arduino core advertises (see the results).
* **Speed.** On the nRF52840, which has an FPU, inference is moderately
  faster than the float baseline, not dramatically (the measured ratios are
  under the device table; a toy model shows no difference). The gain is
  mostly flash. Small models with many features spend a visible share of the
  time ranking features.
* **AVR.** The generated C compiles for 8-bit AVR, but there `const` tables
  are copied to RAM (the report for `--target avr` counts them as RAM), so
  only very small models are usable. AVR was not a target of this work and
  nothing was run on one.
* **Timing was measured on one MCU and one compiler.** ESP32 numbers are
  sizes of compiled code only. Zephyr, ESP-IDF and the ESP32 Arduino core were
  build-tested; only the nRF52840 Arduino builds were run on hardware.
  nRF Connect SDK itself was not installed; upstream Zephyr was used.

## What was verified, and where

| claim | host | nRF52840 (measured) | ESP32 |
|---|---|---|---|
| same class as xgboost on every unflagged test row | yes: every zoo model and variant, every row | yes: 9 models, every row sent to the board | not run |
| integer margins equal the tool's exact simulation, bit for bit | yes | yes | not run |
| split decisions equal xgboost's at and around thresholds | yes: property tests against xgboost itself | boundary rows are part of what is sent to the board | not run |
| compiles without warnings under `-Wall -Wextra` | yes (clang, also `-Wpedantic -Wconversion`, and as C++) | yes (`arm-none-eabi-gcc -Werror`, and the Arduino build) | compiled only (`-Werror`, Xtensa and RISC-V) |
| no floating-point operations | objects for FPU-less targets reference no soft-float routine; no FPU instruction on Cortex-M4F | the same code runs there | compiled only |
| reported flash size equals the built size | the declared tables equal `.rodata` for all three compilers (only tables of at most 8 entries, which a compiler may fold away, are exempt) | reported total equals the model's symbols in the linked image, to the byte | table sizes checked, compiled only |
| stack use | from `-fstack-usage` | measured on the board; never above the report, equal to it at `-Os` | compiled only |
| inference time | not measured | measured with a hardware timer | not measured |
| float baseline, m2cgen, TL2cgen sizes | - | float baseline run on a board; m2cgen and TL2cgen objects only | not built |
| platform packages | examples run against stub headers | Arduino example run on a board (prints `self-check OK`); Zephyr sample built for the XIAO and the nRF52840 DK, not verified on hardware (see below) | Arduino and ESP-IDF examples built, not run |

One attempt was made to run the Zephyr sample on a XIAO board. The image
booted (the board enumerated on USB as Zephyr's CDC ACM device), but macOS
never created a serial port for it, so its output could not be read and the
sample remains unverified on hardware.

## Tests

```sh
make test        # unit and property tests (about 2 minutes)
make parity      # the model zoo against xgboost on the host (about 1 minute once models are trained)
make mutation    # do the threshold tests catch deliberately broken rounding? (about 2 minutes)
make oldver      # models trained by older xgboost releases (needs uv and network on first run)
```

* `tests/test_thresholds.py`: property-based tests (Hypothesis). They build
  real XGBoost models whose thresholds sit at arbitrary float32 values,
  including zeros, denormals, the largest finite floats and thresholds that
  are adjacent floats of each other, feed inputs equal to each threshold and
  one float32 below and above it, and require the compiled C to reach the
  same leaf as xgboost itself. The same for integer inputs.
  `python -m harness.mutation` checks that these tests and the guarantee
  tests have teeth: it applies 24 deliberately wrong variants one at a time
  (for example `<` instead of `<=` in the rank search, or a float-noise bound
  of zero) and each one makes the tests fail.
* `tests/test_guarantee.py`: one model per ingredient of the guarantee (leaf
  rounding bound, XGBoost's float32 noise, its softmax/sigmoid ties,
  `SAFE_GAP`, the header macros) on which that ingredient decides the
  outcome, and a check that the harness fails when the ingredient is removed
  or inflated until it is meaningless. `harness/independent.py` recomputes
  the bounds without the exporter's bound code for every zoo export.
* `tests/test_model.py`: exact decimal-to-float32 parsing of the JSON
  (including values a hair away from a float32 midpoint) and one test per
  rejected model variant.
* `tests/test_export.py`, `tests/test_portability.py`,
  `tests/test_package.py`, `tests/test_bench.py`: round selection, layout
  choices, the report, compilation for every installed target, the packages
  and the benchmark baselines.
* `harness/parity.py`: the zoo. `harness/oldver.py`: models from older
  XGBoost releases. `harness/device.py`: the same checks on a board.

To repeat the hardware runs (XIAO nRF52840 boards are selected by the USB
serial numbers in `harness/device.py`; `make device device-float optlevels
example` does the same):

```sh
python -m harness.device suite --board A                 # xgb-edge, every row, timing
python -m harness.device suite --board A --kind float bin_d2_r5_hist mc3_d3_r10_hist \
    mc5_d4_r50_hist_missing bin_d6_r200_exact_missing mc10_digits_d6_r100_hist mc3_dart mc4_int16
python -m harness.device example --board A               # the packaged Arduino example
python -m harness.flash_limit --board A 798720 798724    # what the bootloader accepts
python -m bench.sizes --tl2cgen-parallel-comp 32         # flash sizes of the alternatives
python -m harness.build_packages                         # Arduino, ESP-IDF and Zephyr builds
python -m harness.results --update-readme                # rewrite the tables in this README
python -m harness.results --check                        # results complete, current, and what the README shows?
```

## Repository layout

```
xgb_edge/        the exporter: model.py (loading, validation), quantize.py (thresholds, leaves),
                 pack.py (node stream), codegen.py (C), analysis.py (error bounds, tie risk),
                 package.py (Arduino / Zephyr / ESP-IDF), cli.py
firmware/        the on-device test firmware (Arduino sketch)
harness/         parity.py (host), independent.py (bounds recomputed independently), device.py (boards),
                 zoo.py (test models), synth.py (hand-built models), oldver.py (older xgboost),
                 mutation.py, build_packages.py, results.py
bench/           float baseline, m2cgen and TL2cgen generation, size measurement
tests/           pytest suite
docs/            integration guide for the three platforms
```
