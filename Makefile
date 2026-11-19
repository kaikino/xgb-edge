PY ?= .venv/bin/python

.PHONY: test parity mutation oldver device device-float optlevels example sizes packages results check all

BOARD ?= A

# Unit and property tests.
test:
	$(PY) -m pytest -q

# One command: train the model zoo, export, compile for the host and compare with xgboost.
parity:
	$(PY) -m harness.parity

# Apply deliberately wrong rounding variants and require the threshold tests to fail on each.
mutation:
	$(PY) -m harness.mutation

# Models trained and predicted by older xgboost releases (uses uv to fetch them).
oldver:
	$(PY) -m harness.oldver

# On a Seeed XIAO nRF52840 (see harness/device.py for the board serial numbers).
device:
	$(PY) -m harness.device suite --board $(BOARD)

device-float:
	$(PY) -m harness.device suite --board $(BOARD) --kind float bin_d2_r5_hist mc3_d3_r10_hist \
	    mc5_d4_r50_hist_missing bin_d6_r200_exact_missing mc10_digits_d6_r100_hist mc3_dart mc4_int16

# One model at -O2 and -Ofast, xgb-edge and float baseline (the -Os runs come from the two targets above).
optlevels:
	for opt in -O2 -Ofast; do \
	  $(PY) -m harness.device run mc5_d4_r50_hist_missing --board $(BOARD) --opt=$$opt && \
	  $(PY) -m harness.device run mc5_d4_r50_hist_missing --board $(BOARD) --opt=$$opt --kind float || exit 1; \
	done

example:
	$(PY) -m harness.device example --board $(BOARD)

# Flash size of xgb-edge, the float baseline, m2cgen and TL2cgen (cross-compiled; slow for large models).
sizes:
	$(PY) -m bench.sizes --tl2cgen-parallel-comp 32

# Build the Arduino, ESP-IDF and Zephyr packages (installs the SDKs under .sdk/ on first run).
packages:
	$(PY) -m harness.build_packages

# Rewrite the result tables in README.md from the JSON files the commands above leave in build/.
results:
	$(PY) -m harness.results --update-readme

# Are the result files complete, produced by the current code, and what the README shows?
check:
	$(PY) -m harness.results --check

all: test parity mutation
