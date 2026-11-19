PY ?= .venv/bin/python

.PHONY: parity test all

# One command: train the model zoo, export, compile for the host and compare with xgboost.
parity:
	$(PY) -m harness.parity

test:
	$(PY) -m pytest -q

all: test parity
