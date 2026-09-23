# FoveaMap command surface (README Appendix D). Works with GNU make on Linux/macOS and with
# mingw32-make + Git Bash on Windows. Script flags go through ARGS, e.g.
#   make doctor ARGS="--require-data"        (D-006)

VENV ?= venv
ifeq ($(OS),Windows_NT)
  VENV_PY := $(VENV)/Scripts/python.exe
  SYS_PY ?= python
else
  VENV_PY := $(VENV)/bin/python
  SYS_PY ?= python3
endif
PYTHON ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),$(SYS_PY))
ARGS ?=
SEQUENCES ?= 04 07 08
FAST_MARKERS := not slow and not gpu and not data and not cpp

.PHONY: help doctor install build-cpp data-verify cache-preds test test-fast lint format typecheck \
        render benchmark report demo video ci fresh-clone clean

help:
	@echo "Targets: doctor install build-cpp data-verify cache-preds test test-fast lint format typecheck"
	@echo "         render benchmark report demo video ci fresh-clone clean   (README Appendix D)"

doctor:
	$(PYTHON) scripts/doctor.py $(ARGS)

install:
	$(SYS_PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	$(VENV_PY) -m pip install -e .

build-cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j4

data-verify:
	$(PYTHON) scripts/verify_data.py --sequences $(SEQUENCES) $(ARGS)

cache-preds:
	$(PYTHON) scripts/cache_predictions.py --sequences $(SEQUENCES) $(ARGS)

test:
	$(PYTHON) -m pytest $(ARGS)

test-fast:
	$(PYTHON) -m pytest -q -m "$(FAST_MARKERS)" $(ARGS)

lint:
	$(PYTHON) -m ruff check src scripts tests

format:
	$(PYTHON) -m ruff check --fix src scripts tests
	$(PYTHON) -m black src scripts tests

typecheck:
	$(PYTHON) -m mypy src/foveamap/io src/foveamap/grid

render:
	$(PYTHON) -m foveamap.cli render --mode oracle --sequence 08 --frames 0 1 2 $(ARGS)

benchmark:
	$(PYTHON) -m foveamap.eval.report --benchmark $(ARGS)

report:
	$(PYTHON) -m foveamap.eval.report $(ARGS)

demo:
	@echo "make demo is implemented in Phase 14 (backend + dashboard)."; exit 2

video:
	$(PYTHON) -m foveamap.viz.video $(ARGS)

ci: lint typecheck test-fast
	@echo "report --check joins ci in Phase 12 (T12.9)."

fresh-clone:
	bash scripts/fresh_clone_test.sh

clean:
	rm -rf build .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage src/dashboard/dist
	find . -name __pycache__ -type d -not -path "./venv/*" -not -path "./data/*" -prune -exec rm -rf {} +
