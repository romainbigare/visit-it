# One-command setup for a fresh environment. We move boxes often; everything
# below is idempotent and safe to re-run.
#
#   make setup     install python deps + tesseract
#   make vendor    pull MoGe-2 source (not on PyPI)
#   make data      download the auto-fetchable datasets
#   make golden    rebuild the 30-listing UK golden set from Rightmove
#   make validate  run the model validation suite and regenerate figures
#   make test      unit tests
#   make all       setup + vendor + data

VISITIT_DATA_HOME ?= $(HOME)/.cache/visit-it/datasets
export VISITIT_DATA_HOME
export HF_HOME ?= $(HOME)/.cache/hf
PY ?= python3

.PHONY: all setup vendor data data-full golden media validate test clean-cache status

all: setup vendor data

setup:
	$(PY) -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install --quiet -r requirements.txt
	@which tesseract >/dev/null 2>&1 || (echo ">> installing tesseract"; apt-get update -qq && apt-get install -y -qq tesseract-ocr)
	@echo "setup done"

vendor:
	$(PY) tools/vendor_moge.py --dest vendor/moge

data:
	$(PY) -m pipeline.datasets bootstrap

data-full:
	$(PY) -m pipeline.datasets bootstrap --full

status:
	$(PY) -m pipeline.datasets status
	@$(PY) -m pipeline.datasets verify || true

golden:
	$(PY) -m pipeline.ingest.collect --target 30 --min-floorplan-frac 0.65 \
		--regions london,manchester,birmingham,leeds,glasgow,bristol \
		--max-pages 3 --out-dir data/golden

media:
	$(PY) -m pipeline.ingest.fetch_media --set data/golden/golden_set.json

validate:
	$(PY) -m eval.models.triage
	$(PY) -m eval.models.plan_ocr
	$(PY) -m eval.models.geometry --limit 40
	PYTHONPATH=vendor/moge $(PY) -m eval.models.moge_geometry --limit 20
	PYTHONPATH=vendor/moge $(PY) -m eval.figures

test:
	$(PY) -m unittest discover -s tests -q

clean-cache:
	rm -rf $(VISITIT_DATA_HOME)/_archives
