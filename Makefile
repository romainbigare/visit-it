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

# --- Phase 1 -----------------------------------------------------------------
.PHONY: holdout run run-dev score score-plan batch annotate console viewer viewer-build sheets

holdout:                       ## freeze the dev/holdout split (once, then verify)
	$(PY) -m eval.holdout freeze || $(PY) -m eval.holdout verify

run:                           ## one listing end to end: make run LISTING=87977241
	$(PY) -m pipeline run $(LISTING) --profile $(or $(PROFILE),standard)

run-dev:                       ## the whole dev split
	$(PY) -m pipeline run --all --split dev

score:                         ## M1-M5 + the G1 criteria on the dev split
	$(PY) -m eval.harness --split dev

score-plan:                    ## the plan channel on its own (isolates C from B)
	$(PY) -m eval.harness --split dev --channel plan

batch:                         ## reprocess + score + regression check (nightly)
	$(PY) -m eval.batch --split dev --check

annotate:                      ## seed annotations from each plan's printed text
	$(PY) -m eval.annotations derive && $(PY) -m eval.annotations status

sheets:                        ## contact sheets for every listing that has run
	$(PY) -m tools.build_sheets

console:                       ## the review console
	$(PY) -m services.review.server --port 8080

viewer:                        ## the viewer, against the hand-authored fixture
	$(PY) -m tools.export_scene fixture && cd viewer && npm install && npm run dev

viewer-build:
	$(PY) -m tools.export_scene export --all && cd viewer && npm run build

clean-cache:
	rm -rf $(VISITIT_DATA_HOME)/_archives

# --- GPU validation on Modal (module mode is required: relative imports) ---
.PHONY: gpu-upload gpu-deploy gpu-all
gpu-upload:
	$(PY) -m eval.models.grouping
	modal run -m modal_app.gpu_validate::upload

gpu-deploy:
	modal deploy -m modal_app.web

gpu-all:
	modal run -m modal_app.gpu_validate::run_all
