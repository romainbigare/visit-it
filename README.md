# visit-it

Automatic 3D reconstruction of flat listings: an unlabelled bag of estate-agent photos + (sometimes) a floor plan in → a scaled architectural shell with photorealistic per-room Gaussian splats, navigated by waypoints in the browser (Three.js + Spark).

## Documents

| Document | What it is |
|---|---|
| [`flat-3d-reconstruction-feasibility.html`](flat-3d-reconstruction-feasibility.html) | The feasibility report: what's possible from uncontrolled listing photos, stage-by-stage state of the art, costs, law. **Reviewed and corrected 19 Aug 2026** — inline corrections are marked `✓ VERIFIED (review, 19 Aug 2026)` and summarised in the report's Review Addendum (§12). |
| [`ROADMAP.md`](ROADMAP.md) | The full development roadmap: phases P0–P3, sprint-by-sprint plans, gates G0–G3 with kill criteria, workstream parallelisation, staffing variants, risk register. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Decisions of record (AD-1…AD-18): product shape, pipeline DAG and artifact contracts, model choices, viewer stack, service profiles. |
| [`docs/LICENSING.md`](docs/LICENSING.md) | Availability record for every model, dataset and library — what is open, what needs an application, what checkpoint to pin, and what breaks when you try to fetch it. |
| [`docs/VARIANTS.md`](docs/VARIANTS.md) | Service profiles (instant / standard / premium), unit economics at different price points (incl. the 5–10 s / £0.05 target), and three alternative development journeys with a recommendation. |
| [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md) | Where listings, photos and floor plans come from: portal APIs (mostly closed), public datasets, the in-house scraping decision, and what the first scraper run actually measured. |
| [`docs/PHASE-1-REPORT.md`](docs/PHASE-1-REPORT.md) | **Start here for status** — what Phase 1 built, what it measures, the honest G1 assessment, and what to do next. |
| [`docs/PHASE-0-REPORT.md`](docs/PHASE-0-REPORT.md) | Phase 0: the de-risking spikes, the model validation numbers, and the G0 assessment. |
| [`docs/FAILURE-TAXONOMY.md`](docs/FAILURE-TAXONOMY.md) | Every failure mode seen twice, the QA flag that detects it, and whether it is fixed. The top of this list is what the next sprint burns down. |
| [`docs/runbooks/phase1-pipeline.md`](docs/runbooks/phase1-pipeline.md) | How to run the pipeline, score it, and debug a listing that came out wrong. |
| [`eval/results/VALIDATION-REPORT.md`](eval/results/VALIDATION-REPORT.md) | **Model validation results** — what actually runs, measured on the real UK golden set, with figures. |
| [`docs/PRIOR-ART.md`](docs/PRIOR-ART.md) | Rent3D, Plan2Scene and the research line that attacked this problem in 2015–2021: what they built, why it stalled, why every commercial player controls capture instead, and what that implies for us. |

## The idea in four sentences

Photos alone cannot tell you where rooms sit relative to each other — the floor plan is the spine of the product, and listings without one get a visibly-marked *inferred* arrangement instead. Per-room geometry comes from pointmap foundation models (MapAnything, `-apache` checkpoint), appearance from depth-regularised Gaussian splatting (gsplat) culled against the room's layout polygon, and metric scale from a single global solve anchored to the stated floor area and to the dimensions printed on the plan. Navigation is teleport-between-waypoints because that is the format sparse listing photos can actually support. Every surface carries a provenance tag — photographed, reconstructed, inferred — so that when a room looks wrong you can tell whether you are looking at something real.

## Quick start

```bash
make setup      # deps (torch CPU) + tesseract
make vendor     # MoGe-2 source (not on PyPI)
make data       # auto-fetchable datasets -> $VISITIT_DATA_HOME
make golden     # rebuild the 30-listing UK golden set
make holdout    # freeze (or verify) the dev/holdout split
```

Then run the pipeline:

```bash
python -m pipeline run 87977241        # one listing, stages 0-9, ~100 s on 4 CPU cores
python -m pipeline show 87977241       # what it produced, with its QA flags
make score                             # M1-M5 and the G1 criteria on the dev split
make console                           # review console: queue, contact sheets, fix actions
make viewer                            # the shell walkthrough in a browser
```

Datasets are cached under `$VISITIT_DATA_HOME` (default `~/.cache/visit-it/datasets`), checksummed into `datasets.lock.json`, and resume if interrupted — `make status` reports what a given box has. Pipeline artifacts live under `data/runs/` (or `$VISITIT_RUN_HOME`), content-addressed and versioned, so `--from 6` re-runs the cheap end without touching the expensive one.

## The pipeline

Ten stages, each emitting one schema-validated artifact with a confidence and a list of QA flags. Streams B (photos) and C (the plan) meet only at stage 6.

| | stage | reads | produces |
|---|---|---|---|
| 0 | triage | the listing | `manifest.json` — what each image is, what the listing says about itself |
| 1 | conditioning | 0 | `calibration.json` — the field-of-view prior and the rectification gate |
| 2 | grouping | 0 | `groups.json` — photos grouped into rooms |
| 3 | geometry | 1, 2 | per-room point maps (MoGe-2; MapAnything in Phase 2) |
| 4 | layout | 3 | room polygons, ceiling heights, apertures |
| 5 | plan | 0 | `plan.json` — **the spine**: room polygons, adjacency, doors, metric scale |
| 6 | assembly | 4, 5 | `assembly.json` — which reconstructed room goes in which plan polygon |
| 7 | scale | 4, 5, 6 | `scale.json` — one global scalar, and the three §0b checks |
| 8 | shell | 5, 6, 7 | the glTF shell, apertures cut, provenance per face |
| 9 | package | 8 | `scene.json` — the viewer's only input |

## Status

**Phase 1 built; gate G1 not passed.** All ten stages run end to end on CPU — 30/30
golden listings processed, 22 reaching a walkable glTF shell in ~88 s each. The
viewer, review console, eval harness and batch runner are live.

On the frozen holdout, **one of five G1 criteria passes**. Self-consistency comes
in at a median 14.3% against a ≤10% bar (9.0% on the dev split — the gap between
the two is itself the finding). The failures concentrate almost entirely in the
plan channel: colour-filled plans, open-plan spaces split into several polygons,
and small unlabelled rooms that never become polygons at all. Assembly — the stage
the roadmap front-loaded because it feared it most — is not the bottleneck.

The G1 kill criterion does not fire, and the pivot rule is not needed: the shell
exists and is usable. See [`docs/PHASE-1-REPORT.md`](docs/PHASE-1-REPORT.md).

Next: **train the plan vectoriser** (ROADMAP S3's original plan, deferred in
Phase 1 to unblock everything downstream). Every failing criterion is downstream of
that one component.

**Scope:** internal and non-commercial. Correctness is judged by plausibility, self-consistency against dimensions printed on the floor plan, and cross-model agreement — not by measuring flats. See [`ROADMAP.md`](ROADMAP.md) §0b.
