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
| [`docs/PHASE-0-REPORT.md`](docs/PHASE-0-REPORT.md) | **Start here for status** — what Phase 0 did, what the numbers said, the honest G0 assessment, and what to do next. |
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
make validate   # run model validation, regenerate figures
```

Datasets are cached under `$VISITIT_DATA_HOME` (default `~/.cache/visit-it/datasets`), checksummed into `datasets.lock.json`, and resume if interrupted — `make status` reports what a given box has.

## Status

**Phase 0 complete, gate G0 passed.** Every stage from triage to Gaussian splatting has been run on real UK listings and measured: triage F1 0.96, MapAnything 1.35 s per group, gsplat train-view PSNR 25.9 dB, **$0.009–0.021 of GPU per listing**. Two items carry into Sprint 1 — freezing the holdout split, and SPZ export plus a viewer test. See [`docs/PHASE-0-REPORT.md`](docs/PHASE-0-REPORT.md).

Next: Sprint 1 (see [`ROADMAP.md`](ROADMAP.md) §9) — freeze the holdout, build stage 4 (the room polygon), and wire plan OCR into the scale solve.

**Scope:** internal and non-commercial. Correctness is judged by plausibility, self-consistency against dimensions printed on the floor plan, and cross-model agreement — not by measuring flats. See [`ROADMAP.md`](ROADMAP.md) §0b.
