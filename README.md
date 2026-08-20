# visit-it

Automatic 3D reconstruction of flat listings: an unlabelled bag of estate-agent photos + (sometimes) a floor plan in → a scaled architectural shell with photorealistic per-room Gaussian splats, navigated by waypoints in the browser (Three.js + Spark).

## Documents

| Document | What it is |
|---|---|
| [`flat-3d-reconstruction-feasibility.html`](flat-3d-reconstruction-feasibility.html) | The feasibility report: what's possible from uncontrolled listing photos, stage-by-stage state of the art, costs, law. **Reviewed and corrected 19 Aug 2026** — inline corrections are marked `✓ VERIFIED (review, 19 Aug 2026)` and summarised in the report's Review Addendum (§12). |
| [`ROADMAP.md`](ROADMAP.md) | The full development roadmap: phases P0–P3, sprint-by-sprint plans, gates G0–G3 with kill criteria, workstream parallelisation, staffing variants, risk register. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Decisions of record (AD-1…AD-15): product shape, pipeline DAG and artifact contracts, model choices, viewer stack, compliance posture. |
| [`docs/LICENSING.md`](docs/LICENSING.md) | The licence ledger — every model/dataset/library with code licence, weights licence, and a PROD_OK / RESEARCH_ONLY verdict, verified against primary sources 2026-08-19. CI enforces it. |
| [`docs/VARIANTS.md`](docs/VARIANTS.md) | Service profiles (instant / standard / premium), unit economics at different price points (incl. the 5–10 s / £0.05 target), and three alternative development journeys with a recommendation. |
| [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md) | Where listings, photos and floor plans come from: portal APIs (mostly closed), commercially-licensed datasets, the estate-agency CRM route, the in-house scraping decision, and who owns listing photograph copyright. |
| [`eval/results/VALIDATION-REPORT.md`](eval/results/VALIDATION-REPORT.md) | **Model validation results** — what actually runs, measured on the real UK golden set, with figures. |
| [`docs/PRIOR-ART.md`](docs/PRIOR-ART.md) | Rent3D, Plan2Scene and the research line that attacked this problem in 2015–2021: what they built, why it stalled, why every commercial player controls capture instead, and what that implies for us. |

## The idea in four sentences

Photos alone cannot tell you where rooms sit relative to each other — the floor plan is the spine of the product, and listings without one get a visibly-marked *inferred* arrangement instead. Per-room geometry comes from pointmap foundation models (MapAnything, Apache checkpoint), appearance from depth-regularised Gaussian splatting (gsplat) culled against the room's layout polygon, and metric scale from a single global solve anchored to the legally-published surface area. Navigation is teleport-between-waypoints because that is the format sparse listing photos can actually support. Every surface carries a provenance tag — photographed, reconstructed, inferred — and the viewer renders the difference honestly.

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

Phase 0 in progress: golden set collected, models validated (see the validation report). Next step: Sprint 1 (see [`ROADMAP.md`](ROADMAP.md) §9, kickoff checklist) — golden-set collection, agent outreach via the CRM route, the derivative-works rights grant, and the de-risking spikes.
