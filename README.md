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
| [`docs/PLAN-READING-REPORT.md`](docs/PLAN-READING-REPORT.md) | Why the plan reading was bad, what already exists to fix it (survey of downloadable models), what we tested, and the pretrained wall segmenter now shipped in stage 5 — including the regression it introduced. |
| [`notebooks/plan_reading_modal.ipynb`](notebooks/plan_reading_modal.ipynb) | **Modal GPU: everything that can be tuned about the room-finder, one change at a time** — all five published checkpoints, the picture it is shown, the plan in pieces, four views merged, its answer tidied, and finally its corners pulled onto our wall map. Every step draws all 25 plans before and after and then stops at a `True`/`False` cell: **you** keep or drop it, and the next step builds on what you kept. Ends in a table, a chart and a per-plan comparison. |
| [`notebooks/finetune_wallnet_colab.ipynb`](notebooks/finetune_wallnet_colab.ipynb) | Colab: teach the wall model our own plans, using corrections made with `tools/annotate_walls.py`. Only worth running if the notebook above runs out of road. |
| [`docs/PHASE-1-FIXES.md`](docs/PHASE-1-FIXES.md) | The prioritised fix list coming out of Phase 1 — what to change, why, and what each costs. Scheduled as ROADMAP Sprint 6b. |
| [`docs/FAILURE-TAXONOMY.md`](docs/FAILURE-TAXONOMY.md) | Every failure mode seen twice, the QA flag that detects it, and whether it is fixed. The top of this list is what the next sprint burns down. |
| [`docs/runbooks/phase1-pipeline.md`](docs/runbooks/phase1-pipeline.md) | How to run the pipeline, score it, and debug a listing that came out wrong. |
| [`eval/results/VALIDATION-REPORT.md`](eval/results/VALIDATION-REPORT.md) | **Model validation results** — what actually runs, measured on the real UK golden set, with figures. |
| [`docs/PRIOR-ART.md`](docs/PRIOR-ART.md) | Rent3D, Plan2Scene and the research line that attacked this problem in 2015–2021: what they built, why it stalled, why every commercial player controls capture instead, and what that implies for us. |

## The idea in four sentences

Photos alone cannot tell you where rooms sit relative to each other — the floor plan is the spine of the product, and listings without one get a visibly-marked *inferred* arrangement instead. Per-room geometry comes from pointmap foundation models (MapAnything, `-apache` checkpoint), appearance from depth-regularised Gaussian splatting (gsplat) culled against the room's layout polygon, and metric scale from a single global solve anchored to the stated floor area and to the dimensions printed on the plan. Navigation is teleport-between-waypoints because that is the format sparse listing photos can actually support. Every surface carries a provenance tag — photographed, reconstructed, inferred — so that when a room looks wrong you can tell whether you are looking at something real.

## Try it yourself

Everything below runs on a laptop with no GPU. Budget ~15 minutes of setup, most
of which is downloads.

### 1. Set up (once)

```bash
make setup      # python deps + tesseract
make vendor     # MoGe-2 source (it isn't on PyPI)
python -m pipeline.ingest.fetch_media --set data/golden/golden_set.json   # ~87 MB of listing photos
make doctor     # tells you if anything is still missing, and what to run
```

`make doctor` is the one to run when something doesn't work — it checks all six
things that usually go wrong and prints the fix next to each.

The MoGe-2 weights (~1.3 GB) download themselves the first time you run the
pipeline, into `$HF_HOME`.

### 2. Run one listing

```bash
python -m pipeline run 87977241
```

About 90 seconds on four CPU cores, ~80 of which is stage 3 (a GPU makes that
40× faster). You should see all ten stages report `o`:

```
OK  87977241  101.7s  0:o 1:o 2:o 3:o 4:o 5:o 6:o 7:o 8:o 9:o  flags=...
```

Then look at what it made:

```bash
python -m pipeline show 87977241     # every artifact, its confidence and QA flags
```

### 3. Walk through the flat

```bash
python -m tools.export_scene export --all
cd viewer && npm install && npm run build && npm run preview
```

Open the URL it prints. The dropdown top-right switches listings. Drag to look
around, click a room on the minimap to teleport, press **d** for the dollhouse
view. Add `?dev=1` to the URL to see per-room areas, confidences and QA flags.

If you'd rather not run the pipeline first, `make viewer` starts it against a
hand-authored example flat with no pipeline involved at all.

### 4. Look at what went wrong

```bash
make console        # http://127.0.0.1:8080
```

The queue is ordered by how likely a person is to be needed. Click a listing to
get its contact sheet — every stage's output on one page, which is the fastest way
to see *why* a reconstruction is bad rather than *that* it is. The same page lets
you relabel a mis-read room or drag a room onto a different plan polygon; both
re-run in seconds.

### 5. Reproduce the numbers

```bash
make test                                  # 77 unit tests, ~3 s
python -m tests.perf_budgets               # per-stage latency budgets
python -m eval.holdout verify              # the frozen split is still sealed

python -m pipeline run --all --split dev   # ~15 min: the 10 development listings
make score                                 # M1-M5 and the G1 criteria
python -m eval.phase1_summary              # every table in the Phase 1 report
```

To reproduce the gate measurement in [`docs/PHASE-1-REPORT.md`](docs/PHASE-1-REPORT.md),
run `--all` (all 30 listings, ~45 min) then `python -m eval.harness --split holdout`.

### Useful flags

| flag | what it does |
|---|---|
| `--from 6` | resume from a stage, reusing what's on disk — skips the 90 s geometry pass |
| `--only 5-plan` | run exactly one stage |
| `--max-rooms 4` | cap rooms reconstructed, for a quick look |
| `--no-triage-model` | skip the image classifier if you don't want the 800 MB download |
| `--profile instant` | the fast-path bindings and budgets |

Datasets cache under `$VISITIT_DATA_HOME` (default `~/.cache/visit-it/datasets`),
checksummed into `datasets.lock.json`. Pipeline artifacts live under `data/runs/`
(or `$VISITIT_RUN_HOME`), content-addressed and versioned.

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

Next, both scheduled as ROADMAP **Sprint 6b** (Amendment B): **build the shell from
plan polygons rather than photo-derived ones** (measured: photo rooms are 31% too
big and leave holes), and **train the plan vectoriser** — S3's original plan,
deferred in Phase 1 to unblock everything downstream. Every failing criterion is
downstream of those two. See [`docs/PHASE-1-FIXES.md`](docs/PHASE-1-FIXES.md).

**Scope:** internal and non-commercial. Correctness is judged by plausibility, self-consistency against dimensions printed on the floor plan, and cross-model agreement — not by measuring flats. See [`ROADMAP.md`](ROADMAP.md) §0b.
