# Architecture — decisions of record

**Project:** visit-it — automatic 3D reconstruction of flat listings from unlabelled photos + optional floor plan, delivered in the browser.
**Status:** accepted baseline, 19 August 2026. Each decision below is an ADR; revisit only through a written amendment (add a dated note, never edit history).
**Companion documents:** [`ROADMAP.md`](../ROADMAP.md) (phases, sprints, gates), [`LICENSING.md`](LICENSING.md) (model/package ledger), `flat-3d-reconstruction-feasibility.html` (the feasibility report this architecture implements).

---

## 1. The decisions, in one table

| # | Decision | Choice | Rejected alternatives |
|---|----------|--------|----------------------|
| AD-1 | Product shape | Hybrid **shell + splats**, waypoint navigation, two honesty tiers | Free-roam mesh; splats-only; NeRF |
| AD-2 | Global structure | **Floor-plan-first spine**, photos attach to it | Photo-first with plan as bonus |
| AD-3 | Pipeline | Staged DAG, typed JSON artifacts, content-addressed store, every stage re-runnable & scorable | Monolithic end-to-end model; ad-hoc scripts |
| AD-4 | Multi-view geometry | **MapAnything (Apache weights)** per room; COLMAP/GLOMAP fallback for photo-rich rooms | MASt3R/DUSt3R (NC licence); VGGT default checkpoint (NC) |
| AD-5 | Monocular fallback | **MoGe-2** (MIT, metric + intrinsics); Depth Anything V2 *Small* as cross-check | Depth Pro (Apple custom personal-use licence — verified 19 Aug 2026, not shippable); UniDepth (CC-BY-NC) |
| AD-6 | Appearance | **gsplat** (Apache), depth-regularised, culled against the room layout polygon; SPZ/SOGS delivery | Inria 3DGS reference (research-only licence); mesh-only texturing |
| AD-7 | Metric scale | **One global scale scalar**, weighted least squares over Carrez area + door heights + ceiling priors + metric depth | Trusting metric depth per-stage; per-room scales |
| AD-8 | Assembly | Hungarian assignment to plan polygons + joint SE(2) refinement; explicit `method` + confidence on the result | Learned end-to-end placement; VLM-guessed layout |
| AD-9 | Viewer | **Three.js + Spark** hybrid renderer, waypoint teleport, dollhouse, minimap, honesty shading | Free roam; PlayCanvas; Unity WebGL |
| AD-10 | VLM usage | Frontier VLM API for **semantics and adjudication only** (triage, room labels, same-room checks); never for metric quantities | VLM-estimated dimensions/layout |
| AD-11 | Licensing policy | Prod allowlist: Apache-2.0 / MIT / BSD only. NC-licensed code lives in `research/`, never imported by `pipeline/` | "Sort it out before launch" |
| AD-12 | Data acquisition | Agency/portal **data partnership**; no scraping | Scraping portals (ToS + litigation risk, see report §10) |
| AD-13 | Backend | Python 3.11 + PyTorch monorepo; FastAPI control plane; Redis-backed GPU worker queue; Postgres metadata; S3-compatible artifact store | Kubeflow/Temporal/Airflow (overkill at this stage) |
| AD-14 | Human review | Review console is a product component from Phase 1; every artifact inspectable; corrections feed the eval set | Review as an afterthought |
| AD-15 | Provenance & compliance | Every surface tagged `photographed / reconstructed / inferred / generated` from day one; GDPR blur pre-processing; no generative content in v1 | Retro-fitting provenance later |

The rest of this document expands the non-obvious ones.

---

## 2. AD-1 / AD-2 — Product shape and the floor-plan spine

The feasibility report's central finding stands: photographs contain **zero** signal about inter-room placement, and ~60% of every room is never photographed. The product is therefore an *inference product with a reconstruction component*, and its architecture must make the honesty boundary a first-class concept:

- **Tier A — "Verified layout"**: listing has a floor plan. Rooms are assigned to plan polygons; the arrangement is real; area is anchored to the published (Carrez) figure. This is the flagship output.
- **Tier B — "Inferred layout"**: no plan. Individually reconstructed rooms in a synthesised arrangement, visually marked as inferred (desaturated connective tissue, explicit label, optional 30-second drag-and-drop arrangement step for the agent).

Waypoint navigation is load-bearing, not cosmetic: the scene only has to look right from the waypoints and sightlines between them, which is exactly what sparse listing photos can support. This decision propagates backwards — splat optimisation targets waypoint views, and the culling volume comes from the layout polygon.

## 3. AD-3 — Pipeline as a staged DAG with typed artifacts

Ten stages (0–9), exactly as in report §5. Non-negotiable properties:

1. **Typed artifacts.** Every stage consumes and emits JSON-schema-validated artifacts (+ binary blobs referenced by content hash). Schemas live in `schemas/`, versioned; a stage declares which schema versions it reads/writes.
2. **Idempotence & partial re-runs.** Re-running stage 6 on a listing must not require re-running stages 0–5. Artifacts are immutable; re-runs create new versions.
3. **Confidence everywhere.** Every stage output carries a confidence and machine-readable QA flags (`mirror_suspected`, `perspective_corrected`, `scale_constraints_disagree`, …). These drive routing to the review console and survive into the viewer.
4. **Inspectability.** The review console renders any artifact for any listing at any stage. When a listing looks wrong, you look at the stage boundary where it went wrong.

### Artifact contracts (v0 sketch — formalised in `schemas/` during Sprint 1)

| Stage | Artifact | Key contents |
|-------|----------|--------------|
| 0 Triage | `manifest.json` | per-image: `{type, room_label, quality_flags[], phash, confidence}`; listing: `{advertised_area_m2, room_count, floor, source_text_refs}` |
| 1 Conditioning | `conditioned/` + `calibration.json` | undistorted images, per-image intrinsics (GeoCalib + pointmap-model estimate + agreement score), photometric alignment report |
| 2 Grouping | `groups.json` | room groups, pairwise match scores, VLM adjudication log, singletons |
| 3 Per-room geometry | `rooms/<id>/geometry.npz` + `poses.json` | camera poses, point maps, per-point confidence, engine used (`mapanything\|monocular\|colmap`) |
| 4 Structure | `rooms/<id>/layout.json` | floor/ceiling/wall planes, room polygon (2D + height), door/window apertures, regularisation report, `approximate: bool` |
| 5 Plan channel | `plan.json` | vector room polygons, adjacency graph, door positions, OCR'd `{label, area_m2}` per room, plan-pixel→metre scale candidates |
| 6 Assembly | `assembly.json` | room→polygon assignment with per-match cost breakdown, per-room SE(2) pose, `method: plan\|doorway_chain\|inferred`, global confidence |
| 7 Scale | `scale.json` | scale factor, per-constraint residuals (Carrez, doors, ceilings, depth prior), quality score |
| 8 Appearance | `rooms/<id>/splats.spz` (+ SOGS), `shell.glb` | per-room splats (culled), shell mesh with per-face provenance tags |
| 9 Packaging | `scene.json` + CDN payload | waypoint graph, room streaming manifest, provenance masks, tour metadata |

### Orchestration

Deliberately boring: **FastAPI** control plane, **Celery + Redis** for CPU/GPU work queues (separate queues per resource class), **Postgres** for listing/artifact metadata and review state, **S3-compatible** object store (MinIO in dev) for artifacts, content-addressed. One `pipeline run <listing> [--from stage]` CLI drives the same code path locally and in workers. Observability: structured logs per stage + a per-listing HTML "contact sheet" (thumbnails of every intermediate) generated automatically — this is the single most valuable debugging artifact.

If volume later justifies it, swap Celery for something heavier behind the same stage interface. Do not start there.

## 4. AD-4 / AD-5 — Geometry engines

- **Primary:** MapAnything, Apache-licensed checkpoint, batches of 3–8 images per room. It regresses **metric** geometry and accepts optional priors (intrinsics), which feeds the scale solve.
- **Monocular fallback** (1-photo rooms — bathrooms, hallways): MoGe-2 primary (MIT; metric point map + intrinsics from one image); Depth Anything V2 *Small* (Apache) as a cheap cross-check. Disagreement between the two lowers the room's confidence. (Depth Pro was rejected on licence verification — Apple custom personal-use terms; see `LICENSING.md`.)
- **Classical fallback:** COLMAP/GLOMAP path for the rare listing with dense overlapping coverage (new-build marketing shoots) — classical SfM beats pointmap models when overlap is good.
- **Benchmark-only:** MASt3R/DUSt3R/VGGT-default stay in `research/` under the licence firewall (AD-11).
- **Engine abstraction:** stage 3 exposes one interface (`reconstruct(images, priors) -> {poses, pointmap, conf}`); engines are plugins. This is cheap insurance in the fastest-moving corner of the stack — expect to swap models within 12 months.

## 5. AD-7 — The scale solve (the report's §4.8, made concrete)

One scalar `s` solved by weighted least squares over:

| Constraint | Source | Prior / weight rationale |
|-----------|--------|--------------------------|
| Total floor area | Carrez figure from listing text and/or plan OCR | Strongest when present (legally measured, ±5% tolerance in law) |
| Door leaf heights | Door detections (stage 4 apertures) | ~2.04 m EU standard, per-door observation |
| Ceiling heights | Layout polygons | 2.50–2.70 m modern / higher haussmannien — wide prior, plausibility check |
| Fixture dimensions | Counters ~0.90 m, switches ~1.10–1.30 m | Weak, opportunistic |
| Metric depth | Stage 3 metric output | Honest ±10–15% uncertainty |

Residuals are the listing's quality score; large disagreement ⇒ QA flag ⇒ review queue, never silent shipping. **The viewer never displays a surface area other than the advertised legal figure** (the model is scaled *to* it; we do not publish a competing measurement — this is a legal posture, not just engineering).

## 6. AD-9 — Viewer

- **Stack:** TypeScript, Three.js, Spark for hybrid splat+mesh rendering (SPZ/SOGS input), glTF + Draco + KTX2 shell. (Fallback if Spark's licence disappoints on verification: three.js' native WebGPU splat support / `@playcanvas` tooling — decision recorded in `LICENSING.md`.)
- **Navigation:** teleport between waypoints generated from camera poses + doorway midpoints; dollhouse overview; floor-plan minimap synced to position.
- **Budgets (hard, CI-enforced):** 200–500 K splats per room; one room's splats GPU-resident at a time (adjacent room prefetched); shell ≤ 1 MB; first-room-interactive ≤ 5 s on 4G mid-range mobile; 30 fps on the reference device matrix (defined in Sprint 1; includes an iPhone with its ~100–200 MB Safari memory ceiling).
- **Honesty rendering:** provenance tags per surface drive a visual treatment — photographed surfaces full-colour, inferred fill desaturated/hazed, plus a plain-language legend. This is a compliance feature (report §10) as much as UX.

## 7. AD-11 — The licence firewall

The best models in this field ship non-commercial. Policy:

1. `docs/LICENSING.md` is the ledger: every model/dataset/library → code licence, weights licence, training-data caveats, date checked, verdict (`PROD_OK / RESEARCH_ONLY / CHECK`).
2. Prod images are built from `pipeline/` + `services/` + `viewer/` only. `research/` is excluded from builds and from prod dependency manifests; CI fails if anything under `pipeline/` imports from `research/` or pulls a dependency whose ledger verdict isn't `PROD_OK`.
3. Ledger re-verification is a release-checklist item (this field re-licenses frequently — VGGT gained a commercial checkpoint post-release; DINOv2 went Apache post-release).
4. Datasets: ZInD, Structured3D, CubiCasa5K are all encumbered for commercial use (see ledger) — they inform research; **the golden set we build ourselves is the product's evaluation basis**, and partner-plan annotations are the fine-tuning basis.

## 8. AD-13 — Repository layout

```
visit-it/
├── ROADMAP.md                  # phases, sprints, gates (companion to this doc)
├── docs/
│   ├── ARCHITECTURE.md         # this file
│   ├── LICENSING.md            # the ledger (AD-11)
│   ├── spikes/                 # dated de-risk spike write-ups
│   └── runbooks/
├── schemas/                    # versioned JSON schemas for all artifacts
├── pipeline/                   # stages 0–9, one package each, pure + typed
│   ├── triage/ conditioning/ grouping/ room_geometry/ layout/
│   ├── floorplan/ assembly/ scale/ appearance/ packaging/
│   └── core/                   # artifact IO, stage runner, confidence model
├── eval/                       # golden set tooling, metrics M1–M11, harness, scoreboard
├── services/
│   ├── api/                    # FastAPI control plane
│   ├── workers/                # Celery GPU/CPU workers
│   └── review/                 # review console (web)
├── viewer/                     # Three.js/TS walkthrough viewer
├── research/                   # NC-licensed experiments — firewall, never shipped
└── infra/                      # IaC, docker, CI
```

## 9. Evaluation model (referenced by every gate in ROADMAP.md)

| ID | Metric | Definition |
|----|--------|-----------|
| M1 | Total area error | \|model area − ground truth\| / ground truth, % |
| M2 | Per-room area error | median per-room %, matched rooms |
| M3 | Adjacency accuracy | correct room-adjacency edges / GT edges |
| M4 | Layout IoU | 2D IoU of assembled footprint vs GT plan |
| M5 | Assignment accuracy | photos placed in the correct plan room, % |
| M6 | Grouping quality | pairwise same-room precision/recall/F1 |
| M7 | Triage accuracy | image type + room label top-1 |
| M8 | Render quality | LPIPS/PSNR on held-out photo views + blind-panel rubric (1–5) |
| M9 | Yield | listings shippable with zero human touch, % |
| M10 | Unit economics | GPU-minutes, API €, wall-clock per listing |
| M11 | Review cost | median human minutes per reviewed listing |

Harness rules: frozen holdout split untouched by development; nightly run on the dev split; every PR that touches `pipeline/` posts its eval delta; gates are measured on the holdout only.

## 10. Security, privacy, compliance hooks (built in, not bolted on)

- **GDPR:** blur pass (faces, family photos, documents, screens) runs in stage 1 before anything is stored long-term; we are processor for partner imagery — DPA template needed before first real data (Sprint 1 legal task).
- **EU AI Act Art. 50 (in force for these obligations since 2 Aug 2026):** provenance tags (AD-15) give us surface-level machine-readable marking if/when any generated content ships; v1 ships none.
- **Never-rules (product invariants, testable):** never generate content that changes perceived size, layout or condition; never display an area figure other than the advertised one; never present Tier-B arrangement as measured.
