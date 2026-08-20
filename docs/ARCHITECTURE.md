# Architecture — decisions of record

**Project:** visit-it — automatic 3D reconstruction of flat listings from unlabelled photos + optional floor plan, delivered in the browser.
**Status:** accepted baseline, 19 August 2026. Each decision below is an ADR; revisit only through a written amendment (add a dated note, never edit history).
> **Scope note (20 Aug 2026): internal, non-commercial.** Decisions that existed
> to make this commercially shippable — the licence firewall, the data-partnership
> route, GDPR process and the disclosure never-rules — are relaxed below. What
> survives is what makes the system *work* and *debuggable*, not what made it
> sellable.

**Companion documents:** [`ROADMAP.md`](../ROADMAP.md) (phases, sprints, gates), [`LICENSING.md`](LICENSING.md) (model/package ledger), `flat-3d-reconstruction-feasibility.html` (the feasibility report this architecture implements).

---

## 1. The decisions, in one table

| # | Decision | Choice | Rejected alternatives |
|---|----------|--------|----------------------|
| AD-1 | Product shape | Hybrid **shell + splats**, waypoint navigation, two honesty tiers | Free-roam mesh; splats-only; NeRF |
| AD-2 | Global structure | **Floor-plan-first spine**, photos attach to it | Photo-first with plan as bonus |
| AD-3 | Pipeline | Staged DAG, typed JSON artifacts, content-addressed store, every stage re-runnable & scorable | Monolithic end-to-end model; ad-hoc scripts |
| AD-4 | Multi-view geometry | **MapAnything** per room; COLMAP/GLOMAP fallback for photo-rich rooms | MASt3R/DUSt3R, VGGT — now equally usable; MapAnything chosen on merit (validated 20 Aug 2026: 12/12 groups, 1.35 s each) |
| AD-5 | Monocular fallback | **MoGe-2** — chosen because it predicts camera intrinsics, which Phase 0 proved is not optional | Depth-only models: estimated ceiling height scales with the assumed lens (2.76 m at 60°, 5.93 m at 104°) |
| AD-6 | Appearance | **gsplat**, depth-regularised, culled against the room layout polygon; SPZ/SOG delivery | Mesh-only texturing |
| AD-7 | Metric scale | **One global scale scalar**, weighted least squares over floor-plan printed dimensions + door heights + ceiling priors + metric depth | Trusting metric depth per-stage; per-room scales |
| AD-8 | Assembly | Hungarian assignment to plan polygons + joint SE(2) refinement; explicit `method` + confidence on the result | Learned end-to-end placement; VLM-guessed layout |
| AD-9 | Viewer | **Three.js + Spark** hybrid renderer, waypoint teleport, dollhouse, minimap, honesty shading | Free roam; PlayCanvas; Unity WebGL |
| AD-10 | VLM usage | Frontier VLM API for **semantics and adjudication only** (triage, room labels, same-room checks); never for metric quantities | VLM-estimated dimensions/layout |
| AD-11 | Model availability | Any model we can obtain and run is fair game. `docs/LICENSING.md` is kept as an *availability* record — what needs an application form, what has dead links — not as a gate | — |
| AD-12 | Data acquisition | In-house scraping (Rightmove works; Zoopla is Cloudflare-blocked). Rate-limited and robots-compliant, because that is what keeps it working | Agency partnerships (unnecessary for internal use) |
| AD-13 | Backend | Python 3.11 + PyTorch monorepo; FastAPI control plane; Redis-backed GPU worker queue; Postgres metadata; S3-compatible artifact store | Kubeflow/Temporal/Airflow (overkill at this stage) |
| AD-14 | Human review | Review console from Phase 1; every artifact inspectable; corrections feed the eval set | Review as an afterthought |
| AD-15 | Provenance for debuggability | Every surface tagged `photographed / reconstructed / inferred / generated`. Kept not for compliance but because a viewer that hides which surfaces are real is one we cannot debug | Untagged output |
| AD-16 *(added 19 Aug 2026)* | Service profiles | One DAG, three **profiles** — `instant` / `standard` / `premium` — differing only in per-stage engine bindings, budgets and SLOs (see `docs/VARIANTS.md`) | Separate pipelines per tier; one-size-fits-all latency |
| AD-17 *(added 19 Aug 2026)* | Hot-path policy | The synchronous (`instant`) path bans: frontier-VLM calls, per-scene optimisation loops, human gates, cold model loads. Self-hosted small models only, warm pools, per-room GPU fan-out | "Optimise later" |
| AD-18 *(added 19 Aug 2026)* | Reconstruction cache & serving split | Reconstructions cached by listing content hash (photos+plan bytes); viewers served purely from CDN — GPU cost is per *unique listing*, never per viewer | Recompute per request |

The rest of this document expands the non-obvious ones.

---

## 2. AD-1 / AD-2 — Product shape and the floor-plan spine

The feasibility report's central finding stands: photographs contain **zero** signal about inter-room placement, and ~60% of every room is never photographed. The product is therefore an *inference product with a reconstruction component*, and its architecture must make the honesty boundary a first-class concept:

- **Tier A — "Plan-anchored layout"**: listing has a floor plan. Rooms are assigned to plan polygons; the arrangement is real; area is anchored to the stated floor area and to any dimensions printed on the plan. This is the flagship output.
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
| 0 Triage | `manifest.json` | per-image: `{type, room_label, quality_flags[], phash, confidence, provenance}` where `provenance ∈ {scraped, self_captured}` — kept for traceability when debugging a bad listing; listing: `{advertised_area_m2, room_count, floor, source_text_refs}` |
| 1 Conditioning | `conditioned/` + `calibration.json` | undistorted images, per-image intrinsics (GeoCalib + pointmap-model estimate + agreement score), photometric alignment report |
| 2 Grouping | `groups.json` | room groups, pairwise match scores, VLM adjudication log, singletons |
| 3 Per-room geometry | `rooms/<id>/geometry.npz` + `poses.json` | camera poses, point maps, per-point confidence, engine used (`mapanything\|monocular\|colmap`) |
| 4 Structure | `rooms/<id>/layout.json` | floor/ceiling/wall planes, room polygon (2D + height), door/window apertures, regularisation report, `approximate: bool` |
| 5 Plan channel | `plan.json` | vector room polygons, adjacency graph, door positions, OCR'd `{label, area_m2}` per room, plan-pixel→metre scale candidates |
| 6 Assembly | `assembly.json` | room→polygon assignment with per-match cost breakdown, per-room SE(2) pose, `method: plan\|doorway_chain\|inferred`, global confidence |
| 7 Scale | `scale.json` | scale factor, per-constraint residuals (stated area, plan dimensions, doors, ceilings, depth prior), quality score |
| 8 Appearance | `rooms/<id>/splats.spz` (+ SOGS), `shell.glb` | per-room splats (culled), shell mesh with per-face provenance tags |
| 9 Packaging | `scene.json` + CDN payload | waypoint graph, room streaming manifest, provenance masks, tour metadata |

### Orchestration

Deliberately boring: **FastAPI** control plane, **Celery + Redis** for CPU/GPU work queues (separate queues per resource class), **Postgres** for listing/artifact metadata and review state, **S3-compatible** object store (MinIO in dev) for artifacts, content-addressed. One `pipeline run <listing> [--from stage]` CLI drives the same code path locally and in workers. Observability: structured logs per stage + a per-listing HTML "contact sheet" (thumbnails of every intermediate) generated automatically — this is the single most valuable debugging artifact.

If volume later justifies it, swap Celery for something heavier behind the same stage interface. Do not start there.

## 4. AD-4 / AD-5 — Geometry engines

- **Primary:** MapAnything, batches of 3–8 images per room. It regresses **metric** geometry and accepts optional priors (intrinsics), which feeds the scale solve.
- **Monocular fallback** (1-photo rooms — bathrooms, hallways): MoGe-2 primary (metric point map + intrinsics from one image); Depth Anything V2 *Small* as a cheap cross-check. Disagreement between the two lowers the room's confidence. Phase 0 measured what this choice is actually worth: Depth Anything alone gives ~6 m ceilings at the true 98.6° field of view, because depth without intrinsics is not geometry.
- **Classical fallback:** COLMAP/GLOMAP path for the rare listing with dense overlapping coverage (new-build marketing shoots) — classical SfM beats pointmap models when overlap is good.
- **Benchmark-only:** MASt3R/DUSt3R/VGGT-default stay in `research/` as comparison baselines, not production paths (AD-11).
- **Engine abstraction:** stage 3 exposes one interface (`reconstruct(images, priors) -> {poses, pointmap, conf}`); engines are plugins. This is cheap insurance in the fastest-moving corner of the stack — expect to swap models within 12 months.

## 5. AD-7 — The scale solve (the report's §4.8, made concrete)

One scalar `s` solved by weighted least squares over:

| Constraint | Source | Prior / weight rationale |
|-----------|--------|--------------------------|
| Total floor area | Stated floor area from listing text and/or plan OCR | Strongest when present; ~53% of listings state one |
| Door leaf heights | Door detections (stage 4 apertures) | ~2.04 m EU standard, per-door observation |
| Room dimensions printed on the plan | Plan OCR (54% of plans, 62% of high-resolution ones) | Direct per-room constraint; the main self-consistency check |
| Ceiling heights | Layout polygons | 2.50–2.70 m modern / higher in period stock — wide prior, plausibility check |
| Fixture dimensions | Counters ~0.90 m, switches ~1.10–1.30 m | Weak, opportunistic |
| Metric depth | Stage 3 metric output | Honest ±10–15% uncertainty |

Residuals are the listing's quality score; large disagreement ⇒ QA flag ⇒ review queue, never silent shipping. **The viewer never displays a surface area other than the advertised legal figure** (the model is scaled *to* it; we do not publish a competing measurement — this is a legal posture, not just engineering).

## 6. AD-9 — Viewer

- **Stack:** TypeScript, Three.js, Spark for hybrid splat+mesh rendering (SPZ/SOGS input), glTF + Draco + KTX2 shell. (Fallback if Spark disappoints on device testing: three.js' native WebGPU splat support / `@playcanvas` tooling.)
- **Navigation:** teleport between waypoints generated from camera poses + doorway midpoints; dollhouse overview; floor-plan minimap synced to position.
- **Budgets (hard, CI-enforced):** 200–500 K splats per room; one room's splats GPU-resident at a time (adjacent room prefetched); shell ≤ 1 MB; first-room-interactive ≤ 5 s on 4G mid-range mobile; 30 fps on the reference device matrix (defined in Sprint 1; includes an iPhone with its ~100–200 MB Safari memory ceiling).
- **Honesty rendering:** provenance tags per surface drive a visual treatment — photographed surfaces full-colour, inferred fill desaturated/hazed, plus a plain-language legend. This is a compliance feature (report §10) as much as UX.

## 7. AD-11 — Model availability (was: the licence firewall)

*Superseded 20 Aug 2026. This project is internal and non-commercial, so the
Apache/MIT-only allowlist and the CI gate that enforced it are dropped. Any model
we can obtain and run is fair game — including MASt3R, DUSt3R, VGGT's default
checkpoint, Depth Pro, CubiCasa5K and ZInD.*

`docs/LICENSING.md` survives as an **availability record**, which is still useful:
it says what is downloadable today, what sits behind an application form, and
what has a dead link (Rent3D's advertised archive 404s). That is operational
information, not a compliance gate.

Datasets: ResPlan, Swiss Dwellings, CubiCasa5K, Structured3D and ZInD are all
usable. The vectoriser should be trained on whichever mix works best, with our
own annotated plans for UK style.

## 8. AD-13 — Repository layout

```
visit-it/
├── ROADMAP.md                  # phases, sprints, gates (companion to this doc)
├── docs/
│   ├── ARCHITECTURE.md         # this file
│   ├── LICENSING.md            # model/dataset availability record (AD-11)
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
├── research/                   # experiments and benchmark baselines — never shipped
└── infra/                      # IaC, docker, CI
```

## 9. Evaluation model (referenced by every gate in ROADMAP.md)

| ID | Metric | Definition |
|----|--------|-----------|
| M1 | Total area error | \|model area − stated/plan area\| / stated area, % |
| M2 | Per-room area error | median per-room %, matched rooms |
| M3 | Adjacency accuracy | correct room-adjacency edges / edges in the listing's own floor plan |
| M4 | Layout IoU | 2D IoU of assembled footprint vs the listing's own floor plan |
| M5 | Assignment accuracy | photos placed in the correct plan room, % |
| M6 | Grouping quality | pairwise same-room precision/recall/F1 |
| M7 | Triage accuracy | image type + room label top-1 |
| M8 | Render quality | LPIPS/PSNR on held-out photo views + blind-panel rubric (1–5), **reported per observed-coverage level** (0.0/0.2/0.4/0.6/0.8/1.0, protocol adopted from Rent3D++ — see `docs/PRIOR-ART.md` §6) |
| M9 | Yield | listings shippable with zero human touch, % |
| M10 | Unit economics | GPU-seconds, API cost, **utilisation-adjusted COGS £/listing**, per profile |
| M11 | Review cost | median human minutes per reviewed listing |
| M12 *(added 19 Aug 2026)* | Latency | p50/p95 wall-clock per stage and end-to-end, per profile, warm-pool conditions stated |

Harness rules: frozen holdout split untouched by development; nightly run on the dev split; every PR that touches `pipeline/` posts its eval delta; gates are measured on the holdout only.

## 9b. AD-16/17/18 — Profiles, the hot path, and the serving split *(amendment, 19 Aug 2026)*

The product must serve two economic regimes at once: a **synchronous** regime (a paying user waits 5–10 s for an analysis, marginal cost must sit in single-digit pence) and a **quality** regime (minutes are fine, fidelity sells). These are not two pipelines — they are two *profiles* over the same DAG:

- A profile is a config: per-stage **engine binding** (e.g. stage 8 = `feedforward_splats` vs `optimised_gsplat`), per-stage **latency budget**, and an SLO. `pipeline run <listing> --profile instant|standard|premium`.
- Every stage in `pipeline/` must ship its **fast binding first**; quality bindings are additive. CI perf tests assert each stage's fast binding stays inside its latency budget on the reference GPU.
- **Hot-path bans (instant profile):** no frontier-VLM calls (fine-tuned SigLIP-class classifiers instead), no per-scene optimisation loops, no human gates, no cold model loads (warm pools with all weights resident), no stage that cannot fan out per room. Confidence gating replaces review: a listing that fails checks **degrades** (shell-only, or Tier-B badge) or is **refused with a reason** — it never silently ships wrong and never waits for a human.
- **Serving split:** reconstruction writes static artifacts to the CDN; viewers cost egress only. The cache key is a content hash of (photos + plan + listing text); a repeated analysis is a cache hit and costs ~nothing. GPU economics therefore scale with *unique listings*, not with traffic.

Numbers, price points and the dev-journey variants live in [`VARIANTS.md`](VARIANTS.md).

## 10. What we keep from the compliance design, and why

*Amended 20 Aug 2026. GDPR process, AI-Act marking and the misrepresentation
never-rules are dropped — they were commercial-launch requirements. Two habits
survive on engineering merit alone:*

- **Provenance tagging on every surface** (`photographed / reconstructed /
  inferred / generated`). Not for disclosure — because a viewer that renders
  invented geometry identically to measured geometry is one we cannot debug. When
  a room looks wrong, the first question is always "is that surface real?".
- **Inferred regions render differently** for the same reason: it makes the
  failure visible instead of plausible.

Everything else in this section — the blur pass, the DPA, the machine-readable
marking, the never-generate rules — is removed. If this ever becomes something
we publish, they come back, and `docs/PHASE-0-REPORT.md` records what they were.
