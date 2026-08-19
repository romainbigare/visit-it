# Development roadmap — visit-it

Automatic 3D reconstruction of flat listings: unlabelled photos + (sometimes) a floor plan in → scaled architectural shell with photorealistic per-room Gaussian splats, waypoint-navigated in the browser.

**Basis:** `flat-3d-reconstruction-feasibility.html` (feasibility report, reviewed and corrected 19 Aug 2026) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (decisions of record AD-1…AD-15) · [`docs/LICENSING.md`](docs/LICENSING.md) (what may ship).
**Planning assumptions:** 2-week sprints; core team of 4 (see §6 for the 2-person and 5-person variants); one rented L40S/A100-class GPU per engineer + one shared batch GPU; a French data-partner agency is attainable.

---

## 0. The one-page version

| Phase | Sprints | Weeks | Deliverable | Gate |
|---|---|---|---|---|
| **P0 Foundations** | S1–S2 | 1–4 | Golden set + eval harness + de-risk spikes + platform skeleton | **G0** go/kill: data partner + 30 GT listings + harness live |
| **P1 Measurable shell** | S3–S6 | 5–12 | Scaled, correctly-arranged 3D shell from plan+photos, in the viewer | **G1** ±8% area, 70% correct arrangement (kill criterion) |
| **P2 Photorealism** | S7–S11 | 13–22 | Per-room splats inside the shell; full waypoint walkthrough on mobile | **G2** blind panel prefers it to the photo gallery |
| **P3 No-plan branch + hardening** | S12–S15 | 23–30 | Inferred-layout tier, review console, GDPR blur, partner API | **G3** ≥60% zero-touch yield; review ≤3 min |
| **P4 Generative completion** | not scheduled | — | Only inside the never-rules, after legal review | Explicit go decision by founders + counsel |

Two products fall out along the way: the **P1 shell** is already sellable (scaled interactive floor plan — what Archilogic/CubiCasa sell), and **P2** is the differentiated walkthrough. This ordering deliberately front-loads the stage most likely to kill the project (global assembly) and the discipline most likely to save it (the eval harness).

> **Amendment A (19 Aug 2026) — service profiles and price points.** A new target was set: serve paying customers at **5–10 s per analysis under £0.05** at crowd scale. The full analysis — three service profiles (`instant`/`standard`/`premium`), price-point variants, unit economics on verified Aug-2026 GPU pricing, and three alternative dev journeys — is in [`docs/VARIANTS.md`](docs/VARIANTS.md); the architectural decisions are AD-16/17/18 in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). **Journey C (twin-track) deltas are adopted into this plan**, marked ⚡ below: latency (M12) and COGS instrumentation from S1; feed-forward-splat and serverless cold-start spikes in S1; per-profile criteria added to G0–G3; the D-stream builds two stage-8 engines in P2 (+1 sprint, both-profiles GA ≈ week 32). The journey choice (A/B/C) is revisited once at G0 when the first customer conversation lands — until then C keeps all doors open at a cost of two spikes and some instrumentation.

**Sequencing principle** (from the report, endorsed on review): build what is most valuable *and most verifiable* first; every stage ships with a fallback and a confidence; nothing enters `pipeline/` without a `PROD_OK` licence verdict.

---

## 1. Workstreams

Six streams, deliberately decoupled by the artifact contracts in `docs/ARCHITECTURE.md` §3 so they parallelise:

| Stream | Scope | Pipeline stages |
|---|---|---|
| **A — Data & Eval** | Golden set, ground truth, metrics M1–M11, harness, annotation tooling, panel studies | (cross-cutting) |
| **B — Photo geometry** | Calibration, conditioning, grouping, per-room reconstruction, layout extraction | 1, 2, 3, 4 |
| **C — Plan channel & assembly** | Triage, plan detection/vectorisation/OCR, assignment, scale solve | 0, 5, 6, 7 |
| **D — Appearance** | Splat training service, culling, shell texturing, compression | 8 |
| **E — Viewer & delivery** | Three.js/Spark app, waypoints, streaming, honesty rendering, device perf | 9 |
| **F — Platform & product ops** | Orchestrator, artifact store, review console, licence CI, GDPR blur, partner API | (cross-cutting) |

**Decoupling contracts** (what lets streams run in parallel rather than in a chain):

- E never waits for the pipeline: from S1 it works against **hand-authored fixture artifacts** (`schemas/fixtures/`) — a fake `scene.json`, a Matterport-derived shell, sample splats trained from a dense photo set. The contract, not the pipeline, is the interface.
- D never waits for grouping: it develops against **hand-grouped photos with COLMAP poses** from the golden set until stage 2/3 land, then swaps input source.
- B and C join only at stage 6 (assembly), which consumes both `rooms/*/layout.json` and `plan.json`; each side tests against synthetic counterparts of the other until then.
- A is upstream of everyone and therefore starts first and staffs heaviest in P0.

### Dependency graph (critical path in bold)

```mermaid
graph LR
  GS[A: golden set + harness] --> B3[B: per-room geometry]
  GS --> C5[C: plan vectorise + OCR]
  GS --> D8[D: splat service]
  C0[C: triage] --> B2[B: grouping]
  B2 --> B3
  B3 --> B4[B: layout polygons]
  C5 --> C6[C: assembly]
  B4 --> C6
  C6 --> C7[C: scale solve]
  C7 --> SHELL[shell build → G1]
  B3 --> D8
  B4 --> D8
  D8 --> E9[E: walkthrough viewer]
  SHELL --> E9
  E9 --> G2[G2 panel]
  style GS fill:#f9f,stroke:#333
  style SHELL fill:#ff9,stroke:#333
  style G2 fill:#ff9,stroke:#333
```

Critical path: **golden set → plan channel → assembly → scale → shell (G1) → splat quality inside the shell → viewer (G2)**. Grouping (B2) and splatting (D8) have slack — they must not be allowed to *become* the critical path, which is why both get fixture-based early starts.

---

## 2. Phase 0 — Foundations (S1–S2, weeks 1–4)

> Goal: make every later claim measurable, and kill the licence/feasibility unknowns while the codebase is still small. **Build the scoring harness before the pipeline.**

### Sprint 1

| Stream | Work | Done means |
|---|---|---|
| A | Metric definitions doc (M1–M11 formulas, edge cases) — **including the Rent3D++ coverage-level protocol** (evaluate appearance at observed fractions 0.0/0.2/0.4/0.6/0.8/1.0, see [`docs/PRIOR-ART.md`](docs/PRIOR-ART.md) §6: the principled way to measure M8 against photo coverage, which is the report's central appearance risk); annotation spec (room polygons, adjacency, door positions, per-room areas); collect first 10 listings internally; **request Rent3D (email the Toronto authors) and Rent3D++ (Plan2Scene Google Form) in week 1** — 215 annotated London flats would give stages 4–6 a regression set before our own golden set is ready; **start partner outreach week 1, day 1** — now targeted per [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md): approach agents *by which CRM they use* (Street.co.uk / Reapit in the UK, Apimo / Hektor in France), since those expose OAuth APIs with photos and floor plans. Also week 1: **email the Rent3D authors** for licence terms (215 London flats with photo↔plan alignment) | Spec reviewed; 10 listings ingested; ≥3 agent conversations booked; Rent3D enquiry sent |
| B | **Spike:** MapAnything (`-apache` checkpoint) on 3 real listings' room groups (hand-grouped); GeoCalib on 20 listing photos; record runtime/VRAM/qualitative notes in `docs/spikes/`. ⚡ Measure MapAnything's **GPU seconds per forward pass** explicitly — no published figure exists and the instant profile's latency budget leans on it | Spike write-ups with numbers, go/no-go note per model |
| C | **Spike:** VLM triage prompt v0 over 10 listings (image type, room label, staging/mirror flags, structured JSON); plan-area OCR spike on 10 French plans (the `Séjour 24,5 m²` trick) | ≥90% plan detection on the sample; OCR area extraction works on ≥7/10 plans |
| D | **Spike:** gsplat on one photo-rich room (COLMAP poses) and one 4-photo room (MapAnything poses); floaters documented; SPZ export → file size. ⚡ **Feed-forward spike:** AnySplat and DepthSplat on the same rooms — seconds/room, VRAM, side-by-side quality vs gsplat; email AnySplat authors re weights licence (ledger blocker) | Splat spike write-up incl. feed-forward comparison; format pipeline proven end-to-end |
| E | **Spike:** Spark hello-world — one mesh shell + one splat cloud, correct occlusion, on desktop + one mid-range Android + one iPhone; define the **reference device matrix** | Hybrid render proven on all three; device matrix in `docs/` |
| F | Monorepo scaffold per ARCHITECTURE §8; `schemas/` v0 for all 10 artifacts; stage-runner CLI (`pipeline run <listing> --from <stage> --profile <p>` ⚡ profile flag + per-stage wall-clock logging (M12) from day one); artifact store (MinIO) + Postgres; CI with **licence gate** live (LICENSING.md §Process); ⚡ serverless cold-start spike: our weight set on RunPod/Modal, cold-to-first-inference measured; **CRM ingestion connector spike (Street.co.uk or Reapit OAuth → listing images + plans)**; GDPR: DPA template **plus the derivative-3D-works rights grant** to counsel (DATA-SOURCES §3.7 — blocker before real images are ingested) | `pipeline run` executes stub stages end-to-end on a fixture listing; licence CI red-tests correctly; latency log visible in the contact sheet; connector pulls one real agent's listing |

### Sprint 2

| Stream | Work | Done means |
|---|---|---|
| A | Golden set to 30–50 listings: photos + plan + advertised area for all; per-room laser measurements for ≥10; Matterport (or equivalent) scans for ≥5; frozen **holdout split** (≥20) sealed; harness computes M1–M8 against stubs; naive baseline recorded (monocular depth only, no plan) | Harness runs nightly; scoreboard page exists; baseline numbers published internally |
| F | **Scraper build** (Rightmove + Zoopla per DATA-SOURCES §3.9): listing fetch, image + floor-plan download, listing-text parse for area/room count, rate limiting, robots.txt handling, and **`provenance: scraped` stamped on every asset at ingest**. CI rule: assets tagged `scraped` are rejected by training-data loaders | Scraper fills the golden set; provenance filter demonstrably blocks scraped assets from a training run |
| B | Turn spikes into stage skeletons: stage 1 (GeoCalib undistort + intrinsics artifact), stage 3 API (`reconstruct(images, priors)` plugin interface with MapAnything + MoGe-2 monocular backends) | Stages 1,3 produce schema-valid artifacts on 10 listings |
| C | Stage 0 triage v1 (VLM, structured outputs, phash dedup, listing-text parser for Carrez area/room count); measured against golden labels (M7) | M7 report: plan detection, image type, room label accuracies |
| D | Splat trainer as a service stub (queue in, SPZ out) on hand-posed rooms | 5 golden rooms splatted reproducibly by job id |
| E | Viewer skeleton app (TS, Three.js+Spark, loads fixture `scene.json`+shell+splats); waypoint teleport mechanics v0 | Deployed preview URL teammates can open on a phone |
| F | Review console stub: per-listing artifact browser ("contact sheet" of every stage output) — this is the debugging backbone for everything after | Any listing's stage outputs viewable in one page |

### Gate G0 (end of week 4) — go / kill

- [ ] ≥30 listings with usable ground truth in the harness; holdout frozen.
- [ ] **Data access (reworded twice, 19 Aug 2026).** Two constraints, now on different clocks:
  - **(a) Ground truth for P0 — no longer partner-blocked.** With the in-house scraper (§3.9) plus our own measurements, ≥30 listings with ground truth is an engineering task, not a negotiation. **Kill criterion:** cannot assemble 30 ground-truth listings with verified measurements ⇒ stop.
  - **(b) A rights grant covering derivative 3D works — a launch blocker, not a P0 blocker.** Scraping gets pixels; it does not get the right to publish a 3D model derived from someone else's photograph. Drafting starts in P0, signature required before any customer-facing output (moved to **G2**). Track it from week 1 — it is still the longest-lead item, just no longer on the critical path to a working prototype.
- [ ] Ingestion connector against Street.co.uk **or** Reapit OAuth, pulling a real agent's listings with images — the production channel, developed alongside the scraper rather than instead of it.
- [ ] Spikes green: MapAnything runs on our inputs at seconds-per-group; gsplat→SPZ→Spark proven; hybrid rendering works on the device matrix.
- [ ] Licence CI enforcing; zero non-`PROD_OK` deps in `pipeline/`.
- [ ] Measured (not estimated) GPU cost/listing for the spike path recorded — sanity-check against the report's €0.30–1.50.
- [ ] ⚡ Latency harness live (M12 per stage); utilisation-adjusted COGS model in the dashboard; feed-forward and cold-start spike numbers recorded — these decide whether the instant profile's 10 s / £0.05 envelope survives contact (VARIANTS.md §6 updated with our measurements).
- [ ] ⚡ Journey decision reviewed (A/B/C per VARIANTS.md §4) against the first real customer conversations; default remains C.

---

## 3. Phase 1 — The measurable shell (S3–S6, weeks 5–12)

> Goal: listing in → **scaled, correctly-arranged, untextured 3D shell + interactive floor plan** out, in the browser. No splats. This is the report's "deliberately unglamorous and right" first product, and it exercises the killer stage (assembly) earliest.

### Sprint 3 — plan channel front half

| Stream | Work |
|---|---|
| C | Plan vectorisation v1: preprocessing (deskew, binarise) + wall/room extraction. Model plan: RoomFormer-class architecture (code MIT), **pretrained on ResPlan (17K plans, CC BY 4.0) + Swiss Dwellings (42K apartments, CC BY 4.0)** — both commercially licensed, verified 19 Aug 2026 — then **fine-tuned on partner plans** for French/UK agency style. CubiCasa5K/Structured3D remain NC and stay out of the shipped model. Annotation tooling for plans lands this sprint (A helps). Target: ≥80% room F1 on our plan validation split. |
| C | Plan OCR productionised: room labels + areas, total-area cross-check vs listing text; adjacency graph + door extraction v1. |
| B | Stage 4 v1 (monocular era): MoGe-2 point map → floor plane + wall RANSAC → room polygon under Atlanta-world regularisation; `approximate` flagging. Works on singleton rooms — multi-view comes in P2. |
| A | Annotation drive: plan polygons + adjacency for all golden listings; per-room GT areas table finalised. |
| E | Shell viewer: glTF shell loading, minimap component from `plan.json`, room click-through. |
| F | Orchestrator: DAG execution with per-stage retries, artifact versioning, run ledger; contact-sheet auto-generation per run. |

### Sprint 4 — plan channel back half + scale

| Stream | Work |
|---|---|
| C | Vectorisation v2 (door/window detection on plans; non-Manhattan support for haussmannien stock); confidence outputs. |
| C | **Scale solve v1** (stage 7): weighted least squares over Carrez area + OCR'd room areas + ceiling prior; residual report + QA flags. Door-height constraints join in P2 (needs photo detections). |
| B | Door/window aperture detection in photos v0 (SAM2 + Grounding DINO) — feeds assembly matching and, later, door-height scale constraints. |
| A | Harness: M1/M2/M4 computed on plan-channel outputs alone (plan → scaled polygons vs GT) — this isolates plan-channel error from photo-channel error, which G1 debugging will need. |
| E | Dollhouse view; measurement affordance behind a dev flag (display rule: only the advertised area is ever user-facing — ARCHITECTURE §5). |
| F | Review console: plan-vectorisation overlay editor (fix a wall, re-run downstream) — the first human-in-the-loop tool, and the fallback that keeps G1 achievable if vectorisation accuracy lags. |

### Sprint 5 — assembly

| Stream | Work |
|---|---|
| C | **Stage 6 v1:** Hungarian assignment of reconstructed rooms → plan polygons; cost = room-type match + area ratio + aspect ratio + window/door count; joint SE(2) refinement; `assembly.json` with per-match cost breakdown (instrument heavily — the report is right that this is the wrong-but-confident stage). |
| B | Room-polygon quality pass on golden set; failure taxonomy entries for the polygon failure modes (furniture occlusion, mirrors). |
| C+E | Shell builder (stage 8-lite/9): extrude assembled polygons, cut door/window apertures, flat-shade, glTF+Draco+KTX2 export ≤1 MB. |
| E | Viewer v0.9: walkthrough of the shell (waypoint per room centroid + doorway midpoints), minimap position sync. |
| A | First end-to-end eval: M1–M5 on the dev split, full pipeline. Publish the number, however bad. |
| F | Batch runner: whole golden set reprocessed nightly; regression alerts on metric drops >2σ. |

### Sprint 6 — iterate to the gate

All streams: burn the top of the failure taxonomy (expected leaders per the report: room-type confusions hall/bedroom and dining/living feeding wrong assignments; vectorisation misses on stylised plans; monocular polygon scale noise). A: run G1 eval on the **holdout** at sprint end. E: polish pass so the G1 demo is honest but presentable. F: assignment-nudge UI in review console (drag a room chip onto a different polygon, re-run 6→9 in seconds).

### Gate G1 (end of week 12) — the report's kill criterion, kept verbatim

On holdout listings **that have a floor plan**:

- [ ] Median total-area error ≤ **±8%** (M1) — with scale solve, target is actually ±3–5%; 8% is the kill line.
- [ ] Correct room arrangement on ≥ **70%** of listings (M3/M5 composite).
- [ ] Shell loads < 2 s desktop / < 5 s mid-range mobile.
- [ ] Every shipped shell displays only the advertised area figure.
- [ ] ⚡ Instant-profile shell: end-to-end **p95 ≤ 10 s** (warm/cached conditions stated) and **measured COGS ≤ £0.02** — the shell path is the instant product's core, so this is cheap to demand here and expensive to discover later.
- **Kill criterion:** if after S6 we cannot hit this, the rest does not matter — stop and reassess (the honest fallbacks: human-assisted assembly as default posture, or pivot to the shell-from-plan-only product without photo assignment).

---

## 4. Phase 2 — Photorealism and the walkthrough (S7–S11, weeks 13–22)

> Goal: per-room Gaussian splats anchored inside the G1 shell; the waypoint walkthrough that feels like a product. D and E now carry the critical path; B's grouping/multi-view work feeds them.
>
> ⚡ **Amendment A:** the D-stream builds **two stage-8 engines behind one interface** — feed-forward (AnySplat/DepthSplat-class, instant profile) in S7–S8, optimised gsplat (standard profile) in S9–S10 — and the viewer gains hot-swap upgrade-in-place (instant result first, standard splats replace them minutes later). This adds one sprint to P2 under Journey C (G2 lands end of week 24; G3 week 32). Sprint contents below are otherwise unchanged.

### Sprint 7 — conditioning + grouping

| Stream | Work |
|---|---|
| B | Stage 1 full: exposure/white-balance alignment within candidate room groups; composite/perspective-correction detector (GeoCalib-vs-pointmap-intrinsics disagreement gate from report §4.2) → `texture_only` demotion path. |
| B | Stage 2: DINOv2/MegaLoc retrieval → LightGlue+ALIKED (RoMa v1 for hard pairs) verification → graph clustering; VLM adjudication of low-confidence pairs; singleton routing. Eval M6 against golden hand-groups (target pairwise F1 ≥ 0.8 on textured rooms; measure, don't assume, on white-wall rooms). |
| D | Splat trainer v1: depth-regularised gsplat (reimplement DNGaussian-style regularisation natively — ledger forbids vendoring), init from stage 3; **polygon culling** against stage 4 layouts (the floaters fix, report §4.9). |
| A | Blind-panel protocol design (report G2 criterion): paired A/B "feel for the flat", n≥15 raters, rubric; dry run on 3 listings. |
| E | Room-at-a-time splat streaming (load current + prefetch adjacent); memory watchdog for iOS Safari ceilings. |
| F | GPU autoscaling for splat jobs; cost metering per listing (M10 dashboards). |

### Sprint 8 — multi-view geometry lands

| Stream | Work |
|---|---|
| B | Stage 3 full: MapAnything per group (3–8 images); confidence-masked point maps; COLMAP/GLOMAP alternate path when a room has ≥10 overlapping photos; cross-check MoGe-2 vs MapAnything on singletons. |
| B | Stage 4 v2: plane fitting on multi-view pointmaps replaces monocular polygons where available; aperture placement on wall planes; polygon accuracy re-measured (expect the report's benchmark-to-reality drop — measure it, publish it internally). |
| C | Assembly v2: door positions from photos join the matching cost; **door-height constraints join the scale solve** (2.04 m prior). |
| D | Per-room quality tiers: ≥6 views → full splat; 3–5 → splat with heavier regularisation; 1–2 → shell + projected texture only (report's view-count curve). Unobserved-surface fill: flat-shade + texture-extend (options 1–3 only; provenance-tagged). |
| E | Honesty rendering: provenance-driven treatment (photographed vs inferred), legend, Tier A/B badges. |
| A | M8 harness: held-out-view LPIPS/PSNR where a room has ≥4 photos (leave-one-out). |

### Sprint 9 — appearance pipeline complete

| Stream | Work |
|---|---|
| D | Shell texturing: MVS-Texturing projective pass where photos cover shell faces; seam/exposure blending; KTX2 compression; splat budgets enforced (200–500 K/room), SPZ + chunked-SOG (splat-transform) export. |
| B | Mirror detection v0 (SAM2-based segmentation + reflection heuristics) → mask from geometry, flag to viewer. |
| E | Waypoint graph generation from camera poses + doorway midpoints; transition animations; dollhouse ↔ walkthrough ↔ minimap sync complete. |
| C | Tier-B scaffolding begins ahead of P3: inferred-arrangement data model in `assembly.json` (`method: inferred`), so the viewer treatment can be built now. |
| A | Panel dry-run #2 on 10 listings; iterate rubric; recruit external raters (not team). |
| F | CDN packaging + cache headers; per-listing payload report in CI (fail >25 MB typical). |

### Sprint 10 — integration at scale

Process 50+ listings end-to-end. Streams swarm the integration failure list. E: perf pass on device matrix (30 fps target, first-room-interactive ≤5 s on 4G). D: floater/artefact triage on the worst 10 rooms. B: grouping errors that survived to render get adjudication-prompt fixes. F: review console gains splat-quality flagging + re-run-with-overrides.

### Sprint 11 — the panel and the gate

Freeze the pipeline; run the **blind panel** on ≥30 holdout listings (external raters; walkthrough vs photo gallery; "which gave you a better feel for this flat?" + would-show-to-a-buyer rubric). Fix nothing mid-measurement. Publish results internally, warts included.

### Gate G2 (end of week 22)

- [ ] Blind panel prefers the walkthrough on a **majority** of holdout listings (report kill criterion: if a splat with floaters is worse than the gallery, we've added cost and subtracted value).
- [ ] ≥70% of rooms with ≥3 usable photos reach "acceptable" on the panel rubric.
- [ ] Device matrix: 30 fps, first-room-interactive ≤5 s on 4G mobile, zero OOM tab-kills across the test suite.
- [ ] Measured cost ≤ €2/listing at current quality (M10), and G1 metrics have not regressed (shell accuracy is non-negotiable ballast).
- [ ] ⚡ Per-profile matrix published: **instant** — p95 ≤ 10 s, COGS ≤ £0.03, beats the "plan + photo lightbox" baseline in the panel; **standard** — ≤ 5 min, COGS ≤ £0.15, meets the beats-the-gallery bar above. Upgrade-in-place demonstrated on a live listing.
- [ ] **Rights grant signed** (moved here from G0): at least one agent/agency agreement covering process, store, create derivative 3D works, and publish, with the de-listing position stated. **No customer-facing output ships before this**, and nothing customer-facing derives from scraped imagery (provenance filter verified in CI).
- **Pivot rule:** if the panel prefers the gallery, ship the Phase-1 shell product (+ plan minimap + photo lightbox anchored to rooms) commercially while appearance quality is reworked — the shell alone is a sellable product and the company does not stall.

---

## 5. Phase 3 — No-plan branch, review economics, production (S12–S15, weeks 23–30)

> Goal: the degraded-gracefully Tier B product; the review console that makes unit economics work (the report's sharpest operational finding: human review costs rival all compute — reducing review rate is worth more than any GPU optimisation); production hardening.

### Sprint 12 — Tier B (no plan)

- B: doorway chaining — same-door detection across room groups (aperture signature + RoMa verification) → relative pose per shared doorway (expect ~⅓ of adjacencies, per report).
- C: layout inference — constraint-based arrangement synthesis (room count/types from triage, chained adjacencies, total advertised area, corridor conventions); output explicitly `method: inferred`.
- E: Tier B viewer treatment (visibly inferred arrangement, plain-language note); agent-facing 30-second drag-and-drop arrangement step (turns the unsolvable problem into a trivial one — report §7.1).
- A: Tier B metrics: room-level accuracy still measurable (per-room area, layout IoU per room); arrangement scored only as "plausible/implausible" by rubric.

### Sprint 13 — review console v1 (the margin machine)

- F/E: unified review flow: QA-flag queues → one-screen fix actions (reassign room, nudge arrangement, drop a photo from geometry, accept/reject) → targeted re-run. Target: median ≤3 min per reviewed listing (M11), instrumented.
- C: auto-QA routing thresholds tuned on residuals (scale disagreement, assignment cost margins, splat quality score) — the goal is *calibrated* flags: high recall on the listings a human would reject.
- A: reviewer time-and-motion measurement baked into the console.

### Sprint 14 — trust, privacy, robustness

- B: mirror/glass handling v1 (mask from geometry, flat reflective plane in render); virtual-staging detector v1 (multi-view object-consistency check + VLM flag) → staged objects excluded from geometry, disclosure badge in viewer (AB 723 is the template; French counsel review this sprint).
- F: GDPR blur pass in stage 1 (faces, family photos, screens, documents) before long-term storage; DPA executed with partner; deletion/retention runbook.
- D: robustness sweep — HDR/flambient extremes, minimalist white rooms, haussmannien irregular polygons; document per-condition yield.

### Sprint 15 — production

- F: partner API (submit listing → webhook on ready/needs-review), batch mode, monitoring/alerting, SLOs; load test at 200 listings/day simulated.
- A: G3 eval at 200-listing scale (partner-supplied fresh listings, not golden set).
- All: failure-taxonomy burn-down; docs/runbooks complete; security pass (`/security-review` on the services).

### Gate G3 (end of week 30) — production readiness

- [ ] Yield: ≥60% of plan-bearing listings ship with **zero human touch** (M9); reviewed listings median ≤3 min (M11); combined review rate ≤25%.
- [ ] Tier B: no-plan listings produce labelled inferred layouts; with Carrez area present, total-area error still ≤±10%.
- [ ] Privacy: audit sample of shipped scenes shows zero readable personal data; blur runs pre-storage.
- [ ] Compliance posture: counsel sign-off on viewer disclosures (FR consumer law + AI Act Art. 50 position); provenance tags verified machine-readable end-to-end.
- [ ] Ops: 99% pipeline completion without manual intervention at 200/day; cost dashboard green.
- [ ] ⚡ Economics at declared price points over a 200-listing/day week: instant COGS ≤ £0.03 with SLO held; standard ≤ £0.15; premium COGS review-dominated as modelled (VARIANTS.md §3 margins validated with the partner). Review never gates the instant profile (confidence-degrade/refuse only).

---

## 6. Parallelisation and staffing

### With 4 engineers (baseline plan above)

| | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| Eng 1 (ML geometry) | B spikes | B: stages 1,4 + apertures | B: stages 2,3,4v2 | B: doorway chain, mirrors, staging |
| Eng 2 (ML vision) | C spikes + A harness | C: plan channel, scale, assembly | C: assembly v2 + D support | C: layout inference + auto-QA |
| Eng 3 (3D/web) | E spikes | E: shell viewer + shell builder | E: walkthrough, streaming, honesty UX | E: Tier B UX + review UI |
| Eng 4 (platform) | F: scaffold, CI, store | F: orchestrator, review stub, batch | D: splat service + F: autoscale/CDN | F: console v1, API, GDPR, ops |
| Founder/PM (part-time) | A: partner deal, GT collection | A: annotations, eval reviews | A: panel studies | A: G3 eval, counsel loop |

Note the deliberate imbalance: stream A is founder-heavy early (partner + ground truth are relationship work, not code), and stream D is *shared* rather than owned until P2 — splatting has slack in the dependency graph.

### With 2 engineers (stretch: ~45 weeks)

Serialise D behind B (one ML engineer owns B→D), merge E+F (one product engineer). Phases become P0: 6 wks, P1: 12, P2: 14, P3: 12. Gates unchanged — never trade the gates for the calendar; cut scope inside sprints instead (e.g. defer dollhouse, defer Tier B drag-and-drop).

### With 5+ engineers

Fifth engineer takes D as a dedicated stream from S1 (fixture-driven), pulling splat quality work earlier and de-risking G2; P2 can compress to 4 sprints. Do **not** add people to stream C — assembly and scale are conceptually serial and gain little from splitting.

### Standing cross-cutting rules (every sprint's definition of done)

1. New/changed stage ⇒ schema versioned, confidence + QA flags emitted, contact-sheet renders it.
2. Eval harness updated in the same PR as the capability; nightly regression stays green; PRs post their eval delta.
3. New dependency ⇒ ledger row in `docs/LICENSING.md` in the same PR; licence CI green.
4. Failure taxonomy updated when a new failure mode is seen twice.
5. Demo listing regenerated at sprint end (the same 3 listings throughout the project — progress must be visible on *stable* examples).
6. The never-rules hold (ARCHITECTURE §10): no generated content that changes perceived size/layout/condition; only the advertised area is user-facing; Tier B never presented as measured.
7. ⚡ Every stage lands its **fast binding first** (AD-17); per-stage latency budgets asserted in CI perf tests; a PR that pushes a stage over budget needs an explicit waiver in the PR description.

---

## 7. Risk register (top 8, owners assigned at kickoff)

| # | Risk | Likelihood | Impact | Mitigation / early warning |
|---|---|---|---|---|
| R1 | No data partner materialises | Med | Fatal | Start outreach day 1; G0 kill criterion enforced; fallback: buy Matterport scans + scrape-free listing sets from a friendly independent agency |
| R2 | Assembly accuracy stalls below 70% | Med | Fatal to autonomy, not product | Isolated plan-channel metrics (S4) localise blame; human-assist assignment UI exists from S6; C3Po-style learned matching is the research lever |
| R3 | Benchmark-to-reality drop worse than expected on layout (report warns 20–40%) | High | High | Measured on golden set from S3; monocular fallback polygons + cuboid fallback keep yield non-zero; review-console wall editor |
| R4 | Splat quality below "beats the gallery" | Med | High (G2) | Quality tiers by view count; polygon culling; pivot rule at G2 keeps the company shipping |
| R5 | Licence landscape shifts (model re-licensed, ledger entry invalidated) | Med | Med | Ledger re-verification each release; engine plugin abstraction (AD-4) makes swaps cheap |
| R6 | ~~Vectoriser training data gap~~ **largely resolved 19 Aug 2026** | Low | Low | ResPlan + Swiss Dwellings are CC BY 4.0 and commercially usable; residual risk is only *stylistic* (neither is French/UK agency-drawn), handled by partner fine-tuning |
| R11 | **Image rights don't extend to derivative 3D works** — photographer owns copyright, agent holds a marketing-only, listing-lifetime licence (DATA-SOURCES §3.7). Unaffected by the scraping decision, which solves access and not rights | Med | **High** | Rights grant drafted in S1, signed before G2; de-listing behaviour defined in product; fallback is per-agent SaaS where the agent supplies and controls their own imagery |
| R12 | **Scraping exposure** — portal ToS, EU database right (Entreparticuliers v. Leboncoin, €50k), image copyright (CoStar v. Zillow), plus IP blocking as an ongoing engineering cost (DATA-SOURCES §3.9) | Accepted by founder decision | Med–High | Provenance tagging + CI split keeps scraped assets out of training and out of customer-facing output; rate limiting and robots.txt; counsel briefed before any output derives from scraped imagery |
| R7 | Mobile memory ceilings kill the viewer on iOS | Med | High | Budgets CI-enforced from S9; room-at-a-time streaming from S7; device matrix testing every sprint from P2 |
| R8 | Review economics don't converge (>25% review rate persists) | Med | Margin | M11 instrumented from S13; auto-QA calibration is a first-class task, not a cleanup; pricing model keeps a human-reviewed tier |
| R9 ⚡ | Instant-engine licence gaps (AnySplat weights unstated; FastGS licence chain possibly Inria-tainted) | Med | Med | Author contact + legal read in S1; fallbacks already identified: DepthSplat (MIT) for feed-forward, techniques ported onto gsplat for fast optimisation |
| R10 ⚡ | Feed-forward splat quality too low even for the instant tier's honest bar | Med | Med (V1 variant only) | S1 spike gives the answer 6 months early; fallback: instant profile ships shell + projected textures (still a real product), splats stay standard-only |

---

## 8. What we are explicitly *not* doing (v1 scope fence)

- No generative completion of unseen content (Phase 4 is unscheduled; requires founders + counsel go, AI-Act Art. 50 marking machinery, and the never-rules).
- No free-roam navigation; no watertight meshing of splats (SuGaR etc.) — the shell is the mesh.
- ~~No scraping of portals.~~ **Superseded 19 Aug 2026 by founder decision** (DATA-SOURCES §3.9): we build our own Rightmove/Zoopla scraper for acquisition. Scope fence becomes narrower but still binding — **scraped imagery is for R&D, the golden set and demos only**; it does not go into training corpora for shipped models, and no customer-facing 3D output derives from it without a rights grant. Provenance tagging enforces the split.
- No panorama capture path (the whole point is photos-that-already-exist; panoramas would be a different product with a mature competitive field).
- No self-hosted VLM in v1 (API with an abstraction seam; revisit at volume).
- No displayed measurements other than the advertised legal figure.

---

## 9. Kickoff checklist (week 1)

- [ ] Repo scaffold merged (ARCHITECTURE §8 layout), licence CI green on empty pipeline.
- [ ] Golden-set listing sources identified for the first 10; annotation tool chosen (start with CVAT/Label Studio, custom plan-overlay later).
- [ ] Partner shortlist (≥5 French agencies/portals) + outreach owner + LOI draft.
- [ ] GPU environment reproducible (`infra/`, one command); MapAnything-apache + MoGe-2 + GeoCalib + gsplat weights pinned and hash-locked.
- [ ] Reference device matrix purchased/borrowed (mid-range Android, 2-gen-old iPhone, low-end laptop).
- [ ] DPA template with counsel; GDPR record of processing started.
- [ ] Sprint 1 board cut from §2 of this document.
