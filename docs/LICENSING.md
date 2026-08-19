# Licence ledger

Single source of truth for what may ship. **Verdicts:** `PROD_OK` (allowlisted for production), `RESEARCH_ONLY` (may live in `research/`, never imported by `pipeline/`, never in prod images), `CONDITIONAL` (usable only under the stated condition), `AVOID`.

All entries below were verified against primary sources (LICENSE files, model cards, official pages) on **2026-08-19** unless noted. Rules of the ledger:

1. Record **code licence and weights licence separately** — they frequently differ.
2. Re-verify every entry touched by a release; this field re-licenses often (VGGT gained a commercial checkpoint post-release, DINOv2 went Apache post-release, SOGS was archived in favour of splat-transform).
3. CI fails the build if `pipeline/`, `services/` or `viewer/` depend on anything not marked `PROD_OK` (or `CONDITIONAL` with the condition satisfied and noted).
4. A model fine-tuned or distilled on a non-commercial dataset inherits the taint — track training-data provenance for anything we train.

## Reconstruction & geometry

| Component | Code licence | Weights licence | Verdict | Notes |
|---|---|---|---|---|
| **MapAnything** (Meta) | Apache-2.0 | `facebook/map-anything`: CC-BY-NC-4.0; **`facebook/map-anything-apache`: Apache-2.0** | **PROD_OK** (Apache checkpoint only) | Pin the `-apache` checkpoint in code; CI check that the NC one never appears in configs. |
| VGGT | VGGT Research License (non-comm.) | Default CC-BY-NC-4.0; `VGGT-1B-Commercial` under VGGT-AUP (application/approval, non-military) | CONDITIONAL | Backup engine. Usable only via the approved commercial checkpoint; keep approval paperwork. |
| DUSt3R / MASt3R (Naver) | CC-BY-NC-SA-4.0 | CC-BY-NC-SA-4.0 (+ restrictive training-data caveats) | RESEARCH_ONLY | Benchmark baseline only. |
| Pi3 / π³ | BSD-2 (commercial: contact authors) | CC-BY-NC-4.0 | RESEARCH_ONLY | Weights are NC regardless of code licence. |
| Fast3R (Meta) | FAIR NC Research License | FAIR NC Research License | RESEARCH_ONLY | Both code and weights NC. |
| CUT3R | CC-BY-NC-SA-4.0 | presumed same (no official model card) | RESEARCH_ONLY | |
| Pow3R (Naver) | NAVER NC License | NAVER NC License | RESEARCH_ONLY | Auto-termination clause; contribution grant to NAVER. |
| **MoGe / MoGe-2** (Microsoft) | MIT (DINOv2 parts Apache-2.0) | MIT per repo; not separately licensed on HF | **PROD_OK** | Monocular fallback of record. Metric point maps + intrinsics from one image. Re-check model card at GA. |
| Depth Pro (Apple) | Apple custom "personal, non-exclusive" | same | **AVOID** | **Not Apache-2.0** (feasibility report corrected 2026-08-19). Commercial use legally problematic. |
| Depth Anything V2 | Apache-2.0 | Small: Apache-2.0; Base/Large/Giant: CC-BY-NC-4.0 | CONDITIONAL | Small checkpoint only. Relative depth by default. |
| UniDepth / V2 | CC-BY-NC-4.0 | CC-BY-NC-4.0 | RESEARCH_ONLY | |
| Metric3D v2 | BSD-2 | Unspecified — authors ask to be contacted | CONDITIONAL | Weights unusable until authors grant terms in writing. |
| COLMAP | New BSD | n/a | **PROD_OK** | Dense-coverage fallback path. |
| GLOMAP | BSD-3 | n/a | **PROD_OK** | ~8× COLMAP on IMC 2023 (own paper, verified). |
| **GeoCalib** | Apache-2.0 | CC-BY-4.0 (attribution required) | **PROD_OK** | Add attribution to NOTICE file. Pinhole + fisheye/radial. |
| NeurVPS | MIT | MIT | PROD_OK | Cross-check only. |

## Matching, retrieval, segmentation

| Component | Code licence | Weights licence | Verdict | Notes |
|---|---|---|---|---|
| **LightGlue** | Apache-2.0 | — | **PROD_OK** | Only with ALIKED or DISK features (below). |
| SuperPoint weights (Magic Leap) | — | "Academic or non-profit noncommercial research use only" | **AVOID** | The classic trap, confirmed. Never pair with LightGlue in prod. |
| **ALIKED** | BSD-3 | BSD-3 | **PROD_OK** | Default local feature. |
| **DISK** | Apache-2.0 | Apache-2.0 | **PROD_OK** | Alternative local feature. |
| RoMa (v1) | MIT | uses DINOv2 (Apache-2.0) | **PROD_OK** | Dense matcher for hard pairs. |
| RoMa v2 | MIT | uses **DINOv3** (Meta custom licence) | CONDITIONAL | DINOv3 licence is bespoke (no military, non-transferable, acknowledgment clauses). Legal read required before adopting. |
| **DINOv2** | Apache-2.0 | Apache-2.0 (except Cell/XRay variants: NC) | **PROD_OK** | Global descriptor backbone. |
| DINOv3 | Meta custom commercial licence | same | CONDITIONAL | Readable for commercial use but bespoke terms; legal review before use. |
| **MegaLoc** | MIT | MIT | **PROD_OK** | Place-recognition head option. |
| AnyLoc | BSD-3 | BSD-3 | PROD_OK | |
| **SAM 2** (Meta) | Apache-2.0 (+BSD-3 parts) | Apache-2.0 | **PROD_OK** | Door/window/mirror/fixture segmentation. |
| **Grounding DINO** | Apache-2.0 | Apache-2.0 | **PROD_OK** | Open-vocabulary detection for the scale solve. |
| OmniGlue | Apache-2.0 | check weights on adoption | CONDITIONAL | Corrected benchmark: ScanNet-1500 AUC@5° 8.6 vs SuperGlue 7.2. |

## Layout & floor plans

| Component | Code licence | Dataset/weights | Verdict | Notes |
|---|---|---|---|---|
| RoomFormer | MIT | trained on Structured3D (NC data) | CONDITIONAL | Code fine; **retrain on our own/partner data** before shipping the model. |
| PolyRoom | **No licence stated** | — | AVOID | No licence = no rights. Watch the repo. |
| Plane-DUSt3R | MIT | builds on DUSt3R checkpoints (CC-BY-NC) | RESEARCH_ONLY | Architecture reference for our stage 4. |
| HorizonNet | MIT | check checkpoint provenance | CONDITIONAL | Panorama channel, if we ever ingest panos. |
| AtlantaNet | MIT | check checkpoint provenance | CONDITIONAL | Non-Manhattan layouts (haussmannien). |
| SceneScript (Meta) | CC-BY-NC (code IS released — report corrected) | CC-BY-NC | RESEARCH_ONLY | Steal the token representation, reimplement. |
| LASER (Zillow) | CC-BY-NC-ND-4.0 | — | RESEARCH_ONLY | |
| F3Loc | MIT | check | CONDITIONAL | Photo-in-plan localisation, later phases. |
| **C3Po** (arXiv 2511.18559, NeurIPS 2025) | see repo | **dataset CC-BY-4.0 — commercial use permitted with attribution** ✅ (verified 19 Aug 2026) | **PROD_OK** (with attribution) | 90K photo/plan pairs, 597 scenes, on HF (~427 GB). Domain gap: internet/landmark buildings, not agent flat photography — pretrain here, fine-tune on partner data. |
| **ResPlan** | MIT | **dataset CC-BY-4.0** ✅ | **PROD_OK** (with attribution) | 17,000 vector floor plans with room graphs and metric scale. US-derived. Primary vectoriser training corpus. |
| **Swiss Dwellings** (Archilyse) | — | **CC-BY-4.0** ✅ | **PROD_OK** (attribute Archilyse AG) | 42,207 apartments / 242,257 rooms, European. Second vectoriser corpus. |
| MSD (Modified Swiss Dwellings) | — | CC-BY-SA-4.0 | CONDITIONAL | Share-alike: assess whether it reaches released model weights before training on it. European multi-apartment stock, vector+raster+graph. |
| **CubiCasa5K** | code: see repo | **dataset CC-BY-NC-4.0** | RESEARCH_ONLY | Cannot train the commercial vectoriser on it — **and no longer needed**, ResPlan + Swiss Dwellings cover it. |
| Structured3D | code MIT | dataset: non-commercial Terms of Use | RESEARCH_ONLY | |
| ZInD (Zillow) | code Apache-2.0 | dataset: non-commercial; commercial enquiries ZInD@zillowgroup.com | RESEARCH_ONLY | Consider a paid licence if it would accelerate Phase 1. |
| Rent3D / Rent3D++ (Toronto) | — | **Not stated — must ask the authors** | CHECK (S1 task) | 215 London flats, ~1,570 photos + aligned floor plans. Closest public analogue to our problem, in a target market. |
| LIFULL HOME'S (NII Japan) | — | Academic institutions only | RESEARCH_ONLY | 5.33M listings; wrong market regardless. |
| Places365 / MIT Indoor67 | — | Reported non-commercial | AVOID | Verify directly if ever needed; assume closed for a shipped classifier. |
| Unlicensed Kaggle/HF image sets | — | **No stated licence** | AVOID | No licence means no rights — worse than a restrictive licence. |

## Splatting & delivery

| Component | Licence | Verdict | Notes |
|---|---|---|---|
| **gsplat** (Nerfstudio) | Apache-2.0 | **PROD_OK** | The production rasteriser/trainer. Note: gsplat does not claim clean-room independence from the Inria paper — it is an independent implementation of the published method; monitor for any IP noise, but Apache LICENSE is what it ships. |
| Inria/MPI 3DGS reference | Custom: research/evaluation only; commercial requires written consent (stip-sophia.transfert@inria.fr) | **AVOID** | Confirmed trap. |
| DNGaussian | Inria-style NC research licence | RESEARCH_ONLY | Reimplement depth regularisation natively on gsplat. |
| SparseGS | Licence file not accessible at review | AVOID until clarified | |
| InstantSplat (NVlabs) | Unconfirmed + depends on MASt3R (NC) | RESEARCH_ONLY | Pipeline idea (pointmap init → joint pose+splat optimisation) is what we reimplement on MapAnything + gsplat. |
| **AnySplat** (SIGGRAPH Asia 2025) | Code MIT; **weights licence unstated** (HF `lhjiang/anysplat`) | CONDITIONAL | Prime instant-profile engine candidate (unposed, 2–64 views). Blocked on weights-licence clarification — ask the authors in S1; do not ship until stated. |
| **DepthSplat** (cvg) | MIT | CONDITIONAL → PROD_OK pending checkpoint provenance | Verified 0.6 s / 12 views on A100. Second instant-engine candidate (posed input — fine, stage 3 provides poses). |
| pixelSplat / latentSplat | MIT / MIT | CONDITIONAL | Stereo-pair only; partial coverage. Check checkpoint provenance. |
| GS-LRM (Adobe) | Adobe Research License, no official open release | AVOID | |
| Long-LRM / Long-LRM++ | Unclear; unofficial reimplementations | AVOID until clarified | |
| MV-DUSt3R+ (Meta) | CC-BY-NC-4.0 | RESEARCH_ONLY | |
| PreF3R | Licence unstated | AVOID until clarified | |
| FastGS | Labeled MIT but README requires adherence to 3DGS/Taming-3DGS/Speedy-Splat licences — possible Inria taint in the chain | CONDITIONAL | Standard-profile fast-optimisation candidate. Legal read of the licence chain required; if tainted, port the *techniques* onto gsplat instead. |
| Taming-3DGS | Licence unstated | RESEARCH_ONLY | Comparison point (7–13 min/scene A100). |
| Splatt3R | CC-BY-NC-4.0 | RESEARCH_ONLY | |
| DepthSplat | MIT | CONDITIONAL | Check its checkpoint provenance before prod. |
| MVSplat | MIT | CONDITIONAL | Same. |
| **SPZ** (Niantic) | MIT | **PROD_OK** | Primary splat delivery format. |
| SOGS (PlayCanvas) | Apache-2.0 — **repo archived** | superseded | Use splat-transform. |
| **splat-transform** (PlayCanvas) | MIT | **PROD_OK** | Maintained; multi-LOD chunked SOG streaming. |
| **Spark** (World Labs) | MIT | **PROD_OK** | Hybrid splat+mesh Three.js renderer; reads PLY/SOG/SPZ/SPLAT/KSPLAT. |
| @mkkellogg/GaussianSplats3D | MIT | PROD_OK (fallback) | No longer actively developed; README points to Spark. |
| Three.js | MIT | **PROD_OK** | Claim of native WebGPU splat renderer in r186 unconfirmed at review — don't plan around it. |
| KHR_gaussian_splatting (Khronos) | spec | WATCH | Release Candidate, not ratified. Adopt when ratified + supported. |
| **glTF-Transform** | MIT | **PROD_OK** | |
| **meshoptimizer / gltfpack** | MIT | **PROD_OK** | |
| **xatlas** | MIT | **PROD_OK** | |
| **KTX-Software** (Khronos) | Apache-2.0 (+ per-component) | **PROD_OK** | |
| **MVS-Texturing** | BSD-3 | **PROD_OK** | Projective texturing of the shell. |
| OpenMVS | **AGPL-3.0** | AVOID in service | AGPL propagates over network use for our SaaS shape; MVS-Texturing covers the need. |

## Generative / world models (Phase 4 only, all gated)

| Component | Licence | Verdict | Notes |
|---|---|---|---|
| World Labs Marble | Commercial product; Free/$20/$35/$95 per month tiers; generation API | CONDITIONAL | Only within the never-rules (no size/layout/condition changes) + AI-Act Art. 50 marking. |
| Stable Virtual Camera | Stability AI Non-Commercial License | RESEARCH_ONLY | No published commercial path. |
| HunyuanWorld 1.0 (Tencent) | Community License — **explicitly not granted in the EU, UK, South Korea** | **AVOID** | Unusable for a French company. Confirmed from licence text. |
| CAT3D / CAT4D (Google) | Research, no code | AVOID | |

## Datasets we build (untainted by construction)

| Asset | Terms | Notes |
|---|---|---|
| Golden set (30–50 listings + ground truth) | Partner DPA + our annotation | The product's evaluation basis. GDPR: blur before storage. |
| Partner plan corpus + annotations | Partner agreement | Fine-tuning basis for the plan vectoriser. |
| Synthetic plan generator output | Ours | Augmentation for vectoriser training. |

## Process

- **Owner:** whoever adds a dependency adds the ledger row in the same PR.
- **CI:** `infra/ci/licence_gate.py` — builds the prod dependency closure, fails on anything not `PROD_OK`/`CONDITIONAL`-satisfied; greps checkpoint identifiers against a pinned allowlist (e.g. rejects `facebook/map-anything` where `-apache` is required).
- **Release checklist:** re-verify every `CONDITIONAL` and any row older than 90 days.
- **NOTICE file:** attribution for CC-BY-4.0 (GeoCalib weights) and Apache NOTICE requirements, generated at build.
