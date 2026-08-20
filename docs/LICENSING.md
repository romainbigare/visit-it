# Model and dataset availability record

> **Scope note (20 Aug 2026): internal, non-commercial.** This file used to be a
> licence ledger with a `PROD_OK` / `RESEARCH_ONLY` / `AVOID` verdict per row and
> a CI gate enforcing it (AD-11). The project is built for ourselves and not
> shipped, so the gate is gone and the verdicts with it. What survives is the
> part that saves real time: **can we actually get this, and what breaks when we
> try?** Dead links, application forms, checkpoints that differ from the default,
> archived repos, weights that don't match their code repo.
>
> If this ever becomes something we publish, the licence question comes back and
> has to be redone from primary sources — the old verdicts are not preserved
> here, and several would have changed anyway. Git history has them.

**Availability values:** `Open` (download and go) · `Variant` (the default artifact is not the one we want — pin the named one) · `Application` (form, email or approval before download) · `Dead` (link broken; needs an author enquiry or a mirror) · `Archived` (superseded upstream).

Verified against primary sources on **2026-08-19**, with download reality re-checked **2026-08-20** where the fetcher touched it. This field moves fast — re-verify anything you are about to depend on.

## Reconstruction & geometry

| Component | Availability | Notes |
|---|---|---|
| **MapAnything** (Meta) | **Variant** — pin `facebook/map-anything-apache` | Primary multi-view engine. Validated on T4: 12/12 groups, 1.35 s each. Output key names are **not stable across versions** — resolve them by alias, don't hard-code (`tools/gpu_validate_standalone.py::_pick`). Real keys as of 20 Aug: `cam_quats, cam_trans, camera_poses, conf, depth_along_ray, depth_z, img_no_norm, intrinsics, mask, metric_scaling_factor, non_ambiguous_mask, pts3d, pts3d_cam, ray_directions` |
| **MoGe / MoGe-2** (Microsoft) | **Open**, but **not on PyPI** — vendored via `make vendor` | Monocular fallback of record. Metric point maps **plus intrinsics** from one image, which is the whole reason it beats depth-only models (AD-5). Ships its own `utils3d_moge` fork — do **not** `pip install utils3d` alongside it, they conflict |
| VGGT | **Variant/Application** — default checkpoint differs from the approved one | Backup engine |
| DUSt3R / MASt3R (Naver) | Open | Benchmark baseline |
| Pi3 / π³ | Open | Benchmark baseline |
| Fast3R (Meta) | Application | |
| CUT3R | Open | No official model card |
| Pow3R (Naver) | Application | |
| Depth Pro (Apple) | Open | Rejected on merit, not availability: no intrinsics |
| Depth Anything V2 | **Variant** — Small vs Base/Large/Giant differ | **Rejected for metric use.** Measured: estimated ceiling height scales almost linearly with the *assumed* field of view (2.76 m at 60°, 5.93 m at 104°). MoGe measured the true FOV at 98.6°, where Depth Anything gives ~6 m ceilings. Usable only as a relative-depth cross-check |
| UniDepth / V2 | Open | |
| Metric3D v2 | **Application** — weights unspecified, authors ask to be contacted | |
| COLMAP | Open | Dense-coverage fallback path |
| GLOMAP | Open | ~8× COLMAP on IMC 2023 (own paper, verified) |
| **GeoCalib** | Open | Pinhole + fisheye/radial. Attribution requested by the weights; keep the NOTICE entry as a courtesy |
| NeurVPS | Open | Cross-check only |

## Matching, retrieval, segmentation

| Component | Availability | Notes |
|---|---|---|
| **LightGlue** | Open | Pair with ALIKED or DISK |
| SuperPoint (Magic Leap) | Open | |
| **ALIKED** | Open | Default local feature |
| **DISK** | Open | Alternative local feature |
| RoMa (v1) | Open | Dense matcher for hard pairs; uses DINOv2 |
| RoMa v2 | **Variant** — pulls **DINOv3**, a separate download with its own gate | |
| **DINOv2** | **Variant** — Cell/XRay variants are separate | Global descriptor backbone |
| DINOv3 | Application | |
| **MegaLoc** | Open | Place-recognition head option |
| AnyLoc | Open | |
| **SAM 2** (Meta) | Open | Door/window/mirror/fixture segmentation |
| **Grounding DINO** | Open | Open-vocabulary detection for the scale solve |
| **SigLIP** | Open | Zero-shot triage. Validated: **F1 0.96, recall 1.00** after adjudication, 177 ms/image on CPU. Scores are poorly calibrated in absolute terms — **use the argmax, never an absolute threshold** (a 0.02 floor cut room grouping from 26 groups to 2) |
| OmniGlue | Open — check weights on adoption | ScanNet-1500 AUC@5° 8.6 vs SuperGlue 7.2 |

## Layout & floor plans

| Component | Availability | Notes |
|---|---|---|
| RoomFormer | Open | Trained on Structured3D; retrain on our own corpus for UK plan style |
| PolyRoom | Open — no releases, watch the repo | |
| Plane-DUSt3R | Open | Architecture reference for our stage 4 |
| HorizonNet | Open | Panorama channel, if we ever ingest panos |
| AtlantaNet | Open | Non-Manhattan layouts |
| SceneScript (Meta) | Open (code *is* released — the feasibility report said otherwise and was corrected) | Steal the token representation |
| LASER (Zillow) | Open | |
| F3Loc | Open | Photo-in-plan localisation, later phases |
| **C3Po** (NeurIPS 2025) | Open — **~427 GB** on HF, budget disk before fetching | 90K photo/plan pairs, 597 scenes. Domain gap: internet/landmark buildings, not agent flat photography — pretrain here, fine-tune on scraped listings |
| **ResPlan** | Open | 17,000 vector floor plans with room graphs and metric scale. US-derived. Primary vectoriser corpus |
| **Swiss Dwellings** (Archilyse) | Open — **fetched and verified end to end**, 792 MB, 2.5 M rows | 42,207 apartments / 242,257 rooms, European. Second vectoriser corpus |
| MSD (Modified Swiss Dwellings) | Open | European multi-apartment stock, vector + raster + graph |
| **CubiCasa5K** | Open | 5,000 **raster** plans with vector annotation — the raster half is what ResPlan lacks, so this is back in the corpus rather than out of it |
| Structured3D | Application | |
| ZInD (Zillow) | Application | Panoramas, not agent photography |
| **Rent3D** (Toronto, CVPR 2015) | **Dead** — 404 from both Toronto hostnames, confirmed twice on 20 Aug. `pipeline/datasets/fetch.py` falls back to printing a ready-to-send author enquiry | 215 London flats, ~1,570 photos + aligned plans; 1,312 rooms, 6,628 walls, 1,923 doors, 1,268 windows annotated. See `PRIOR-ART.md` |
| **Rent3D++ / Plan2Scene** (SFU, CVPR 2021) | Application (Google Form) | Same 215 apartments, richer annotations, **and the coverage-level evaluation protocol we adopted as M8**. Code is open, so the pipeline is reusable even without the data |
| LIFULL HOME'S (NII Japan) | Application (academic) | Wrong market regardless |
| Places365 / MIT Indoor67 | Open | Largely moot — zero-shot SigLIP is already at F1 0.96 |
| Kaggle/HF room-image sets | Open, quality varies | House Rooms & Streets ~25K; MMIS 160K |

## Splatting & delivery

| Component | Availability | Notes |
|---|---|---|
| **gsplat** (Nerfstudio) | Open — **build config matters** | The rasteriser/trainer of record. Its default CUDA arch list **excluded T4 and P100**, precisely the GPUs free tiers hand out; we build `6.0;7.0;7.5;8.0;8.6;8.9;9.0+PTX`. Also pick AMP dtype by compute capability — **T4/V100 have no native bf16** |
| Inria/MPI 3DGS reference | Open | Reference implementation of the published method |
| DNGaussian | Open | Depth regularisation; reimplement natively on gsplat |
| SparseGS | Open | |
| InstantSplat (NVlabs) | Open | Pipeline idea (pointmap init → joint pose+splat optimisation) is what we reimplement on MapAnything + gsplat |
| **AnySplat** (SIGGRAPH Asia 2025) | Open | Prime instant-profile engine candidate (unposed, 2–64 views) |
| **DepthSplat** (cvg) | Open — check checkpoint provenance | Verified 0.6 s / 12 views on A100. Second instant-engine candidate (posed input — fine, stage 3 provides poses) |
| pixelSplat / latentSplat | Open | Stereo-pair only; partial coverage |
| GS-LRM (Adobe) | **Dead** — no official open release | |
| Long-LRM / Long-LRM++ | Unofficial reimplementations only | |
| MV-DUSt3R+ (Meta) | Open | |
| PreF3R | Open | |
| FastGS | Open | Standard-profile fast-optimisation candidate |
| Taming-3DGS | Open | Comparison point (7–13 min/scene A100) |
| Splatt3R | Open | |
| MVSplat | Open | |
| **SPZ** (Niantic) | Open | Primary splat delivery format |
| SOGS (PlayCanvas) | **Archived** | Use splat-transform |
| **splat-transform** (PlayCanvas) | Open | Maintained; multi-LOD chunked SOG streaming |
| **Spark** (World Labs) | Open | Hybrid splat+mesh Three.js renderer; reads PLY/SOG/SPZ/SPLAT/KSPLAT |
| @mkkellogg/GaussianSplats3D | Open | No longer actively developed; README points to Spark |
| Three.js | Open | Claim of a native WebGPU splat renderer in r186 unconfirmed at review — don't plan around it |
| KHR_gaussian_splatting (Khronos) | **Release Candidate, not ratified** | Adopt when ratified and supported |
| **glTF-Transform** / **meshoptimizer** / **xatlas** / **KTX-Software** | Open | |
| **MVS-Texturing** | Open | Projective texturing of the shell |
| OpenMVS | Open | MVS-Texturing covers the need; no reason to add it |

## Generative / world models (Phase 4 only)

| Component | Availability | Notes |
|---|---|---|
| World Labs Marble | Commercial product; Free / $20 / $35 / $95 per month tiers, generation API | Paid tiers only above trivial volume |
| Stable Virtual Camera | Open | |
| HunyuanWorld 1.0 (Tencent) | **Region-gated** — not distributed in the EU, UK or South Korea | Practically unobtainable from here |
| CAT3D / CAT4D (Google) | **Dead** — research paper, no code | |

## Data we build

| Asset | Where it lives | Notes |
|---|---|---|
| Golden set (30 UK listings, 553 images) | `data/golden/` — manifest committed, **media gitignored** | Rebuild with `make golden`. Media is re-fetchable from the URLs in the manifest, which is also how the GPU script runs without an upload |
| Scraped plan corpus | `$VISITIT_DATA_HOME` | Fine-tuning basis for the plan vectoriser, once there is volume |
| Synthetic plan generator output | ours | Augmentation for vectoriser training |

## Process

- **Owner:** whoever adds a dependency adds the row in the same PR — the value here is the "what broke when I tried" column, and it is only worth anything if it is written while the memory is fresh.
- **No CI gate.** The licence gate described in earlier revisions is not built and is not planned.
- **Fetching:** everything auto-fetchable is registered in `pipeline/datasets/registry.py` and pulled by `make data`. Downloads resume via HTTP Range, checksum into `datasets.lock.json`, and a manually-supplied archive always wins — which matters, because several upstream links above are dead or gated.
- **NOTICE file:** still generated at build. Attribution costs nothing and several of these ask for it.
