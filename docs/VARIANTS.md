# Service variants, price points, and alternative development journeys

**Added 19 Aug 2026.** Trigger: the target of serving paying customers at **5–10 seconds per analysis, under £0.05 each, at crowd scale**. This document works out what that target actually implies, defines three service **profiles** over the one pipeline (ARCHITECTURE AD-16/17/18), prices them, and lays out three alternative development journeys with a recommendation.

Cost basis: the feasibility report's §8 figures (L40S/A100-class at ~$1–2/hr rented) plus stage timings from the report; £1 ≈ $1.27. Figures marked **(est.)** are engineering estimates to be replaced by measured numbers in P0 — an evidence appendix at the end tracks verification status. Everything here is COGS (cost of goods sold), not price.

---

## 1. First, the two reframings that change the maths

### 1.1 Per-analysis vs per-view

A listing is reconstructed **once** and viewed by **many**. The crowd of paying customers never touches a GPU — viewers are served static artifacts (shell glTF + SPZ splats) from a CDN. So there are two different unit costs, and the £0.05 question must be asked of the right one:

- **Per unique listing (reconstruction):** the GPU cost. This is where 5–10 s and £0.05 bind.
- **Per view (serving):** CDN egress of a 5–25 MB payload ≈ **£0.000–0.002/view** (est.; near zero on an R2-class zero-egress CDN). Effectively free at any crowd size.

With a **content-hash cache** (AD-18: hash of photos + plan + listing text), a repeated analysis is a cache hit at ~zero cost. In a portal integration, every listing is analysed exactly once regardless of traffic; in a consumer "paste any listing" app, popular listings amortise the same way. **The £0.05 target therefore binds only on cache-miss reconstructions.**

### 1.2 What actually breaks a £0.05 budget

£0.05 ≈ $0.064 buys (verified pricing, Aug 2026): **~230 GPU-seconds** of on-demand L40S (RunPod $0.99/hr), **~90 s** of serverless A100 (Modal $0.000694/s = $2.50/hr), or **~50 s** of serverless H100 (RunPod $0.00126/s). Two structural facts matter more than the exact rates:

- **Serverless GPU is billed per-second with scale-to-zero**, so at low volume you pay the premium per-second rate but *no idle cost* — the classic utilisation discount largely disappears from the maths; at sustained volume you switch to on-demand/reserved at ~2.5× cheaper per second.
- **Cold starts are no longer fatal**: RunPod FlashBoot advertises 0.5–2 s, Modal 3–4 s with cached images (verified from vendor pages; multi-GB *weight loading into VRAM* still needs its own P0 measurement — vendor cold-start figures assume cached/snapshotted workers).

Against that budget, the cost structure of the committed pipeline:

| Cost item | Typical (verified basis + est.) | Fits in £0.05? |
|---|---|---|
| Triage via frontier/mid VLM (20 imgs, full res) | £0.02–0.06/listing (Sonnet-class, ~1 tok/750 px) | **No** — eats most or all of the budget alone |
| Triage via Haiku-class VLM, images downscaled to ~768px | ~£0.01–0.02/listing | Borderline — OK for Standard, not Instant |
| Human review (25% rate × 3 min at £20–25/hr) | ~£0.25 amortised | **Absolutely not** — 5× the whole budget |
| Per-scene optimised splatting, classic (5–10 GPU-min) | £0.08–0.28 | **No** |
| Per-scene optimised splatting, FastGS-class (~100 s/scene, verified claim) | ~£0.02–0.04 | Marginal — this is the Standard profile's budget |
| Feed-forward pipeline, all stages (~15–30 GPU-s total; DepthSplat runs 12 views in **0.6 s on an A100**, verified) | **£0.004–0.02** | **Yes, with margin** |
| Fine-tuned classifier triage (self-hosted, batched) | ~£0.001 | Yes |
| CPU stages, storage, orchestration | ~£0.002–0.005 | Yes |

Conclusion: **£0.05 is comfortably achievable for compute — but only under three disciplines**: no frontier-VLM in the path, no human review in the path, no per-scene optimisation loops. That is a specific product profile, not a cheaper version of the same product.

### 1.3 The latency constraint is harder than the cost constraint

5–10 s wall-clock forces: feed-forward models only (a single forward pass per stage) and **per-room fan-out** (5 rooms reconstructed on 5 workers in parallel — serverless makes burst fan-out natural). The cold-start question is now empirical rather than fatal: vendor-verified figures (FlashBoot 0.5–2 s, Modal 3–4 s) suggest scale-to-zero *may* be compatible with a 10 s p95, but those figures assume cached workers — the P0 spike must measure real cold-to-first-inference with our 5–10 GB of weights, and the fallback is a small warm pool (one always-on L40S ≈ £560/month at RunPod's verified $0.99/hr, amortised over volume). Either way, per-second billing means the **cost** target holds even at pilot volume; it is only the **latency tail** (p95 vs p50) that scale improves.

---

## 2. The three service profiles (one pipeline, per-stage bindings)

| | **INSTANT** | **STANDARD** | **PREMIUM** |
|---|---|---|---|
| Latency SLO | **p95 ≤ 10 s** (warm) | ≤ 3–5 min, async | ≤ 24 h, incl. human pass |
| COGS target (at scale) | **≤ £0.02–0.03** | ≤ £0.10–0.15 | £1.0–1.7 |
| Geometry | MapAnything fan-out per room; MoGe-2 singletons | same | same + COLMAP path where dense |
| Appearance | Shell + projected textures + **feed-forward splats** — candidate engines, verified: **AnySplat** (MIT code, *unposed*, 2–64 views — weights licence unstated, ledger check pending) and **DepthSplat** (MIT, posed — we have poses from stage 3; 0.6 s/12 views on A100). Falls back to textures-only | + **optimised gsplat** (FastGS-class fast config ≈100 s/scene → ~10–30 s/room, depth-regularised, polygon-culled) | + high-iteration splats, careful texture blending, mirror handling at full strength |
| Triage/adjudication | Fine-tuned SigLIP-class classifiers only | + small fast VLM on low-confidence cases | + frontier VLM adjudication |
| Human involvement | **None.** Confidence-gated: degrade to shell-only / Tier-B badge, or refuse with a reason | Sampled audit (2–5%), non-blocking | Review console pass on every listing |
| Quality posture | Honest and legible; softer splats acceptable; refusal is a feature | The G2 "beats the gallery" bar | Guaranteed arrangement, agency-branded |
| Failure mode | Visible degradation, never silent wrongness | flagged + queued | human-corrected |

The `instant` profile's output is a strict subset of `standard`'s stages with cheaper bindings — so a listing analysed instantly can be **upgraded in place** (standard splats arrive 3 minutes later; the viewer hot-swaps them). That "instant result, quality follows" pattern is the best of both and costs nothing extra architecturally: same DAG, second pass on stages 8–9 only.

## 3. Price-point variants (business shapes these profiles support)

| Variant | Profile | Price point | Margin logic | Notes |
|---|---|---|---|---|
| **V1 — Portal/API at scale** | Instant (+background upgrade to Standard) | **£0.03–0.10 per unique listing**; cache hits free; or per-1,000-views | COGS £0.02–0.05 blended ⇒ thin per-unit, volume business; the user's stated target lives here | Needs sustained volume for warm-pool utilisation; contract minimums solve that |
| **V2 — Agency self-serve** | Standard | **£0.50–1.50 per listing** (or bundles: e.g. £99/mo for 100) | COGS ≤£0.15 ⇒ 80%+ gross margin at modest volume | The default wedge; matches the committed roadmap |
| **V3 — Premium marketing** | Premium | **£5–20 per listing** | COGS ~£1.1–1.7 (review-dominated) ⇒ 70–90% margin | Reference points: a Matterport shoot is £100–300; virtual staging £15–30/photo. Easy value story |
| **V4 — Consumer one-off** | Standard (async ~2–3 min) | **£0.99–2.99 per listing** | Impulse price, COGS ≤£0.15 | Buyer pastes a listing they're viewing. ⚠️ input-rights question (user-supplied images of someone else's listing) — counsel before launch; V1–V3 use partner-licensed inputs |

These are not mutually exclusive — they are one pipeline sold three ways, and V2 revenue funds the utilisation that makes V1's price point real.

---

## 4. Three development journeys

The committed roadmap (ROADMAP.md, 19 Aug 2026) is quality-first. The £0.05/10 s target gives two alternatives. Deltas are expressed against the committed plan.

### Journey A — Instant-first ("consumer wedge")

Build the 5–10 s path from day one; optimised splatting waits.

- **P0 (S1–S2):** unchanged, plus: feed-forward-splat spike replaces the gsplat-optimisation spike; latency instrumentation in the stage runner; warm-pool-vs-serverless infra spike.
- **P1 (S3–S6):** shell exactly as committed **but with the p95 ≤ 10 s SLO as a G1 criterion** (the shell path is seconds-scale by construction — this is cheap here).
- **P2 (S7–S10, one sprint shorter):** grouping/geometry as committed; appearance = feed-forward splats + projected textures only; viewer as committed. G2 quality bar *lowered honestly*: instant output must beat a "floor plan + photo lightbox" baseline, not the full gallery-preference bar.
- **P3:** optimised-splat engine added as the Standard profile; review console becomes an internal QA tool only.
- **Timeline: instant product GA ~week 20; standard quality ~week 30.**
- **Choose when:** the buyer is a portal/consumer product and speed-at-price is the wedge.
- **Risks:** quality ceiling early (feed-forward splats are visibly softer — the wow factor that sells B2B demos arrives late); the G2 "beats the gallery" proof is deferred, so the appearance-quality risk (R4) is discovered latest, which is the wrong order for the riskiest unknown.

### Journey B — Quality-first (the committed roadmap, unchanged)

- Instant profile is retrofitted in P3+ (engine swaps + serving work: warm pools, classifier triage, fan-out).
- **Timeline: standard GA ~week 30; instant ~week 36–38.**
- **Choose when:** the first customer is an agency and the demo must dazzle.
- **Risks:** hot-path retrofit late (sync serving, no-VLM triage, per-stage latency budgets touch every stage after the fact — the classic "we'll optimise later" trap AD-17 exists to prevent); the £0.05 market stays unserved for 9 months.

### Journey C — Twin-track profiles ★ recommended

One DAG, both bindings, latency as a first-class metric from S1. Concretely, against the committed plan:

- **P0:** S1 adds the feed-forward-splat spike (alongside, not instead of, the gsplat spike) and the warm-pool infra spike; the stage runner logs per-stage wall-clock from day one (M12). **G0 adds:** latency harness live; utilisation cost model in the dashboard.
- **P1:** unchanged in content — the shell path *is* the instant path. **G1 adds:** shell profile end-to-end **p95 ≤ 10 s warm** and **measured COGS ≤ £0.02** at simulated 50% utilisation.
- **P2 (+1 sprint, S7–S12):** the D-stream builds **two stage-8 engines behind one interface**: feed-forward (S7–S8, instant) and optimised gsplat (S9–S10, standard). Viewer gains the hot-swap upgrade path. **G2 measured per profile:** standard must beat the photo gallery (unchanged); instant must beat the plan+photos baseline and hold p95 ≤ 10 s / COGS ≤ £0.03.
- **P3:** review console scoped to Standard/Premium only (never gates instant); S15 adds warm-pool autoscaling + SLO monitoring. **G3 adds:** per-profile COGS at declared price points, measured over a 200-listing/day week.
- **Timeline: both profiles GA ~week 32 (+2 weeks vs committed plan).**
- **Why recommended:** +2 weeks buys both markets; the latency discipline is architectural (per-stage budgets, fan-out, warm pools) and near-free to impose early but expensive to retrofit (Journey B's trap); and the riskiest unknown (does splat quality beat the gallery?) is still tested at the same point as the committed plan (Journey A's trap avoided).

### Decision table

| If the first paying customer is… | …take | First revenue | Both profiles live |
|---|---|---|---|
| A portal / high-volume API buyer | Journey A | wk 20 (instant) | wk 30 |
| A single agency wanting beautiful tours | Journey B | wk 30 (standard) | wk 36–38 |
| Unknown / both plausible | **Journey C** | wk 22–24 (instant shell+FF) | **wk 32** |

The decision is commercially led: **pick the journey when the first customer conversation lands (P0 exit at the latest)** — G0 already forces the partner question, so the information arrives exactly when the fork must be chosen. Until then, Journey C's P0/P1 deltas (latency instrumentation, feed-forward spike, SLO criteria) are cheap and keep all three journeys open — they are adopted into the plan **now**.

---

## 5. What the £0.05 / 10 s target explicitly gives up (so nobody is surprised later)

1. **No frontier-VLM judgement per listing** — classifier-grade triage means slightly more wrong room labels; the confidence gate turns those into Tier-B badges or refusals, not errors.
2. **No human safety net** — the instant profile's honesty machinery (provenance shading, refusal, degradation) *is* the quality control. This is why it was built into the architecture rather than bolted on.
3. **Softer appearance** — feed-forward splats in 2026 are below optimised splats; the upgrade-in-place path exists precisely so instant customers can still get standard quality minutes later.
4. **Latency-tail scale dependence** — per-second serverless billing keeps *COGS* honest even at pilot volume, but p95 latency at low volume depends on cold-start behaviour with our weights (P0 spike); if it disappoints, a small warm pool (~£560/month/GPU) is the fix and gets amortised only with volume.
5. **Premium economics are review-bound, not compute-bound** — the report's core operational finding survives every variant: cutting the review *rate* is worth more than any GPU optimisation.

---

## 6. Evidence appendix — verified 19 Aug 2026 (Haiku research agents, primary sources)

### GPU & serving economics

| Fact | Verified figure | Source |
|---|---|---|
| L40S on-demand | $0.99/hr (RunPod); from $0.40/hr (Vast.ai marketplace) | runpod.io/pricing |
| A100 80GB on-demand | $1.39/hr (RunPod); $1.99–2.79 (Lambda); $3.18 (Paperspace) | vendor pricing pages |
| H100 on-demand | $2.89–3.29 (RunPod); $3.29–3.99 (Lambda) | vendor pricing pages |
| Serverless per-second, scale-to-zero | Modal A100 $0.000694/s ($2.50/hr), H100 $0.001097/s; RunPod H100 $0.00126/s; Replicate A100 $0.0014/s | modal.com, runpod.io, replicate.com pricing |
| Serverless cold start | RunPod FlashBoot 0.5–2 s; Modal 3–4 s typical (cached images). Industry moved from 30–60 s to sub-2 s in 2025–26. Vendor figures assume cached workers — measure with our weights in P0 | vendor blogs/docs |
| CDN egress | Cloudflare R2: **$0.00/GB**; Bunny $0.01/GB ($0.005 at 500 TB+); CloudFront $0.085/GB first 10 TB | vendor pricing pages |
| VLM vision pricing | Haiku-class $1/MTok in, $5/MTok out; images ≈ 1 token/750 px (a 768×768 image ≈ 0.8k tokens → 20-image listing ≈ £0.01–0.02 at Haiku rates; full-res or mid-tier models 3–5×) | platform.claude.com pricing |

Implication check: instant-profile GPU budget of ~15–30 GPU-s costs $0.010–0.021 on serverless A100 = **£0.008–0.017** — inside the £0.05 envelope with room for CPU, storage and payment fees. Per-view serving on an R2-class CDN is ~£0.000.

### Feed-forward splatting & fast optimisation

| Model | Speed (verified) | Views / posing | Licence | Fit |
|---|---|---|---|---|
| **AnySplat** (SIGGRAPH Asia 2025) | "real-time" class; exact s/scene unpublished — P0 spike measures | 2–64, **unposed** | Code MIT; **weights licence unstated on HF** | Prime instant-engine candidate, pending weights-licence check |
| **DepthSplat** (CVPR 2025) | **0.6 s for 12 views @512×960 on A100** | posed (we have stage-3 poses) | MIT | Strong instant-engine candidate |
| MVSplat | ~22 fps claim | sparse, posed | paper CC-BY; code licence to confirm | Candidate |
| pixelSplat / latentSplat | 0.1 s encode / <100 ms | 2, posed | MIT / MIT | Pairs only — partial coverage |
| GS-LRM (Adobe) | 0.23 s on A100 | 2–4 posed | **Adobe Research License** (no official open release) | Ruled out |
| Long-LRM / ++ | 1 s / 32 views; 5 s / 64 views (A100) | posed | unclear / unofficial reimpl. | Watch |
| MV-DUSt3R+ (Meta) | 0.89–1.54 s (12–20 views) | **unposed** | CC-BY-NC-4.0 | Research only |
| PreF3R | 20 fps incremental (H100) | unposed sequences | unstated | Watch |
| **FastGS** (per-scene opt.) | **~100 s/scene**, 3.3–15× vs vanilla 3DGS | n/a | "MIT" but says it must adhere to 3DGS/Taming/Speedy-Splat licences — **legal check required** (possible Inria taint) | Standard-profile candidate, conditional |
| Taming-3DGS (per-scene opt.) | 7–13 min on A100 | n/a | unstated | Comparison point |
| MapAnything | GPU per-pass time **not published** — P0 spike measures (only a CPU figure found) | 1–2000, unposed | Apache (code + `-apache` weights) | Committed engine; timing assumption must be validated in S1 |

All rows above must clear `docs/LICENSING.md` (rows added there) before an engine choice is final; the two open items that could move the plan are AnySplat's weights licence and FastGS's licence chain.
