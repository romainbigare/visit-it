# Phase 0 — what we did and what we found

**Period:** 19–20 August 2026 · **Market:** UK · **Compute:** 4-vCPU sandbox, then a free Colab T4

Phase 0 exists to make every later claim measurable and to kill the biggest unknowns while the codebase is still small. This records what was built, what the numbers said, and — as importantly — what was *not* done.

**Headline:** the technical risk came down a long way. Three of the four questions Phase 0 was meant to answer are settled, and the fourth is open exactly where the roadmap always said it would be decided. **Gate G0 passes on its revised criteria**; two infrastructure items are outstanding and are the first work of Sprint 1. See §6.

> **Scope note (20 Aug 2026).** This report was written against a commercial
> gate: verified tape/laser measurements, an agency-CRM connector, a
> derivative-works rights grant, and licence CI. The project is internal and
> non-commercial, so those criteria are withdrawn — §6 records the revised
> assessment, and §6b records what the original criteria were, since the
> measurements would genuinely have told us something the replacements cannot.

---

## 1. Data acquisition

**Rightmove works; Zoopla does not.** Rightmove's `robots.txt` permits the search and property pages for a generic agent (named AI crawlers — GPTBot, CCbot, SpriftCrawler — are disallowed outright, and we don't impersonate them). Search results parse from an embedded `__NEXT_DATA__` blob; detail pages use a `devalue`-flattened payload where every value is an index into one array, which needed a resolver. Zoopla sits behind Cloudflare and returns HTTP 403 to every request from a datacentre IP — including for `robots.txt` itself. Its adapter is written but its field mapping is **unverified**, and marked as such in the code.

**The golden set: 30 listings, 553 images.** Spread across London, Manchester, Birmingham, Leeds, Glasgow and Bristol, stratified by price band. 24 have floor plans (80%); the 6 without are deliberate — they're the test cases for the no-plan branch.

**Floor-plan coverage across everything seen, unfiltered: 92.5%** (98 of 106; 8 missing). Better than the plan assumed. Median 17 photos per listing. About 53% of listings carry some stated floor area.

Two things that took real effort to get right:

- **Python's standard `robotparser` silently ignores `*` wildcards**, so it reported Rightmove's disallowed paths as fetchable. We wrote our own matcher with `*`/`$` and longest-match precedence.
- **An unstratified London search returns almost nothing but prime-central trophy flats** — One Hyde Park, Mayfair, 20+ professionally shot photos each. That would have flattered the pipeline badly at G1. Collection is now stratified across cities and price bands, with a filter for blocks, portfolios and off-plan "6% rental yield" listings, which are typically illustrated with CGI renders rather than photographs.

Scraped imagery is not committed to the repo — it keeps the repository small and the images are re-fetchable from the manifest — and every asset carries a `provenance` tag, which is what lets us answer "where did this input come from?" when a reconstruction misbehaves.

## 2. Dataset pipeline

Ten corpora registered with live-verified URLs: Swiss Dwellings, CubiCasa5K, ResPlan, MSD, C3Po, ZInD, Structured3D, Rent3D, Rent3D++, Hypersim. Built for moving environments — one env var sets the cache root, downloads resume via HTTP Range (verified by killing a 5.5 GB transfer at 40 MB and watching it continue), every file is checksummed into a committed `datasets.lock.json`, and manually-supplied archives always win, which matters because several upstream links are dead.

Swiss Dwellings is downloaded and verified end to end (792 MB, 2.5 M rows).

**Rent3D's advertised download is a dead link** — 404 from both Toronto hostnames, confirmed twice. The fetcher falls back to printing a ready-to-send enquiry to the authors.

## 3. Model validation — CPU

| Stage | Model | Result |
|---|---|---|
| Triage | SigLIP zero-shot | **F1 0.96**, recall 1.00 after adjudication, 177 ms/image |
| Monocular geometry | MoGe-2 | Median ceiling **2.71 m**, 85% plausible |
| Monocular geometry | Depth Anything V2 | **Rejected** — see below |
| Floor-plan OCR | Tesseract | Labels on 83%, dimensions on 54% |

**The triage result contains a finding worth keeping: the model is more accurate than the portal's own metadata.** Of six disagreements, three were real floor plans that Rightmove had filed as photos, and one "floorplan" was an entirely black image — a corrupt asset. Only two were genuine errors. **Portal floor-plan metadata is not a reliable reference.**

**The most consequential finding was about camera lenses.** Depth Anything V2 gives metric depth but no camera intrinsics, so turning its depth into a point cloud requires *assuming* a field of view — and the estimated room height scales almost linearly with that assumption: 2.76 m at 60°, 5.93 m at 104°. Only a narrow lens gives plausible ceilings. But **MoGe-2, which predicts intrinsics, measured the real field of view at a median of 98.6°** — confirming from our own data that estate agents shoot ultra-wide. At the true field of view, Depth Anything's geometry gives ~6 m ceilings, which is nonsense. **A depth-only model is unusable for metric reconstruction from listing photos.** That is decision AD-5 validated empirically rather than argued.

**Floor-plan OCR largely settles the UK scale problem.** There is no legally-mandated area figure in a UK listing, but printed room dimensions appear on 54% of plans (62% of high-resolution ones) — better than the EPC-only fallback we assumed. They do double duty: a scale constraint *and* the self-consistency check that stands in for tape measurements (ROADMAP §0b). Nearly half carry an explicit "not to scale" disclaimer, so resolution matters and the scraper should always take the largest plan asset.

## 4. Model validation — GPU (Tesla T4, free Colab tier)

Modal was set up first but turned out to require a payment method for all compute, not just GPUs. The work runs on a free Colab T4 via a portable script that needs no upload — it re-fetches the images it needs from the URLs already in the manifest.

**MoGe-2: 0.364 s/image versus 14.4 s on 4 CPU cores — a 39× speedup.** Median predicted field of view 98.5°, against 98.6° on CPU. Agreement to a tenth of a degree across different hardware and precision is good evidence the ultra-wide finding is a property of the photographs, not an artefact.

**MapAnything: 12 of 12 groups, 1.35 s each.** No collapsed reconstructions — camera baselines 3.2–12.7 m, none near zero, which was the failure mode to watch for. Vertical extent median 2.91 m, independently consistent with MoGe's 2.71 m ceiling estimate from single images. Reconstructed size tracks the real properties: 1-bed flats come out small (4.2 × 3.3 m), big new-builds large. The 10–16 m horizontal extents on glazed city apartments are almost certainly content seen through windows — a stage-4 clipping job the architecture already specifies.

**gsplat: the chain is proven; novel-view quality is not.**

| | median |
|---|---|
| Training-view PSNR | **25.9 dB** |
| Held-out-view PSNR | **9.8 dB** |
| Training time | 10.4 s per room |

The 25.9 dB fit is the result we needed. If MapAnything's poses were wrong, or its intrinsics, or the cam2world→viewmat inversion, or the point-to-pixel colour mapping, the training views could not converge at all. They converge. **The stage 3 → stage 8 chain is correct with no COLMAP anywhere** — decision AD-6 validated.

The 9.8 dB novel-view score is the regime the feasibility report named: *"below 3 views floaters and smearing dominate."* Each scene had three views with one held out — **two supervising views**, against the hundred-plus a normal splat capture uses. Both designed mitigations are unbuilt: the T4's 16 GB caps MapAnything at 3 views, and stage 4 doesn't exist so nothing culls splats against the room polygon. Appearance quality is therefore **untested rather than poor**.

## 5. Cost — measured, not estimated

From the measured per-stage timings, for a median listing (17 photos, ~5 rooms):

| Splat schedule | GPU seconds | T4 (~$0.50/hr) | L4 (~$0.80/hr) | L40S (~$1.95/hr) |
|---|---|---|---|---|
| 1500 iterations | 65 s | **$0.009** | $0.014 | $0.035 |
| 4000 iterations | 152 s | **$0.021** | $0.034 | $0.082 |

**Comfortably inside the £0.05 target**, and on the weakest current GPU. The feasibility report estimated €0.30–1.50 per listing; the measured figure is an order of magnitude lower, because feed-forward models replaced the per-scene optimisation that estimate assumed. Compute is not the constraint — human review time will be, exactly as the report argued.

## 6. Gate G0 — honest assessment

**Passed, with two items carried into Sprint 1.** The gate was rewritten on 20 August when the project's scope became explicitly internal and non-commercial (ROADMAP §0b, §4). Criteria that existed to make this shippable — verified measurements, a rights grant, an agency connector, licence CI — are withdrawn. What is left is the engineering question the gate was always really asking: *does the chain work, at what cost, and can we tell when it doesn't?*

| Criterion | Status |
|---|---|
| ≥30 listings collected, stratified, with images | ✅ 30 across 6 cities and 4 price bands |
| ≥65% carrying a floor plan | ✅ 80% (24/30); coverage across everything seen 92.5% |
| Dataset pipeline reproducible on a fresh machine | ✅ `make data`; resumable, checksummed, manual-override |
| Triage usable without training data | ✅ F1 0.96, and more accurate than the portal's own metadata |
| Monocular geometry gives plausible ceilings | ✅ MoGe-2 median 2.71 m, 85% plausible |
| MapAnything runs at seconds-per-group | ✅ 1.35 s, 12/12, no collapsed reconstructions |
| Cross-model agreement on scale | ✅ MoGe 2.71 m vs MapAnything 2.91 m from independent methods |
| gsplat chain proven end to end without COLMAP | ✅ Train-view PSNR 25.9 dB — see §4 for why that is the load-bearing number |
| Measured GPU cost per listing | ✅ $0.009–0.021 on a T4 |
| Holdout split frozen | ❌ **Not done.** First Sprint 1 task — it costs an afternoon and it stops being possible to do honestly the moment tuning starts |
| gsplat → SPZ → Spark viewer proven | ⚠️ gsplat runs; no SPZ export, no viewer test, no device matrix |
| Latency instrumentation | ⚠️ Per-stage timings exist; no dashboard |

**The one that matters is the holdout freeze.** Everything else outstanding is additive; that one degrades with time, because a split chosen after you have seen how the pipeline behaves is not a split.

**What we can and cannot claim.** With no measured reference, every number here is *plausibility* and *self-consistency*, not *accuracy*. A 2.71 m ceiling is a credible ceiling; we have not established that it is the right ceiling. Two independent models agreeing on 2.71 m and 2.91 m is real evidence — they share no architecture and one uses a single image while the other uses three — but agreement is not correctness, and both could share a bias inherited from similar training data. The honest ceiling on our claims is therefore "consistent and plausible", and ROADMAP §0b sets the gates accordingly.

### 6b. What the withdrawn criteria would have bought

Recorded because dropping them was a scope decision, not a discovery that they were worthless:

- **Verified measurements** (tape/laser on ~10 flats, 2–3 paid scans) were the only route to an *accuracy* number. Without them we cannot distinguish a systematic 8% scale bias from a correct reconstruction — both look plausible and both are self-consistent. If a number ever needs defending, this is the half-day of physical work that defends it.
- **The rights grant, agency CRM connector, GDPR process and licence CI** were all launch requirements. They bought nothing technical and are correctly gone.

## 7. Decisions this changed

- **AD-5 confirmed empirically.** MoGe-2 over depth-only models, because intrinsics are not optional.
- **AD-6 confirmed mechanically.** The no-COLMAP chain works.
- **Cost model revised down** by roughly an order of magnitude against the feasibility report's estimate.
- **The vectoriser corpus is settled.** ResPlan (17K vector) plus Swiss Dwellings (42K apartments, European) plus CubiCasa5K's raster half cover what the plan vectoriser needs, with scraped UK plans for style fine-tuning.
- **Portal metadata demoted.** It cannot be used as an evaluation reference.
- **Stage 4 promoted.** Building the room polygon now unblocks Phase 1 *and* fixes the dominant appearance artefact, and needs no GPU. It is the highest value-per-effort task available.

## 8. Bugs worth remembering

Several were found only by looking at outputs rather than at metrics:

- **A flat grey render scored ~12 dB**, which reads as merely-poor in a results table. Only opening the PNG revealed it was nothing at all. The stage now measures render contrast against ground-truth contrast and flags it.
- **Gaussian colours were never initialised from the photographs** — every splat started mid-grey and was expected to discover its colour through optimisation. It couldn't.
- **Gaussian size came from distance-to-centroid** rather than nearest-neighbour spacing, seeding half-metre blobs on a 10 m room. This was also a *speed* bug: fixing it cut training from 143 s to 10 s.
- **A 0.02 confidence threshold** cut the room-grouping yield from 26 groups to 2, because SigLIP's scores are poorly calibrated in absolute terms. The argmax is the signal; the magnitude is not.
- **Modal's `TimeoutError` does not subclass the stdlib one**, and `OutputExpiredError` *does* subclass Modal's — so a naive except-ordering reports an expired job as "still running" forever.
- **The CUDA architecture list excluded T4 and P100**, precisely the GPUs free tiers hand out.

## 9. What to do next, in order

1. **Freeze the holdout split.** An afternoon, and it expires — do it before any tuning.
2. **Build stage 4 (the room polygon).** No GPU needed. Unblocks Phase 1 and targets the artefact dominating the splat renders: nothing currently culls splats to the room, so windows and reflections bleed into the geometry.
3. **Build plan OCR into the scale solve.** Dimensions printed on 54% of plans are both a constraint and the self-consistency check that the gates now rest on (ROADMAP §0b). CPU-only, small, high value.
4. **Re-run reconstruction at 6–8 views** on a card with more than 16 GB, then re-measure appearance. The 9.8 dB held-out score was measured with two supervising views; it is untested, not poor.
5. **SPZ export and a viewer smoke test**, to close the last unproven link in the delivery chain.

Item 1 is the only one that gets harder if deferred.
