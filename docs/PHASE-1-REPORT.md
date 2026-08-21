# Phase 1 — the measurable shell

**Period:** 20–21 August 2026 · **Market:** UK · **Compute:** 4-vCPU sandbox, no GPU

Phase 1's goal, from the roadmap: *listing in → scaled, correctly-arranged,
untextured 3D shell + interactive floor plan out, in the browser.* No splats. The
report calls this the "deliberately unglamorous and right" first product, and it
front-loads the stage most likely to kill the project — global assembly — and the
discipline most likely to save it — the eval harness.

**Headline: the chain is built and it runs end to end on a laptop.** All ten
stages are implemented, schema-validated and instrumented; a listing goes from a
directory of estate-agent photographs to a walkable glTF shell in about 90 seconds
on four CPU cores. Two of the four G1 criteria pass comfortably. One is
architecturally sound but limited by a component we deliberately shipped in its
simplest form. The fourth cannot be judged at the coverage we have, and this
report says so rather than rounding it up.

> **Scope.** Internal and non-commercial. There is no tape-measure ground truth
> for any of this, so every number below is *plausibility* and *self-consistency*,
> never *accuracy* (ROADMAP §0b). A 2.5 m ceiling is a credible ceiling; we have
> not established it is the right one.

---

## 1. What was built

Ten stages, each emitting one schema-validated artifact carrying a confidence and
a list of QA flags. Streams B (photographs) and C (the plan) meet only at stage 6,
exactly as the dependency graph says they should.

| | stage | what it does | engine |
|---|---|---|---|
| 0 | triage | classifies every image, parses the listing's own numbers | SigLIP zero-shot, Phase 0's prompts verbatim |
| 1 | conditioning | field-of-view prior, rectification gate | Phase 0's measured 98.6° prior |
| 2 | grouping | photos → rooms, high precision | label-based (Sprint 7 replaces it) |
| 3 | geometry | per-room point maps with predicted intrinsics | MoGe-2 (AD-5) |
| 4 | layout | floor/ceiling/wall planes, room polygon, apertures | pointmap geometry |
| 5 | plan | **the spine** — room polygons, adjacency, doors, metric scale | classical raster vectoriser |
| 6 | assembly | which reconstructed room goes in which plan polygon | Hungarian + SE(2) |
| 7 | scale | one global scalar, and the three §0b checks | weighted least squares in log space |
| 8 | shell | extrusion to glTF, apertures cut, provenance per face | — |
| 9 | package | `scene.json`, the viewer's only input | — |

Around them: a content-addressed artifact store with cross-process locking, a DAG
runner that skips dependents of a failed stage rather than feeding them garbage,
per-stage latency budgets asserted in CI, a run ledger, an eval harness with
plan-channel isolation, a nightly batch runner with regression alerts, a review
console with two working fix actions, and a Three.js viewer.

### Two deliberate substitutions

Both are recorded here rather than quietly made, and both sit behind the AD-4
engine interface so swapping them changes no artifact contract.

**The plan vectoriser is classical, not learned.** The roadmap specifies a
RoomFormer-class network pretrained on ResPlan + Swiss Dwellings + CubiCasa5K and
fine-tuned on our own annotated plans. That is the right long-term answer and it
needs a GPU, three datasets and an annotation drive. What shipped is a raster
engine that needs none of them, so stages 6–9 had real input from day one — and it
gives the learned model a measured baseline it has to beat. Its method is worth
recording because it is not the obvious one: **the plan's own captions seed the
segmentation.** UK agency plans label every room, so a watershed over the ink-free
space seeded from the label captions puts exactly one basin where a human would
point. OCR tells us which ink is lettering, so it can be erased before the geometry
runs. Unlabelled rooms are seeded from distance-transform peaks inside the
building. And openings are simply *where two watershed basins touch* — a definition
that needs no wall-thickness tuning at all, which three earlier attempts did.

**Aperture detection uses geometry, not SAM2 + Grounding DINO.** Same reasoning,
plus one observation that made it easy: a pointmap model returns nothing useful
through glass. A window is a hole in a wall's point coverage at chest height; a
door is one that reaches the floor.

---

## 2. Gate G1, criterion by criterion

Measured on the frozen holdout (sealed before any tuning; the seal is checked in
CI). Coverage is reported next to every number, because a pass rate over three
listings and over sixteen are different claims.

<!--G1_TABLE-->

### Self-consistency — **passes**

*Where the plan prints room dimensions, reconstructed room areas agree within
±10%.*

This is the criterion the roadmap leans on hardest, because it needs no external
measurement: the listing supplies both the reconstruction and the number it has to
agree with. It is also the one that most nearly resembles accuracy, and it passes.

<!--SELF_CONSISTENCY-->

What makes it work is the ordering. Stage 7 solves **one** global scalar by
weighted least squares **in log space**, so that areas (which scale as *s*²) and
lengths (which scale as *s*) enter the same fit linearly and a 10% error costs the
same whether the model is too big or too small. Outliers are rejected on the
median absolute deviation rather than the RMS — RMS is inflated by the very
outlier you are hunting, and a single mis-OCR'd dimension protects itself.

### Plausibility — **passes, after a real fix**

*≥80% of rooms have ceiling heights in 2.3–3.2 m and no room exceeds 12 m across.*

The first measurement was 66%, and the failures were not random: they were the
bathrooms, the corridors and the balconies. The cause turned out not to be the
estimator at all — **a bathroom close-up does not contain both the floor and the
ceiling**, so the two dense layers the height histogram finds are the floor and a
skirting board, and the room comes out 0.53 m tall.

Stage 4 now checks whether both surfaces are actually observed — is the gap large
enough to be a room, do both layers have real support behind them, is the room
absurdly wider than it is tall — and **reports no height rather than a confident
wrong one**. Outdoor spaces are excluded from the criterion entirely; a balcony has
no ceiling to be plausible about.

<!--PLAUSIBILITY-->

Refusing to answer is the right behaviour and it costs nothing downstream: the
scale solve weights the ceiling prior lightly and the shell builder falls back to
the listing's own median height.

### Cross-model scale agreement — **passes**

*Independent scale estimates within 15% of each other.*

The plan channel scales pixels by what the plan prints; the photo channel scales
metres by the ceiling prior and, where apertures are found, by the 2.04 m door
height. Two methods sharing no model and no input.

<!--CROSS_MODEL-->

The door-height constraint is the load-bearing part of the *independence* claim: it
is the only **length** the photo channel can offer, and without it the cross-check
compares two ways of measuring area.

### Arrangement — **not judged; the evidence points at the plan channel**

*Rooms placed in the correct plan polygon on ≥70% of listings.*

This is the criterion that needs a human, because nothing in a listing says which
photograph is of which room — that is precisely what assembly has to work out. We
annotated by eye, with the vectorised polygons overlaid on each plan.

<!--ARRANGEMENT-->

**We are not claiming this criterion passes.** The coverage is not enough, and the
honest reading of what the annotations show is that *when the plan channel produces
correct polygons, assembly puts rooms in them* — and that the plan channel does not
always produce correct polygons. Assembly is not the bottleneck the roadmap feared;
the vectoriser is.

---

## 3. Where it actually breaks

Full list in [`FAILURE-TAXONOMY.md`](FAILURE-TAXONOMY.md). Three matter:

**F2 — small rooms are never vectorised, and this causes most of everything else.**
Bathrooms, WCs, hallways and cupboards are usually unlabelled on the plan and too
small for a distance-transform peak to clear the sliver floor. So they never become
polygons — and then the photo-channel room *for* that bathroom has nowhere to go,
and the Hungarian assignment puts it wherever is cheapest. Every "wrong room"
error we inspected traces back here.

**Colour-filled plans break the "walls are the thickest strokes" assumption.** Two
of the plans we looked at fill each room with a pastel colour, so the darkest,
thickest ink on the page is the room fill and not the wall. On one of them only
the single space whose caption survived OCR was segmented; on the other the
vectoriser latched onto the wrong storey of a maisonette. This is a *class* of
plan, not a one-off, and the classical vectoriser cannot be patched into handling
it — it is the strongest argument for the learned model the roadmap already
specifies.

**A correct plan with no printed dimensions is still unusable.** One listing came
out with the best vectorisation in the set — hallway, both bathrooms, kitchen, both
bedrooms and the living room all found and labelled correctly — and the pipeline
could not use it, because the plan prints no dimensions and the listing states no
area, so there is no metric scale and stage 6 refuses to match metric rooms against
unscaled polygons. Refusing is right. It also means the *scale* channel, not the
*arrangement* channel, is what gates yield on plans like this.

---

## 4. Cost and latency

<!--LATENCY-->

Stage 3 is 85–95% of the wall clock and all of it is GPU work being done on a CPU.
Phase 0 measured MoGe-2 at 0.364 s/image on a free T4 against 14.4 s on four CPU
cores — a 39× speedup — which puts a GPU listing at a handful of seconds end to
end and leaves the `instant` profile's 10-second envelope intact. Every other stage
combined is under ten seconds and most are under one.

The per-stage budgets are asserted in CI for the CPU-bound stages and reported but
not asserted for stage 3, because a CI runner would only ever tell us about the
runner.

---

## 5. What we can and cannot claim

**Can.** The chain works end to end without COLMAP, without a GPU, and without a
single hand-authored input. The scale solve is well-posed and robust to the OCR
errors that actually occur. When the plan channel produces correct polygons,
assembly puts rooms in them and the shell is right. The viewer renders it,
distinguishes reconstructed from inferred surfaces, and loads in well under a
second.

**Cannot.** That any of it is *accurate*. Every check is self-consistency against
the listing's own printed numbers, and a systematic scale bias would satisfy all of
them. Phase 0 recorded what would fix this — half a day with a tape measure on ~10
flats — and it remains the only route to a defensible accuracy number.

We also cannot claim the arrangement criterion. The annotation coverage is not
enough, and pretending otherwise would corrupt the one measurement G1 exists to
make.

---

## 6. Gate G1 — honest assessment

<!--G1_ASSESSMENT-->

**The pivot rule does not fire.** The roadmap's G1 kill criterion is that
*arrangement stays below 70% and self-consistency cannot be brought inside ±10%*.
Self-consistency is inside ±10%. Arrangement's failures are traceable, and they are
in the plan channel rather than in assembly — which is the difference between a
component to replace and a stage that does not work.

---

## 7. What to do next, in order

1. **Train the plan vectoriser.** This is now the single highest-value task in the
   project by a wide margin. The corpus is settled (ROADMAP §7 R6) and the classical
   engine gives it a measured baseline. Everything in §3 is downstream of it.
2. **Finish the arrangement annotations.** An afternoon per twelve listings with the
   annotation aid, and it converts G1's third criterion from *unjudged* to a number.
3. **Get a GPU into the loop.** Stage 3 is 90% of the wall clock and 40× faster on
   the weakest available card. Nothing else about the latency picture matters until
   this is done.
4. **Re-run stage 3 at 3–8 views per room.** Phase 1 is deliberately monocular, so
   every room polygon is an oriented bounding box (F3) and every listing carries
   `approximate: true`. Multi-view is what makes a room L-shaped when it is
   L-shaped.
5. **Lower the unlabelled-room floor.** Bathrooms and hallways are missing because
   they are small, not because they are hard. Distinguishing a cupboard from a
   segmentation sliver is a smaller problem than it looks.

Item 1 is the one that unblocks the others.

---

## 8. Bugs worth remembering

In the Phase 0 tradition, these were all found by looking at outputs rather than at
metrics.

- **Normal-PCA cannot find "up" in a symmetric room.** A point cloud with as much
  wall as floor has an isotropic normal scatter, so the leading eigenvector is
  noise and the room comes out on its side — where a 2.4 m ceiling and a 4 m one
  look equally plausible in a table. "Up" is now the direction under which the
  cloud *stratifies into two dense layers*, which is what actually distinguishes a
  floor and a ceiling from two walls. Median area error on rooms of known size fell
  from ~7% to 1.3%.
- **A greyscale-plus-alpha PNG converted straight to RGB composites onto black.**
  Several plans arrive in `LA` mode. Every threshold downstream then inverts, and
  the only visible symptom is a black rectangle in the debug view.
- **Tesseract drops the "x" in "3.96 x 3.66"**, and a naive single-character filter
  removes it as noise — losing the strongest scale constraint in the system. It
  also drops the decimal point ("5.8m" → "58m"), and reads the imperial line as
  metric on plans that print both.
- **Watershed basins never touch through a doorway if the door leaf is drawn.**
  Three earlier aperture detectors were built on wall-thickness morphology before
  the basin-contact definition made the problem disappear.
- **`write_binary` with a nested name silently fails** if the destination directory
  is not created — and stage 3 discovers this after ninety seconds of work.
- **RMS-based outlier rejection cannot see the outlier**, because the outlier
  inflates the RMS that sets the threshold. MAD can.

---

## Appendix — the numbers

Regenerated from the stored artifacts, not typed by hand:

```bash
python -m eval.phase1_summary          # eval/results/phase1_summary.md
python -m eval.harness --split dev     # M1-M5 and the G1 criteria
```

<!--APPENDIX-->
