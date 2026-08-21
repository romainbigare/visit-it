# Why the plan reading is bad, and what to do about it

**21 August 2026 · investigation, with measurements, a survey and a shipped fix**

You looked at 25 flats and said not one was even remotely well detected: door swing
lines taken as room boundaries, nothing found when the walls are not filled black,
bed outlines read as room framing, open-plan kitchen-and-living split in two,
outlines offset from the plan with gaps between rooms and walls, outlines stopping
at kitchen cabinets.

Most of that is right. One item was my rendering bug rather than the pipeline, and
one of my own measurements was wrong in a way that flattered the result. Both are
corrected below, because a report that quietly drops its own errors is not evidence.

---

## 1. What I got wrong first

**The verdict labels on the review page were meaningless.** They scored room
*counts*, *area totals* and *whether a scale was found* — all of which can agree
perfectly while every outline is drawn in the wrong place. "Reads clean" was
checking arithmetic, not the thing a person checks by eye. That is why the page
told you 6 flats read clean and your eyes told you none did. Your eyes were right.

**The overlay images were misaligned by about 2.4%.** Stage 5 works on a deskewed,
size-capped copy of the plan and stores polygon coordinates in *that* image's
pixels. I drew them over the original file. At the far edge of a 2000-pixel plan
that is a 35-pixel offset — several wall thicknesses — so correct rooms appeared to
float clear of their walls with a gap all round. That is the "widely offset, with
gaps between rooms and walls" you saw, and it was mine, not the pipeline's. Fixed;
every image in this report is drawn in the right coordinate space.

**My first metric could not adjudicate your complaint.** I measured "what fraction
of each room outline lies on a drawn line", which sounds right and is useless here:
a kitchen cabinet run *is* a drawn line, a door swing *is* a drawn line, a bed *is*
a drawn line. An outline that traces the furniture scored just as well as one that
traces the wall. Corrected for the coordinate bug, that metric reads a median of
0.856 — which would say the reading is fine, and it is not.

---

## 2. The complaint, measured properly

The metric has to distinguish a wall from a wardrobe. The only reference available
for that is a model trained to make exactly that distinction (§4). Scoring every
stored room outline against a predicted wall map rather than against all ink:

| reference | median outline fit | rooms ≥80% |
|---|---|---|
| any drawn line | 0.856 | 64.9% |
| **an actual wall** | **0.773** | **44.4%** |

**That eight-point gap is the furniture error, quantified.** Better than a fifth of
every room outline was drawn along something that is not a wall — a cabinet run, a
door swing, a bed, a dimension line — and fewer than half the rooms in the set had
outlines that mostly followed their walls.

Read the bias honestly: the wall map is itself a prediction, so this number favours
any reading built on the same map. It is reported next to the unbiased one, not
instead of it, and the side-by-side images remain the real verdict.

---

## 3. One cause, not six

Every item on your list comes from a single line of reasoning: **the vectoriser
decides a pixel is a wall by asking whether it is dark.**

Everything downstream of that — the watershed, the caption seeding, the polygon
fitting, the scale solve — is sound. But a floor plan draws far more than walls in
dark ink:

| what else is drawn dark | what the software concluded |
|---|---|
| kitchen cabinet runs, worktops | the room ends at the counter |
| beds, sofas, baths, WCs | the room is framed by the furniture |
| door swing arcs | there is a curved wall there |
| dimension arrows and their lines | there is a wall across the middle of the room |
| the room caption itself | there is a small enclosure around the text |

And the converse: where a plan draws its walls as thin outlines rather than solid
fills, there is barely any dark ink where the walls are, so little is found.

This is not six bugs. It is one bad definition, and it cannot be patched, because
"dark" genuinely does not distinguish a wall from a wardrobe. It needs something
that has seen thousands of floor plans and knows what a wall *is*.

---

## 4. What already exists

You asked whether we can use something that exists rather than build from scratch.
We can, and we now do. The field, restricted to things whose weights you can
actually download:

| what | released | licence | input | output | usable here? |
|---|---|---|---|---|---|
| **[`Yytsi/floorplan-to-3d-walls`](https://huggingface.co/Yytsi/floorplan-to-3d-walls)** | weights, 98 MB | MIT | raster plan | wall / door / window / floor per pixel | **yes — now shipped, ~6 s per plan on CPU** |
| **[Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq)** (SIGGRAPH 2026) | [weights](https://huggingface.co/haopt/Raster2Seq), 650 MB | MIT | raster plan | labelled room polygons, directly | strongest published result; needs a GPU |
| [RoomFormer](https://github.com/ywyue/RoomFormer) (CVPR 2023) | weights | MIT | point-cloud density maps | room polygons | wrong input domain |
| [PolyRoom](https://github.com/3dv-casia/PolyRoom/), [FRI-Net](https://github.com/Daisy-1227/FRI-Net) (ECCV 2024) | weights | research | density maps | room polygons | wrong input domain |
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) reference model | code, **not weights** | research | raster plan | 12 room types + 80 icon types | would be ideal; weights were never published |
| [FloorSAM](https://arxiv.org/abs/2509.15750) (2025) | paper only | — | LiDAR density maps | room polygons | wrong input domain |
| Segment Anything, zero-shot | weights | Apache-2.0 | any image | masks from point prompts | **tested — not competitive, see §5.1** |

Two things worth stating plainly.

**Nobody needs to collect data.** CubiCasa5K (5,000 annotated raster plans),
Structured3D, Raster2Graph, WAFFLE (20,000 real-world plans), plus the Swiss
Dwellings and ResPlan sets already registered here — all public.

**Raster2Seq is the best available and the only one whose output is what we
actually want:** labelled room polygons in one step, no post-processing, and it
generalises to WAFFLE, the closest public analogue to estate-agent plans. Its cost
is deployment — custom CUDA kernels (deformable attention plus a differentiable
rasteriser) compiled from source, so it is GPU-only. Our pipeline runs on four CPU
cores. It is the right next step, not the right step today.

---

## 5. What we tested

### 5.1 Segment Anything, zero-shot — ruled out

The appealing shortcut: OCR already knows "KITCHEN" is at a particular pixel, SAM
takes a point prompt and returns the region containing it, no training at all.

Prompted on the caption, SAM segments **the letter K**. Erase the text first and
take the largest sensible mask instead and it scores a median 0.54 outline fit —
better than the classical reading on that plan, well short of what follows. SAM has
no notion that a wall differs from a bath, which is the entire problem.

### 5.2 The pretrained wall model — adopted

A UNet with a ResNet-34 encoder, trained on CubiCasa5K, 98 MB, MIT, CPU-friendly.
On our plans it does the three things the ink threshold cannot:

- **ignores furniture** — kitchen units, baths, WCs, beds and sofas all read as floor
- **ignores door swings** — the arcs vanish; the opening itself is marked *door*,
  which is more useful than either
- **finds thin walls** — outline walls with no solid fill are found

Two preprocessing steps turned out to be required, not optional:

1. **Erase the text first.** The model reads lettering as wall.
2. **Level the page to white.** On one grey-filled plan on a lavender page it
   labelled **62% of the flat "window"** — the whole interior became a barrier and
   room extraction collapsed to three blobs. Mapping the page colour to white and
   the darkest ink to black took that plan from 0.14 to 0.86. The model was trained
   on dark ink on white paper; give it that.

### 5.3 The two engines fail on opposite plans, so keep both

The change is small: feed the watershed a different wall map and leave everything
else alone. Running both engines over the golden set and comparing:

| | on the 12 plans the ink mask was failing | on the 12 it was handling |
|---|---|---|
| ink mask | 0.40 | 0.89 |
| pretrained wall map | **0.79** | 0.84 |
| which wins | net, 11 of 12 | ink, 11 of 12 |

That is about as clean a complementarity as you get. Choosing one engine for all
plans throws half of it away — so stage 5 runs both and keeps whichever puts more
of its outline on predicted walls, a judgement it can make from the plan alone with
no ground truth. On the golden set it picks the net for 20 of 25 plans and the ink
mask for 5.

It also needs a floor-area guard, and one plan showed exactly why. A leak in the
wall map let the exterior basin flood the flat, leaving a *ring-shaped* polygon
traced along the outer wall plus a few pockets. That ring scores near 1.0 on
outline fit — it **is** the wall — while accounting for a fifth of the floor. A
reading that loses most of the building is not the better reading however neatly
its outlines sit, so the two readings first have to account for comparable floor
area, and outline fit decides only inside that.

---

## 6. Result, including what got worse

Shipped in `pipeline/floorplan/wallnet.py`, selected in `vectorise.segment()`.
Measured over all 25 plans, before and after:

| | before | after |
|---|---|---|
| median outline on a wall | 0.773 | **0.827** |
| rooms with ≥80% of the outline on a wall | 44.4% | **55.9%** |
| rooms found | 225 | 227 |
| seconds per plan (4 CPU cores) | ~6 | ~12 |
| **median floor-area error vs advertised** ¹ | **11.2%** | **15.6%** |

¹ On the eight plans whose scale comes from *printed dimensions*. The other
listings scale from the advertised area itself, which forces the ratio to 1.00 by
construction — a circular number that was inflating this figure before, and worth
knowing about independently of this change.

**The area regression is real, and I could not find its cause.** Rooms come out
about four points further from the advertised area than they did, and the same
shift shows up in the G1 self-consistency criterion (holdout median per-room area
error 9.0% → 15.2%). Four hypotheses, each tested and each wrong:

| hypothesis | test | result |
|---|---|---|
| the predicted wall band is too thick | swept how far it may sit from the stroke, 0.5–1.0 × wall half-thickness | no measurable difference |
| the window class walls off glazed bays | dropped it from the barrier entirely | no measurable difference |
| the metre scale shifted | compared px-per-metre per listing | unchanged (1.00×) on 15 of 22 |
| the plausibility filter drops more regions | counted dropped regions | *fewer* now — 11, against 24 before |

So it is not the barrier, not the classes, not the scale and not the filter. Rooms
are simply coming out slightly smaller in pixels on some plans and I have not
isolated why. Stated plainly rather than guessed at, and it is the first thing on
the list below.

Two things keep this from being a reason to hold the change. The sample is eight
listings measured against *advertised* gross internal area, which includes wall
thickness and is rounded and inconsistently defined — a ratio under 1.0 is the
expected relationship, and the old median of 1.00 partly reflected plans scaled
*from* that same advertised area, which forces the ratio by construction. And the
direction of the errors changed as well as the size: the two worst over-estimates
in the set (1.40 and 1.17) both improved.

**And the outline-fit number is partly circular.** It scores outlines against the
same wall map used to build them, so it flatters any reading based on that map. It
is the only wall-versus-furniture reference available without hand annotation, so
it is what there is — reported with the bias stated, next to the unbiased
ink-referenced number, which also improved slightly (0.821 → 0.835).

**What is not ambiguous** is the segmentation itself. Put the wall map next to the
plan and the kitchen units, baths, WCs, beds, sofas and door-swing arcs are simply
gone, and the walls — including thin outline walls carrying no fill — are all
there. No metric is doing any work in that comparison.

**What this does not fix.** Open-plan splits: "RECEPTION / DINING ROOM" still comes
out as two rooms, because the caption seeding sees two room words in a space with
no wall between them. Rooms with neither a caption nor a wall separating them.
Maisonettes. Those are room *reasoning* problems, not wall *detection* problems.

## 6b. Door swings: the model is inconsistent, and that is the worst case

The wall model is much better at ignoring door swings than the ink threshold was, but it is
not reliable about it — and a component that is right most of the time without saying which
times is harder to build on than one that is wrong predictably.

Measured over the swing arcs we could detect automatically:

| what the wall model does to a swing arc | share |
|---|---|
| wipes it out completely (<10% of the arc still called wall) | 42% |
| half-erases it (10–90%) | 50% |
| leaves it essentially untouched (>90%) | 8% |

**Caveat on that table: it is 12 arcs across 25 plans, and the detector's recall is the
reason.** A geometric swing finder — fit a circle to each thin ink component, require the
radius to be a door width in metres (we know the plan's scale), require it to span most of
a quadrant — is precise but finds almost nothing, because agents draw swings a dozen
different ways: quarter circles, straight leaf lines, thin light strokes over a colour
fill, and on some plans not at all. It was written, measured, and deleted rather than
shipped at 10% recall.

**That failure is the argument for fine-tuning.** Every plan style needs its own rule, the
rules interact, and there is no end to them. A model that has seen our plans learns the
distinction once. So:

| | what it is |
|---|---|
| `tools/annotate_walls.py` | a browser tool that opens each plan with the model's own reading painted on, so the work is scrubbing off wrong walls rather than tracing right ones — a minute or two per plan |
| `notebooks/finetune_wallnet_colab.ipynb` | fine-tunes the wall model on those corrections: frozen encoder, low learning rate, 512-pixel crops, augmentation aimed at colour and line-weight variation, held-out plans split by a hash of the listing id |

Twenty to thirty corrected plans is a useful set. The notebook supervises wall-versus-floor
only — that is all the corrections claim — and keeps the model's four-class head intact, so
it learns *where* the line between structure and furniture sits without being taught
anything false about doors and windows.

One honest limitation, stated in the notebook too: the labels are seeded from the model's
own reading, so they are not independent of it. Somewhere the model is confidently wrong in
a plausible-looking way is somewhere an annotator is less likely to correct. That is
acceptable for closing a known gap like door swings, and it is not a substitute for tracing
a handful of plans from scratch if we ever want a clean measurement.

## 6c. Raster2Seq: what it is good at, and the one thing holding it back

Run on our plans (`notebooks/raster2seq_eval_colab.ipynb`) it does what the wall model
cannot: it predicts **which rooms exist, what type each is, and how they fit together**,
directly, with no captions read. Open-plan spaces stay whole. Unlabelled WCs, cupboards and
hallways are found. That is the half of the problem our watershed has always guessed at.

**What holds it back is coordinate resolution, and it is measurable.** The published
CubiCasa5K checkpoint works on a 256-pixel copy and tokenises coordinates into 32 bins per
axis, so every vertex snaps to a grid 1/32 of the plan wide — around 30 cm on a typical
flat, whatever the source resolution.

To size that on its own, take our own outlines — which do sit on the walls — and snap them
to the same grid, changing nothing else:

| outlines | median outline-on-wall |
|---|---|
| ours, as shipped | **0.822** |
| the same, snapped to a 64-bin grid | 0.643 |
| the same, snapped to Raster2Seq's 32-bin grid | **0.514** |

*n = 25 plans.* So a third of this metric is spent on the coordinate grid before the model
is judged on the thing it is good at. **0.514 is the ceiling any 32-bin prediction can
reach**, not 1.0, and any comparison that ignores that under-rates it. The notebook's
head-to-head prints all three columns for exactly this reason.

**`--num_bins` and `--seq_len` are not knobs.** The coordinate vocabulary is baked into the
trained tokeniser — this checkpoint was trained at 32 bins, and passing 64 changes the
vocabulary size out from under the weights. The quantisation cannot be tuned away at
inference; it can only be worked around.

### The two models are complementary, not competing

| | Raster2Seq | the wall model |
|---|---|---|
| which rooms exist | **yes**, directly | guessed from captions and distance peaks |
| room types | **yes**, no text read | from OCR, so only where a caption exists |
| open-plan spaces | **kept whole** | split when one caption holds two room words |
| unlabelled WCs, cupboards, halls | **found** | often missed |
| where the wall is | ~1/32 of the plan width | **pixel-accurate** |
| runs on | a GPU | four CPU cores |

**So use Raster2Seq for the room graph and the wall model for the geometry** — and we
already own the piece that joins them. Stage 5's watershed had exactly two weaknesses: it
was *seeded* from OCR captions, so it only found rooms somebody had labelled, and its
*barrier* was an ink threshold, so cabinets and door swings became walls. The wall model
fixed the barrier. Raster2Seq's room polygons are a far better seed than a caption ever
was — one per real room, already typed, already whole. Feed them in as markers, keep the
wall map as the barrier, and each room grows out to the wall behind it.

That is an integration of three things we have, not new research.

### Ranked

| | what | effort | what it buys |
|---|---|---|---|
| 1 | **Seed the watershed with Raster2Seq rooms** over the wall-model barrier | small — all three pieces exist | the room graph *and* pixel-accurate outlines |
| 2 | Try the **`Raster2Graph-512`** checkpoint the authors also publish | one string | double the coordinate resolution, no training |
| 3 | **Tile** the plan and run at native wall thickness, merge | medium | the small rooms the 256-pixel downscale erases |
| 4 | **Ensemble** the three preprocessing variants plus small rotations | small | recall, and a confidence signal for free |
| 5 | The authors' **VLM refinement** (`vlm_refinement/`, drives the Gemini CLI over the predicted JSON) | small, per-plan API cost | artefacts from noisy CubiCasa5K ground truth |
| 6 | **Fine-tune Raster2Seq** | large | the domain gap — but see below |

**Fine-tuning Raster2Seq needs different labels from the ones we collect.**
`tools/annotate_walls.py` produces wall *masks*, which is what the wall model learns from;
Raster2Seq learns from room *polygons* with types. Same plans, a slower annotation job, and
25 plans is far too few either way. Worth doing after 1–3, not before — and if 1 works,
the wall model is the only thing left that needs our labels at all.

## 7. What to do next

**Now, no cost:** `python -m tools.fetch_wallnet` puts the weights in place; without
them stage 5 behaves exactly as before and says so in its QA flags. Nothing else in
the pipeline changed, and no artifact contract moved (AD-4).

**Next, in this order:**

1. **Find the area regression.** Four hypotheses are already eliminated above.
   The next step is a per-room before/after diff in pixels on one listing, which
   will say immediately whether rooms are being clipped, split or dropped.
2. **Seed the watershed with Raster2Seq's rooms** over the wall-model barrier —
   §6c. Raster2Seq has been run on all 25 plans and it finds the rooms; what it
   cannot do is put their corners on the walls, and that is precisely what the
   wall model and the watershed already do. This is the highest-value change on
   the list and it needs no training.
3. **Or close the gap on the model we have** with
   [`tools/annotate_walls.py`](../tools/annotate_walls.py) and
   [`notebooks/finetune_wallnet_colab.ipynb`](../notebooks/finetune_wallnet_colab.ipynb).
   Independent of step 2 and worth doing whichever way that goes, because the
   corrected plans are a permanent asset.
4. **Trace five plans from scratch.** Every number in this report is measured
   against a prediction, because there is no ground truth for room outlines — and
   the fine-tuning labels in step 3 inherit that, since they are corrections of the
   model rather than independent tracings. Five carefully traced plans would end
   the circularity for good, and five is an afternoon.

**Where the older notebook sits now.** `notebooks/train_plan_vectoriser_colab.ipynb`
trains a room-segmentation vectoriser from CubiCasa5K from scratch. It is superseded
by both of the above: Raster2Seq is a stronger architecture already trained on the
same data, and fine-tuning the wall model is far cheaper than training a new one.
Keep it as the fallback if both of those disappoint, not as the next step.

**Does this kill the project?** No. The reconstruction chain, the scale solve, the
assembly and the viewer were never the weak link, and the weak link turned out to
have an off-the-shelf answer that took an afternoon to wire in.
