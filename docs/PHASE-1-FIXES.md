# Phase 1 → Phase 2: the fix list

What Phase 1 measured, what it says to change, and what each change costs. Written
in dependency order — F1 unblocks most of the rest.

Evidence for every claim is in [`PHASE-1-REPORT.md`](PHASE-1-REPORT.md) and
regenerable with `python -m eval.phase1_summary`.

---

## The one-line summary

The 3D reconstruction works. **Reading the floor plan is the weak link, and the
shell is built from the wrong source.** Neither is a research problem.

---

## F1 — Train the plan vectoriser · **large · unblocks everything**

**Evidence.** Every failing G1 criterion traces to a listing where the vectoriser
returned something that was not a room. Two of the twenty-two plans we rendered
were colour-filled — the software assumes the darkest, thickest strokes on a page
are walls, and on those plans the *room fills* are darker than the walls. One
returned a single polygon for a six-room flat. Another latched onto the wrong
storey of a maisonette.

**Why the classical engine cannot be patched into this.** It has one global
assumption about what ink means. A colour-filled plan violates it, a hatched plan
violates it differently, and a hand-drawn plan violates it a third way. Each
patch is a new special case, and the special cases interact.

**What to build.** Exactly what ROADMAP §3 Sprint 3 already specifies and Phase 1
deferred: a RoomFormer-class network predicting room polygons directly.

| corpus | size | what it gives | status |
|---|---|---|---|
| ResPlan | 17K vector plans | room polygons + connectivity graphs | registered, auto-fetch |
| Swiss Dwellings | 42K apartments, 242K rooms | European residential, vector | **downloaded and verified** (792 MB) |
| CubiCasa5K | 5K raster plans, 80+ categories | the raster half the other two lack | registered, auto-fetch (5.5 GB) |
| our own scraped plans | 24 annotated | UK agency style, for fine-tuning | annotation format and tool exist |

The classical engine stays as the fallback binding and as the baseline the learned
model has to beat. Both sit behind the same interface, so swapping changes no
artifact contract (AD-4).

**Target.** ≥80% room F1 on our plan validation split — the number ROADMAP Sprint 3
already set.

---

## F2 — Build the shell from plan polygons, not photo polygons · **small · high value**

**This is a correction to how Phase 1 implemented AD-2, not a change to AD-2.**

AD-2 says *floor-plan-first spine, photos attach to it*. Phase 1 read that as
"shape from the photos, position from the plan". Measured across 22 listings, that
was the wrong reading:

| what we measured | result |
|---|---|
| photo-derived room area ÷ its own plan polygon area | **median 1.31** — photo rooms are 31% too big |
| shell coverage of the plan footprint | median 87%, but ranging 12%–165% |
| listings where the shell covers under 70% of the plan | **7 of 22** |

Two causes, both structural rather than tunable:

1. **Every photo-derived room is a rectangle.** One wide-angle photograph does not
   support a concave footprint, so stage 4 returns the oriented bounding box. A
   room with an alcove reads 10–20% large. The plan polygon *is* the room outline,
   alcoves included.
2. **Rooms nobody photographed leave holes.** The shell only contains rooms that
   got matched to a photograph, so a five-bedroom flat photographed in four rooms
   comes out as four floating boxes. The plan knew the other rooms were there.

**What to change.** Extrude the **plan polygon**, at the ceiling height the
**photographs** measured (or the listing's median, or the 2.55 m prior — in that
order). Rooms with no photograph are still built, tagged `inferred` so the viewer
desaturates them and the honesty rendering stays truthful.

**What this costs.** The photo-derived polygon stops being the shell's shape and
becomes three other things, all of which it is better at: the ceiling height, an
independent cross-check on the plan's scale, and — the reason it exists — the
geometry the Phase 2 splats are trained and culled against.

**What this means for the metrics.** Self-consistency currently compares a
photo-derived area against a printed dimension. Under F2 the shell's area comes
from the plan, so that comparison becomes the plan channel checking its own
vectoriser against its own printed text. That is still a real check — it is what
catches a mis-vectorised room — but it is no longer evidence about the photo
channel, and the harness must report it as such. The photo channel's honest check
becomes the ceiling height and the cross-model scale agreement.

---

## F3 — Multi-view geometry · **medium · already scheduled (Sprint 8)**

Phase 1 is deliberately monocular: one photograph per room via MoGe-2. That is
what makes every room a rectangle (F2 cause 1) and is the most likely source of
the systematic 16.7% cross-model scale offset. MapAnything over 3–8 views per room
is the fix and is already in ROADMAP Sprint 8. Needs a GPU (F4).

---

## F4 — Put a GPU in the loop · **small · pure cost**

Stage 3 is **88% of an 87.6-second run** and all of it is GPU work being done on a
CPU. Phase 0 measured MoGe-2 at 0.364 s/image on a free T4 against 14.4 s on four
CPU cores. A GPU listing is a handful of seconds end to end, which leaves the
`instant` profile's 10-second envelope intact. Nothing about the latency picture
means anything until this is done, and F3 needs it.

---

## F5 — Find the small rooms · **medium**

Bathrooms, WCs, hallways and cupboards are usually unlabelled on the plan and too
small to clear the vectoriser's sliver floor, so they never become polygons. 63% of
vectorised rooms carry a label; the rest are unnamed regions. This is the same
problem F1 solves — a trained model finds a 3 m² WC because it has seen thousands —
so it is listed separately only because a lower floor plus a shape test would
recover some of them sooner and cheaply.

---

## F6 — Re-measure on fresh listings · **small · expires**

Two Phase 1 fixes were found by inspecting holdout failures, which costs the
holdout some of its independence. It remains a good regression set; it is no
longer a clean gate. The scraper works and thirty more listings is an afternoon.
**This gets harder the longer it waits**, in exactly the way the original holdout
freeze did.

---

## F7 — Finish the arrangement annotations · **small**

Four of fourteen plan-bearing holdout listings are annotated, which is not enough
to judge G1's arrangement criterion. `tools/annotation_aid.py` draws each plan with
its polygons named; ten more listings makes the criterion measurable.

---

## Not on this list

Things Phase 1 got right and that no measurement argues against: the staged DAG and
artifact contracts, the single-scalar scale solve in log space, Hungarian assignment
with stored cost matrices, the confidence-and-QA-flag discipline, the review console,
and the waypoint viewer. The failure taxonomy
([`FAILURE-TAXONOMY.md`](FAILURE-TAXONOMY.md)) records what broke inside them and
what was done about it.
