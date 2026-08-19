# Prior art — Rent3D, Plan2Scene, and why nobody has shipped this

**Added 19 Aug 2026**, in response to: *Rent3D looks like exactly what we're trying to do — why hasn't it been pushed further, and is it enough for us?*

Short answer: **Rent3D is the right problem attacked with 2015 tools, and the reasons the research line stalled were reasons about building research datasets, not reasons about the task being impossible.** Every one of those blockers has either dissolved or is one we now route around. That is, in fact, the clearest statement of why this company can exist in 2026.

---

## 1. Why the line of work stalled

The trajectory runs Rent3D (2015) → Plan2Scene / Rent3D++ (2021) → and then the field walks away from listing photos entirely, toward **controlled capture** and **synthetic data**:

| Dataset | Year | Scale | Capture |
|---|---|---|---|
| **Rent3D** | 2015 | 215 apartments, ~1,570 photos | **Found listing photos** (uncontrolled) |
| Structured3D | 2020 | 3,500 CAD houses, 21,835 panoramas | **Synthetic** — auto-labelled, no annotation cost |
| ZInD (Zillow) | 2021 | 1,575 homes, 67,448 panoramas | **Controlled** 360° capture, unfurnished |
| ScanNet | — | 1,513 scenes, 2.5M RGB-D frames | **Controlled** RGB-D scanning |
| M3DLayout | 2026 | 21,367 scenes | Multi-source |

The pivot had four causes, and it is worth being precise about which of them are about *the research* and which about *the problem*:

1. **Annotation cost.** Hand-annotating uncontrolled photos with 3D ground truth does not scale. Synthetic data (Structured3D) gets dense ground truth for free; controlled capture (ZInD) gets it cheaply. → *A research-economics problem, not a task problem.*
2. **Licensing.** Listing photos and plans are copyrighted and hard to license at research scale — the same wall documented in [`DATA-SOURCES.md`](DATA-SOURCES.md) §3.7. Academics could not build a big corpus legally. → *A data-access problem, and one a company with agency relationships solves differently than a lab does.*
3. **Deep learning's appetite.** By ~2018 methods needed 1,000s of scenes minimum; 215 was no longer a viable training set, so the dataset aged out even as the problem stayed interesting. → *Now largely moot: we train almost nothing from scratch. MapAnything, MoGe-2 and the splatting stack are pretrained foundation models; our only trained components are the plan vectoriser and the triage classifier, both of which have commercially-licensed corpora available (ResPlan, Swiss Dwellings).*
4. **The enabling technology did not exist.** Pointmap models (2024–25) and Gaussian splatting (2023–) are what make sparse, unposed, wide-baseline indoor reconstruction tractable at all.

**None of the four is "the task turned out to be impossible."** Three are dataset-construction problems that a company solves differently from a lab, and the fourth has been fixed by the field itself in the last two years.

## 2. The commercial confirmation

Independent of the research trajectory: **no commercial product reconstructs from uncontrolled, pre-existing listing photos.** Every success in the category controls capture —

- **Matterport** — proprietary structured capture (public 2021, acquired by CoStar).
- **Zillow** — built 3D Home on **consumer 360° cameras with an agent capture protocol**, not on the listing photos they already had. They published ZInD as research and productised something else entirely.
- **Giraffe360** — sells a 360° camera plus 50+ ML models for stitching and floor-plan measurement.

That Zillow — who owned both the research (ZInD, LASER) and the largest listing-photo archive in the world — chose to ship a *camera-based* product is the single most informative data point available. It can be read two ways, and honesty requires holding both:

- **The opportunity reading:** the gap is real and nobody occupies it, exactly as the feasibility report argued.
- **The warning reading:** the best-resourced player in the category looked at this and decided controlled capture was the surer path.

The reconciling detail is timing. Zillow's product decisions were made in 2019–2022, before pointmap models and before Gaussian splatting. The bet this company is making is not that Zillow was wrong then — it is that **the input requirements changed in 2024–25**, and nobody has re-run the calculation since. That bet is testable, cheaply, at gate G1, which is exactly where the roadmap puts it.

## 3. Successor research worth tracking

The floor-plan-localisation thread did continue, just on different data:

- **LASER** (Zillow, CVPR 2022) — Monte Carlo localisation of an image within a floor map; reported 97% recall at 5 cm median error **on ZInD**, i.e. on controlled unfurnished panoramas. Licence CC-BY-NC-ND (research only for us).
- **F3Loc** (CVPR 2024) — probabilistic filtering over the same problem; MIT-licensed.
- **C3Po** (NeurIPS 2025) — pixel-level photo↔floor-plan correspondence, 90K pairs, reports 34% error reduction over prior methods. **CC BY 4.0, commercially usable** — the most directly useful successor for our stage 6.

These solve *localisation within a plan*, which is a strict sub-problem of our stage 6 (assembly). They are worth adopting rather than reinventing.

## 4. What Rent3D actually does — and why it is a smaller problem than ours

It resembles our project in shape, but it is solving roughly **stages 4 and part of 6, with stages 0, 2, 3, 5, 7, 8 and 9 either assumed away or absent.**

**Task.** Given a photo *and the floor plan* — **and told which room the photo shows** — estimate that room's 3D layout and the camera's pose within it. The room correspondence is an *input*, not an output.

**Method (all pre-deep-learning):**
1. Vanishing points under a Manhattan-world assumption → camera intrinsics and rotation.
2. Hand-crafted appearance features (an 11-dimensional vector per wall face: 5 from an orientation map, 6 from geometric context).
3. An MRF energy over layout + camera pose, with terms for appearance, window alignment against the plan, and an **aspect-ratio prior read off the floor plan**.
4. Exact inference via branch-and-bound with integral-geometry bounds, exploiting the aspect-ratio constraint to parameterise with 3 variables instead of 4. **~3.1 ms per apartment.**

**Results.** Pixel-wise layout classification error on the test split: **13.88%** with no floor-plan prior → **11.79%** with the aspect-ratio prior → **11.73%** adding window alignment. Splits were 100 train / 30 val / 85 test apartments.

**Dataset provenance, worth noting given our own decision:** the plans and photos were **crawled from a London rental listings website**. The canonical academic dataset in this space was built by scraping a UK portal — which is context for [`DATA-SOURCES.md`](DATA-SOURCES.md) §3.9, though a 2015 academic crawl and a 2026 commercial product are different propositions legally.

### Side-by-side with our pipeline

| Our stage | Rent3D | Gap |
|---|---|---|
| 0 Triage | **Assumed** — room identity given as input | We must infer it (VLM/classifier) |
| 1 Calibration | Vanishing points, Manhattan assumption | We use GeoCalib + pointmap intrinsics; handles non-Manhattan and wide-angle |
| 2 Grouping | **Assumed** — correspondence given | We must infer which photos share a room |
| 3 Per-room geometry | None — no multi-view reconstruction | **The single biggest difference.** Pointmap models did not exist; they had no way to get real 3D structure from 2–3 unposed photos |
| 4 Layout | ✅ Their core contribution — cuboid layout + camera pose | Ours must handle non-rectangular (haussmannien) rooms; theirs is Manhattan-only |
| 5 Plan channel | Plans hand-annotated | We must vectorise and OCR automatically |
| 6 Assembly | Partly — localises a photo to a wall, but arrangement comes free from the given correspondence | We solve the assignment they were handed |
| 7 Metric scale | Plans carried annotated real-world scale | We solve scale globally against Carrez/door/ceiling priors |
| 8 Appearance | **None** | Splatting did not exist |
| 9 Delivery | None | — |

**The honest read on their headline number.** The floor plan bought them only ~2 points of layout error (15% relative). Taken at face value that looks like a weak endorsement of the floor-plan-as-spine thesis — but it is measuring a different contribution. In Rent3D the plan is a *refinement* prior on single-room layout, because inter-room arrangement was handed to them by assumption. In our system the plan's value is almost entirely in the thing they assumed away: **which room is which and where it sits**. Their number neither supports nor undermines our architecture; it simply measures a different quantity. Worth remembering if anyone cites it at us.

## 5. Verdict: is Rent3D enough for us?

**As a training set, no. As an early evaluation set, very likely yes — and 215 is not the reason.**

Sizing it against our three data needs ([`DATA-SOURCES.md`](DATA-SOURCES.md) §1):

- **N1 golden set (target 30–50):** 215 apartments is *four times* what we planned for. Comfortably sufficient, and it comes with room-layout, wall, door and window annotations plus photo↔room correspondence already done — annotation we would otherwise pay for.
- **N2 training corpus:** far too small to train anything from scratch — but we aren't. Our only trained components are the plan vectoriser (pretrained on ResPlan's 17K and Swiss Dwellings' 42K) and the triage classifier. 215 real agency plans is a *useful fine-tuning increment*, not a base corpus.
- **N3 production:** irrelevant, it's a static dataset.

**The real objections are not scale:**

1. **Age.** 2015 listing photography predates the HDR/flambient and ultra-wide conventions that the feasibility report identifies as actively hostile to reconstruction (§3, "estate agent photography is adversarial"). A pipeline tuned on Rent3D would be tuned on an easier input distribution than it will meet in production. **Mitigation:** use it as a *floor*, never as the primary benchmark.
2. **Rental, not sale.** Rental shoots are typically thinner and lower-effort than sale shoots — fewer photos per room, which puts us in the hardest part of the view-count curve, and often no floor plan at all in the modern market.
3. **Ground truth is agent-drawn plans, not measurements.** This is the important one. Rent3D can validate *layout shape and arrangement* (M3, M4, M5), but **cannot validate metric accuracy** (M1, M2) to the ±3–5% we intend to claim, because its own scale reference carries unquantified error. Our laser-measured and Matterport-scanned subset remains mandatory.
4. **Licence unknown** — not stated on the project page. One email to the authors, in week 1.

**Recommendation.** Pursue it, at a cost of one email, and if granted use it as an **early regression and smoke-test set** that lets stages 4–6 be developed before our own golden set is fully assembled — a genuine schedule win in P0/P1. It does **not** replace the golden set, and it must not become the benchmark we optimise against. Its annotations (1,312 rooms, 6,628 walls, 1,923 doors, 1,268 windows) are the real prize; the photos are the weakest part.

## 6. Plan2Scene / Rent3D++ — the closest published work, and the most useful warning

**Plan2Scene** (CVPR 2021, Simon Fraser University) takes a floor plan plus a sparse set of listing photos and outputs a **textured 3D mesh** — which is, on paper, our product. Code is **MIT**; the Rent3D++ dataset is available by Google Form request.

**Pipeline:** vectorise the plan → lift to 3D geometry → place fixed objects (doors, windows, sanitaryware) from ShapeNet CAD → assign photos to rooms → **synthesise tileable textures** for observed surfaces → **propagate textures to unobserved surfaces via a graph neural network** over the room-door-room adjacency graph.

**Rent3D++ adds** (same 215 apartments, no new photos): object bounding boxes, explicit photo→room assignments, rectified surface crops, curated texture datasets, and — most useful to us — **an unobserved-surface simulation framework with coverage levels 0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0**. That is a ready-made experimental harness for the feasibility report's central appearance problem ("~60% of surfaces are never photographed"), and we should reuse the protocol even if we never use their data.

**The critical divergence.** Plan2Scene's answer to unobserved surfaces is to **infer a plausible texture** from room type and neighbouring rooms. Their own numbers show the cost: colour fidelity degrades from 0.431 on observed surfaces to 0.653 on unobserved — roughly 51% worse — and overall FID sits around 196. More importantly than the numbers: **the output is a plausible-looking apartment, not the actual apartment.** Furniture is generic CAD stand-ins rather than the real contents. For an academic scene-synthesis result that is entirely legitimate. For property marketing it is the wrong side of the line the feasibility report draws (§4.10) and that regulators are moving against (§10, and California AB 723): *never generate anything that changes the perceived size, layout or condition of the property.*

**And the successors went further in that direction, not ours.** **HouseCrafter** (ICCV 2025) replaces the GNN with a 2D diffusion model generating consistent multi-view RGB-D along the floor plan; the wider 2023–25 line (SemLayoutDiff, MiDiffusion, NeuralField-LDM) is all generative scene *synthesis*. These optimise for **plausibility**; a property product must optimise for **fidelity**. That divergence is, I think, the real reason this research line has not turned into a listing product: the academic reward function and the commercial/legal one point in opposite directions.

**What this means for our design — three concrete confirmations:**
1. **Our architecture is on the right side of that line** by construction: real Gaussian splats from the real photographs, an architectural shell from the real plan, and unobserved surfaces flat-shaded or texture-extended and *visibly marked* rather than invented. Plan2Scene is the counterfactual showing what we are choosing not to do.
2. **Their weakest stage is our strongest new capability.** Plan2Scene had no way to recover real 3D from the photos, so appearance had to be synthesised. We have pointmap models and splatting — we can reproduce observed surfaces rather than imagining them.
3. **Steal the evaluation, not the method:** the coverage-level protocol (0.0→1.0 observed fraction) belongs in our eval harness in P0, as the principled way to measure M8 against photo coverage.

---

## 7. Summary — answering the original question

- **Is Rent3D the same thing we're building?** No. It is our stages 4 and part of 6, with the hardest parts (which room is this? where does it sit? what does it look like?) either given as input or absent. Plan2Scene extends it to a textured mesh and lands closest to our product, but reaches photorealism by *inventing* it.
- **Why hasn't it been pushed further?** The stall was about research data economics — annotation cost, copyright, and deep learning's appetite outgrowing 215 apartments — plus the absence of the enabling technology. The field responded by moving to synthetic and controlled-capture data, and commercially everyone (Matterport, Zillow, Giraffe360) chose to control capture instead. **Nobody concluded the task was impossible; they concluded the data was inconvenient.**
- **Is 215 enough?** For evaluation, yes — four times our golden-set target, with annotations we would otherwise pay for. For training, irrelevant, because we train almost nothing from scratch. The real objections are age (2015 photography is *easier* than 2026 photography), rental-not-sale, and ground truth that cannot validate metric accuracy.
- **Action:** one email to the Rent3D authors and one Google Form to the Plan2Scene team, both in week 1. Adopt the Rent3D++ coverage-level evaluation protocol regardless.
