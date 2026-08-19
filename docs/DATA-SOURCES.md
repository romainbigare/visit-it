# Data sources — how we get listings, photos and floor plans

**Status:** research in progress, 19 Aug 2026. This document answers the question the whole plan hangs on (ROADMAP gate G0): *where do the listings come from, legally, for development, for evaluation, and at production volume?*

## 1. What we actually need — three distinct needs, often confused

| Need | Volume | Must have | Licence bar | When |
|---|---|---|---|---|
| **N1 — Golden set** (evaluation ground truth) | 30–50 listings | Photos + floor plan + advertised area, **plus** independently verified measurements (our own tape/laser, or a paid Matterport scan) | Must permit our commercial evaluation use; small enough to license or shoot ourselves | P0, weeks 1–4 |
| **N2 — Training corpus** (fine-tuning the floor-plan vectoriser and the triage classifier) | 1,000s of plans; 10,000s of images | Floor plans in the styles our market actually uses; room-labelled interior photos | **Commercial-use licence required** — this taints the shipped model, so NC datasets are unusable (this already rules out CubiCasa5K, ZInD, Structured3D) | P1, weeks 5–12 |
| **N3 — Production input** (live listings customers pay us to process) | ongoing, at volume | Photos + plan, with the right to process and to publish a derived 3D model | Contract with whoever owns/controls the images; the derived-work right is the crux | P2 onward |

These have different answers. A research dataset can serve N1 and never N3. A portal API might serve N3 and never N2. **Failing to separate them is how projects discover at launch that their model was trained on data they cannot ship.**

## 2. The evaluation rubric for any candidate source

Every candidate in §3 is scored on these, because any single "no" changes what the source is good for:

1. **Images included?** Many "property data" products are prices, addresses and EPCs only — no photos. This kills most of the obvious names immediately.
2. **Floor plans included?** Rarer than photos, and the floor plan is our spine.
3. **Commercially licensable?** Distinguish: (a) may we *process* the images, (b) may we *store* them, (c) may we *train* on them, (d) may we *publish a derived 3D model*. (d) is the one nobody's standard terms contemplate.
4. **Who owns the copyright?** Estate agent photographs are usually the photographer's or agency's work, licensed to the portal — so the portal often *cannot* grant us rights it doesn't hold. The contract may need to be with agencies, not portals.
5. **Access reality:** open API / paid API / enterprise sales only / dead / agent-side only.
6. **Geography:** UK and France first (the loi Carrez surface-area trick is French; the UK has no equivalent legal area figure — see §5).
7. **Volume and cost.**

## 3. Findings by route

*Being populated from the research pass launched 19 Aug 2026 — five parallel investigations: UK portals, EU portals, academic datasets, open-data platforms, commercial resellers and the agency-software route.*

## 4. Recommendation

*To follow.*

## 5. Note: the UK has no loi Carrez

The architecture's scale solve (ARCHITECTURE §5) leans on France's legally-published Carrez surface area as its strongest constraint. **The UK has no direct equivalent** — floor area is commonly quoted in listings and on floor plans (sq ft / m²) but is not a statutory measured figure, and RICS/IPMS practice is not universally applied to residential listings. For UK stock the scale solve therefore falls back to: floor-plan printed dimensions and room areas where present, door-height priors, and ceiling-height priors — with a wider published uncertainty. This is a real accuracy difference between the two markets and should inform which market we target first.
