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

Research pass of 19 Aug 2026 (seven parallel investigations). **Caveat on method:** several "API provider" claims in the wild are scraping resellers wearing an API coat. Where a source's provenance is not established, it is marked ⚠ and must not be relied on — an Apify scraper existing for a portal is *not* evidence that the portal sanctions automated access.

### 3.1 UK portals — effectively closed

| Source | API? | Photos/plans? | Verdict |
|---|---|---|---|
| **Rightmove** | Real Time Data Feed exists, and its spec does carry image and floor-plan URLs | Yes, in spec | **Inbound only** — RTDF is how *agents upload to* Rightmove, not how third parties read from it. No outbound listing API. Dead end unless we become an agent-side integration |
| **Zoopla** | Historic public API retired; no current developer portal found | Unknown | Dead / opaque |
| **OnTheMarket** | None | — | Dead end |
| **Nestoria** (Lifull Connect) | Yes, free, no registration | **Yes — image URLs** | Only free UK route with images. **But terms reportedly cap caching at 24 hours** and require linking back — which forbids building a corpus and is awkward even for on-demand processing. Verify terms directly before any use |
| PropertyData (£28/mo) | Yes | **No images.** Does OCR floor plans to extract square footage | Interesting as a *cross-check on our own area estimates*, not as an image source |
| PaTMa, Sprift, PriceHubble, LandTech, Homedata | Yes | **No images** — attributes, valuations, EPC, planning | Not relevant to us |

**Conclusion:** there is no legitimate bulk route to UK listing photos and floor plans via portals. The UK route is agency-side (below) or nothing.

### 3.2 EU portals and aggregators — better, via two routes

- **Agency CRM/software APIs are the cleanest legal path.** France's **Apimo** and **Hektor / La Boîte Immo** both publish APIs that expose an agency's own listings **including photos and floor plans**, gated on that agency's consent. This matters because it inverts the rights problem: the agency owns or licenses the images, so the agency can grant us rights the portal never could. One integration then serves every agency on that platform.
- **Casafari** (pan-European, 20+ countries) markets a documented data API covering photos, floor plans and brochures. **Ruled out on its own terms** — see §3.8.
- **Idealista** (Spain) and **ImmobilienScout24** (Germany) have real partner APIs, but vetted-partner-only; IS24's terms reportedly forbid onward data sharing.
- **SeLoger, Logic-Immo, Leboncoin, PAP, Figaro, Immobiliare.it, Funda, Pararius:** no third-party APIs. Funda's partner API is discontinued.
- **Bien'ici has no public API** — the "free public API with photos" claim from the first research pass was wrong; it came from Apify scraper listings. Access is via unauthorised scrapers only. Do not use.
- **Structural fact:** Europe has no MLS. Listings are fragmented across hundreds of portals with no shared identifier, which is why aggregators exist and why agency-side integration is durable in a way portal-side never is.

### 3.3 Datasets that pair photos with floor plans

| Dataset | Contents | Geography | Licence | Use to us |
|---|---|---|---|---|
| **Rent3D** (CVPR 2015, Toronto) | **215 apartments, ~1,570 interior photos, annotated floor plans** with photo-to-plan alignment | **London** | Not stated on the project page — **must ask the authors** | Closest public analogue to our exact problem, in our target market. Small, and ten years old (photography styles have changed), but ideal as an early dev/regression set if licensable |
| **Rent3D++** (via Plan2Scene, CVPR 2021) | Rent3D extended: better alignment, more photos, object icons | London | Code MIT; dataset terms unstated; access by request form | Same as above |
| **C3Po** (NeurIPS 2025) | **90K photo/floor-plan pairs, 597 scenes, 153M correspondences, 85K poses** | Internet-sourced (Wikimedia plans + MegaScenes/YFCC photos) — **not estate-agent listings** | **CC BY 4.0 — commercial use permitted with attribution** ✅ | Best available for *pretraining photo↔plan correspondence*. Domain gap is real: public/landmark buildings, not flats shot at 16 mm by an agent. Pretrain here, fine-tune on partner data |
| **LIFULL HOME'S** (NII Japan) | 5.33M listings, 83M photos, 5.31M floor plans | Japan | **Academic institutions only — not commercially usable** | Research reference only; wrong market anyway |
| **ZInD** (Zillow) | 67,448 panoramas, 1,575 homes, 2,500+ plans | USA | **Non-commercial** (verified earlier) | Research only |

### 3.4 Floor-plan corpora — the training-data problem is smaller than we thought

The earlier plan assumed CubiCasa5K's non-commercial licence left us with no legal corpus for the plan vectoriser. That was too pessimistic — two commercially-licensed corpora exist:

| Corpus | Size | Licence | Note |
|---|---|---|---|
| **ResPlan** | **17,000 vector floor plans** with room graphs, metric scale, 17 semantic classes | **CC BY 4.0 — commercial OK** ✅ | Derived from US listings; vector only, no source raster images |
| **Swiss Dwellings** (Archilyse) | **42,207 apartments / 242,257 rooms** across 3,093 buildings, with geometry | **CC BY 4.0 — commercial OK** ✅ | **European**, and large |
| **MSD (Modified Swiss Dwellings)** (ECCV 2024) | 5,372 plans / 18.9K apartments, vector + raster + graph | CC BY-SA 4.0 (share-alike — check implications for derived models) | European multi-apartment stock |
| CubiCasa5K | 5,000 plans | CC BY-NC 4.0 ❌ | Research only |
| RPLAN, ROBIN, SESYD, FloorPlanCAD | various | NC or unstated | Research only |

**Consequence:** a commercially-clean plan vectoriser is trainable today on ResPlan + Swiss Dwellings + MSD + synthetic generation, then fine-tuned on partner plans for local style. That removes a blocker and de-risks Sprint 3 considerably. The remaining gap is *stylistic*: none of these are French or UK agency-drawn plans, which is exactly what fine-tuning on partner data is for.

### 3.5 Interior-photo corpora (for triage/room classification)

Weaker picture. **Places365 and MIT Indoor67 are both reported non-commercial** — the two classic scene-classification sets are therefore off-limits for a shipped classifier (verify directly before relying on either way). Kaggle/HuggingFace room-classification sets (House Rooms & Streets ~25K images; MMIS 160K interior images) mostly carry **no stated licence at all**, which is worse than a restrictive one — no licence means no rights. **Inside Airbnb is metadata only, no photos.**

**Consequence:** the triage classifier should be trained on **partner-supplied images we have rights to**, with a VLM used zero-shot to bootstrap labels. This is cheap (triage is the easiest stage) but it does mean triage quality is gated on the partner deal too.

### 3.6 The agency-software route — much more open than expected ★

This is the finding that changes the plan. UK estate-agency CRMs expose **OAuth APIs through which an agent can authorise a third party to read that agent's own listings, including property media**:

| Platform | Access model | Media | Cost |
|---|---|---|---|
| **Street.co.uk** | Modern REST API, OAuth, webhooks, two-way sync | Property data available to connected systems | **Free to all Street customers — no API tier, no per-call charges** |
| **Reapit** (Foundations) | OAuth2 + OpenID Connect, **self-serve developer portal**, "onboarded in minutes" | **Dedicated property-images endpoint** | Free partner enrolment |
| Dezrez (Rezi) | Open Core API, OAuth2, webhooks | Documents/images supported | Not published |
| Vebra / Alto | XML/REST feed, polls every 15 min | **Links to images and documents** | ~£100 setup + £128/yr per branch |
| Jupix | API with per-customer keys, IP whitelisting | Images linkable | Not published |
| France: **Apimo**, **Hektor** | Documented APIs, agency-gated | **Photos and floor plans**, HD, real-time sync | Not published |

Why this matters: it turns "sign a data partnership" from a 6–12 month enterprise negotiation with a portal into **one friendly agent clicking Authorise**, plus one integration that then works for every agent on the same platform. It is also the *legally* correct counterparty (§3.7). Sprint 1's partner outreach should therefore lead with "which CRM do you use?" and target Street.co.uk or Reapit users first.

### 3.7 Who actually owns the photographs — the constraint nobody budgeted for ⚠

Both UK and French law point the same way, and it is not where the plan assumed:

- **The photographer owns the copyright by default** — in the UK unless there is a written transfer; in France under CPI art. L111-1, and notably *even an employer does not automatically acquire it* from an employee photographer without a contractual transfer.
- **The agency typically holds only a licence**, commonly limited to marketing *that listing* while it is live — and expiring when the property de-lists.
- **The portal holds less still**, and therefore cannot grant us rights it does not have. This is the real reason portal access wouldn't solve our problem even if it were available.

The consequence for us is sharper than "get permission". We need a right nobody's standard paperwork contemplates: **to create and publish a derived 3D work from the photographs, and to keep serving it**. Two specific risks follow:

1. **Scope:** a marketing-only licence arguably does not cover generating a derived 3D model, and certainly does not cover our using the images to train models.
2. **Duration:** if the image licence dies when the listing de-lists, does our 3D model have to come down with it? For a portal/agency product that is probably fine and even desirable; for anything archival it is fatal.

**Action:** the partner agreement (Sprint 1 legal task, already on the plan) must explicitly grant: process, store, create derivative 3D works, publish those works, and — separately negotiated — train on. And it must state what happens at de-listing. This should be drafted before the first real images are ingested, not after. Add it to the DPA workstream.

### 3.8 Aggregators and resellers — due diligence result: avoid the category ⛔

A deliberately skeptical pass on the "European property data API" vendors found none that can give us what we need:

| Vendor | Verdict | Why |
|---|---|---|
| **Casafari** | **Ruled out** | Their own data-supply terms state: *"CASAFARI does not grant any rights to the underlying images or descriptions of real estate postings"* — image use remains "subject to the copyright owner's permission". Since **images are the entire input to our pipeline**, a data licence that explicitly excludes image rights is of no use to us. Their sourcing methodology (30,000+ sources) is also undisclosed |
| **Melo** | Likely scraped | Markets itself as "continuously interrogating" 1,500+ portals; a 2–10 person company is unlikely to hold 1,500 licences; no partnerships disclosed |
| **Stream.estate** | Likely scraped | Its own blog positions the product as *"the maintenance-free scraping alternative"*; no company registration, legal entity or team information locatable |
| **RealtyAPI** | **Avoid** | Claims "compliant, no legal risk" access to Rightmove and Zoopla data. Neither portal licenses bulk data to small resellers, and both explicitly prohibit scraping. The claim is not credible |
| Apify-style scrapers | Avoid | Direct terms violations |

**The legal backdrop is worse than generic ToS risk**, and there are three distinct exposures:
1. **Copyright** in the photographs (§3.7).
2. **The EU sui generis database right** (Directive 96/9/EC) — separate from copyright, and it protects the portal's *collection*. French precedent is directly on point: in **Entreparticuliers.com v. Leboncoin (2021)** the court awarded **€50,000** against systematic extraction of property listings.
3. **AI-specific terms.** Zoopla's terms prohibit text/data mining and use of the site "for the purposes of developing or contributing towards a solution utilising artificial intelligence" without a licence — drafted squarely at products like ours.

Together with **CoStar v. Zillow** (53,000+ images claimed, active in 2026), this closes the question: **the aggregator/scraper category is not a shortcut, it is the single largest legal risk available to us**, and the roadmap's existing no-scraping rule stands reinforced.

### 3.9 Decision of record: in-house scraping of Rightmove and Zoopla (19 Aug 2026)

**Decided by the founder: we will build our own scraper for Rightmove and Zoopla.** Recorded here as the project's data-acquisition position, superseding the earlier "no scraping" scope fence for the acquisition of *input* imagery. The rest of this section is the engineering consequence, not a re-argument.

**What scraping does and does not solve.** It solves **access** — getting pixels and plans onto disk quickly, without waiting on a partner. It does **not** solve **rights**, and those are separate problems (§3.7): the photographer owns the copyright, so no amount of access creates permission to *publish a derived 3D model* of their photograph to end users. Planning must therefore keep two tracks:

| Use of scraped data | Status | Note |
|---|---|---|
| Internal R&D, algorithm development, the golden set, regression testing | Workable now | No redistribution, no publication of outputs |
| Training shipped models | ⚠ Avoid | Zoopla's terms specifically prohibit use of the site for "developing or contributing towards a solution utilising artificial intelligence" without a licence. A model trained on scraped data carries that taint into the product |
| Publishing 3D outputs to end users | ⚠ Needs a rights grant regardless of how the images were obtained | This is the copyright question, not the access question |

**The exposures, so they're costed rather than discovered.** Three distinct ones, all verified: portal terms explicitly prohibit scraping (both portals); the **EU sui generis database right** (Directive 96/9/EC) protects the portal's collection independently of image copyright — the French precedent **Entreparticuliers.com v. Leboncoin (2021), €50,000** was exactly systematic extraction of property listings; and image copyright itself, which is what **CoStar v. Zillow** (53,000+ images, active 2026) turns on. Practical exposure also includes IP blocking and anti-bot measures (both portals run active detection), which is an ongoing engineering cost rather than a one-off build.

**Risk reducers worth building in from the start** (cheap now, expensive to retrofit):
1. **Tag every asset with its provenance** (`scraped` / `partner_granted` / `self_captured`) in the manifest, and make the pipeline able to filter on it. This is what lets you later prove a shipped model saw no scraped data, or purge a source if a position changes. It reuses the provenance machinery already in the architecture.
2. **Keep a hard split**: scraped data in the R&D/eval lane; partner-granted data in the training-and-publishing lane. Enforce it in CI the same way the licence gate works.
3. **Rate-limit and respect robots.txt.** Reduces the "systematic extraction" characterisation and the blocking arms race, though it does not remove database-right exposure.
4. **Note the clean path stays available for the product:** where the *agent is the customer*, they supply their own listing imagery — identical pixels, zero exposure. Scraping is only needed for listings where we have no relationship (speculative demos, consumer "paste any listing" use). That makes it a **bootstrapping and demo tool**, not the production input channel — the CRM route (§3.6) remains the production answer.

**Owner:** founder decision; counsel should still be briefed before any customer-facing output derives from scraped imagery.

## 4. Recommendation

**Headline: no dataset or API removes the need for a data partner — but the partner just got much easier to land, and the training-data blocker has largely dissolved.**

### 4.1 What to do, per need

**N1 — Golden set (Sprints 1–2).** Three tracks in parallel, cheapest first:
1. **Email Rent3D's authors** (University of Toronto) in week 1 asking licence terms. 215 London flats with photos aligned to floor plans is startlingly close to our exact problem; even a research-use grant makes it a valuable early regression set. One email, potentially large payoff.
2. **Recruit 2–3 friendly agents** via the §3.6 CRM route and have them authorise access to their own live listings. This is the real deliverable — current-style photography, real plans, *and* the start of the commercial relationship.
3. **Shoot and measure 10–15 ourselves** (laser measurements plus a few paid Matterport scans) for unimpeachable ground truth with zero licence questions.

**N2 — Training corpus.** Much better than the earlier plan assumed:
- **Floor-plan vectoriser:** train on **ResPlan** (17K, CC BY 4.0) + **Swiss Dwellings** (42K apartments, CC BY 4.0, European) + **MSD** (CC BY-SA — check share-alike implications for released weights) + synthetic generation; fine-tune on partner plans for French/UK agency style. **CubiCasa5K is no longer a blocker — it is simply unnecessary.**
- **Photo↔plan correspondence:** pretrain on **C3Po** (90K pairs, CC BY 4.0, commercial use permitted), accepting the domain gap (internet building photography, not agent flat photography); fine-tune on partner data.
- **Triage classifier:** partner images with VLM-bootstrapped labels. Avoid Places365 and MIT Indoor67 (non-commercial) and every unlicensed Kaggle/HF set — no stated licence means no rights.

**N3 — Production input.** Build the **agency-CRM integration** as the primary and probably only channel: **Street.co.uk** and **Reapit** in the UK (free, self-serve, OAuth, documented media endpoints), **Apimo** and **Hektor** in France (photos *and* floor plans). There is no credible aggregator fallback (§3.8); breadth comes from adding CRM connectors, not from buying data.

### 4.2 What this changes in the roadmap

1. **Gate G0's kill criterion needs rewording.** "No data partner ⇒ stop" was right in spirit but wrong in target: the binding constraints are (a) enough ground-truth listings and (b) a signed rights grant covering derivative 3D works — not a portal deal. Two or three small agents on Street/Reapit satisfy both, a far lower bar than the original framing, and reachable inside P0.
2. **Sprint 1 gains a concrete build target** instead of open-ended outreach: the Street.co.uk *or* Reapit OAuth connector as our ingestion path. Documented, free, self-serve — days, not months — and it turns the agent conversation into "click Authorise" rather than "please export some files".
3. **Sprint 3's vectoriser plan improves**: pretrain on ResPlan + Swiss Dwellings rather than building a corpus from scratch. Lower risk, faster, and it removes a dependency on partner data arriving early.
4. **The rights grant becomes a Sprint 1 blocker, not a Sprint 14 compliance task** (§3.7). Draft the licence language — process, store, derive, publish, and separately train — before ingesting a single real image.
5. **Market sequencing is now an informed choice** (§5): France has the better scale anchor, the UK has the better data plumbing. The golden set should span both so we measure the difference instead of guessing it.

## 5. The scale anchor differs by country — and it argues for France first

The architecture's scale solve (ARCHITECTURE §5) leans on a published, legally-meaningful surface area. Verified position in each market:

**France — two anchors, both mandatory in the advertisement itself:**
- **Loi Carrez** — for lots in copropriété (i.e. most flats), the private surface must appear in the *annonce* as well as the sale documents. A shortfall >5% entitles the buyer to a proportional price reduction, so agents have real incentive to state it accurately. This is the strong anchor the architecture was designed around, and it is confirmed.
- **DPE** (energy certificate) — also mandatory in every sale/rental advertisement, also records surface area, and ADEME publishes a database of ~25M+ records. A free cross-check on the Carrez figure.

**UK — no legal equivalent, one usable proxy:**
- There is **no requirement to state floor area** in a UK listing. NTSELAT's material-information guidance never included it and was withdrawn in May 2025 anyway (superseded by the DMCC Act 2024, under which the CMA has declined to issue property-specific guidance). RICS removed residential from its Code of Measuring Practice's scope in 2018, and RICS measurement never applied to listings regardless.
- Rightmove and Zoopla *do* display floor area when available, sourced from the agent's floor plan or the EPC — but it is optional and inconsistent.
- **The proxy: EPC total floor area.** Every domestic EPC records it; ~22M+ records for England & Wales are published as open data under the Open Government Licence, per address, with bulk download and API access (Scotland has its own register). **But accuracy is documented as poor** — research suggests millions of EPCs carry floor-area errors, with real cases of 180 m² recorded against 120 m² actual.

**Consequence for the plan.** EPC area is usable as a *weak, wide-uncertainty* constraint in the scale solve, never as the strong anchor Carrez provides — and our own reconstructed area may well be more accurate than the EPC figure, which is a marketing angle but also a reason never to display a competing number (ARCHITECTURE §5's rule holds). Practically: **French listings will reconstruct to ±3–5%; UK listings to something meaningfully worse** unless the listing carries a floor plan with printed dimensions. Combined with the fact that the cleanest data route (§3.6) is UK-flavoured, the sensible split is: **France for accuracy-led product proof, UK for data-access ease** — and the P0 golden set should include both so the difference is measured rather than assumed.
