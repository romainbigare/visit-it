# Data sources — how we get listings, photos and floor plans

**Status:** research of 19 Aug 2026, first scraper run 20 Aug 2026.

> **Scope note (20 Aug 2026): internal, non-commercial.** This document used to
> be half data-acquisition research and half legal analysis — licence bars per
> dataset, photograph copyright, database rights, the rights grant needed to
> publish derived 3D works, and the agency-CRM route that existed to obtain
> those rights. None of that applies to a project we build for ourselves and do
> not ship. Those sections are removed. What remains is the engineering
> question: **what data can we actually obtain and run on, and what is it like?**

## 1. What we need

| Need | Volume | Must have | When |
|---|---|---|---|
| **N1 — Golden set** (the evaluation corpus) | 30–50 listings | Photos + floor plan + stated area | P0, weeks 1–4 |
| **N2 — Training corpus** (floor-plan vectoriser, triage classifier) | 1,000s of plans; 10,000s of images | Floor plans in the styles our market uses; room-labelled interior photos | P1, weeks 5–12 |
| **N3 — Pipeline input** (listings we process) | ongoing | Photos + plan | P2 onward |

The correctness question these feed — how we judge a reconstruction without
tape-measuring flats — is answered in [`ROADMAP.md`](../ROADMAP.md) §0b:
plausibility, self-consistency against dimensions printed on the plan, and
cross-model agreement.

## 2. The rubric for a candidate source

1. **Images included?** Many "property data" products are prices, addresses and EPCs only — no photos. This kills most of the obvious names immediately.
2. **Floor plans included?** Rarer than photos, and the floor plan is our spine.
3. **Access reality:** open API / paid API / enterprise sales only / dead / agent-side only.
4. **Geography:** UK first, France second (the surface-area anchor differs — see §5).
5. **Volume and cost.**

## 3. Findings by route

**Caveat on method:** several "API provider" claims in the wild are scraping resellers wearing an API coat. Where a source's provenance is not established it is marked ⚠ — an Apify scraper existing for a portal is *not* evidence that the portal offers automated access.

### 3.1 UK portals — effectively closed as APIs

| Source | API? | Photos/plans? | Verdict |
|---|---|---|---|
| **Rightmove** | Real Time Data Feed exists, and its spec does carry image and floor-plan URLs | Yes, in spec | **Inbound only** — RTDF is how *agents upload to* Rightmove, not how third parties read from it. No outbound listing API |
| **Zoopla** | Historic public API retired; no current developer portal found | Unknown | Dead / opaque |
| **OnTheMarket** | None | — | Dead end |
| **Nestoria** (Lifull Connect) | Yes, free, no registration | **Yes — image URLs** | The only free UK API route with images. Useful as a secondary channel |
| PropertyData (£28/mo) | Yes | **No images.** Does OCR floor plans to extract square footage | Interesting as a *cross-check on our own area estimates*, not as an image source |
| PaTMa, Sprift, PriceHubble, LandTech, Homedata | Yes | **No images** — attributes, valuations, EPC, planning | Not relevant to us |

**Conclusion:** there is no bulk API route to UK listing photos and floor plans. We scrape (§3.6).

### 3.2 EU portals

- **Idealista** (Spain) and **ImmobilienScout24** (Germany) have real partner APIs, vetted-partner-only.
- **SeLoger, Logic-Immo, Leboncoin, PAP, Figaro, Immobiliare.it, Funda, Pararius:** no third-party APIs. Funda's partner API is discontinued.
- **Bien'ici has no public API** — the "free public API with photos" claim from the first research pass was wrong; it came from Apify scraper listings.
- **Structural fact:** Europe has no MLS. Listings are fragmented across hundreds of portals with no shared identifier, so there is no single integration that reaches the market.

### 3.3 Datasets that pair photos with floor plans

| Dataset | Contents | Geography | Availability | Use to us |
|---|---|---|---|---|
| **Rent3D** (CVPR 2015, Toronto) | **215 apartments, ~1,570 interior photos, annotated floor plans** with photo-to-plan alignment | **London** | ⚠ **Advertised download is dead** — 404 from both Toronto hostnames, confirmed twice (20 Aug). The fetcher prints an enquiry to the authors | Closest public analogue to our exact problem, in our target market. Small and ten years old (photography styles have changed) but ideal as an early regression set |
| **Rent3D++** (via Plan2Scene, CVPR 2021) | Rent3D extended: better alignment, more photos, object icons | London | Access by request form | Same as above; also the source of the coverage-level evaluation protocol we adopted (M8) |
| **C3Po** (NeurIPS 2025) | **90K photo/floor-plan pairs, 597 scenes, 153M correspondences, 85K poses** | Internet-sourced (Wikimedia plans + MegaScenes/YFCC photos) — **not estate-agent listings** | Open | Best available for *pretraining photo↔plan correspondence*. Domain gap is real: public/landmark buildings, not flats shot at 16 mm by an agent. Pretrain here, fine-tune on scraped listings |
| **LIFULL HOME'S** (NII Japan) | 5.33M listings, 83M photos, 5.31M floor plans | Japan | Academic application | Wrong market; reference only |
| **ZInD** (Zillow) | 67,448 panoramas, 1,575 homes, 2,500+ plans | USA | Request form | Panoramas, not agent photography — different capture regime |

### 3.4 Floor-plan corpora

| Corpus | Size | Note |
|---|---|---|
| **ResPlan** | **17,000 vector floor plans** with room graphs, metric scale, 17 semantic classes | Derived from US listings; vector only, no source raster images |
| **Swiss Dwellings** (Archilyse) | **42,207 apartments / 242,257 rooms** across 3,093 buildings, with geometry | **European**, and large. Downloaded and verified end to end (792 MB, 2.5 M rows) |
| **MSD (Modified Swiss Dwellings)** (ECCV 2024) | 5,372 plans / 18.9K apartments, vector + raster + graph | European multi-apartment stock |
| **CubiCasa5K** | 5,000 plans, raster + vector | The classic raster→vector benchmark; useful precisely because it *is* raster, which ResPlan is not |
| RPLAN, ROBIN, SESYD, FloorPlanCAD | various | Style variety for augmentation |

All are registered in `pipeline/datasets/registry.py` with live-verified URLs.

**Consequence:** the plan vectoriser has a large corpus available today — ResPlan + Swiss Dwellings + MSD + CubiCasa5K + synthetic generation — then fine-tuning on scraped UK plans for local style. The remaining gap is *stylistic*: none of these are UK agency-drawn plans, which is what the fine-tune is for.

### 3.5 Interior-photo corpora (for triage/room classification)

**Places365** and **MIT Indoor67** are the two classic scene-classification sets. Kaggle/HuggingFace room-classification sets (House Rooms & Streets ~25K images; MMIS 160K interior images) are larger but noisier. **Inside Airbnb is metadata only, no photos.**

In practice Phase 0 showed triage barely needs a training corpus: **zero-shot SigLIP scored F1 0.96** on our own listings, and was *more accurate than Rightmove's own image metadata*. Fine-tuning is an optimisation, not a prerequisite — and the natural corpus is our own scraped set with SigLIP-bootstrapped labels.

### 3.6 Decision of record: in-house scraping of Rightmove and Zoopla (19 Aug 2026)

**Decided: we build our own scraper.** It is the only route that puts real UK listing pixels and plans on disk without a portal relationship, and portals are the only place current-style agent photography exists at volume.

Practical constraints, which are engineering constraints rather than paperwork:

1. **Rate-limit and respect `robots.txt`.** This is what keeps the scraper working — aggressive crawling gets the IP blocked and costs us the channel.
2. **Never impersonate a named crawler.** Rightmove disallows GPTBot, CCbot, SpriftCrawler and TrovitBot outright; sending their user-agent strings is both dishonest and a fast route to a block.
3. **Tag every asset with its provenance** (`scraped` / `self_captured`) in the manifest. Kept for debuggability — when a reconstruction misbehaves, the first question is where its inputs came from — and because it lets us purge a source cleanly.
4. **Never commit scraped imagery to the repo.** `.gitignore` covers `data/golden/media/`. This keeps the repo small and keeps other people's photographs out of our history.

#### Measured reality, 20 Aug 2026 (first scraper run)

Built and run against both portals. Findings, all from live requests:

| | Result |
|---|---|
| **Rightmove** | **Works.** `robots.txt` permits `/property-for-sale/find.html` and `/properties/<id>` for a generic user-agent. Search results embed `__NEXT_DATA__`; detail pages embed a devalue-flattened `window.__PAGE_MODEL`. Both parse cleanly. No rate limiting or blocking encountered at ~1 req/sec |
| **Zoopla** | **Blocked.** Cloudflare returns HTTP 403 with a "Just a moment…" interstitial to *every* request from a datacentre IP — including `robots.txt` itself. No header combination gets through. It needs a real browser plus a residential IP. The adapter is written but its field mapping is **unverified**, since we have never seen a real response |
| **Floor-plan coverage** | **92.5%** of listings seen carried at least one floor plan (98 of 106; 8 missing). Better than assumed — the UK plan channel is well supplied |
| **Stated floor area** | ~53% of search results, and 63% of fetched listings, carried a usable area figure — from Rightmove's `sizings` field or parsed out of the description |
| **Dimensions printed on the plan** | 54% of plans, rising to 62% of high-resolution ones. This is the single most useful scale signal in the UK (§5), and it is why the scraper always takes the largest plan asset |
| **Photos per listing** | median 16.5, range 5–54. Comfortably above the 6–12 views per room where splatting becomes viable, *if* they are spread across rooms rather than concentrated in the reception |

Two engineering notes worth carrying forward: Python's stdlib `urllib.robotparser` **silently ignores `*` wildcards**, so it would have permitted paths Rightmove disallows — we wrote our own matcher (`pipeline/ingest/robots.py`). And listing search results need a non-dwelling filter: blocks, portfolios and off-plan "6% rental yield" listings (typically illustrated with CGI renders, not photographs) pollute a golden set badly.

## 4. Recommendation

### 4.1 What to do, per need

**N1 — Golden set.** Done: 30 Rightmove listings, 6 cities, price-band stratified, 553 images, 24 with floor plans. Rebuild with `make golden`. The remaining work is *stratification quality*, not volume — an unstratified London search returns almost nothing but prime-central trophy flats, which would flatter the pipeline badly at G1.

**N2 — Training corpus.**
- **Floor-plan vectoriser:** ResPlan (17K vector) + Swiss Dwellings (42K apartments, European) + MSD + CubiCasa5K (raster) + synthetic generation; fine-tune on scraped UK plans for local style.
- **Photo↔plan correspondence:** pretrain on **C3Po** (90K pairs), accepting the domain gap; fine-tune on scraped listings.
- **Triage classifier:** not needed yet — zero-shot SigLIP is already at F1 0.96 (§3.5).

**N3 — Pipeline input.** The scraper (§3.6), with Nestoria as a secondary channel if breadth is ever needed.

### 4.2 What this changes in the roadmap

1. **Sprint 3's vectoriser plan improves**: pretrain on ResPlan + Swiss Dwellings rather than building a corpus from scratch. Lower risk, faster.
2. **Plan OCR is promoted.** Dimensions printed on 54% of plans are both a scale constraint *and* the self-consistency check that replaces tape measurements (ROADMAP §0b). It is a small, CPU-only stage with outsized value.
3. **Market sequencing is an informed choice** (§5): France has the better scale anchor, the UK has the better scraping surface. We are UK-first for that reason, and the France question can be revisited when the pipeline works.

## 5. The scale anchor differs by country

The architecture's scale solve (ARCHITECTURE §5) leans on a published surface area. Verified position in each market:

**France — two anchors, both mandatory in the advertisement itself:**
- **Loi Carrez** — for lots in copropriété (i.e. most flats), the private surface must appear in the *annonce*. A shortfall >5% entitles the buyer to a proportional price reduction, so agents have real incentive to state it accurately. This is the strong anchor the architecture was originally designed around.
- **DPE** (energy certificate) — also mandatory in every advertisement, also records surface area, and ADEME publishes a database of ~25M+ records. A free cross-check.

**UK — no legal equivalent, two usable proxies:**
- There is **no requirement to state floor area** in a UK listing. NTSELAT's material-information guidance never included it and was withdrawn in May 2025. RICS removed residential from its Code of Measuring Practice's scope in 2018.
- Rightmove and Zoopla *do* display floor area when available, sourced from the agent's floor plan or the EPC — optional and inconsistent, present on ~53% of what we scraped.
- **EPC total floor area.** Every domestic EPC records it; ~22M+ records for England & Wales are published as open data, per address, with bulk download and API access. **But accuracy is documented as poor** — real cases of 180 m² recorded against 120 m² actual.
- **Dimensions printed on the floor plan** — present on 54% of ours (62% of high-resolution plans). Measured in Phase 0, and better than the EPC-only fallback the plan originally assumed. This is the UK's real anchor.

**Consequence for the plan.** EPC area is a *weak, wide-uncertainty* constraint in the scale solve; plan-printed dimensions are the strong one where they exist; the stated area sits between. Practically: **UK listings with a dimensioned plan should reconstruct about as well as French ones; UK listings without will be meaningfully worse.** The scraper's preference for the largest plan asset exists to keep as many listings as possible in the first category.
