"""Build the Phase 0 golden set and report floor-plan coverage.

    python -m pipeline.ingest.collect --target 30 --min-floorplan-frac 0.65

Two outputs land in --out-dir:
  golden_set.json      the listings themselves
  coverage_report.md   floor-plan coverage over EVERYTHING SEEN, plus the
                       selected-set summary

Coverage is measured on every search result inspected, before any filtering.
Reporting coverage over a set we deliberately filtered for floor plans would
be meaningless, so the two numbers are kept separate throughout.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import re as _re

from .http import BlockedError, PoliteSession
from .models import CoverageStats, Listing
from . import rightmove, zoopla

log = logging.getLogger("collect")

# Listings that are not a single dwelling: blocks, portfolios, off-plan
# developments, land. These pollute a golden set - a "19 bedroom" result is a
# block of flats, not a flat, and nothing downstream can do anything sensible
# with it.
_NOT_A_DWELLING = _re.compile(
    # Whole listings that are not a single dwelling.
    r"\b(?:portfolio|investment opportunity|off[- ]plan|block of|development site|"
    r"land for sale|hmo|guest house|freehold block|ground rent investment|"
    r"apartment block|student (?:scheme|investment))\b"
    # Off-plan / investment marketing: usually illustrated with CGI renders
    # rather than photographs, which is worse than useless to us. No trailing
    # \b here - these phrases often end at a '|' or a plural, where a word
    # boundary cannot match.
    r"|\brental yield|\byields?\s*\||\bbuy[- ]to[- ]let\b"
    r"|\d+(?:\.\d+)?\s*%\s*(?:rental\s*)?yield"
    r"|\bassured (?:rent|yield)|\bcomputer[- ]generated",
    _re.I)
MAX_SENSIBLE_BEDROOMS = 6


def is_single_dwelling(rec: dict) -> tuple[bool, str]:
    """Cheap sanity filter applied to search results before any detail fetch."""
    beds = rec.get("bedrooms")
    if isinstance(beds, int) and beds > MAX_SENSIBLE_BEDROOMS:
        return False, f"bedrooms={beds}"
    blob = " ".join(str(rec.get(k) or "") for k in
                    ("displayAddress", "summary", "propertyTypeFullDescription", "propertySubType"))
    m = _NOT_A_DWELLING.search(blob)
    if m:
        return False, f"matched '{m.group(0)}'"
    sub = (rec.get("propertySubType") or "").lower()
    if sub in ("land", "block of apartments", "commercial property"):
        return False, f"subtype={sub}"
    return True, ""


def _select(candidates: list[dict], target: int, fp_frac: float) -> tuple[list[dict], list[dict]]:
    """Split candidates into (with-plan, without-plan) picks hitting the quota.

    We deliberately keep some plan-less listings: they are the test cases for
    the Tier B / no-plan branch (ROADMAP P3), so a set that is 100% plans would
    be the wrong shape even though it would trivially pass the quota.
    """
    def interleave(items: list[dict]) -> list[dict]:
        """Round-robin by stratum so one city/price band cannot dominate."""
        buckets: dict[str, list[dict]] = {}
        for it in items:
            buckets.setdefault(it.get("_stratum", "?"), []).append(it)
        out, order = [], list(buckets)
        while any(buckets[k] for k in order):
            for k in order:
                if buckets[k]:
                    out.append(buckets[k].pop(0))
        return out

    with_fp = interleave([c for c in candidates if (c.get("numberOfFloorplans") or 0) > 0])
    without_fp = interleave([c for c in candidates if not (c.get("numberOfFloorplans") or 0)])
    n_with = min(len(with_fp), math.ceil(target * fp_frac))
    n_without = min(len(without_fp), target - n_with)
    n_with = min(len(with_fp), target - n_without)  # backfill if too few plan-less
    return with_fp[:n_with], without_fp[:n_without]


# Price bands (GBP). A golden set drawn from one band would be unrepresentative:
# an unstratified London search returns almost nothing but prime-central trophy
# flats, which carry 20+ professionally shot photos and would flatter the
# pipeline badly at gate G1.
PRICE_BANDS: tuple[tuple[int | None, int | None], ...] = (
    (None, 250_000),
    (250_000, 450_000),
    (450_000, 750_000),
    (750_000, 1_500_000),
)


def collect(target: int = 30, min_fp_frac: float = 0.65, fp_target_frac: float = 0.75,
            regions: tuple[str, ...] = ("london",), out_dir: Path = Path("data/golden"),
            max_pages: int = 8, download_media: bool = False, min_photos: int = 4,
            include_zoopla: bool = False, min_interval: float = 1.5,
            stratify: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = PoliteSession(min_interval=min_interval)
    stats: dict[str, CoverageStats] = {}
    listings: list[Listing] = []
    notes: list[str] = []

    # ---- Rightmove -----------------------------------------------------
    rm_stats = CoverageStats(source="rightmove")
    stats["rightmove"] = rm_stats
    candidates: list[dict] = []
    rejected: list[str] = []
    seen_ids: set[str] = set()
    strata = ([(r, lo, hi) for r in regions for (lo, hi) in PRICE_BANDS]
              if stratify else [(r, None, None) for r in regions])
    per_stratum = max(2, math.ceil(target * 2.5 / max(1, len(strata))))
    for region, lo, hi in strata:
        got = 0
        try:
            for rec in rightmove.iter_search_results(
                    session, region, max_pages=max_pages, stats=rm_stats,
                    max_price=hi, min_price=lo):
                rid = str(rec.get("id"))
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                if (rec.get("numberOfImages") or 0) < min_photos:
                    continue  # too few photos to reconstruct anything
                ok, why = is_single_dwelling(rec)
                if not ok:
                    rejected.append(f"{rec.get('id')}: {why}")
                    continue
                rec["_stratum"] = f"{region}:{(lo or 0)//1000}-{(hi or 0)//1000}k"
                candidates.append(rec)
                got += 1
                if got >= per_stratum:
                    break
        except BlockedError as e:
            notes.append(f"rightmove/{region}: BLOCKED - {e}")
            log.error("rightmove blocked: %s", e)
        except Exception as e:  # noqa: BLE001 - one bad stratum must not sink the run
            notes.append(f"rightmove/{region}/{lo}-{hi}: {type(e).__name__}: {e}")
            log.exception("rightmove stratum %s %s-%s failed", region, lo, hi)

    picks_with, picks_without = _select(candidates, target, fp_target_frac)
    log.info("candidates=%d (rejected %d non-dwellings) -> picking %d with plan + %d without",
             len(candidates), len(rejected), len(picks_with), len(picks_without))
    if rejected:
        notes.append(f"filtered out {len(rejected)} non-dwelling listings "
                     f"(blocks/portfolios/off-plan): {', '.join(rejected[:8])}"
                     + (" ..." if len(rejected) > 8 else ""))

    # Fetch detail pages, topping up from spares if any fail.
    queue = picks_with + picks_without
    spares = [c for c in candidates if c not in queue]
    qi = 0
    while len(listings) < min(target, len(candidates)) and (qi < len(queue) or spares):
        rec = queue[qi] if qi < len(queue) else spares.pop(0)
        qi += 1
        try:
            lst = rightmove.fetch_listing(session, rec["id"])
            rm_stats.detail_fetched += 1
            listings.append(lst)
            log.info("[%2d/%d] %s  photos=%2d plan=%s  %s", len(listings), target,
                     lst.listing_id, lst.n_photos, "Y" if lst.has_floorplan else "n",
                     (lst.display_address or "")[:52])
        except Exception as e:  # noqa: BLE001
            rm_stats.detail_failed += 1
            notes.append(f"detail {rec.get('id')}: {type(e).__name__}: {e}")
            log.warning("detail fetch failed for %s: %s", rec.get("id"), e)

    # ---- Zoopla (expected to be blocked; see zoopla.py) -----------------
    if include_zoopla:
        z_stats = CoverageStats(source="zoopla")
        stats["zoopla"] = z_stats
        try:
            for _ in zoopla.iter_search_results(session, "london", max_pages=1, stats=z_stats):
                pass
        except BlockedError as e:
            notes.append(f"zoopla: BLOCKED - {e}")
            log.error("zoopla blocked: %s", e)
        except Exception as e:  # noqa: BLE001
            notes.append(f"zoopla: {type(e).__name__}: {e}")
            log.error("zoopla failed: %s", e)

    # ---- media ---------------------------------------------------------
    if download_media:
        media_root = out_dir / "media"
        for lst in listings:
            for i, ref in enumerate(lst.photos + lst.floorplans):
                ext = Path(ref.url.split("?")[0]).suffix or ".jpg"
                dest = media_root / lst.listing_id / f"{ref.kind}_{i:02d}{ext}"
                try:
                    session.download(ref.url, dest)
                    ref.local_path = str(dest.relative_to(out_dir))
                except Exception as e:  # noqa: BLE001
                    log.warning("media failed %s: %s", ref.url, e)

    # ---- write ---------------------------------------------------------
    n_fp = sum(1 for x in listings if x.has_floorplan)
    frac = (n_fp / len(listings)) if listings else 0.0
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": "scraped",
        "market": "UK",
        "selection": {
            "target": target,
            "collected": len(listings),
            "with_floorplan": n_fp,
            "without_floorplan": len(listings) - n_fp,
            "floorplan_fraction": round(frac, 4),
            "min_floorplan_fraction_required": min_fp_frac,
            "quota_met": frac >= min_fp_frac,
        },
        "coverage_over_all_seen": {k: v.to_dict() for k, v in stats.items()},
        "notes": notes,
        "listings": [x.to_dict() for x in listings],
    }
    (out_dir / "golden_set.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "coverage_report.md").write_text(render_report(payload))
    return payload


def render_report(p: dict) -> str:
    sel = p["selection"]
    L = [
        "# UK golden set - collection report",
        "",
        f"Generated {p['generated_at']} · provenance `{p['provenance']}` · market {p['market']}",
        "",
        "## Selected set",
        "",
        f"| Target | Collected | With floor plan | Without | Fraction | Quota (>={sel['min_floorplan_fraction_required']:.0%}) |",
        "|---|---|---|---|---|---|",
        f"| {sel['target']} | {sel['collected']} | {sel['with_floorplan']} | "
        f"{sel['without_floorplan']} | {sel['floorplan_fraction']:.1%} | "
        f"{'PASS' if sel['quota_met'] else 'FAIL'} |",
        "",
        "## Floor-plan coverage across everything seen",
        "",
        "Measured on every search result inspected, *before* any filtering - this is the",
        "honest market rate, not the rate in our deliberately-filtered set.",
        "",
        "| Source | Seen | With plan | Missing plan | Coverage | Multi-plan | Stated area | Median photos |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in p["coverage_over_all_seen"].values():
        L.append(
            f"| {s['source']} | {s['listings_seen']} | {s['with_floorplan']} | "
            f"{s['without_floorplan']} | {s['floorplan_rate']:.1%} | "
            f"{s['with_multiple_floorplans']} | {s['area_rate']:.1%} | "
            f"{s['median_photos_per_listing']} |"
        )
    L += ["", "## Listings", "",
          "| # | ID | Address | Beds | Photos | Plan | Area m² | Area source |",
          "|---|---|---|---|---|---|---|---|"]
    for i, x in enumerate(p["listings"], 1):
        L.append(f"| {i} | {x['listing_id']} | {(x['display_address'] or '')[:44]} | "
                 f"{x['bedrooms'] or ''} | {x['n_photos']} | "
                 f"{'Y' if x['has_floorplan'] else '-'} | {x['floor_area_sqm'] or ''} | "
                 f"{x['floor_area_source'] or ''} |")
    if p["notes"]:
        L += ["", "## Notes and failures", ""] + [f"- {n}" for n in p["notes"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--min-floorplan-frac", type=float, default=0.65)
    ap.add_argument("--floorplan-target-frac", type=float, default=0.75,
                    help="aim above the floor so the quota survives detail-fetch failures")
    ap.add_argument("--regions", default="london", help="comma-separated Rightmove regions")
    ap.add_argument("--out-dir", type=Path, default=Path("data/golden"))
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--min-photos", type=int, default=4)
    ap.add_argument("--min-interval", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--download-media", action="store_true")
    ap.add_argument("--no-stratify", action="store_true",
                    help="disable region/price stratification (not recommended)")
    ap.add_argument("--include-zoopla", action="store_true",
                    help="attempt Zoopla (blocked from datacentre IPs; see zoopla.py)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = collect(target=a.target, min_fp_frac=a.min_floorplan_frac,
                fp_target_frac=a.floorplan_target_frac,
                regions=tuple(r.strip() for r in a.regions.split(",")),
                out_dir=a.out_dir, max_pages=a.max_pages,
                download_media=a.download_media, min_photos=a.min_photos,
                include_zoopla=a.include_zoopla, min_interval=a.min_interval,
                stratify=not a.no_stratify)

    sel = p["selection"]
    print("\n" + render_report(p).split("## Listings")[0])
    if not sel["quota_met"]:
        print(f"QUOTA NOT MET: {sel['floorplan_fraction']:.1%} < "
              f"{sel['min_floorplan_fraction_required']:.0%}", file=sys.stderr)
        return 1
    if sel["collected"] < a.target:
        print(f"UNDER TARGET: {sel['collected']} < {a.target}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
