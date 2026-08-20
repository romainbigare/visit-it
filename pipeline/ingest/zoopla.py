"""Zoopla adapter.

STATUS (verified 2026-08-20): Zoopla sits behind Cloudflare bot protection.
Every request from this project's cloud environment - including
https://www.zoopla.co.uk/robots.txt - returns HTTP 403 with a
"Just a moment..." interstitial. Plain HTTP scraping of Zoopla does not work
from a datacentre IP, no matter the headers.

Consequences, stated plainly so nobody wastes a day rediscovering them:
  * The HTTP path below will raise BlockedError on any cloud/CI runner.
  * The Playwright path needs a real browser AND a residential IP to stand a
    chance; Cloudflare fingerprints the TLS stack and the IP reputation, not
    just the user-agent.
  * Because the site could not be reached, THE FIELD MAPPING BELOW IS
    UNVERIFIED. It follows Zoopla's documented Next.js structure but has never
    been run against a real response. Treat `parse_search`/`parse_property`
    as a starting point to fix, not as working code.

Rightmove alone is sufficient for the Phase 0 golden set; Zoopla is a
second-source nice-to-have. See docs/DATA-SOURCES.md §3.9.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterator

from .http import BlockedError, PoliteSession
from .models import CoverageStats, Listing, MediaRef

log = logging.getLogger(__name__)

BASE = "https://www.zoopla.co.uk"
UNVERIFIED = True  # flip to False once a real response has been parsed successfully


def _next_data(html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _dig(obj: Any, *keys: str) -> Any:
    """Depth-first search for the first dict value under any of `keys`."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k in keys:
                if k in cur:
                    return cur[k]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def parse_search(html: str) -> list[dict]:
    """UNVERIFIED - see module docstring."""
    d = _next_data(html)
    if not d:
        raise ValueError("Zoopla search payload not found")
    listings = _dig(d, "regularListingsFormatted", "listings", "results")
    if isinstance(listings, dict):
        listings = listings.get("regular") or listings.get("results") or []
    return listings or []


def parse_property(html: str) -> dict:
    """UNVERIFIED - see module docstring."""
    d = _next_data(html)
    if not d:
        raise ValueError("Zoopla property payload not found")
    pd = _dig(d, "listingDetails", "listing", "propertyDetails")
    if not isinstance(pd, dict):
        raise ValueError("Zoopla property detail object not found")
    return pd


def to_listing(pd: dict) -> Listing:
    """UNVERIFIED - see module docstring."""
    lid = str(pd.get("listingId") or pd.get("id") or "")
    images, floorplans = [], []
    for im in (pd.get("propertyImage") or pd.get("images") or []):
        url = im.get("original") or im.get("filename") or im.get("src") if isinstance(im, dict) else im
        if url:
            images.append(MediaRef(url=url, kind="photo",
                                   caption=(im.get("caption") if isinstance(im, dict) else None)))
    for fp in (pd.get("floorPlan") or pd.get("floorplans") or []):
        url = fp.get("original") or fp.get("filename") or fp.get("src") if isinstance(fp, dict) else fp
        if url:
            floorplans.append(MediaRef(url=url, kind="floorplan"))
    return Listing(
        listing_id=lid,
        source="zoopla",
        url=pd.get("detailUrl") or f"{BASE}/for-sale/details/{lid}/",
        display_address=pd.get("displayAddress") or pd.get("address"),
        outcode=pd.get("outcode"),
        country="United Kingdom",
        property_subtype=pd.get("propertyType"),
        transaction_type="buy",
        tenure=pd.get("tenure"),
        bedrooms=pd.get("numBedrooms") or pd.get("bedrooms"),
        bathrooms=pd.get("numBathrooms") or pd.get("bathrooms"),
        price_gbp=(int(re.sub(r"[^\d]", "", str(pd.get("price"))))
                   if re.search(r"\d", str(pd.get("price") or "")) else None),
        photos=images,
        floorplans=floorplans,
        description=pd.get("description"),
        agent=(pd.get("branch") or {}).get("name") if isinstance(pd.get("branch"), dict) else None,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def fetch_html_via_browser(url: str, timeout_ms: int = 45000) -> str:
    """Render `url` in a real browser. Requires `pip install playwright`.

    Even this is not guaranteed against Cloudflare from a datacentre IP.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Zoopla needs a real browser: pip install playwright "
            "(Chromium is preinstalled at $PLAYWRIGHT_BROWSERS_PATH)"
        ) from e
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(locale="en-GB", viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()
    if "just a moment" in html[:4000].lower():
        raise BlockedError("Cloudflare challenge survived browser rendering; needs a residential IP")
    return html


def iter_search_results(session: PoliteSession, region: str = "london", *,
                        max_pages: int = 5, stats: CoverageStats | None = None,
                        use_browser: bool = False) -> Iterator[dict]:
    for page in range(1, max_pages + 1):
        url = f"{BASE}/for-sale/property/{region}/?page_size=25&pn={page}"
        html = fetch_html_via_browser(url) if use_browser else session.get(url).text
        for rec in parse_search(html):
            if stats is not None:
                fps = rec.get("floorPlan") or rec.get("floorplans") or []
                imgs = rec.get("propertyImage") or rec.get("images") or []
                stats.observe(n_floorplans=len(fps), n_photos=len(imgs),
                              has_area=bool(rec.get("floorArea")))
            yield rec
