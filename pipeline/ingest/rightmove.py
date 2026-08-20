"""Rightmove adapter.

Rightmove serves two different embedded payloads:
  * search results  -> <script id="__NEXT_DATA__">  (plain JSON)
  * property detail -> window.__PAGE_MODEL          (devalue-style flat array)

Both are parsed here so the caller only ever sees `Listing` objects.
robots.txt (checked 2026-08-20) permits /property-for-sale/find.html and
/properties/<id> for a generic user-agent; named crawlers such as GPTBot and
CCbot are disallowed outright and this adapter must not impersonate them.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterator

from .http import PoliteSession
from .models import CoverageStats, Listing, MediaRef

log = logging.getLogger(__name__)

BASE = "https://www.rightmove.co.uk"
SEARCH = BASE + "/property-for-sale/find.html"

# A few useful region identifiers. Rightmove's own codes; stable in practice.
REGIONS = {
    "london": "REGION^87490",
    "manchester": "REGION^904",
    "birmingham": "REGION^162",
    "bristol": "REGION^219",
    "leeds": "REGION^787",
    "glasgow": "REGION^550",
    "edinburgh": "REGION^475",
    "liverpool": "REGION^814",
}

_SPECIAL = {-1: None, -2: None, -3: float("nan"), -4: float("inf"), -5: float("-inf"), -6: -0.0}


# ---------------------------------------------------------------- payloads
def _extract_js_object(html: str, marker: str) -> dict | list | None:
    """Pull the first balanced JSON object/array following `marker`."""
    i = html.find(marker)
    if i < 0:
        return None
    start = None
    for j in range(i + len(marker), min(i + len(marker) + 200, len(html))):
        if html[j] in "{[":
            start = j
            break
    if start is None:
        return None
    opening = html[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    instr = False
    esc = False
    for j in range(start, len(html)):
        c = html[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        else:
            if c == '"':
                instr = True
            elif c == opening:
                depth += 1
            elif c == closing:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(html[start:j + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _unflatten(arr: list) -> Any:
    """Resolve a devalue-style flat array where members are indices into `arr`."""
    memo: dict[int, Any] = {}

    def rec(idx):
        if isinstance(idx, bool) or not isinstance(idx, int):
            return idx
        if idx < 0:
            return _SPECIAL.get(idx)
        if idx in memo:
            return memo[idx]
        if idx >= len(arr):
            return None
        v = arr[idx]
        if isinstance(v, dict):
            out: dict = {}
            memo[idx] = out
            for k, ref in v.items():
                out[k] = rec(ref)
            return out
        if isinstance(v, list):
            lst: list = []
            memo[idx] = lst
            for ref in v:
                lst.append(rec(ref))
            return lst
        memo[idx] = v
        return v

    return rec(0)


def parse_search(html: str) -> dict:
    data = _extract_js_object(html, '<script id="__NEXT_DATA__" type="application/json">')
    if data is None:
        data = _extract_js_object(html, '<script id="__NEXT_DATA__"')
    if not isinstance(data, dict):
        raise ValueError("Rightmove search payload not found (page layout may have changed)")
    return data["props"]["pageProps"]["searchResults"]


def parse_property(html: str) -> dict:
    pm = _extract_js_object(html, "window.__PAGE_MODEL")
    if isinstance(pm, dict) and "data" in pm:
        data = pm["data"]
        if isinstance(data, str):
            data = json.loads(data)
        obj = _unflatten(data) if isinstance(data, list) else data
    else:
        obj = pm
    if not isinstance(obj, dict) or "propertyData" not in obj:
        raise ValueError("Rightmove property payload not found (page layout may have changed)")
    return obj["propertyData"]


# ---------------------------------------------------------------- mapping
_AREA_RX = re.compile(r"([\d,]+(?:\.\d+)?)\s*(sq\.?\s*(?:ft|feet)|sqft|ft2|sq\.?\s*m|sqm|m2|m²)", re.I)


def _area_sqm(pd: dict) -> tuple[float | None, str | None]:
    """Rightmove exposes `sizings`; it is usually empty for UK residential.

    Falls back to parsing the description, which is where agents often put it.
    """
    for s in pd.get("sizings") or []:
        unit = (s.get("unit") or "").lower()
        val = s.get("minimumSize") or s.get("maximumSize")
        if not val:
            continue
        if "sqft" in unit or "sq ft" in unit or unit == "ft2":
            return round(float(val) * 0.09290304, 2), "sizings:sqft"
        if "sqm" in unit or "sq m" in unit or unit == "m2":
            return round(float(val), 2), "sizings:sqm"
    text = (pd.get("text") or {}).get("description") or ""
    m = _AREA_RX.search(re.sub(r"<[^>]+>", " ", text))
    if m:
        val = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower().replace(" ", "").replace(".", "")
        if unit in ("sqft", "sqfeet", "ft2"):
            return round(val * 0.09290304, 2), "description:sqft"
        return round(val, 2), "description:sqm"
    return None, None


def to_listing(pd: dict) -> Listing:
    addr = pd.get("address") or {}
    prices = pd.get("prices") or {}
    price = prices.get("primaryPrice") or ""
    price_int = int(re.sub(r"[^\d]", "", price)) if re.search(r"\d", str(price)) else None
    area, area_src = _area_sqm(pd)
    text = pd.get("text") or {}
    cust = pd.get("customer") or {}
    return Listing(
        listing_id=str(pd.get("id")),
        source="rightmove",
        url=f"{BASE}/properties/{pd.get('id')}",
        display_address=addr.get("displayAddress"),
        outcode=addr.get("outcode"),
        country=addr.get("ukCountry"),
        property_subtype=pd.get("propertySubType"),
        transaction_type=pd.get("transactionType"),
        tenure=((pd.get("tenure") or {}).get("tenureType")),
        bedrooms=pd.get("bedrooms"),
        bathrooms=pd.get("bathrooms"),
        price_gbp=price_int,
        floor_area_sqm=area,
        floor_area_source=area_src,
        photos=[MediaRef(url=i["url"], kind="photo", caption=i.get("caption"))
                for i in (pd.get("images") or []) if i.get("url")],
        floorplans=[MediaRef(url=f["url"], kind="floorplan", caption=f.get("caption"))
                    for f in (pd.get("floorplans") or []) if f.get("url")],
        description=re.sub(r"<[^>]+>", " ", text.get("description") or "").strip() or None,
        key_features=pd.get("keyFeatures") or [],
        agent=(cust.get("branchDisplayName") or cust.get("companyName")),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------- crawling
def iter_search_results(session: PoliteSession, region: str = "london", *,
                        max_pages: int = 8, stats: CoverageStats | None = None,
                        property_types: str = "flat", max_price: int | None = None,
                        min_price: int | None = None,
                        min_bedrooms: int | None = None) -> Iterator[dict]:
    """Yield raw search-result records, recording coverage on everything seen."""
    loc = REGIONS.get(region.lower(), region)
    for page in range(max_pages):
        params = [f"locationIdentifier={loc.replace('^', '%5E')}", f"index={page * 24}"]
        if property_types:
            params.append(f"propertyTypes={property_types}")
        if max_price:
            params.append(f"maxPrice={max_price}")
        if min_price:
            params.append(f"minPrice={min_price}")
        if min_bedrooms:
            params.append(f"minBedrooms={min_bedrooms}")
        url = f"{SEARCH}?{'&'.join(params)}"
        log.info("search page %d: %s", page + 1, url)
        html = session.get(url).text
        sr = parse_search(html)
        props = sr.get("properties") or []
        if not props:
            return
        for p in props:
            if stats is not None:
                stats.observe(
                    n_floorplans=p.get("numberOfFloorplans") or 0,
                    n_photos=p.get("numberOfImages") or 0,
                    has_area=bool(p.get("displaySize")),
                )
            yield p
        pag = sr.get("pagination") or {}
        if pag.get("next") in (None, "", pag.get("last")):
            return


def fetch_listing(session: PoliteSession, property_id: str | int) -> Listing:
    html = session.get(f"{BASE}/properties/{property_id}").text
    return to_listing(parse_property(html))
