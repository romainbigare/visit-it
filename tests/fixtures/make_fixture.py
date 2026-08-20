"""Generate a synthetic Rightmove-shaped detail page for tests.

Deliberately synthetic: committing a real scraped page would put a
photographer's copyrighted description and imagery into the repo, which
docs/DATA-SOURCES.md §3.7 says is not ours to redistribute. The *structure*
(devalue-flattened window.__PAGE_MODEL) is what the parser cares about, and
that is reproduced faithfully here.

    python tests/fixtures/make_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path


def flatten(obj) -> list:
    """Encode `obj` the way Rightmove's devalue-style payload does: one flat
    array where every object/array member is an index into that array."""
    out: list = []
    memo: dict[int, int] = {}

    def add(v) -> int:
        if isinstance(v, (str, int, float, bool)) or v is None:
            out.append(v)
            return len(out) - 1
        if id(v) in memo:
            return memo[id(v)]
        idx = len(out)
        out.append(None)
        memo[id(v)] = idx
        if isinstance(v, dict):
            out[idx] = {k: add(val) for k, val in v.items()}
        elif isinstance(v, list):
            out[idx] = [add(x) for x in v]
        return idx

    add(obj)
    return out


PROPERTY = {
    "propertyData": {
        "id": 100000001,
        "bedrooms": 2,
        "bathrooms": 1,
        "propertySubType": "Flat",
        "transactionType": "BUY",
        "tenure": {"tenureType": "LEASEHOLD"},
        "address": {
            "displayAddress": "12 Example Road, Testville, EX1",
            "outcode": "EX1", "incode": "2AB", "ukCountry": "England",
            "countryCode": "GB",
        },
        "prices": {"primaryPrice": "£325,000"},
        "sizings": [{"unit": "sqm", "minimumSize": 68, "maximumSize": 68}],
        "keyFeatures": ["Two bedrooms", "Allocated parking"],
        "text": {"description": "<p>A synthetic listing used for tests. 732 sq ft.</p>"},
        "customer": {"branchDisplayName": "Example Estates, Testville"},
        "images": [
            {"url": f"https://media.example.invalid/photo/{i}.jpeg",
             "caption": f"Picture No. {i}"} for i in range(1, 13)
        ],
        "floorplans": [
            {"url": "https://media.example.invalid/floorplan/1.jpeg", "caption": "Floorplan",
             "type": "IMAGE"}
        ],
    }
}

HTML = """<!DOCTYPE html><html lang="en"><head><title>Synthetic fixture</title></head>
<body><div id="root"></div>
<script>window.rightmoveDataLayer = {{}};</script>
<script>window.__PAGE_MODEL = {payload};</script>
</body></html>
"""

if __name__ == "__main__":
    payload = {"data": flatten(PROPERTY), "encoding": "on"}
    dest = Path(__file__).parent / "rightmove_property.html"
    dest.write_text(HTML.format(payload=json.dumps(payload)), encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
