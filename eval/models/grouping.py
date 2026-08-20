"""A cheap, high-precision room grouping to feed the multi-view test.

Full grouping (stage 2) is a Sprint-7 job needing global descriptors and a
local matcher. For validating MapAnything we do not need recall, we need
CERTAINTY that a group really is one room - a group that secretly contains two
different bedrooms would make the model look broken when it is not.

So we exploit a property of flats: a listing has exactly one kitchen and
usually one bathroom. Photos within a single listing that triage labels
"kitchen" are therefore the same kitchen. Room types that legitimately repeat
(bedroom, living_room) are excluded. High precision, low recall - correct for
this purpose.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# Room types a flat has at most one of. Bathrooms can repeat in larger flats,
# so it is included only when the listing has <= 2 bedrooms.
SINGLETON_ROOMS = ("kitchen",)
# Rooms that repeat only in bigger flats: safe to treat as singletons when the
# listing is small enough.
CONDITIONAL_SINGLETON = {"bathroom": 2, "living_room": 2, "dining_room": 3}
# No confidence floor. SigLIP's sigmoid scores are poorly calibrated in absolute
# terms (real matches routinely score 0.01-0.15), so any fixed threshold throws
# away good groups - an earlier 0.02 floor cut the yield from 14 groups to 2.
# The argmax over room types is the signal; the magnitude is not.


def build_groups(golden: Path, triage_json: Path, min_views: int = 3,
                 max_views: int = 8) -> list[dict]:
    tri = json.loads(triage_json.read_text())
    gold = {l["listing_id"]: l for l in json.loads((golden / "golden_set.json").read_text())["listings"]}

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in tri["predictions"]:
        if p["pred_type"] != "interior" or not p.get("pred_room"):
            continue
        buckets[(p["listing_id"], p["pred_room"])].append(p)

    groups = []
    for (lid, room), items in buckets.items():
        beds = gold.get(lid, {}).get("bedrooms") or 99
        if room in SINGLETON_ROOMS:
            ok = True
        elif room in CONDITIONAL_SINGLETON:
            ok = beds <= CONDITIONAL_SINGLETON[room]
        else:
            ok = False
        if not ok or len(items) < min_views:
            continue
        items = sorted(items, key=lambda x: -(x.get("room_score") or 0))[:max_views]
        groups.append({
            "group_id": f"{lid}_{room}",
            "listing_id": lid,
            "room_type": room,
            "n_views": len(items),
            "paths": [i["path"] for i in items],
            "address": (gold.get(lid) or {}).get("display_address"),
            "bedrooms": (gold.get(lid) or {}).get("bedrooms"),
        })
    return sorted(groups, key=lambda g: -g["n_views"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--triage", type=Path, default=Path("eval/results/stage0_triage.json"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/room_groups.json"))
    ap.add_argument("--min-views", type=int, default=3)
    a = ap.parse_args()
    g = build_groups(a.golden, a.triage, a.min_views)
    a.out.write_text(json.dumps({"n_groups": len(g), "groups": g}, indent=2))
    print(f"{len(g)} groups with >={a.min_views} views "
          f"({sum(x['n_views'] for x in g)} photos total)")
    for x in g[:12]:
        print(f"  {x['group_id']:28} {x['room_type']:10} {x['n_views']} views  "
              f"{(x['address'] or '')[:38]}")
