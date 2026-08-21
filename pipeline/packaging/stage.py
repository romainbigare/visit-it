"""Stage 9 — package the scene the viewer loads.

`scene.json` is the viewer's *only* input, so it carries everything and nothing
more: the shell, the rooms with their provenance, the waypoint graph, the minimap
footprint, and the QA state.

Two rules from ARCHITECTURE are enforced here rather than trusted to the front end:

* **The viewer never displays a surface area other than the advertised figure**
  (§5). The model is scaled *to* the advertised area; we do not publish a
  competing measurement. The per-room areas we do emit are marked as such.
* **Waypoints must be inside their rooms.** A waypoint in a wall is a bug you only
  discover in the viewer, so the position comes from a representative point, not a
  centroid — an L-shaped room's centroid is outside it.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

import numpy as np

from ..core import geom
from ..core.stages import StageContext, StageResult, register_stage
from ..shell.mesh import PROVENANCE_RGBA

log = logging.getLogger("packaging")

EYE_HEIGHT_M = 1.55

PROVENANCE_LEGEND = {
    "photographed": "Surfaces visible in a listing photograph.",
    "reconstructed": "Shape estimated from the photographs; position taken from the floor plan.",
    "inferred": "Neither photographed nor reconstructed — filled in so the room closes.",
    "generated": "Invented content. Not produced in Phase 1.",
}

DISPLAY_NAMES = {
    "living_room": "Living room", "bedroom": "Bedroom", "kitchen": "Kitchen",
    "bathroom": "Bathroom", "dining_room": "Dining room", "hallway": "Hallway",
    "reception": "Reception", "study": "Study", "utility": "Utility",
    "wc": "WC", "storage": "Storage", "balcony": "Balcony", "garden": "Garden",
    "other_room": "Room", "unknown": "Room",
}


def _facing(position, towards) -> float:
    return round(math.degrees(math.atan2(towards[1] - position[1],
                                         towards[0] - position[0])), 2)


def build_scene(shell: dict, plan: dict, assembly: dict, scale: dict,
                manifest: dict | None, listing: dict, profile: str = "standard") -> dict:
    t0 = time.perf_counter()
    rooms_out, waypoints, edges = [], [], []
    by_room_wp: dict[str, str] = {}
    photo_by_room: dict[str, list[str]] = {}
    if manifest:
        for im in manifest["images"]:
            if im.get("room_label"):
                photo_by_room.setdefault(im["room_label"], []).append(im["image_id"])

    for r in shell["rooms"]:
        centre = geom.representative_point(r["polygon_m"])
        rooms_out.append({
            "room_id": r["room_id"],
            "label": r.get("label"),
            "display_name": DISPLAY_NAMES.get(r.get("label") or "unknown", "Room"),
            "polygon_m": [[round(x, 3), round(y, 3)] for x, y in r["polygon_m"]],
            "centroid_m": [round(centre[0], 3), round(centre[1], 3)],
            "height_m": r["height_m"],
            "area_m2": r["area_m2"],
            "provenance": r["provenance"],
            "confidence": r["confidence"],
            "photo_ids": photo_by_room.get(r.get("label") or "", []),
            "splats": None,                     # Phase 2 fills this
        })
        wid = f"w_{r['room_id']}"
        by_room_wp[r["room_id"]] = wid
        waypoints.append({
            "waypoint_id": wid, "room_id": r["room_id"],
            "position_m": [round(centre[0], 3), round(centre[1], 3), EYE_HEIGHT_M],
            "look_deg": 0.0, "kind": "room_centre",
            "label": DISPLAY_NAMES.get(r.get("label") or "unknown", "Room"),
        })

    # Doorway waypoints, so a walkthrough goes *through* doors rather than teleporting
    # between room centres through walls.
    plan_to_room = {m["plan_room_id"]: m["room_id"] for m in assembly.get("matches", [])}
    for ap in plan.get("apertures", []):
        if ap["type"] not in ("door", "opening") or not ap.get("position_m"):
            continue
        linked = [plan_to_room[p] for p in ap.get("rooms", []) if p in plan_to_room]
        if len(linked) != 2:
            continue
        wid = f"w_{ap['aperture_id']}"
        pos = ap["position_m"]
        waypoints.append({
            "waypoint_id": wid, "room_id": linked[0],
            "position_m": [round(pos[0], 3), round(pos[1], 3), EYE_HEIGHT_M],
            "look_deg": 0.0, "kind": "doorway", "label": "Doorway",
        })
        for rid in linked:
            other = by_room_wp.get(rid)
            if other:
                edges.append([wid, other])

    # Any rooms the doorway graph left unreachable get joined to their nearest
    # neighbour. A disconnected walkthrough is worse than an approximate one.
    _connect_islands(waypoints, edges)
    # Which way you face when you arrive decides whether a room reads as a room.
    # Facing a neighbouring waypoint puts you nose-first into the nearest wall; the
    # room's farthest corner shows two walls, the floor and the ceiling at once,
    # which is the most of the room a single view can carry.
    poly_by_room = {r["room_id"]: r["polygon_m"] for r in rooms_out}
    for wp in waypoints:
        if wp["kind"] != "room_centre":
            continue
        poly = poly_by_room.get(wp["room_id"] or "")
        if poly:
            far = max(poly, key=lambda v: math.dist(v, wp["position_m"][:2]))
            wp["look_deg"] = _facing(wp["position_m"], far)
        else:
            nbrs = [w for w in waypoints
                    if any(set(e) == {wp["waypoint_id"], w["waypoint_id"]} for e in edges)]
            if nbrs:
                wp["look_deg"] = _facing(wp["position_m"], nbrs[0]["position_m"])

    polys = [r["polygon_m"] for r in rooms_out]
    if polys:
        xs = [p[0] for poly in polys for p in poly]
        ys = [p[1] for poly in polys for p in poly]
        bounds = [[round(min(xs), 3), round(min(ys), 3)],
                  [round(max(xs), 3), round(max(ys), 3)]]
    else:
        bounds = [[0.0, 0.0], [0.0, 0.0]]

    qa = sorted(set(shell.get("qa_flags", []) + scale.get("qa_flags", []) +
                    assembly.get("qa_flags", []) + plan.get("qa_flags", [])))
    conf = float(np.mean([shell.get("confidence", 0), scale.get("quality", 0),
                          assembly.get("confidence", 0), plan.get("confidence", 0)]))

    return {
        "schema": "scene/v1",
        "listing_id": shell["listing_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tier": "A" if assembly.get("method") == "plan" else "B",
        "units": "metres",
        "profile": profile,
        # ARCHITECTURE §5: the advertised figure, never our own measurement.
        "advertised_area_m2": (manifest or {}).get("listing", {}).get("advertised_area_m2"),
        "address": listing.get("display_address"),
        "shell": {"uri": shell["glb"]["uri"], "bytes": shell["glb"]["bytes"],
                  "triangles": shell["glb"]["triangles"]},
        "rooms": rooms_out,
        "waypoints": waypoints,
        "waypoint_edges": [sorted(e) for e in {tuple(sorted(e)) for e in edges}],
        "minimap": {"bounds_m": bounds, "footprint_m": shell.get("footprint_m")},
        "provenance_legend": PROVENANCE_LEGEND,
        "provenance_colours": {k: list(v) for k, v in PROVENANCE_RGBA.items()},
        "confidence": round(conf, 3),
        "qa_flags": qa,
        "timing_s": round(time.perf_counter() - t0, 3),
    }


def _connect_islands(waypoints: list[dict], edges: list[list[str]]) -> None:
    centres = [w for w in waypoints if w["kind"] == "room_centre"]
    if len(centres) < 2:
        return
    adj: dict[str, set[str]] = {w["waypoint_id"]: set() for w in waypoints}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen: set[str] = set()
    components: list[list[dict]] = []
    for w in centres:
        if w["waypoint_id"] in seen:
            continue
        stack, comp = [w["waypoint_id"]], []
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            node = next((x for x in waypoints if x["waypoint_id"] == n), None)
            if node and node["kind"] == "room_centre":
                comp.append(node)
            stack += [m for m in adj.get(n, ()) if m not in seen]
        if comp:
            components.append(comp)
    while len(components) > 1:
        a = components.pop()
        best = None
        for comp in components:
            for wa in a:
                for wb in comp:
                    d = math.dist(wa["position_m"][:2], wb["position_m"][:2])
                    if best is None or d < best[0]:
                        best = (d, wa["waypoint_id"], wb["waypoint_id"], comp)
        if best is None:
            break
        edges.append([best[1], best[2]])
        best[3].extend(a)


@register_stage("9-package", description="Assemble scene.json — the viewer's only input")
def run(ctx: StageContext) -> StageResult:
    scene = build_scene(ctx.require("8-shell"), ctx.require("5-plan"),
                        ctx.require("6-assembly"), ctx.require("7-scale"),
                        ctx.read("0-triage"), ctx.listing, ctx.profile.name)
    return StageResult(payload=scene, engine="package")
