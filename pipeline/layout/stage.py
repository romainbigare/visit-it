"""Stage 4 — room structure: the polygon, the planes, the apertures, the flags.

The Phase 0 report promoted this stage on the grounds that it "unblocks Phase 1
*and* fixes the dominant appearance artefact". Both are still true: stage 6 needs
polygons to assign, stage 7 needs ceiling heights to constrain scale, and Phase 2
needs the polygon as the culling volume that stops splats bleeding out of windows.

Everything here is in the room's own frame and in *unscaled* metres — the engine's
metric output, before stage 7's global scalar is applied. `scale_applied` says so
explicitly so nobody has to guess.
"""
from __future__ import annotations

import io
import logging
import math
import time

import numpy as np

from ..core import geom
from ..core.stages import StageContext, StageResult, register_stage
from . import apertures, planes

log = logging.getLogger("layout.stage")


def layout_from_points(points: np.ndarray, confidence: np.ndarray | None,
                       up_prior: np.ndarray | None, *, room_id: str,
                       room_label: str | None = None, engine: str = "unknown",
                       n_views: int = 1) -> dict:
    """One room's `layout.json`, from one room's point cloud."""
    t0 = time.perf_counter()
    qa: list[str] = []
    rot, up = planes.orient(points, up_prior)
    kept, kconf, clip = planes.clip_outliers(rot, confidence)
    if len(kept) < 200:
        kept, kconf = rot, confidence
        qa.append("clipping_removed_too_much")

    lo, hi = planes.floor_ceiling(kept[:, 2])
    height = (hi - lo) if (lo is not None and hi is not None) else None
    dirs = planes.wall_directions(kept[:, :2])
    poly = planes.footprint_polygon(kept[:, :2], dirs)
    poly = geom.ensure_ccw(poly)
    walls = planes.fit_walls(kept[:, :2], poly)
    aps = (apertures.detect(kept, poly, lo, hi)
           if (lo is not None and hi is not None) else [])
    area = geom.area(poly)
    long_m, short_m, orient_deg = geom.oriented_extent(poly)

    # ROADMAP §0b check 1, applied here rather than at the gate so a broken room
    # is visible in the contact sheet the day it happens.
    if height is None:
        qa.append("no_floor_ceiling_found")
    elif not (planes.CEILING_PLAUSIBLE_M[0] <= height <= planes.CEILING_PLAUSIBLE_M[1]):
        qa.append("ceiling_outside_2.3_3.2m")
        if not (planes.CEILING_POSSIBLE_M[0] <= height <= planes.CEILING_POSSIBLE_M[1]):
            qa.append("ceiling_implausible")
    if long_m > planes.MAX_ROOM_EXTENT_M:
        qa.append("room_over_12m_across")
    if area < 1.5:
        qa.append("room_area_under_1.5m2")
    weak = [w for w in walls if w["inlier_frac"] < 0.03]
    if len(weak) >= 2:
        # Two walls with almost nothing behind them means we closed the polygon
        # rather than observed it — the classic single-photo failure.
        qa.append("walls_largely_unobserved")
    if n_views < 2:
        qa.append("monocular_single_view")
    if not aps:
        qa.append("no_apertures_detected")

    approximate = n_views < 2 or "walls_largely_unobserved" in qa
    conf = _confidence(height, area, walls, n_views, qa)

    return {
        "schema": "layout/v1",
        "listing_id": "",                       # filled by the stage
        "room_id": room_id,
        "room_label": room_label,
        "engine": engine,
        "n_views": int(n_views),
        "up_axis": [round(float(v), 5) for v in up],
        "floor_h_m": round(float(lo), 4) if lo is not None else None,
        "ceiling_h_m": round(float(hi), 4) if hi is not None else None,
        "room_height_m": round(float(height), 4) if height is not None else None,
        "polygon_m": [[round(x, 4), round(y, 4)] for x, y in poly],
        "area_m2": round(area, 4),
        "extent_m": [round(long_m, 4), round(short_m, 4)],
        "orientation_deg": round(float(dirs[0]), 3),
        "walls": walls,
        "apertures": [a.to_dict(i) for i, a in enumerate(aps)],
        "regularisation": {
            "model": "atlanta",
            "n_directions": len(dirs),
            "directions_deg": [round(d, 3) for d in dirs],
            "snap_residual_deg": 0.0,
            "residual_m": round(float(np.median(np.abs(
                [w["inlier_frac"] for w in walls] or [0.0]))), 4),
        },
        "clipping": clip,
        "scale_applied": False,
        "approximate": bool(approximate),
        "confidence": conf,
        "qa_flags": sorted(set(qa)),
        "timing_s": round(time.perf_counter() - t0, 3),
    }


def _confidence(height, area, walls, n_views, qa) -> float:
    c = 0.25 + min(0.25, 0.08 * n_views)
    if height and planes.CEILING_PLAUSIBLE_M[0] <= height <= planes.CEILING_PLAUSIBLE_M[1]:
        c += 0.25
    if 4.0 <= area <= 60.0:
        c += 0.15
    if walls:
        c += 0.15 * min(1.0, float(np.mean([w["inlier_frac"] for w in walls])) / 0.15)
    for f in qa:
        if f in ("ceiling_implausible", "room_over_12m_across", "no_floor_ceiling_found",
                 "room_area_under_1.5m2", "clipping_removed_too_much"):
            c -= 0.2
        elif f in ("ceiling_outside_2.3_3.2m", "walls_largely_unobserved"):
            c -= 0.1
    return round(min(1.0, max(0.02, c)), 3)


@register_stage("4-layout", description="Floor/ceiling/wall planes and the room polygon")
def run(ctx: StageContext) -> StageResult:
    geo = ctx.require("3-geometry")
    layouts: list[dict] = []
    qa: list[str] = []
    for room in geo["rooms"]:
        raw = ctx.store.read_binary(room["geometry_uri"])
        if raw is None:
            qa.append("geometry_blob_missing")
            continue
        with np.load(io.BytesIO(raw)) as z:
            pts = z["points"].astype(float)
            conf = z["confidence"].astype(float) if "confidence" in z else None
            up_prior = z["up_prior"].astype(float) if "up_prior" in z else None
        if up_prior is not None and not np.any(up_prior):
            up_prior = None
        lay = layout_from_points(pts, conf, up_prior, room_id=room["room_id"],
                                 room_label=room.get("room_label"),
                                 engine=room.get("engine", geo.get("engine", "unknown")),
                                 n_views=room.get("n_views", 1))
        lay["listing_id"] = ctx.listing_id
        layouts.append(lay)

    heights = [l["room_height_m"] for l in layouts if l["room_height_m"]]
    plausible = [h for h in heights
                 if planes.CEILING_PLAUSIBLE_M[0] <= h <= planes.CEILING_PLAUSIBLE_M[1]]
    frac = len(plausible) / len(heights) if heights else None
    if frac is not None and frac < 0.8:
        # This is the G1 plausibility criterion, evaluated per listing so it can
        # route to review instead of only surfacing at the gate.
        qa.append("under_80pct_rooms_plausible_ceiling")

    payload = {
        "schema": "layouts/v1",
        "listing_id": ctx.listing_id,
        "engine": geo.get("engine"),
        "rooms": layouts,
        "summary": {
            "n_rooms": len(layouts),
            "median_ceiling_m": round(float(np.median(heights)), 3) if heights else None,
            "ceiling_plausible_frac": round(frac, 3) if frac is not None else None,
            "total_area_m2_unscaled": round(sum(l["area_m2"] for l in layouts), 3),
            "max_extent_m": (round(max(l["extent_m"][0] for l in layouts), 3)
                             if layouts else None),
        },
        "confidence": (round(float(np.mean([l["confidence"] for l in layouts])), 3)
                       if layouts else 0.0),
        "qa_flags": sorted(set(qa) | {f for l in layouts for f in l["qa_flags"]}),
    }
    return StageResult(payload=payload, engine=geo.get("engine"))
