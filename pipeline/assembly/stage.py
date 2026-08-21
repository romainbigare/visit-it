"""Stage 6 — assembly: reconstructed rooms onto plan polygons.

This is the join between the photo channel (stream B) and the plan channel
(stream C), and the stage the roadmap front-loads because it is the one most
likely to kill the project. It emits its full cost matrix as well as its answer,
so the review console's assignment-nudge UI has something to work with and so a
bad arrangement can be debugged rather than merely observed.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..core import geom
from ..core.stages import StageContext, StageResult, register_stage
from . import pose as pose_mod
from .matching import assign, normalise_areas

log = logging.getLogger("assembly.stage")

#: Below this, the arrangement is a coin flip and a human should look.
LOW_MARGIN = 0.08


def _room_features(layouts: dict) -> list[dict]:
    out = []
    for r in layouts["rooms"]:
        poly = r["polygon_m"]
        out.append({
            "room_id": r["room_id"],
            "label": r.get("room_label"),
            "area_m2": r["area_m2"],
            "aspect": geom.aspect_ratio(poly),
            "polygon_m": poly,
            "confidence": r.get("confidence", 0.5),
            "apertures": r.get("apertures", []),
            "height_m": r.get("room_height_m"),
        })
    return out


def _plan_features(plan: dict) -> list[dict]:
    """Plan rooms as match candidates.

    Falls back to the *pixel* polygon when the plan prints no dimensions. Shape and
    proportion are what the matcher uses, and both survive the change of units —
    which is what lets a perfectly-vectorised plan with no printed sizes still be
    matched, instead of being thrown away for want of a number.
    """
    out = []
    metric = plan.get("px_per_metre") is not None
    for p in plan["rooms"]:
        # Prefer metres, fall back to pixels, and never drop a room just because one
        # of the two representations is missing.
        poly = (p.get("polygon_m") or p.get("polygon_px")) if metric \
            else (p.get("polygon_px") or p.get("polygon_m"))
        if not poly or len(poly) < 3:
            continue
        out.append({
            "room_id": p["room_id"],
            "label": p.get("label"),
            "area_m2": p.get("area_m2") if metric else geom.area(poly),
            "aspect": geom.aspect_ratio(poly),
            "polygon_m": poly,
            "aperture_ids": p.get("aperture_ids", []),
            "confidence": p.get("confidence", 0.5),
        })
    return out


def assemble(layouts: dict, plan: dict, override: dict | None = None) -> dict:
    t0 = time.perf_counter()
    rooms = _room_features(layouts)
    plan_rooms = _plan_features(plan)
    qa: list[str] = []
    if not plan_rooms:
        qa.append("plan_has_no_rooms")
    scale_free = plan.get("px_per_metre") is None
    if scale_free:
        qa.append("matched_without_a_plan_scale")
        normalise_areas(rooms, plan_rooms)

    # A reviewer's pin is not a hint — it is the answer. Pinned rooms and their
    # polygons come out of the assignment problem entirely, so the solver spends
    # its freedom on what is actually still in doubt.
    ov = override or {}
    pins: dict[str, str] = dict(ov.get("pin") or {})
    rejected = set(ov.get("reject") or [])
    free_rooms = [r for r in rooms if r["room_id"] not in pins and r["room_id"] not in rejected]
    taken = set(pins.values())
    free_plans = [p for p in plan_rooms if p["room_id"] not in taken]

    matches, unmatched_rooms, unmatched_plans, cost = assign(free_rooms, free_plans)
    if pins:
        qa.append("override_applied")
        from .matching import Match, build_cost_matrix
        room_by_id = {r["room_id"]: r for r in rooms}
        plan_by_id_all = {p["room_id"]: p for p in plan_rooms}
        for rid, pid in pins.items():
            r, p = room_by_id.get(rid), plan_by_id_all.get(pid)
            if not r or not p:
                continue
            c, parts = build_cost_matrix([r], [p])
            matches.append(Match(room_id=rid, plan_room_id=pid, cost=round(float(c[0][0]), 4),
                                 breakdown=parts[0][0], margin=None, confidence=1.0))
        unmatched_rooms = [r for r in unmatched_rooms if r not in pins]
        unmatched_plans = [p for p in unmatched_plans if p not in taken]
    # A pin wins over a reject. Both come from a hand-edited override file, so a
    # room can appear in each; without this it comes out matched *and* unmatched,
    # and the downstream shell builder and scoreboard disagree about the listing.
    unmatched_rooms += sorted(rejected - set(pins))
    by_id = {r["room_id"]: r for r in rooms}
    plan_by_id = {p["room_id"]: p for p in plan_rooms}

    ious_before, ious_after, out = [], [], []
    for m in matches:
        rp = by_id[m.room_id]["polygon_m"]
        pp = plan_by_id[m.plan_room_id]["polygon_m"]
        if scale_free:
            # Comparing a metre polygon against a pixel one has no meaning; the pose
            # is not needed either, since the shell is extruded from the plan.
            before, p, iou = 0.0, {"tx_m": 0.0, "ty_m": 0.0, "theta_deg": 0.0,
                                   "flip": False}, None
        else:
            before = geom.iou(geom.place_at(rp, geom.centroid(pp), 0.0), pp)
            p, iou = pose_mod.refine(rp, pp)
        ious_before.append(before)
        ious_after.append(iou)
        out.append({
            "room_id": m.room_id, "plan_room_id": m.plan_room_id, "cost": m.cost,
            "cost_breakdown": m.breakdown, "margin": m.margin, "pose_se2": p,
            "fit_iou": iou, "confidence": m.confidence,
            "edited_by_human": m.room_id in pins,
            "alternatives": [{"plan_room_id": a, "cost": c} for a, c in m.alternatives],
        })

    low = [m for m in out if (m["margin"] is not None and m["margin"] < LOW_MARGIN)]
    if low:
        qa.append("low_assignment_margin")
    if unmatched_rooms:
        qa.append("unmatched_reconstructed_rooms")
    if unmatched_plans:
        qa.append("unmatched_plan_polygons")
    ious_after = [i for i in ious_after if i is not None]
    ious_before = [i for i in ious_before if i is not None]
    if ious_after and float(np.median(ious_after)) < 0.4:
        qa.append("poor_polygon_fit")

    conf = (round(float(np.mean([m["confidence"] for m in out])), 3) if out else 0.0)
    if unmatched_plans and plan_rooms:
        conf *= max(0.4, 1.0 - len(unmatched_plans) / len(plan_rooms))

    return {
        "schema": "assembly/v1",
        "listing_id": layouts["listing_id"],
        "method": "plan",
        "matches": out,
        "unmatched_rooms": unmatched_rooms,
        "unmatched_plan_rooms": unmatched_plans,
        "cost_matrix": {
            "rooms": [r["room_id"] for r in rooms],
            "plan_rooms": [p["room_id"] for p in plan_rooms],
            "cost": [[round(float(v), 4) for v in row] for row in cost],
        },
        "refinement": {
            "iterations": 2,
            "mean_iou_before": round(float(np.mean(ious_before)), 4) if ious_before else 0.0,
            "mean_iou_after": round(float(np.mean(ious_after)), 4) if ious_after else 0.0,
        },
        "confidence": round(conf, 3),
        "qa_flags": sorted(set(qa)),
        "timing_s": round(time.perf_counter() - t0, 3),
    }


@register_stage("6-assembly", description="Assign reconstructed rooms to plan polygons")
def run(ctx: StageContext) -> StageResult:
    layouts = ctx.require("4-layout")
    plan = ctx.require("5-plan")
    ov = (ctx.options.get("overrides") or {}).get("assembly")
    return StageResult(payload=assemble(layouts, plan, ov), engine="hungarian_se2")
