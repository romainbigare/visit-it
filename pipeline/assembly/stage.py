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
from .matching import assign

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
    out = []
    for p in plan["rooms"]:
        if not p.get("polygon_m"):
            continue
        out.append({
            "room_id": p["room_id"],
            "label": p.get("label"),
            "area_m2": p.get("area_m2"),
            "aspect": geom.aspect_ratio(p["polygon_m"]),
            "polygon_m": p["polygon_m"],
            "aperture_ids": p.get("aperture_ids", []),
            "confidence": p.get("confidence", 0.5),
        })
    return out


def assemble(layouts: dict, plan: dict) -> dict:
    t0 = time.perf_counter()
    rooms = _room_features(layouts)
    plan_rooms = _plan_features(plan)
    qa: list[str] = []
    if not plan_rooms:
        qa.append("plan_has_no_metric_polygons")

    matches, unmatched_rooms, unmatched_plans, cost = assign(rooms, plan_rooms)
    by_id = {r["room_id"]: r for r in rooms}
    plan_by_id = {p["room_id"]: p for p in plan_rooms}

    ious_before, ious_after, out = [], [], []
    for m in matches:
        rp = by_id[m.room_id]["polygon_m"]
        pp = plan_by_id[m.plan_room_id]["polygon_m"]
        before = geom.iou(geom.place_at(rp, geom.centroid(pp), 0.0), pp)
        p, iou = pose_mod.refine(rp, pp)
        ious_before.append(before)
        ious_after.append(iou)
        out.append({
            "room_id": m.room_id, "plan_room_id": m.plan_room_id, "cost": m.cost,
            "cost_breakdown": m.breakdown, "margin": m.margin, "pose_se2": p,
            "fit_iou": iou, "confidence": m.confidence, "edited_by_human": False,
            "alternatives": [{"plan_room_id": a, "cost": c} for a, c in m.alternatives],
        })

    low = [m for m in out if (m["margin"] is not None and m["margin"] < LOW_MARGIN)]
    if low:
        qa.append("low_assignment_margin")
    if unmatched_rooms:
        qa.append("unmatched_reconstructed_rooms")
    if unmatched_plans:
        qa.append("unmatched_plan_polygons")
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
    if plan.get("px_per_metre") is None:
        return StageResult(payload={}, skipped=True,
                           skip_reason="plan has no metric scale, so its polygons "
                                       "cannot be matched against metric rooms")
    return StageResult(payload=assemble(layouts, plan), engine="hungarian_se2")
