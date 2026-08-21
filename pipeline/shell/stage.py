"""Stage 8 (Phase 1 form) — build the shell.

Phase 2 turns stage 8 into the appearance stage; in Phase 1 it is the shell
builder, which is the whole G1 deliverable: a scaled, correctly-arranged,
untextured 3D shell you can walk around in a browser.

Each matched room is scaled by stage 7's scalar, posed by stage 6's SE(2), and
extruded to its own measured ceiling height. Rooms with no match are dropped from
the mesh and listed, because putting a room somewhere we do not believe it goes is
exactly the failure mode AD-15 exists to prevent.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..assembly.pose import apply_pose
from ..core import geom
from ..core.stages import StageContext, StageResult, register_stage
from .mesh import Mesh, build_room, to_glb

log = logging.getLogger("shell.stage")

SHELL_BYTES_BUDGET = 1_048_576          # ARCHITECTURE §6, CI-enforced
DEFAULT_CEILING_M = 2.5


def build(layouts: dict, plan: dict, assembly: dict, scale: dict) -> tuple[dict, bytes]:
    t0 = time.perf_counter()
    s = float(scale.get("scale", 1.0))
    lay_by_id = {r["room_id"]: r for r in layouts["rooms"]}
    plan_by_id = {p["room_id"]: p for p in plan["rooms"]}
    plan_aps = {a["aperture_id"]: a for a in plan.get("apertures", [])}

    mesh = Mesh()
    rooms_out, qa = [], []
    heights = [r["room_height_m"] * s for r in layouts["rooms"] if r.get("room_height_m")]
    fallback_h = float(np.median(heights)) if heights else DEFAULT_CEILING_M
    if not (2.0 <= fallback_h <= 4.0):
        fallback_h = DEFAULT_CEILING_M
        qa.append("ceiling_fallback_used")

    for m in assembly["matches"]:
        lay = lay_by_id.get(m["room_id"])
        pl = plan_by_id.get(m["plan_room_id"])
        if not lay or not pl or not pl.get("polygon_m"):
            continue
        # The room's *shape* comes from the reconstruction, scaled and posed; its
        # *place* comes from the plan. That split is AD-2 in one line of code.
        poly = geom.scale_polygon(lay["polygon_m"], s, about=geom.centroid(lay["polygon_m"]))
        poly = apply_pose(poly, m["pose_se2"])
        poly = geom.simplify(poly, tol=0.03)
        h = (lay.get("room_height_m") or 0) * s
        if not (2.0 <= h <= 4.2):
            h = fallback_h
        aps = [plan_aps[i] for i in pl.get("aperture_ids", []) if i in plan_aps]
        # AD-15 provenance. A monocular room is still *reconstructed* — its shape
        # came from a photograph. "inferred" is reserved for geometry we did not
        # observe at all: a room whose walls have almost no points behind them, or
        # one with no usable floor/ceiling. Calling honest monocular geometry
        # "inferred" would desaturate the whole flat and make the distinction
        # useless exactly where it matters.
        unobserved = {"walls_largely_unobserved", "no_floor_ceiling_found",
                      "clipping_removed_too_much"}
        provenance = "inferred" if unobserved & set(lay.get("qa_flags", [])) \
            else "reconstructed"
        build_room(mesh, poly, 0.0, h, m["room_id"], provenance, aps)
        rooms_out.append({
            "room_id": m["room_id"], "plan_room_id": m["plan_room_id"],
            "label": pl.get("label") or lay.get("room_label"),
            "polygon_m": [[round(x, 3), round(y, 3)] for x, y in poly],
            "centroid_m": [round(v, 3) for v in geom.representative_point(poly)],
            "height_m": round(h, 3), "area_m2": round(geom.area(poly), 3),
            "provenance": provenance,
            "confidence": round(min(m.get("confidence", 0.5), lay.get("confidence", 0.5)), 3),
            "aperture_ids": pl.get("aperture_ids", []),
        })

    glb = to_glb(mesh)
    if len(glb) > SHELL_BYTES_BUDGET:
        qa.append("shell_over_1mb_budget")
    if assembly.get("unmatched_rooms"):
        qa.append("rooms_omitted_from_shell")
    if not rooms_out:
        qa.append("empty_shell")

    footprint = geom.union_polygon([r["polygon_m"] for r in rooms_out]) if rooms_out else None
    payload = {
        "schema": "shell/v1",
        "listing_id": layouts["listing_id"],
        "scale_applied": round(s, 6),
        "rooms": rooms_out,
        "omitted_rooms": assembly.get("unmatched_rooms", []),
        "glb": {"uri": "shell.glb", "bytes": len(glb),
                "triangles": mesh.n_triangles,
                "budget_bytes": SHELL_BYTES_BUDGET,
                "within_budget": len(glb) <= SHELL_BYTES_BUDGET},
        "footprint_m": footprint,
        "total_area_m2": round(sum(r["area_m2"] for r in rooms_out), 3),
        "confidence": (round(float(np.mean([r["confidence"] for r in rooms_out])), 3)
                       if rooms_out else 0.0),
        "qa_flags": sorted(set(qa)),
        "timing_s": round(time.perf_counter() - t0, 3),
    }
    return payload, glb


@register_stage("8-shell", description="Extrude the assembled, scaled rooms into a glTF shell")
def run(ctx: StageContext) -> StageResult:
    payload, glb = build(ctx.require("4-layout"), ctx.require("5-plan"),
                         ctx.require("6-assembly"), ctx.require("7-scale"))
    return StageResult(payload=payload, binaries={"shell.glb": glb}, engine="extrude")
