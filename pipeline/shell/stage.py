"""Stage 8 (Phase 1 form) — build the shell.

**The plan supplies the room's outline. The photographs supply its ceiling height.**

That split is AD-2's amendment of 21 Aug 2026, and it is the difference between a
shell that matches the floor plan and one that does not. Phase 1's first
implementation extruded the *photo-derived* polygon and positioned it on the plan,
which measured badly for two structural reasons:

* every photo-derived room is an oriented bounding box — one ultra-wide photograph
  cannot reveal an alcove — so rooms came out a median **31% larger** than their own
  plan polygon;
* the shell only contained rooms somebody had photographed, so a five-bedroom flat
  shot in four rooms came out as four floating boxes with holes between them.

Both disappear when the plan polygon is the outline. Every room the plan found gets
built, whether or not a photograph reached it; the ones without a photograph are
tagged ``inferred`` so the viewer desaturates them and the honesty rendering stays
truthful (AD-15).

The photo channel is not discarded — it moves to the three jobs it is better at:
the ceiling height, an independent cross-check on scale (stage 7), and the geometry
Phase 2's splats are trained and culled against.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..core import geom
from ..core.stages import StageContext, StageResult, register_stage
from .mesh import Mesh, build_room, to_glb

log = logging.getLogger("shell.stage")

SHELL_BYTES_BUDGET = 1_048_576          # ARCHITECTURE §6, CI-enforced
#: UK ceiling prior, used only when neither this room nor the flat offers a measurement.
DEFAULT_CEILING_M = 2.55
CEILING_ACCEPTABLE_M = (2.0, 4.2)


def _ceiling_for(room_id: str | None, lay_by_id: dict, scale: float,
                 fallback: float) -> tuple[float, str]:
    """This room's measured height, the flat's median, or the prior — in that order."""
    lay = lay_by_id.get(room_id or "")
    if lay and lay.get("room_height_m"):
        h = lay["room_height_m"] * scale
        if CEILING_ACCEPTABLE_M[0] <= h <= CEILING_ACCEPTABLE_M[1]:
            return round(h, 3), "measured"
    return round(fallback, 3), "flat_median" if fallback != DEFAULT_CEILING_M else "uk_prior"


def _plan_polygons_in_metres(plan: dict, scale: dict) -> tuple[dict[str, list], str]:
    """Every plan room's outline in metres, however the scale had to be obtained."""
    ppm = plan.get("px_per_metre") or scale.get("derived_plan_px_per_metre")
    if plan.get("px_per_metre"):
        return ({p["room_id"]: p["polygon_m"] for p in plan["rooms"] if p.get("polygon_m")},
                plan.get("scale_source") or "plan")
    if not ppm:
        return {}, "none"
    # Same frame convention as stage 5: origin at the footprint's lower-left, y up.
    h_px = (plan.get("image_size_px") or [0, 0])[1]
    pts = [pt for p in plan["rooms"] for pt in (p.get("polygon_px") or [])]
    if not pts:
        return {}, "none"
    ox = min(x for x, _ in pts)
    oy = h_px - max(y for _, y in pts)
    out = {}
    for p in plan["rooms"]:
        poly = p.get("polygon_px")
        if poly and len(poly) >= 3:
            out[p["room_id"]] = [[(x - ox) / ppm, (h_px - y - oy) / ppm] for x, y in poly]
    return out, "derived_from_photos"


def build(layouts: dict, plan: dict, assembly: dict, scale: dict) -> tuple[dict, bytes]:
    t0 = time.perf_counter()
    s = float(scale.get("scale", 1.0))
    polys_m, scale_source = _plan_polygons_in_metres(plan, scale)
    lay_by_id = {r["room_id"]: r for r in layouts["rooms"]}
    plan_aps = {a["aperture_id"]: a for a in plan.get("apertures", [])}
    # Which photo-room, if any, was matched to each plan polygon.
    photo_for_plan = {m["plan_room_id"]: m for m in assembly.get("matches", [])}

    qa: list[str] = []
    heights = [r["room_height_m"] * s for r in layouts["rooms"] if r.get("room_height_m")]
    plausible = [h for h in heights if CEILING_ACCEPTABLE_M[0] <= h <= CEILING_ACCEPTABLE_M[1]]
    fallback = float(np.median(plausible)) if plausible else DEFAULT_CEILING_M
    if not plausible:
        qa.append("no_measured_ceiling_anywhere")

    mesh = Mesh()
    rooms_out = []
    n_measured = n_inferred = 0
    for p in plan["rooms"]:
        poly_m = polys_m.get(p["room_id"])
        if not poly_m or len(poly_m) < 3:
            continue
        poly = geom.simplify(geom.ensure_ccw(poly_m), tol=0.03)
        if geom.area(poly) < 0.8:
            continue                      # a sliver, not a room
        m = photo_for_plan.get(p["room_id"])
        lay = lay_by_id.get(m["room_id"]) if m else None
        h, h_src = _ceiling_for(m["room_id"] if m else None, lay_by_id, s, fallback)

        # AD-15. A room a photograph reached is reconstructed; one the plan found and
        # nobody photographed is inferred, and has to look different.
        provenance = "reconstructed" if m else "inferred"
        if m:
            n_measured += 1
        else:
            n_inferred += 1
        aps = [plan_aps[i] for i in p.get("aperture_ids", []) if i in plan_aps]
        build_room(mesh, poly, 0.0, h, p["room_id"], provenance, aps)
        rooms_out.append({
            "room_id": p["room_id"],
            "plan_room_id": p["room_id"],
            "photo_room_id": m["room_id"] if m else None,
            "label": p.get("label") or (lay.get("room_label") if lay else None),
            "polygon_m": [[round(x, 3), round(y, 3)] for x, y in poly],
            "centroid_m": [round(v, 3) for v in geom.representative_point(poly)],
            "height_m": h,
            "height_source": h_src,
            "area_m2": round(geom.area(poly), 3),
            "provenance": provenance,
            "confidence": round(_room_confidence(p, m, lay, h_src), 3),
            "aperture_ids": p.get("aperture_ids", []),
        })

    glb = to_glb(mesh)
    if len(glb) > SHELL_BYTES_BUDGET:
        qa.append("shell_over_1mb_budget")
    if not rooms_out:
        qa.append("empty_shell")
    if scale_source == "derived_from_photos":
        qa.append("plan_scale_derived_from_photos")
    elif scale_source == "none":
        qa.append("no_plan_scale_at_all")
    if n_inferred > n_measured:
        # More than half the flat is rooms nobody photographed. Still built, still
        # honest — but the viewer will be mostly grey and a person should know.
        qa.append("majority_of_rooms_unphotographed")
    orphans = [m["room_id"] for m in assembly.get("matches", [])
               if m["plan_room_id"] not in {r["room_id"] for r in rooms_out}]
    if orphans:
        qa.append("photo_rooms_without_a_plan_polygon")

    footprint = geom.union_polygon([r["polygon_m"] for r in rooms_out]) if rooms_out else None
    payload = {
        "schema": "shell/v1",
        "listing_id": layouts["listing_id"],
        "built_from": "plan_polygons",
        "plan_scale_source": scale_source,
        "scale_applied": round(s, 6),
        "rooms": rooms_out,
        "omitted_rooms": orphans,
        "coverage": {
            "plan_rooms": len(plan["rooms"]),
            "built": len(rooms_out),
            "with_a_photograph": n_measured,
            "inferred": n_inferred,
        },
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


def _room_confidence(plan_room: dict, match: dict | None, lay: dict | None,
                     height_source: str) -> float:
    c = 0.25 + 0.35 * float(plan_room.get("confidence", 0.5))
    if plan_room.get("label"):
        c += 0.1
    if match:
        c += 0.15 * float(match.get("confidence", 0.5))
    if height_source == "measured":
        c += 0.15
    elif height_source == "uk_prior":
        c -= 0.05
    if lay and lay.get("qa_flags"):
        c -= 0.03 * len(lay["qa_flags"])
    return min(1.0, max(0.05, c))


@register_stage("8-shell", description="Extrude the plan's rooms at the measured ceiling height")
def run(ctx: StageContext) -> StageResult:
    payload, glb = build(ctx.require("4-layout"), ctx.require("5-plan"),
                         ctx.require("6-assembly"), ctx.require("7-scale"))
    return StageResult(payload=payload, binaries={"shell.glb": glb}, engine="extrude_plan")
