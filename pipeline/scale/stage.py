"""Stage 7 — the global scale, and the three checks that stand in for a tape measure.

The scalar itself is `solve.py`'s job. This module's job is to turn artifacts into
constraints, and then to run ROADMAP §0b's three checks against the result:

1. **Plausibility** — ceilings in 2.3-3.2 m, no room over 12 m across.
2. **Self-consistency** — where the plan prints a room's dimensions, the scaled
   reconstruction has to agree with them. The listing supplies both sides, so this
   costs nothing and is the closest thing we have to ground truth.
3. **Cross-model agreement** — the plan channel and the photo channel estimate the
   same scalar independently; they should land within 15% of each other.

These are weaker than measurement and the artifact says so. They are strong enough
to tell working from broken, which is what the gates need.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..core import geom
from ..core.stages import StageContext, StageResult, register_stage
from ..layout.planes import CEILING_PLAUSIBLE_M, MAX_ROOM_EXTENT_M
from .solve import CEILING_PRIOR_M, DOOR_HEIGHT_M, Constraint, Solution, quality, solve

log = logging.getLogger("scale.stage")


def build_constraints(layouts: dict, plan: dict | None, manifest: dict | None,
                      assembly: dict | None) -> list[Constraint]:
    cs: list[Constraint] = []
    rooms = layouts["rooms"]
    by_id = {r["room_id"]: r for r in rooms}

    # Per-room areas printed on the plan, routed through the assembly so we compare
    # a room to *its own* polygon rather than to whichever polygon is nearest in size.
    if plan and assembly:
        plan_by_id = {p["room_id"]: p for p in plan.get("rooms", [])}
        for m in assembly.get("matches", []):
            r = by_id.get(m["room_id"])
            p = plan_by_id.get(m["plan_room_id"])
            if not r or not p:
                continue
            printed = None
            if p.get("ocr_dims_m") and len(p["ocr_dims_m"]) == 2:
                printed = p["ocr_dims_m"][0] * p["ocr_dims_m"][1]
            elif p.get("ocr_area_m2"):
                printed = p["ocr_area_m2"]
            if printed and r.get("area_m2"):
                cs.append(Constraint("plan_room_area", printed, r["area_m2"], 2,
                                     detail=f"{m['plan_room_id']}<-{m['room_id']}",
                                     source="plan_ocr"))

    total_recon = sum(r["area_m2"] for r in rooms if r.get("area_m2"))
    if plan and plan.get("totals", {}).get("plan_area_m2") and total_recon > 0:
        cs.append(Constraint("plan_total_area", plan["totals"]["plan_area_m2"],
                             total_recon, 2, detail="footprint vs sum of rooms",
                             source="plan_vectorisation",
                             # Weak: our rooms are internal and miss unlabelled
                             # circulation, so this compares unlike things on purpose.
                             weight=0.8))
    stated = (manifest or {}).get("listing", {}).get("advertised_area_m2")
    if stated and total_recon > 0:
        cs.append(Constraint("stated_area", stated, total_recon, 2,
                             detail="listing floor area", weight=0.9,
                             source=(manifest or {}).get("listing", {}).get("area_source") or "listing"))

    heights = [r["room_height_m"] for r in rooms if r.get("room_height_m")]
    if heights:
        cs.append(Constraint("ceiling_height", CEILING_PRIOR_M, float(np.median(heights)),
                             1, detail="median room height", source="uk_prior"))

    # Door heights from stage-4 apertures. Empty in Phase 1's monocular era; the
    # constraint is wired now so Sprint 8's photo apertures need no new plumbing.
    door_h = [a["height_m"] for r in rooms for a in (r.get("apertures") or [])
              if a.get("type") == "door" and a.get("height_m")]
    if door_h:
        cs.append(Constraint("door_height", DOOR_HEIGHT_M, float(np.median(door_h)), 1,
                             detail=f"{len(door_h)} door(s)", source="photo_apertures"))
    return cs


def checks(layouts: dict, plan: dict | None, sol: Solution, scale: float) -> dict:
    rooms = layouts["rooms"]
    heights = [r["room_height_m"] * scale for r in rooms if r.get("room_height_m")]
    extents = [max(r["extent_m"]) * scale for r in rooms if r.get("extent_m")]
    plausible = [h for h in heights
                 if CEILING_PLAUSIBLE_M[0] <= h <= CEILING_PLAUSIBLE_M[1]]
    plaus = {
        "n_rooms": len(rooms),
        "ceiling_plausible_frac": round(len(plausible) / len(heights), 3) if heights else None,
        "median_ceiling_m": round(float(np.median(heights)), 3) if heights else None,
        "max_extent_m": round(max(extents), 3) if extents else None,
        "any_room_over_12m": bool(extents and max(extents) > MAX_ROOM_EXTENT_M),
    }

    per_room, pcts = [], []
    for c in sol.constraints:
        if c.kind != "plan_room_area":
            continue
        got = c.observed * scale ** 2
        pct = 100.0 * (got - c.target) / c.target if c.target else 0.0
        per_room.append({"detail": c.detail, "printed_m2": round(c.target, 3),
                         "reconstructed_m2": round(got, 3), "error_pct": round(pct, 2),
                         "used": c.used})
        pcts.append(abs(pct))
    self_consistency = {
        "n_rooms_checked": len(pcts),
        "median_abs_pct": round(float(np.median(pcts)), 3) if pcts else None,
        "within_10pct_frac": (round(float(np.mean([p <= 10.0 for p in pcts])), 3)
                              if pcts else None),
        "per_room": per_room,
    }

    # Cross-model agreement: the plan channel scales pixels by what the plan prints;
    # the photo channel scales metres by the ceiling prior. Independent methods,
    # shared quantity.
    estimates: dict[str, float] = {}
    plan_only = [c for c in sol.constraints if c.kind in ("plan_room_area", "plan_total_area")]
    photo_only = [c for c in sol.constraints if c.kind in ("ceiling_height", "door_height")]
    for name, subset in (("plan_channel", plan_only), ("photo_channel", photo_only)):
        if subset:
            s = solve([Constraint(c.kind, c.target, c.observed, c.power, c.detail,
                                  c.source, c.weight) for c in subset])
            estimates[name] = round(s.scale, 4)
    vals = list(estimates.values())
    disagreement = (100.0 * (max(vals) - min(vals)) / max(vals)) if len(vals) > 1 else None
    cross = {"estimates": estimates,
             "max_disagreement_pct": round(disagreement, 2) if disagreement is not None else None,
             "within_15pct": (disagreement <= 15.0) if disagreement is not None else None}
    return {"plausibility": plaus, "self_consistency": self_consistency, "cross_check": cross}


def build_scale(layouts: dict, plan: dict | None, manifest: dict | None,
                assembly: dict | None) -> dict:
    t0 = time.perf_counter()
    cs = build_constraints(layouts, plan, manifest, assembly)
    sol = solve(cs)
    kinds = len({c.kind for c in cs if c.used})
    q = quality(sol, kinds)
    ck = checks(layouts, plan, sol, sol.scale)

    qa: list[str] = []
    if not cs:
        qa.append("no_scale_constraints")
    if sol.residual_rms_pct > 20:
        qa.append("scale_constraints_disagree")
    if sol.rejected:
        qa.append("scale_constraint_rejected")
    sc = ck["self_consistency"]
    if sc["within_10pct_frac"] is not None and sc["within_10pct_frac"] < 0.5:
        qa.append("self_consistency_outside_10pct")
    pl = ck["plausibility"]
    if pl["ceiling_plausible_frac"] is not None and pl["ceiling_plausible_frac"] < 0.8:
        qa.append("under_80pct_plausible_ceilings")
    if pl["any_room_over_12m"]:
        qa.append("room_over_12m_across")
    if ck["cross_check"]["within_15pct"] is False:
        qa.append("cross_model_scale_disagreement")
    if abs(np.log(max(sol.scale, 1e-6))) > np.log(1.6):
        # Needing to change the reconstruction's size by more than ~60% means one
        # of the two sides is wrong, not that the flat is unusual.
        qa.append("large_scale_correction")

    return {
        "schema": "scale/v1",
        "listing_id": layouts["listing_id"],
        "scale": round(sol.scale, 6),
        "log_scale": round(sol.log_scale, 6),
        "sigma_log": round(sol.sigma_log, 6),
        "constraints": [c.to_dict() for c in sol.constraints],
        "residual_rms_pct": round(sol.residual_rms_pct, 3),
        "quality": q,
        **ck,
        "confidence": q,
        "qa_flags": sorted(set(qa)),
        "timing_s": round(time.perf_counter() - t0, 3),
    }


@register_stage("7-scale", description="Solve the global metric scale and run the §0b checks")
def run(ctx: StageContext) -> StageResult:
    layouts = ctx.require("4-layout")
    return StageResult(payload=build_scale(layouts, ctx.read("5-plan"), ctx.read("0-triage"),
                                           ctx.read("6-assembly")),
                       engine="wls_log")
