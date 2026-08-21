"""M1-M5 and the G1 criteria (ARCHITECTURE §9, ROADMAP §3).

**What we can and cannot measure.** Phase 0 recorded the honest ceiling on our
claims: with no tape-measure ground truth, every number here is *plausibility* and
*self-consistency*, not *accuracy*. A 2.71 m ceiling is a credible ceiling; we have
not established it is the right one. So each metric below declares its reference:

``printed``
    The listing's own floor plan printed the number. This is the strongest
    reference we have and it needs no external measurement — the listing supplies
    both the reconstruction and the number it has to agree with.
``stated``
    The listing text or the portal's sizings field. Weaker: Phase 0 demoted portal
    metadata after finding it wrong about its own floor plans.
``annotated``
    A human drew it (see `eval/annotations.py`). Used for arrangement and
    adjacency, which nothing else can supply.

A metric with no reference available reports ``None`` rather than a number, and the
scoreboard prints the coverage alongside every score, because "M2 is 4%" means
something very different at n=3 and at n=24.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from pipeline.core import geom

log = logging.getLogger("eval.metrics")


@dataclass
class Metric:
    id: str
    name: str
    value: float | None
    unit: str
    reference: str
    n: int
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name,
                "value": round(self.value, 4) if self.value is not None else None,
                "unit": self.unit, "reference": self.reference, "n": self.n,
                "detail": self.detail}


def m1_total_area(scene: dict, scale: dict, manifest: dict, plan: dict | None) -> Metric:
    """|model area − reference area| / reference, %.

    The reference is the printed plan total where there is one, else the stated
    area. Note this is *not* independent of stage 7 when the stated area was used
    as a scale constraint — the detail records which constraints were in play so a
    circular reading is visible rather than flattering.
    """
    model = sum(r["area_m2"] for r in scene.get("rooms", []))
    ref, ref_kind = None, "none"
    if plan and plan.get("ocr", {}).get("total_area_m2"):
        ref, ref_kind = plan["ocr"]["total_area_m2"], "printed"
    elif manifest.get("listing", {}).get("advertised_area_m2"):
        ref, ref_kind = manifest["listing"]["advertised_area_m2"], "stated"
    if not ref or model <= 0:
        return Metric("M1", "total area error", None, "%", ref_kind, 0)
    used = sorted({c["kind"] for c in scale.get("constraints", []) if c.get("used")})
    return Metric("M1", "total area error", 100 * abs(model - ref) / ref, "%", ref_kind, 1,
                  {"model_m2": round(model, 2), "reference_m2": round(ref, 2),
                   "scale_constraints_used": used,
                   "independent": "stated_area" not in used and "plan_total_area" not in used})


def m2_room_area(scale: dict) -> Metric:
    """Median per-room area error against the dimensions printed on the plan.

    This is G1's self-consistency criterion. Stage 7 already computed it per room
    (it is the same comparison the scale solve residuals are built from); this
    reads it out so the harness and the gate cannot disagree about the number.
    """
    sc = scale.get("self_consistency", {})
    per = [r for r in sc.get("per_room", [])]
    if not per:
        return Metric("M2", "per-room area error", None, "%", "printed", 0)
    errs = [abs(r["error_pct"]) for r in per]
    return Metric("M2", "per-room area error", float(np.median(errs)), "%", "printed",
                  len(errs),
                  {"within_10pct_frac": round(float(np.mean([e <= 10 for e in errs])), 3),
                   "p90_pct": round(float(np.percentile(errs, 90)), 2),
                   "per_room": per})


def m3_adjacency(scene: dict, plan: dict, assembly: dict,
                 annotation: dict | None = None) -> Metric:
    """Correct room-adjacency edges / edges in the listing's own floor plan.

    The reference is the plan's own adjacency graph, mapped through the assembly.
    An annotation overrides it where one exists, because the vectoriser's adjacency
    is itself an estimate.
    """
    ref_edges = None
    ref_kind = "printed"
    if annotation and annotation.get("adjacency"):
        ref_edges = {tuple(sorted(e)) for e in annotation["adjacency"]}
        ref_kind = "annotated"
    elif plan.get("adjacency"):
        ref_edges = {tuple(sorted((a["a"], a["b"]))) for a in plan["adjacency"]}
    if not ref_edges:
        return Metric("M3", "adjacency accuracy", None, "fraction", ref_kind, 0)

    room_to_plan = {m["room_id"]: m["plan_room_id"] for m in assembly.get("matches", [])}
    built = set()
    polys = {r["room_id"]: r["polygon_m"] for r in scene.get("rooms", [])}
    ids = list(polys)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if geom.shared_edge_length(polys[a], polys[b], tol=0.25) > 0.5:
                pa, pb = room_to_plan.get(a), room_to_plan.get(b)
                if pa and pb:
                    built.add(tuple(sorted((pa, pb))))
    hit = len(built & ref_edges)
    return Metric("M3", "adjacency accuracy", hit / len(ref_edges), "fraction", ref_kind,
                  len(ref_edges),
                  {"reference_edges": len(ref_edges), "built_edges": len(built),
                   "correct": hit,
                   "missing": sorted("|".join(e) for e in (ref_edges - built))[:10]})


def m4_layout_iou(scene: dict, plan: dict, assembly: dict) -> Metric:
    """2D IoU of the assembled footprint against the plan's own footprint.

    Both sides come from the same listing, so this measures whether the *shell*
    reproduces the *plan* — not whether either reproduces the flat. That is the
    honest scope, and it is exactly what the G1 arrangement criterion needs.
    """
    built = [r["polygon_m"] for r in scene.get("rooms", [])]
    ref = [p["polygon_m"] for p in plan.get("rooms", []) if p.get("polygon_m")]
    if not built or not ref:
        return Metric("M4", "layout IoU", None, "IoU", "printed", 0)
    bu = geom.union_polygon(built)
    ru = geom.union_polygon(ref)
    if not bu or not ru:
        return Metric("M4", "layout IoU", None, "IoU", "printed", 0)
    # Compare shapes, not positions: the plan's frame origin is arbitrary, so both
    # footprints are centred before intersecting.
    bc, rc = geom.centroid(bu), geom.centroid(ru)
    bu2 = geom.se2(bu, rc[0] - bc[0], rc[1] - bc[1], 0.0, about=bc)
    per_room = []
    plan_by_id = {p["room_id"]: p for p in plan.get("rooms", [])}
    for m in assembly.get("matches", []):
        sr = next((r for r in scene.get("rooms", []) if r["room_id"] == m["room_id"]), None)
        pr = plan_by_id.get(m["plan_room_id"])
        if sr and pr and pr.get("polygon_m"):
            per_room.append({"room_id": m["room_id"], "plan_room_id": m["plan_room_id"],
                             "iou": round(geom.iou(sr["polygon_m"], pr["polygon_m"]), 4)})
    return Metric("M4", "layout IoU", geom.iou(bu2, ru), "IoU", "printed", len(built),
                  {"per_room": per_room,
                   "median_room_iou": (round(float(np.median([p["iou"] for p in per_room])), 4)
                                       if per_room else None)})


def m5_assignment(assembly: dict, annotation: dict | None) -> Metric:
    """Photos placed in the correct plan room, %.

    Needs an annotation: nothing in the listing says which photograph belongs to
    which polygon, which is the whole reason assembly is hard. Where no annotation
    exists this reports None rather than a flattering proxy.
    """
    if not annotation or not annotation.get("assignment"):
        return Metric("M5", "assignment accuracy", None, "%", "annotated", 0)
    truth = annotation["assignment"]
    got = {m["room_id"]: m["plan_room_id"] for m in assembly.get("matches", [])}
    checked = [(k, v) for k, v in truth.items() if k in got or v is not None]
    if not checked:
        return Metric("M5", "assignment accuracy", None, "%", "annotated", 0)
    correct = sum(1 for k, v in checked if got.get(k) == v)
    return Metric("M5", "assignment accuracy", 100 * correct / len(checked), "%",
                  "annotated", len(checked),
                  {"correct": correct, "total": len(checked),
                   "wrong": [{"room_id": k, "expected": v, "got": got.get(k)}
                             for k, v in checked if got.get(k) != v]})


# --------------------------------------------------------------------------
# G1 criteria (ROADMAP §3)


def g1_criteria(scale: dict, layouts: dict, shell: dict, assembly: dict,
                m4: Metric, annotation: dict | None = None) -> dict:
    """Each G1 bullet, evaluated, with the evidence behind it.

    Returns a dict of ``criterion -> {passed, value, threshold, note}``. Nothing is
    rounded up: a criterion with no evidence reports ``passed: None``, which is a
    different thing from failing and must not be counted as either.
    """
    out: dict[str, dict] = {}

    sc = scale.get("self_consistency", {})
    v = sc.get("median_abs_pct")
    out["self_consistency"] = {
        "passed": (v is not None and v <= 10.0),
        "value": v, "threshold": 10.0, "unit": "% median room-area error",
        "n": sc.get("n_rooms_checked", 0),
        "note": "where the plan prints room dimensions, reconstructed areas agree within ±10%",
    } if v is not None else {"passed": None, "value": None, "threshold": 10.0,
                             "n": 0, "note": "no plan printed room dimensions"}

    pl = scale.get("plausibility", {})
    frac = pl.get("ceiling_plausible_frac")
    over12 = pl.get("any_room_over_12m")
    out["plausibility"] = {
        "passed": (frac is not None and frac >= 0.8 and not over12),
        "value": frac, "threshold": 0.8, "unit": "fraction of rooms 2.3-3.2 m",
        "n": pl.get("n_rooms", 0),
        "any_room_over_12m": over12,
        "note": "≥80% of rooms have plausible ceilings and no room exceeds 12 m across",
    }

    # Arrangement: rooms in the correct plan polygon. An annotation is the real
    # answer; without one we report the shell-vs-plan IoU as supporting evidence
    # and leave the criterion unjudged rather than passing it on a proxy.
    if annotation and annotation.get("assignment"):
        truth = annotation["assignment"]
        got = {m["room_id"]: m["plan_room_id"] for m in assembly.get("matches", [])}
        checked = [(k, v) for k, v in truth.items()]
        correct = sum(1 for k, v in checked if got.get(k) == v)
        frac_ok = correct / len(checked) if checked else None
        out["arrangement"] = {"passed": (frac_ok is not None and frac_ok >= 0.7),
                              "value": frac_ok, "threshold": 0.7,
                              "unit": "fraction of rooms in the right polygon",
                              "n": len(checked), "reference": "annotated",
                              "note": "rooms placed in the correct plan polygon"}
    else:
        out["arrangement"] = {"passed": None, "value": m4.value, "threshold": 0.7,
                              "unit": "layout IoU (supporting evidence only)",
                              "n": m4.n, "reference": "printed",
                              "note": ("no arrangement annotation for this listing; "
                                       "IoU shown as evidence, criterion not judged")}

    cc = scale.get("cross_check", {})
    d = cc.get("max_disagreement_pct")
    out["cross_model_scale"] = {
        "passed": cc.get("within_15pct"), "value": d, "threshold": 15.0,
        "unit": "% disagreement between independent scale estimates",
        "n": len(cc.get("estimates", {})),
        "note": "independent scale estimates within 15% of each other",
    }

    glb = shell.get("glb", {})
    out["shell_budget"] = {
        "passed": glb.get("within_budget"), "value": glb.get("bytes"),
        "threshold": glb.get("budget_bytes"), "unit": "bytes",
        "n": 1, "note": "shell ≤ 1 MB so it loads inside the G1 time budget",
    }
    return out


def summarise_g1(per_listing: list[dict]) -> dict:
    """Roll listing-level criteria into the gate's own numbers."""
    keys = ["self_consistency", "plausibility", "arrangement", "cross_model_scale",
            "shell_budget"]
    out: dict[str, dict] = {}
    for k in keys:
        judged = [l[k] for l in per_listing if l.get(k, {}).get("passed") is not None]
        passed = [c for c in judged if c["passed"]]
        vals = [c["value"] for c in judged if c.get("value") is not None]
        out[k] = {
            "listings_judged": len(judged),
            "listings_unjudged": len(per_listing) - len(judged),
            "pass_rate": round(len(passed) / len(judged), 3) if judged else None,
            "median_value": round(float(np.median(vals)), 3) if vals else None,
            "threshold": judged[0]["threshold"] if judged else None,
            "unit": judged[0].get("unit") if judged else None,
        }
    return out
