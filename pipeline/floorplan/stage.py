"""Stage 5 — the plan channel.

Consumes the largest floor-plan asset on the listing; emits `plan.json`: room
polygons in pixels and (where a scale could be established) in metres, the
adjacency graph, doors and windows, everything OCR read, and an honest account of
how confident any of it is.

This is the spine of the product (AD-2). Photographs contain **zero** signal about
where rooms sit relative to each other, so if this artifact is wrong the listing's
arrangement is wrong no matter how good the photo channel gets.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ..core import geom
from ..core.stages import StageContext, StageResult, register_stage
from . import ocr as ocr_mod
from . import planscale, preprocess, topology, vectorise

log = logging.getLogger("floorplan.stage")

CEILING_DEFAULT_M = 2.5
#: What a room can plausibly be, in m². Used only once a scale exists — before that
#: there is nothing to compare against.
MIN_ROOM_M2 = 1.2
MAX_ROOM_M2 = 120.0
#: ...and no single room is most of a flat.
MAX_ROOM_SHARE = 0.55


def _to_metres(poly_px, ppm: float, origin_px: tuple[float, float], height_px: int):
    """Image pixels to a right-handed metric frame with y pointing up.

    Screen y grows downward and metric y grows up; getting this wrong mirrors
    every room and the mistake is invisible until the viewer, where every flat
    looks subtly wrong and nobody can say why.
    """
    ox, oy = origin_px
    return [[round((x - ox) / ppm, 4), round((height_px - y - oy) / ppm, 4)]
            for x, y in poly_px]


def build_plan(image_path: Path, listing: dict, *, engine: str = "raster",
               override: dict | None = None) -> dict:
    t0 = time.perf_counter()
    pi = preprocess.prepare(image_path)
    text = ocr_mod.read(pi.rgb)
    # Which listing this is, so the room-finder can look up its predictions.
    vectorise.set_listing(listing.get("listing_id"))
    try:
        vec = vectorise.regularise_all(vectorise.segment(pi, text))
    finally:
        vectorise.set_listing(None)

    footprint_px = float(vec.footprint.sum()) if vec.footprint is not None else 0.0
    stated = listing.get("floor_area_sqm")
    best, cands, per_room_scale, disagreement = planscale.solve(
        vec.rooms, footprint_px, text.total_area_m2, stated, text.scale_ratio)
    ppm = best.px_per_metre if best else None

    ov = override or {}
    edited: set[str] = set()
    if ov.get("px_per_metre"):
        ppm = float(ov["px_per_metre"])
        best = planscale.ScaleCandidate("printed_dimensions", ppm, 6.0, "set by a reviewer")
        cands = cands + [best]
        disagreement = None
    for rid, patch in (ov.get("rooms") or {}).items():
        for r in vec.rooms:
            if r.room_id != rid:
                continue
            if "label" in patch:
                r.label = patch["label"]
                r.label_confidence = 1.0
            if "polygon_px" in patch:
                r.polygon_px = [[float(x), float(y)] for x, y in patch["polygon_px"]]
                r.area_px = geom.area(r.polygon_px)
                r.centroid_px = geom.centroid(r.polygon_px)
            if "drop" in patch and patch["drop"]:
                r.area_px = 0.0
            edited.add(rid)
    if edited:
        vec.rooms = [r for r in vec.rooms if r.area_px > 0]

    # Plausibility filter, applied only once a metric scale exists. A "room" of
    # 300 m², or one that is 60% of the flat on its own, is the exterior, a garden,
    # or the sheet's border — not a room. Left in, it dominates the shell and drags
    # every downstream area comparison.
    dropped_implausible: list[str] = []
    if ppm:
        keep = []
        total_m2 = sum(r.area_px for r in vec.rooms) / (ppm ** 2)
        for r in vec.rooms:
            a = r.area_px / (ppm ** 2)
            too_big = a > MAX_ROOM_M2 or (total_m2 > 0 and a / total_m2 > MAX_ROOM_SHARE
                                          and len(vec.rooms) > 2)
            if a < MIN_ROOM_M2 or too_big:
                dropped_implausible.append(r.room_id)
            else:
                keep.append(r)
        if keep:
            vec.rooms = keep

    apertures = topology.find_apertures(vec.labels, vec.rooms, ppm)
    tol = max(3.0, (ppm or 50.0) * 0.06)
    adjs = topology.attach_doors(topology.adjacency(vec.rooms, tol), apertures)

    h_px, w_px = pi.ink.shape
    if vec.footprint is not None and vec.footprint.any():
        ys, xs = np.nonzero(vec.footprint)
        origin = (float(xs.min()), float(h_px - ys.max()))
    else:
        origin = (0.0, 0.0)

    ap_by_room: dict[str, list[str]] = {}
    for ap in apertures:
        for rid in ap.rooms:
            ap_by_room.setdefault(rid, []).append(ap.aperture_id)

    rooms_out = []
    for r in vec.rooms:
        poly_m = _to_metres(r.polygon_px, ppm, origin, h_px) if ppm else None
        area_m2 = round(r.area_px / (ppm ** 2), 3) if ppm else None
        long_px, short_px, _ = geom.oriented_extent(r.polygon_px)
        conf = _room_confidence(r, ppm, area_m2)
        flags = list(r.qa_flags)
        if r.ocr_dims_m and area_m2:
            printed = r.ocr_dims_m[0] * r.ocr_dims_m[1]
            # Self-consistency, room by room: does the polygon we drew agree with
            # the size the plan prints on it? This is ROADMAP §0b check 2 and it is
            # free — the listing supplies both sides.
            if printed > 0 and abs(area_m2 - printed) / printed > 0.35:
                flags.append("room_area_disagrees_with_printed")
        rooms_out.append({
            "room_id": r.room_id,
            "label": r.label,
            "label_text": r.label_text,
            "label_confidence": round(r.label_confidence, 3),
            "polygon_px": [[round(x, 2), round(y, 2)] for x, y in r.polygon_px],
            "polygon_m": poly_m,
            "area_px": round(r.area_px, 2),
            "area_m2": area_m2,
            "ocr_area_m2": r.ocr_area_m2,
            "ocr_dims_m": list(r.ocr_dims_m) if r.ocr_dims_m else None,
            "centroid_px": [round(v, 2) for v in r.centroid_px],
            "centroid_m": (_to_metres([r.centroid_px], ppm, origin, h_px)[0] if ppm else None),
            "extent_m": ([round(long_px / ppm, 3), round(short_px / ppm, 3)] if ppm else None),
            "aperture_ids": ap_by_room.get(r.room_id, []),
            "confidence": 1.0 if r.room_id in edited else conf,
            "qa_flags": sorted(set(flags)),
            "seeded_by": r.seeded_by,
            "edited_by_human": r.room_id in edited,
        })

    plan_area_m2 = round(footprint_px / ppm ** 2, 2) if ppm else None
    ratio = (round(plan_area_m2 / stated, 3) if plan_area_m2 and stated else None)

    qa: list[str] = list(pi.qa_flags) + list(vec.qa_flags)
    if best is None:
        qa.append("no_plan_scale")
    if disagreement is not None and disagreement > 15.0:
        qa.append("scale_candidates_disagree")
    if text.not_to_scale:
        qa.append("not_to_scale_disclaimer")
    if ratio is not None and not (0.7 <= ratio <= 1.3):
        qa.append("plan_area_disagrees_with_stated")
    if not any(r["label"] for r in rooms_out):
        qa.append("no_room_labels")
    if not any(a.kind == "door" for a in apertures):
        qa.append("no_doors_detected")
    if edited or ov.get("px_per_metre"):
        qa.append("override_applied")
    if dropped_implausible:
        qa.append(f"dropped_{len(dropped_implausible)}_implausible_regions")
    qa += sorted({f for r in rooms_out for f in r["qa_flags"]})

    payload = {
        "schema": "plan/v1",
        "listing_id": listing["listing_id"],
        "source_image": str(image_path),
        "image_size_px": [int(w_px), int(h_px)],
        "method": ("learned_vectorise/v1" if "learned_vectoriser" in vec.qa_flags
                   else "room_finder_vectorise/v1" if "rooms_from_room_finder" in vec.qa_flags
                   else f"{engine}_vectorise/v2"),
        "preprocess": {
            "deskew_deg": round(pi.deskew_deg, 3),
            "binarise": "modal-background distance",
            "upscaled_to_px": ocr_mod.OCR_MIN_SIDE,
            "wall_half_px": round(pi.wall_half_px, 2),
            "text_components_erased": vec.text_erased,
        },
        "px_per_metre": round(ppm, 4) if ppm else None,
        "scale_source": best.source if best else "none",
        "scale_candidates": [c.to_dict() for c in cands],
        "rooms": rooms_out,
        "adjacency": [
            {"a": a.a, "b": a.b,
             "shared_wall_m": round(a.shared_wall_px / ppm, 3) if ppm else None,
             "door": a.door,
             "door_position_m": (_to_metres([a.door_position_px], ppm, origin, h_px)[0]
                                 if (ppm and a.door_position_px) else None),
             "confidence": round(a.confidence, 3)}
            for a in adjs
        ],
        "apertures": [
            {"aperture_id": ap.aperture_id, "type": ap.kind,
             "position_px": [round(v, 2) for v in ap.position_px],
             "position_m": (_to_metres([ap.position_px], ppm, origin, h_px)[0] if ppm else None),
             "width_m": round(ap.width_px / ppm, 3) if ppm else None,
             "rooms": ap.rooms, "confidence": round(ap.confidence, 3)}
            for ap in apertures
        ],
        "ocr": {
            "raw_text": text.raw_text,
            "room_labels": text.room_labels,
            "areas_sqm": text.areas_sqm,
            "dims_m": text.dims_m,
            "total_area_m2": text.total_area_m2,
            "scale_ratio": text.scale_ratio,
            "not_to_scale": text.not_to_scale,
            "boxes": [w.to_dict() for w in text.words[:400]],
        },
        "totals": {
            "plan_area_m2": plan_area_m2,
            "stated_area_m2": stated,
            "area_ratio": ratio,
            "room_area_m2": (round(sum(r["area_m2"] for r in rooms_out if r["area_m2"]), 2)
                             if ppm else None),
            "scale_evidence": per_room_scale,
        },
        "non_manhattan": bool(vec.non_manhattan),
        "approximate": bool(text.not_to_scale or best is None),
        "confidence": _plan_confidence(rooms_out, best, disagreement, text, qa),
        "qa_flags": sorted(set(qa)),
        "timing_s": round(time.perf_counter() - t0, 3),
    }
    return payload


def _room_confidence(r, ppm: float | None, area_m2: float | None) -> float:
    c = 0.35
    if r.label:
        c += 0.2 + 0.1 * min(1.0, r.label_confidence)
    if r.ocr_dims_m:
        c += 0.15
    if r.seeded_by == "caption":
        c += 0.1
    if ppm and area_m2 and 2.0 <= area_m2 <= 90.0:
        c += 0.1
    for f in r.qa_flags:
        c -= 0.15
    return round(min(1.0, max(0.05, c)), 3)


def _plan_confidence(rooms, best, disagreement, text, qa) -> float:
    if not rooms:
        return 0.0
    labelled = sum(1 for r in rooms if r["label"]) / len(rooms)
    c = 0.15 + 0.35 * labelled
    if best:
        c += {"printed_dimensions": 0.3, "printed_area": 0.22,
              "scale_ratio": 0.12, "stated_area": 0.15}.get(best.source, 0.0)
    if disagreement is not None:
        c += 0.1 if disagreement < 8 else (-0.1 if disagreement > 20 else 0.0)
    if text.not_to_scale:
        c -= 0.05
    for f in ("no_rooms_found", "unclaimed_interior_area", "footprint_not_enclosed",
              "plan_area_disagrees_with_stated"):
        if f in qa:
            c -= 0.12
    return round(min(1.0, max(0.02, c)), 3)


def pick_plan_asset(ctx: StageContext) -> Path | None:
    """The largest plan asset wins.

    Phase 0: nearly half of plans carry a "not to scale" disclaimer and dimension
    extraction is strongly resolution-dependent, so resolution is the single knob
    worth caring about at selection time. Stage 0's manifest is preferred over the
    portal's own metadata, which Phase 0 measured as unreliable.
    """
    manifest = ctx.read("0-triage")
    candidates: list[Path] = []
    if manifest:
        for p in manifest.get("plans", []):
            candidates.append(ctx.media_path(p["path"]))
    if not candidates:
        for ref in ctx.listing.get("floorplans", []):
            if ref.get("local_path"):
                candidates.append(ctx.media_path(ref["local_path"]))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    from PIL import Image

    def side(p: Path) -> int:
        try:
            with Image.open(p) as im:
                return max(im.size)
        except Exception:  # noqa: BLE001
            return 0
    return max(candidates, key=side)


@register_stage("5-plan", description="Vectorise the floor plan: rooms, adjacency, doors, scale")
def run(ctx: StageContext) -> StageResult:
    path = pick_plan_asset(ctx)
    if path is None:
        return StageResult(payload={}, skipped=True,
                           skip_reason="listing has no floor-plan asset (Tier B, Phase 3)")
    engine = ctx.profile.engine("5-plan", "raster")
    payload = build_plan(path, ctx.listing, engine=engine,
                         override=(ctx.options.get("overrides") or {}).get("plan"))
    return StageResult(payload=payload, engine=engine)
