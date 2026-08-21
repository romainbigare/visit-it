"""Stage 3 — per-room geometry.

Runs the bound engine over each room group and stores the point map as a binary
blob, keeping only summary statistics in the JSON. A point map is a few megabytes
and the artifact index has to stay readable.

Phase 1 is the monocular era (ROADMAP S3): one representative photo per room via
MoGe-2, which predicts intrinsics. Sprint 8 swaps in MapAnything over 3-8 views
per group without changing this artifact's shape.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np

from ..core.stages import StageContext, StageResult, register_stage
from .engines import EngineUnavailable, resolve

log = logging.getLogger("room_geometry.stage")


def _groups_from(ctx: StageContext) -> list[dict]:
    groups = ctx.read("2-grouping")
    if groups:
        return groups["groups"]
    return []


@register_stage("3-geometry", description="Per-room point maps and poses from the bound engine")
def run(ctx: StageContext) -> StageResult:
    preferred = ctx.profile.engine("3-geometry", "moge2")
    try:
        engine, chain = resolve(preferred)
    except EngineUnavailable as e:
        return StageResult(payload={}, skipped=True, skip_reason=str(e))

    manifest = ctx.require("0-triage")
    by_id = {im["image_id"]: im for im in manifest["images"]}
    groups = _groups_from(ctx)
    if not groups:
        return StageResult(payload={}, skipped=True, skip_reason="stage 2 produced no room groups")

    max_rooms = int(ctx.options.get("max_rooms", 0)) or None
    rooms: list[dict] = []
    blobs: dict[str, bytes] = {}
    qa: list[str] = []
    if chain != "preferred":
        qa.append("geometry_engine_fallback")

    for g in groups[:max_rooms]:
        paths = [ctx.media_path(by_id[i]["path"]) for i in g["image_ids"] if i in by_id]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        t0 = time.perf_counter()
        try:
            rec = engine.reconstruct(paths)
        except Exception as e:  # noqa: BLE001
            log.warning("%s %s: %s", ctx.listing_id, g["group_id"], e)
            qa.append("room_reconstruction_failed")
            continue
        secs = time.perf_counter() - t0
        buf = io.BytesIO()
        np.savez_compressed(buf, points=rec.points.astype(np.float32),
                            confidence=rec.confidence.astype(np.float32),
                            up_prior=(rec.up_prior if rec.up_prior is not None
                                      else np.zeros(3, dtype=np.float32)))
        name = f"rooms/{g['group_id']}/geometry.npz"
        blobs[name] = buf.getvalue()
        rooms.append({
            "room_id": g["group_id"],
            "room_label": g.get("room_label"),
            "engine": rec.engine,
            "n_views": rec.n_views,
            "n_points": int(len(rec.points)),
            "fov_x_deg": round(rec.fov_x_deg, 2) if rec.fov_x_deg else None,
            "fov_y_deg": round(rec.fov_y_deg, 2) if rec.fov_y_deg else None,
            "metric": rec.metric,
            "geometry_uri": name,
            "image_ids": g["image_ids"],
            "seconds": round(secs, 3),
            "notes": rec.notes,
        })

    fovs = [r["fov_x_deg"] for r in rooms if r["fov_x_deg"]]
    if fovs and float(np.median(fovs)) < 70:
        # Phase 0 measured a median of 98.6 degrees on real listing photos. A much
        # narrower estimate usually means the image was already perspective-
        # corrected by the agent, which breaks the metric chain (report §4.2).
        qa.append("narrow_fov_unusual_for_listing_photos")

    payload = {
        "schema": "geometry/v1",
        "listing_id": ctx.listing_id,
        "engine": engine.name,
        "engine_chain": chain,
        "rooms": rooms,
        "median_fov_x_deg": round(float(np.median(fovs)), 2) if fovs else None,
        "confidence": round(min(1.0, 0.3 + 0.1 * len(rooms)), 3) if rooms else 0.0,
        "qa_flags": sorted(set(qa)),
    }
    return StageResult(payload=payload, binaries=blobs, engine=engine.name)
