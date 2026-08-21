"""Stage 1 — conditioning and intrinsics.

**Scope note.** The full stage 1 is a Sprint 7 deliverable: exposure and
white-balance alignment within a room group, and the composite/perspective-
correction detector built on GeoCalib-versus-pointmap disagreement (report §4.2).
Phase 1's monocular era does not need photometric alignment — there is one photo
per room — and the intrinsics come from MoGe-2 inside stage 3, which is the whole
reason AD-5 picked a model that predicts them.

So what this stage does in Phase 1 is the part that is load-bearing now: record
each usable image's geometry, apply a **field-of-view prior measured on our own
data** (Phase 0: median 98.6 degrees horizontal), and raise the perspective-
correction flag where an image's aspect ratio or metadata says an agent has
already rectified it — because a rectified photo silently breaks the metric chain.
"""
from __future__ import annotations

import logging

from ..core.stages import StageContext, StageResult, register_stage

log = logging.getLogger("conditioning")

#: Phase 0, measured on our own golden set with MoGe-2: median 98.6 degrees on
#: CPU and 98.5 on a T4. Estate agents shoot ultra-wide, and assuming otherwise
#: is what put Depth Anything's ceilings at 6 m.
FOV_PRIOR_DEG = 98.6
FOV_PLAUSIBLE_DEG = (55.0, 130.0)


@register_stage("1-conditioning", description="Per-image intrinsics prior and the rectification gate")
def run(ctx: StageContext) -> StageResult:
    manifest = ctx.require("0-triage")
    images, qa = [], []
    for im in manifest["images"]:
        if im["type"] != "interior":
            continue
        if any(f.startswith("duplicate_of_") or f == "blank_image" for f in im["quality_flags"]):
            continue
        w, h = im.get("width"), im.get("height")
        flags = []
        if w and h:
            ar = w / h
            # A 1:1 or portrait interior shot from an agency is nearly always a
            # crop or a composite rather than a camera frame.
            if ar < 1.05:
                flags.append("unusual_aspect_ratio")
        fov_y = None
        if w and h and w >= h:
            import math
            fov_y = round(math.degrees(
                2 * math.atan(math.tan(math.radians(FOV_PRIOR_DEG / 2)) * h / w)), 2)
        images.append({
            "image_id": im["image_id"],
            "fov_x_deg": FOV_PRIOR_DEG,
            "fov_y_deg": fov_y,
            "intrinsics": None,
            "engine": "phase0_fov_prior",
            "agreement": None,
            "quality_flags": flags,
        })
    if not images:
        qa.append("no_usable_interior_images")
    if sum(1 for i in images if "unusual_aspect_ratio" in i["quality_flags"]) > len(images) / 2:
        qa.append("most_images_perspective_corrected")
    return StageResult(payload={
        "schema": "calibration/v1",
        "listing_id": ctx.listing_id,
        "images": images,
        "confidence": 0.4 if images else 0.0,   # a prior, not a measurement
        "qa_flags": sorted(set(qa)),
    }, engine="phase0_fov_prior")
