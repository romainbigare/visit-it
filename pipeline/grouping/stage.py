"""Stage 2 — room groups.

**Scope note.** The full stage 2 is Sprint 7: DINOv2/MegaLoc retrieval, then
LightGlue+ALIKED verification, then graph clustering, with VLM adjudication of
low-confidence pairs and a measured pairwise F1 against hand-grouped golden data
(M6). Phase 1 is the monocular era — stage 3 consumes one view per room — so what
is needed here is exactly what Phase 0's `eval/models/grouping.py` did: build
*high-precision* groups from triage labels, accepting low recall.

The precision-over-recall choice is deliberate. A wrong group in the monocular era
costs one bad room polygon; in Phase 2 it will cost a splat trained on two
different rooms. Getting into the habit now is free.
"""
from __future__ import annotations

import logging

from ..core.stages import StageContext, StageResult, register_stage

log = logging.getLogger("grouping")

#: A flat has at most one of these, whatever the photo count says.
ALWAYS_SINGLETON = ("kitchen",)
#: ...and at most one of these when the flat is small enough.
SINGLETON_IF_BEDROOMS_UPTO = {"bathroom": 2, "living_room": 2, "dining_room": 3,
                              "hallway": 3}
MAX_VIEWS = 8


@register_stage("2-grouping", description="Group photos into rooms (label-based, high precision)")
def run(ctx: StageContext) -> StageResult:
    manifest = ctx.require("0-triage")
    calib = ctx.read("1-conditioning")
    usable = {i["image_id"] for i in (calib or {}).get("images", [])} or None

    bedrooms = manifest["listing"].get("bedrooms") or 2
    by_label: dict[str, list[dict]] = {}
    for im in manifest["images"]:
        if im["type"] != "interior" or not im.get("room_label"):
            continue
        if usable is not None and im["image_id"] not in usable:
            continue
        by_label.setdefault(im["room_label"], []).append(im)

    groups, log_entries, qa = [], [], []
    for label, items in sorted(by_label.items()):
        # SigLIP's absolute scores are not calibrated (Phase 0: a 0.02 threshold
        # cut yield from 26 groups to 2), so we rank by them but never threshold.
        items.sort(key=lambda im: -(im.get("room_confidence") or 0.0))
        singleton = label in ALWAYS_SINGLETON or \
            bedrooms <= SINGLETON_IF_BEDROOMS_UPTO.get(label, -1)
        if singleton:
            chosen = [[im for im in items[:MAX_VIEWS]]]
        else:
            # Several rooms share this label and we cannot tell them apart without
            # the Sprint 7 matcher. One room per photo is the high-precision
            # choice: never merges two different bedrooms.
            chosen = [[im] for im in items[:max(1, bedrooms + 1)]]
            if len(items) > 1:
                log_entries.append({
                    "label": label, "n_images": len(items),
                    "decision": "split_per_image",
                    "reason": ("multiple rooms may share this label and pairwise "
                               "verification is a Sprint 7 deliverable"),
                })
        for k, members in enumerate(chosen):
            if not members:
                continue
            gid = f"{label}" if len(chosen) == 1 else f"{label}_{k}"
            groups.append({
                "group_id": gid,
                "room_label": label,
                "image_ids": [m["image_id"] for m in members],
                "confidence": round(0.7 if singleton else 0.45, 3),
                "singleton": len(members) == 1,
            })
    if not groups:
        qa.append("no_room_groups")
    if any(g["confidence"] < 0.5 for g in groups):
        qa.append("label_only_grouping")
    return StageResult(payload={
        "schema": "groups/v1",
        "listing_id": ctx.listing_id,
        "groups": groups,
        "adjudication_log": log_entries,
        "confidence": round(sum(g["confidence"] for g in groups) / len(groups), 3) if groups else 0.0,
        "qa_flags": sorted(set(qa)),
    }, engine="label_grouping")
