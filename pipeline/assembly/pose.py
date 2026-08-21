"""Placing a reconstructed room onto its plan polygon: the SE(2) refinement.

The plan supplies absolute position; the reconstruction supplies shape. So the
pose we solve for is small: which way round the room goes, and a nudge. We search
orientation coarsely and translation finely, scoring by IoU, because IoU is the
thing G1's arrangement criterion is judged on and optimising a proxy for it would
be a choice we would have to defend later.
"""
from __future__ import annotations

import numpy as np

from ..core import geom


def refine(room_poly, plan_poly, *, allow_flip: bool = True,
           angles: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
           fine_deg: float = 6.0, iters: int = 2) -> tuple[dict, float]:
    """Best ``(pose, iou)`` placing ``room_poly`` inside ``plan_poly``.

    Coarse orientations first (a room is placed the right way round or it is not —
    there is no useful gradient between 0 and 90 degrees), then a local refinement
    of angle and offset. Mirroring is allowed because a monocular reconstruction
    has a handedness the photograph does not determine.
    """
    target = geom.centroid(plan_poly)
    base_ang = geom.oriented_extent(plan_poly)[2] - geom.oriented_extent(room_poly)[2]
    best = ({"tx_m": 0.0, "ty_m": 0.0, "theta_deg": 0.0, "flip": False}, -1.0)
    for flip in ((False, True) if allow_flip else (False,)):
        for a in angles:
            theta = base_ang + a
            placed = geom.place_at(room_poly, target, theta, flip=flip)
            score = geom.iou(placed, plan_poly)
            if score > best[1]:
                cx, cy = geom.centroid(room_poly)
                best = ({"tx_m": round(target[0] - cx, 4), "ty_m": round(target[1] - cy, 4),
                         "theta_deg": round(theta % 360.0, 3), "flip": flip}, score)

    pose, score = best
    step = max(0.05, 0.04 * max(geom.extent(plan_poly)))
    for _ in range(iters):
        improved = False
        for dth in (-fine_deg, 0.0, fine_deg):
            for dx, dy in ((0, 0), (step, 0), (-step, 0), (0, step), (0, -step)):
                cand = {**pose, "theta_deg": (pose["theta_deg"] + dth) % 360.0,
                        "tx_m": pose["tx_m"] + dx, "ty_m": pose["ty_m"] + dy}
                placed = _apply(room_poly, cand)
                s = geom.iou(placed, plan_poly)
                if s > score + 1e-6:
                    pose, score, improved = cand, s, True
        if not improved:
            fine_deg /= 2.0
            step /= 2.0
    pose = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pose.items()}
    return pose, round(float(score), 4)


def _apply(room_poly, pose: dict):
    cx, cy = geom.centroid(room_poly)
    return geom.se2(room_poly, pose["tx_m"], pose["ty_m"], pose["theta_deg"],
                    about=(cx, cy), flip=pose["flip"])


def apply_pose(room_poly, pose: dict):
    """Public form of the transform, so the shell builder and the viewer agree."""
    return _apply(room_poly, pose)
