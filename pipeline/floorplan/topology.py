"""Adjacency, doors and windows, read off the vectorised plan.

Stage 6 matches reconstructed rooms to plan polygons partly on aperture counts,
and stage 7 wants door heights as a scale constraint. Both need to know where the
openings are, so this module finds them the same way the preprocessor found the
building: the gaps a wall would have if the wall were continuous.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..core import geom

log = logging.getLogger("floorplan.topology")

#: A UK internal door leaf is 0.686-0.926 m; add frame and drawing slop.
DOOR_M = (0.55, 1.60)
#: Windows on a plan run from a small casement to a whole glazed wall.
WINDOW_M = (0.35, 6.00)


@dataclass
class Aperture:
    aperture_id: str
    kind: str                       # door | window | opening
    position_px: tuple[float, float]
    width_px: float
    rooms: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class Adjacency:
    a: str
    b: str
    shared_wall_px: float
    door: bool = False
    door_position_px: tuple[float, float] | None = None
    confidence: float = 0.5


def adjacency(rooms, tol_px: float) -> list[Adjacency]:
    """Which rooms share a wall, and how much of one.

    Uses a dilated-boundary intersection rather than exact touching: vectorised
    polygons never share an edge to the pixel, and requiring them to would report
    a flat with no adjacencies at all.
    """
    out: list[Adjacency] = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            shared = geom.shared_edge_length(a.polygon_px, b.polygon_px, tol=tol_px)
            if shared < tol_px * 2:
                continue
            out.append(Adjacency(a.room_id, b.room_id, shared,
                                 confidence=min(1.0, shared / (tol_px * 12))))
    return out


def find_apertures(labels: np.ndarray, rooms, px_per_m: float | None,
                   exterior: int = 1) -> list[Aperture]:
    """Openings are where two watershed basins touch.

    The watershed ran over the *ink-free* space, so a basin stops dead at a wall.
    Two room basins can therefore only touch where the wall between them has a
    hole — a doorway. A room basin touching the *exterior* basin means a hole in
    an outside wall: a window, or the front door. This is a much sharper
    definition than "gaps a closing kernel had to bridge", and it needs no wall
    thickness tuning at all.
    """
    if labels is None or not (labels > exterior).any():
        return []
    room_of = {r.mask_label: r.room_id for r in rooms}
    out: list[Aperture] = []
    for (u, v), pts in _basin_contacts(labels):
        if len(pts) < 3:
            continue
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) + 1.0
        ids = sorted({room_of[t] for t in (u, v) if t in room_of})
        n_rooms = len(ids)
        if n_rooms == 0:
            continue
        if exterior not in (u, v) and n_rooms < 2:
            continue          # a contact between two basins we did not keep as rooms
        kind = _classify(span, n_rooms, px_per_m)
        if kind is None:
            continue
        cy, cx = pts.mean(axis=0)
        out.append(Aperture(f"a{len(out)}", kind, (float(cx), float(cy)), span, ids,
                            0.7 if n_rooms == 2 else 0.45))
    return out


def _basin_contacts(labels: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    """Contact patches between watershed basins, one entry per (pair, patch).

    Two rooms can be joined by a door *and* a wide archway; those are two
    apertures, so contacts are split into spatially connected patches rather than
    lumped together per pair.
    """
    h, w = labels.shape
    pair_map = np.zeros((h, w), dtype=np.int64)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys = slice(max(0, -dy), h - max(0, dy))
        xs = slice(max(0, -dx), w - max(0, dx))
        ys2 = slice(max(0, dy), h - max(0, -dy))
        xs2 = slice(max(0, dx), w - max(0, -dx))
        a, b = labels[ys, xs], labels[ys2, xs2]
        hit = (a > 0) & (b > 0) & (a != b)
        if not hit.any():
            continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        key = lo.astype(np.int64) * 100000 + hi.astype(np.int64)
        block = pair_map[ys, xs]
        np.copyto(block, key, where=hit & (block == 0))
        pair_map[ys, xs] = block
    if not pair_map.any():
        return []
    out: list[tuple[tuple[int, int], np.ndarray]] = []
    for key in np.unique(pair_map):
        if key == 0:
            continue
        m = (pair_map == key)
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        u, v = int(key // 100000), int(key % 100000)
        for c in range(1, n + 1):
            ys_, xs_ = np.nonzero(lab == c)
            out.append(((u, v), np.column_stack([ys_, xs_]).astype(float)))
    return out


def _classify(width_px: float, n_rooms: int, px_per_m: float | None) -> str | None:
    """Door, window or archway — by width once we have a scale, by topology before.

    Widths are checked in metres because that is the only way the numbers mean
    anything: a UK internal door leaf is 0.686-0.926 m, and a 3 m gap between two
    rooms is a knocked-through archway, not a door.
    """
    if px_per_m and px_per_m > 1:
        w = width_px / px_per_m
        if n_rooms >= 2:
            if DOOR_M[0] <= w <= DOOR_M[1]:
                return "door"
            return "opening" if w > DOOR_M[1] else None
        return "window" if WINDOW_M[0] <= w <= WINDOW_M[1] else None
    # No metric scale yet: fall back on topology alone. Confidence upstream is set
    # lower for these, and stage 7 re-runs the classification once scale is known.
    return "door" if n_rooms >= 2 else "window"


def attach_doors(adjs: list[Adjacency], apertures: list[Aperture]) -> list[Adjacency]:
    """Mark an adjacency as doored when a door aperture spans exactly its two rooms."""
    by_pair: dict[tuple[str, str], Aperture] = {}
    for ap in apertures:
        if ap.kind in ("door", "opening") and len(ap.rooms) == 2:
            key = tuple(sorted(ap.rooms))
            prev = by_pair.get(key)  # type: ignore[arg-type]
            if prev is None or ap.width_px > prev.width_px:
                by_pair[key] = ap  # type: ignore[index]
    for adj in adjs:
        ap = by_pair.get(tuple(sorted((adj.a, adj.b))))  # type: ignore[arg-type]
        if ap:
            adj.door = True
            adj.door_position_px = ap.position_px
            adj.confidence = max(adj.confidence, ap.confidence)
    return adjs
