"""Stage 4b — doors and windows found in the room's own geometry.

**What the roadmap asks for** (S4): "door/window aperture detection in photos v0
(SAM2 + Grounding DINO)". Those are the right models and they want a GPU and two
sets of weights. This is the v0 that needs neither, and it is not a stopgap so
much as the observation that *we already have the signal*:

a pointmap model returns nothing useful where it is looking through glass. A
window is a hole in the wall's point coverage at chest height; a door is a hole in
the same wall that reaches the floor. Both are found by projecting the room's
points onto each wall plane and looking for rectangular gaps in the coverage.

What this buys, in order:

1. **Stage 6** gets an aperture count per room, which is one of the assignment
   cost terms.
2. **Stage 7** gets door-height constraints — a UK door leaf is 1981 mm, and it is
   the only *length* constraint the photo channel can offer, which is what makes
   it independent of the plan's area constraints in the cross-check.
3. **Stage 8** cuts the openings, so a doorway is a doorway rather than a wall.

The detector is deliberately conservative: a missed door costs one constraint, an
invented one puts a hole in a wall and drags the scale solve.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("layout.apertures")

#: UK internal door leaf: 1981 x 762/838 mm. Plus frame, minus drawing slop.
DOOR_W_M = (0.60, 1.30)
DOOR_H_M = (1.75, 2.35)
#: Windows sit above a sill and stop below the ceiling.
WINDOW_SILL_M = (0.40, 1.40)
WINDOW_W_M = (0.35, 4.50)
WINDOW_H_M = (0.50, 2.40)

#: Cell size for the wall coverage grid. Fine enough to resolve a door, coarse
#: enough that a sparse point cloud does not look like Swiss cheese.
CELL_M = 0.10
#: A cell counts as "wall" if this many points land in it.
MIN_POINTS_PER_CELL = 2


@dataclass
class Aperture:
    kind: str                     # door | window
    centre_m: tuple[float, float, float]
    width_m: float
    height_m: float
    wall_index: int
    confidence: float

    def to_dict(self, idx: int) -> dict:
        return {"aperture_id": f"pa{idx}", "type": self.kind,
                "centre_m": [round(v, 4) for v in self.centre_m],
                "width_m": round(self.width_m, 3), "height_m": round(self.height_m, 3),
                "wall_index": self.wall_index, "confidence": round(self.confidence, 3),
                "source": "pointmap_wall_coverage"}


def _wall_frame(p: np.ndarray, q: np.ndarray):
    d = q - p
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    t = d / L
    n = np.array([-t[1], t[0]])
    return t, n, L


def detect(points: np.ndarray, polygon_m: list[list[float]], floor_h: float,
           ceiling_h: float, *, band_m: float = 0.30) -> list[Aperture]:
    """Find apertures on each wall of ``polygon_m``.

    ``points`` are in the room's oriented frame (z up), already clipped.
    """
    if len(points) < 400 or ceiling_h - floor_h < 1.6:
        return []
    poly = np.asarray(polygon_m, dtype=float)
    out: list[Aperture] = []
    height = ceiling_h - floor_h

    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fr = _wall_frame(p, q)
        if fr is None:
            continue
        t, n, L = fr
        if L < 0.8:
            continue
        rel = points[:, :2] - p
        along = rel @ t
        perp = rel @ n
        z = points[:, 2] - floor_h
        # Points close to this wall plane, inside its span, between floor and ceiling.
        near = (np.abs(perp) < band_m) & (along > 0.05) & (along < L - 0.05) & \
               (z > 0.05) & (z < height - 0.05)
        if near.sum() < 150:
            continue
        grid, nu, nv = _coverage(along[near], z[near], L, height)
        if grid is None:
            continue
        for kind, u0, u1, v0, v1, cov in _holes(grid, nu, nv, L, height):
            w = (u1 - u0)
            h = (v1 - v0)
            if kind == "door" and not (DOOR_W_M[0] <= w <= DOOR_W_M[1]
                                       and DOOR_H_M[0] <= h <= DOOR_H_M[1]):
                continue
            if kind == "window" and not (WINDOW_W_M[0] <= w <= WINDOW_W_M[1]
                                         and WINDOW_H_M[0] <= h <= WINDOW_H_M[1]):
                continue
            centre_uv = ((u0 + u1) / 2, (v0 + v1) / 2)
            xy = p + t * centre_uv[0]
            out.append(Aperture(kind, (float(xy[0]), float(xy[1]),
                                       float(floor_h + centre_uv[1])),
                                w, h, i, confidence=round(min(0.75, 0.25 + cov), 3)))
    return out


def _coverage(u: np.ndarray, v: np.ndarray, L: float, height: float):
    nu = max(4, int(round(L / CELL_M)))
    nv = max(4, int(round(height / CELL_M)))
    if nu * nv > 40000:
        return None, 0, 0
    hist, _, _ = np.histogram2d(u, v, bins=[nu, nv], range=[[0, L], [0, height]])
    return hist >= MIN_POINTS_PER_CELL, nu, nv


def _holes(grid: np.ndarray, nu: int, nv: int, L: float, height: float):
    """Maximal empty rectangles in the wall's coverage grid.

    Scanned column-run-wise rather than by a general maximal-rectangle algorithm:
    apertures are axis-aligned in the wall's own frame by construction, and the
    simple version does not invent L-shaped windows out of noise.
    """
    from scipy import ndimage as ndi
    empty = ~grid
    # A wall is never fully covered by a monocular point map, so require the wall
    # to be mostly seen before believing any hole in it.
    if grid.mean() < 0.35:
        return
    lab, n = ndi.label(empty)
    if n == 0:
        return
    du, dv = L / nu, height / nv
    for c in range(1, n + 1):
        us, vs = np.nonzero(lab == c)
        if len(us) < 6:
            continue
        u0, u1 = us.min() * du, (us.max() + 1) * du
        v0, v1 = vs.min() * dv, (vs.max() + 1) * dv
        # Only believe a hole that actually fills its own bounding box: a ragged
        # scatter of missing cells is sparse sampling, not an opening.
        fill = len(us) / max(1, (us.max() - us.min() + 1) * (vs.max() - vs.min() + 1))
        if fill < 0.72:
            continue
        touches_floor = vs.min() <= 1
        kind = "door" if touches_floor else "window"
        if kind == "window" and not (WINDOW_SILL_M[0] <= v0 <= WINDOW_SILL_M[1]):
            continue
        # Coverage of the wall around the hole, as a confidence proxy.
        cov = float(grid.mean())
        yield kind, u0, u1, v0, v1, cov
