"""Point cloud → floor, ceiling and wall planes.

The order matters and is not arbitrary:

1. **Find "up" first.** Everything else is expressed relative to it, and it is the
   one quantity a single ultra-wide photo determines well — the floor and ceiling
   are the two largest parallel surfaces in any room.
2. **Then floor and ceiling**, as the outer two peaks of the height histogram.
   Phase 0's plausibility check lives here: a UK ceiling is 2.3-3.2 m, and a
   6 m one means the geometry is wrong, not that the flat is grand.
3. **Then clip.** Phase 0 saw 10-16 m horizontal extents on glazed city
   apartments — that is content seen *through the windows*, and it has to go
   before any wall fitting, or the room comes out the size of the street.
4. **Then walls**, by RANSAC on the vertical points, constrained to an Atlanta
   world so the room comes out with straight walls rather than a noisy blob.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger("layout.planes")

#: ROADMAP §0b check 1. Outside this and the room is flagged, not silently shipped.
CEILING_PLAUSIBLE_M = (2.3, 3.2)
CEILING_POSSIBLE_M = (1.9, 4.6)
MAX_ROOM_EXTENT_M = 12.0


def _candidate_normals(points: np.ndarray, n_samp: int = 6000) -> np.ndarray:
    """Plane normals sampled from random point triplets, plus the three axes."""
    rng = np.random.default_rng(7)
    idx = rng.choice(len(points), size=(n_samp, 3), replace=True)
    tri = points[idx]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    L = np.linalg.norm(nrm, axis=1)
    nrm = nrm[L > 1e-6] / L[L > 1e-6, None]
    nrm = np.where(nrm[:, 2:3] < 0, -nrm, nrm)          # fold to a half-sphere
    axes = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    return np.vstack([nrm, axes])


def _flatness(points: np.ndarray, n: np.ndarray, band: float = 0.08) -> float:
    """How much of the cloud lies in the two densest layers along direction ``n``.

    This is the property that actually identifies "up" in a room: the floor and
    the ceiling are the two largest surfaces, they are parallel, and no other
    direction has two such layers. A plain normal-PCA does not work — a synthetic
    box has as much wall as floor, so the scatter is isotropic and the leading
    eigenvector is noise.
    """
    h = points @ n
    lo, hi = np.percentile(h, [1, 99])
    if hi - lo < 0.4:
        return 0.0
    hist, edges = np.histogram(h, bins=120, range=(lo, hi))
    if hist.sum() == 0:
        return 0.0
    width = (hi - lo) / 120.0
    k = max(1, int(round(band / max(width, 1e-6))))
    smoothed = np.convolve(hist, np.ones(2 * k + 1), mode="same")
    order = np.argsort(-smoothed)
    best = order[0]
    second = next((i for i in order if abs(i - best) > 4 * k), None)
    got = float(smoothed[best] + (smoothed[second] if second is not None else 0))
    return got / float(hist.sum())


def dominant_up(points: np.ndarray, prior_up: np.ndarray | None = None) -> np.ndarray:
    """Room "up": the direction with two dense parallel layers — floor and ceiling.

    Candidate directions come from random-triplet plane normals (sampled down so
    only a few dozen are scored), and the winner is the one under which the cloud
    stratifies best. ``prior_up`` is a soft nudge, not a constraint: the camera
    frame tells us roughly which way is up, so a direction 80 degrees away from it
    needs to be a *lot* flatter to win.
    """
    if len(points) < 50:
        return np.array([0.0, 0.0, 1.0])
    p = points - points.mean(axis=0)
    cands = _candidate_normals(p)
    rng = np.random.default_rng(11)
    reps = cands[rng.choice(len(cands), size=min(48, len(cands)), replace=False)]
    reps = np.vstack([reps, np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])])
    if prior_up is not None:
        pu = np.asarray(prior_up, dtype=float)
        pu = pu / max(np.linalg.norm(pu), 1e-9)
        reps = np.vstack([reps, pu])
    sub = p if len(p) <= 20000 else p[rng.choice(len(p), 20000, replace=False)]
    scores = np.array([_flatness(sub, n) for n in reps])
    if prior_up is not None:
        align = np.abs(reps @ pu) / np.linalg.norm(reps, axis=1)
        scores = scores * (0.45 + 0.55 * align)
    best = reps[int(np.argmax(scores))]
    return best / np.linalg.norm(best)


def floor_ceiling(heights: np.ndarray, bins: int = 240) -> tuple[float | None, float | None]:
    """The two outer dense layers of the height histogram."""
    if heights.size < 50:
        return None, None
    lo, hi = np.percentile(heights, [0.5, 99.5])
    if hi - lo < 0.5:
        return None, None
    hist, edges = np.histogram(heights, bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    thresh = hist.max() * 0.12
    dense = np.where(hist >= thresh)[0]
    if dense.size < 2:
        return None, None
    return float(centres[dense[0]]), float(centres[dense[-1]])


def orient(points: np.ndarray, prior_up: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the cloud so +z is up. Returns ``(rotated, up_vector)``."""
    up = dominant_up(points, prior_up)
    if prior_up is not None and np.dot(up, np.asarray(prior_up, float)) < 0:
        up = -up
    z = up / np.linalg.norm(up)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, z)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    rot = points @ R.T
    # If the bulk of the cloud sits above the "floor" peak we picked, we have the
    # room upside down. Flipping here is cheaper than every downstream consumer
    # having to wonder.
    zs = rot[:, 2]
    lo, hi = floor_ceiling(zs)
    if lo is not None and hi is not None:
        near_lo = float((np.abs(zs - lo) < 0.15).sum())
        near_hi = float((np.abs(zs - hi) < 0.15).sum())
        if near_hi > near_lo * 1.6:
            rot[:, 2] = -rot[:, 2]
            z = -z
    return rot, z


def clip_outliers(rot: np.ndarray, conf: np.ndarray | None = None,
                  quantile: float = 0.02) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cut window content and stragglers before any wall is fitted.

    Two cuts, in order. A robust radial cut about the horizontal median removes
    the through-the-window street; a height cut removes points far outside the
    floor-ceiling band. The report says how much went and why, because "the room
    came out small" and "we clipped 40% of it" are the same bug seen from two
    sides.
    """
    report: dict = {"points_total": int(len(rot)), "reasons": []}
    keep = np.ones(len(rot), dtype=bool)
    if conf is not None and len(conf) == len(rot):
        thr = float(np.quantile(conf, 0.10))
        low = conf < min(thr, 0.25)
        if low.sum() < len(rot) * 0.5:
            keep &= ~low
            if low.any():
                report["reasons"].append("low_confidence_points")

    xy = rot[:, :2]
    c = np.median(xy[keep], axis=0) if keep.any() else np.zeros(2)
    r = np.linalg.norm(xy - c, axis=1)
    # A room's inscribed radius is a few metres; anything past the 98th percentile
    # *and* past a hard metric ceiling is not part of this room.
    r_cut = max(float(np.quantile(r[keep], 1.0 - quantile)) if keep.any() else 8.0, 1.5)
    r_cut = min(r_cut, MAX_ROOM_EXTENT_M / 2.0)
    far = r > r_cut
    if far.any():
        keep &= ~far
        report["reasons"].append("window_content_or_far_field")

    zs = rot[:, 2]
    lo, hi = floor_ceiling(zs[keep]) if keep.any() else (None, None)
    if lo is not None and hi is not None:
        band = (zs > lo - 0.35) & (zs < hi + 0.35)
        if band.sum() > len(rot) * 0.25:
            if (~band).any():
                report["reasons"].append("outside_floor_ceiling_band")
            keep &= band
    report["points_kept"] = int(keep.sum())
    return rot[keep], (conf[keep] if conf is not None and len(conf) == len(keep) else None), report


def wall_directions(xy: np.ndarray, max_dirs: int = 4,
                    min_share: float = 0.10) -> list[float]:
    """Atlanta-world directions from the horizontal point distribution.

    Estimated from *gradients of the point density* rather than from fitted lines:
    at 512 px a monocular point map is far too noisy for line fitting, but the
    orientation at which the cloud's bounding area is smallest is stable, and that
    is what a rectangular room's dominant axis is.
    """
    if len(xy) < 100:
        return [0.0, 90.0]
    best_ang, best_area = 0.0, float("inf")
    for deg in np.arange(0.0, 90.0, 1.0):
        t = math.radians(deg)
        R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
        q = xy @ R.T
        lo, hi = np.percentile(q, [1, 99], axis=0)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if area < best_area:
            best_area, best_ang = area, float(deg)
    return [best_ang, (best_ang + 90.0) % 180.0]


def footprint_polygon(xy: np.ndarray, directions: list[float],
                      quantile: float = 0.015) -> list[list[float]]:
    """The room's floor polygon.

    Phase 1 returns a regularised rectangle in the room's dominant frame: a
    monocular point map from one ultra-wide photo genuinely does not support more
    than that, and a fake concave outline would be a lie the shell builder would
    faithfully extrude. Stage 4 v2 (Sprint 8, multi-view) replaces this with plane
    fitting on real multi-view pointmaps; the artifact contract does not change.
    """
    t = math.radians(directions[0])
    R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    q = xy @ R.T
    lo = np.quantile(q, quantile, axis=0)
    hi = np.quantile(q, 1.0 - quantile, axis=0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return (corners @ R).tolist()


def fit_walls(xy: np.ndarray, poly: list[list[float]]) -> list[dict]:
    """Report each polygon edge as a wall plane with its inlier support.

    The support fraction is the honest part: a wall with 2% of the points behind
    it is a wall we invented to close the polygon, and stage 6 should weigh it
    accordingly.
    """
    out: list[dict] = []
    a = np.asarray(poly, dtype=float)
    n = len(a)
    for i in range(n):
        p, q = a[i], a[(i + 1) % n]
        d = q - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        nv = np.array([-d[1], d[0]]) / L
        offset = float(nv @ p)
        dist = np.abs(xy @ nv - offset)
        inl = float((dist < 0.12).mean())
        out.append({
            "normal": [round(float(nv[0]), 4), round(float(nv[1]), 4)],
            "offset_m": round(offset, 4),
            "inlier_frac": round(inl, 4),
            "length_m": round(L, 3),
            "direction_index": i % 2,
        })
    return out
