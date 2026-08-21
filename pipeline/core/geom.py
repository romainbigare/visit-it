"""2D geometry helpers shared by stages 4-9.

Polygons are lists of ``(x, y)`` in metres, counter-clockwise, first vertex not
repeated. Everything here is pure and side-effect free so it can be unit-tested
without a pipeline around it.
"""
from __future__ import annotations

import math

import numpy as np

Poly = list[tuple[float, float]] | list[list[float]]


def as_array(poly: Poly) -> np.ndarray:
    a = np.asarray(poly, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"polygon must be Nx2, got {a.shape}")
    return a


def signed_area(poly: Poly) -> float:
    """Shoelace. Positive when counter-clockwise."""
    a = as_array(poly)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def area(poly: Poly) -> float:
    return abs(signed_area(poly))


def ensure_ccw(poly: Poly) -> list[list[float]]:
    a = as_array(poly)
    return (a if signed_area(a) >= 0 else a[::-1]).tolist()


def centroid(poly: Poly) -> tuple[float, float]:
    """Area centroid, not vertex mean — an L-shaped room's vertex mean can sit outside it."""
    a = as_array(poly)
    s = signed_area(a)
    if abs(s) < 1e-12:
        return tuple(a.mean(axis=0))
    x, y = a[:, 0], a[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    return (float(np.dot(x + xn, cross) / (6 * s)), float(np.dot(y + yn, cross) / (6 * s)))


def perimeter(poly: Poly) -> float:
    a = as_array(poly)
    return float(np.linalg.norm(np.roll(a, -1, axis=0) - a, axis=1).sum())


def bounds(poly: Poly) -> tuple[float, float, float, float]:
    a = as_array(poly)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def extent(poly: Poly) -> tuple[float, float]:
    minx, miny, maxx, maxy = bounds(poly)
    return maxx - minx, maxy - miny


def oriented_extent(poly: Poly) -> tuple[float, float, float]:
    """Minimum-area bounding box: ``(long_side, short_side, angle_deg)``.

    Axis-aligned extent lies about a room whose walls run at 30 degrees, and
    aspect ratio is one of the assembly cost terms, so it has to be honest.
    """
    a = as_array(poly)
    best = None
    edges = np.roll(a, -1, axis=0) - a
    for ex, ey in edges:
        n = math.hypot(ex, ey)
        if n < 1e-9:
            continue
        c, s = ex / n, ey / n
        rot = a @ np.array([[c, -s], [s, c]])
        w = rot[:, 0].max() - rot[:, 0].min()
        h = rot[:, 1].max() - rot[:, 1].min()
        if best is None or w * h < best[0]:
            best = (w * h, w, h, math.degrees(math.atan2(s, c)))
    if best is None:
        w, h = extent(a)
        return max(w, h), min(w, h), 0.0
    _, w, h, ang = best
    return (max(w, h), min(w, h), ang % 180.0)


def aspect_ratio(poly: Poly) -> float:
    """Long/short of the oriented box. Always >= 1."""
    lo, sh, _ = oriented_extent(poly)
    return lo / sh if sh > 1e-9 else float("inf")


def _shapely(poly: Poly):
    from shapely.geometry import Polygon
    p = Polygon(as_array(poly))
    return p if p.is_valid else p.buffer(0)


def iou(a: Poly, b: Poly) -> float:
    """Intersection over union. 0 when either is degenerate."""
    pa, pb = _shapely(a), _shapely(b)
    if pa.is_empty or pb.is_empty:
        return 0.0
    union = pa.union(pb).area
    return float(pa.intersection(pb).area / union) if union > 1e-12 else 0.0


def union_area(polys: list[Poly]) -> float:
    from shapely.ops import unary_union
    if not polys:
        return 0.0
    return float(unary_union([_shapely(p) for p in polys]).area)


def union_polygon(polys: list[Poly]) -> list[list[float]] | None:
    """Outer ring of the union, for the minimap footprint. None if it is not one piece."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if not polys:
        return None
    u = unary_union([_shapely(p) for p in polys])
    if isinstance(u, Polygon):
        return ensure_ccw(list(u.exterior.coords)[:-1])
    parts = sorted(u.geoms, key=lambda g: g.area, reverse=True)
    return ensure_ccw(list(parts[0].exterior.coords)[:-1]) if parts else None


def shared_edge_length(a: Poly, b: Poly, tol: float = 0.12) -> float:
    """How much wall two rooms share. Rooms drawn as adjacent rarely touch exactly,
    so we dilate one by ``tol`` and measure the overlap of the boundaries."""
    pa, pb = _shapely(a), _shapely(b)
    if pa.is_empty or pb.is_empty:
        return 0.0
    strip = pa.boundary.buffer(tol).intersection(pb.boundary)
    return float(strip.length) if not strip.is_empty else 0.0


def se2(poly: Poly, tx: float, ty: float, theta_deg: float,
        about: tuple[float, float] | None = None, flip: bool = False) -> list[list[float]]:
    """Rotate about ``about`` (default the polygon's own centroid), then translate."""
    a = as_array(poly).copy()
    cx, cy = about if about is not None else centroid(a)
    a[:, 0] -= cx
    a[:, 1] -= cy
    if flip:
        a[:, 0] = -a[:, 0]
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    r = a @ np.array([[c, s], [-s, c]])
    r[:, 0] += cx + tx
    r[:, 1] += cy + ty
    return r.tolist()


def place_at(poly: Poly, target_centroid: tuple[float, float], theta_deg: float,
             flip: bool = False) -> list[list[float]]:
    """Rotate a room about its own centroid and move that centroid onto a target."""
    cx, cy = centroid(poly)
    return se2(poly, target_centroid[0] - cx, target_centroid[1] - cy, theta_deg,
               about=(cx, cy), flip=flip)


def scale_polygon(poly: Poly, s: float, about: tuple[float, float] | None = None) -> list[list[float]]:
    a = as_array(poly).copy()
    cx, cy = about if about is not None else centroid(a)
    a[:, 0] = cx + (a[:, 0] - cx) * s
    a[:, 1] = cy + (a[:, 1] - cy) * s
    return a.tolist()


def simplify(poly: Poly, tol: float = 0.05) -> list[list[float]]:
    """Douglas-Peucker. Keeps polygons small enough for the <=1 MB shell budget."""
    from shapely.geometry import Polygon
    p = _shapely(poly).simplify(tol, preserve_topology=True)
    if not isinstance(p, Polygon) or p.is_empty or len(p.exterior.coords) < 4:
        return ensure_ccw(poly)
    return ensure_ccw(list(p.exterior.coords)[:-1])


def rectangle(w: float, h: float, cx: float = 0.0, cy: float = 0.0) -> list[list[float]]:
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


def point_in_polygon(pt: tuple[float, float], poly: Poly) -> bool:
    x, y = pt
    a = as_array(poly)
    inside = False
    n = len(a)
    for i in range(n):
        x1, y1 = a[i]
        x2, y2 = a[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def representative_point(poly: Poly) -> tuple[float, float]:
    """A point guaranteed inside the polygon — the centroid is not, for an L-shape.
    Waypoints use this; a waypoint in a wall is a bug you only see in the viewer."""
    c = centroid(poly)
    if point_in_polygon(c, poly):
        return c
    p = _shapely(poly).representative_point()
    return (float(p.x), float(p.y))


def wrap_deg(a: float, period: float = 360.0) -> float:
    """Wrap to (-period/2, period/2]. Used for wall-direction differences."""
    return (a + period / 2) % period - period / 2
