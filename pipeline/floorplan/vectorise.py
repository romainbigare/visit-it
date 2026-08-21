"""Raster floor plan → room polygons.

**What this is.** The roadmap's plan for stage 5 is a RoomFormer-class network
trained on ResPlan + Swiss Dwellings + CubiCasa5K and fine-tuned on our own
annotated UK plans. That is the right long-term answer and it is what the
``learned`` engine binding is reserved for. This module is the ``raster`` engine:
a classical vectoriser that needs no GPU, no training set and no annotation drive
to start returning polygons, so stages 6-9 have something real to consume from
day one and so we have a measured baseline the learned model has to beat.

**How it works.** Three ideas, in order of how much they matter:

1. **The plan's own text seeds the segmentation.** UK agency plans label every
   room. A watershed seeded from the label captions puts exactly one basin where
   a human would point, and the basin boundary lands in the doorway neck, which
   is where the wall would be if the door were shut.
2. **Text is erased before the geometry runs.** OCR tells us exactly which ink is
   lettering, so we delete those components. Without this the free space is full
   of holes and the distance transform is meaningless.
3. **Unlabelled rooms get seeded from the distance transform.** Hallways and
   cupboards are often unlabelled; a peak of the free-space distance transform,
   far from every caption, is a room nobody named.

Everything the vectoriser is unsure about comes out as a QA flag rather than a
silent guess — the review console's overlay editor exists precisely so a human
can fix a wall in ten seconds instead of the model being right every time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from ..core import geom
from . import wallnet
from .ocr import PlanText, TextBlock
from .preprocess import PlanImage, count_outlines

log = logging.getLogger("floorplan.vectorise")

#: A basin smaller than this fraction of the total room area is a sliver, not a
#: room. Unlabelled basins have to clear a higher bar than captioned ones: a
#: caption is a human asserting a room exists, a distance peak is only a hint.
MIN_ROOM_FRAC = 0.004
MIN_UNLABELLED_ROOM_FRAC = 0.018
#: ...and one bigger than this is the vectoriser having failed to split anything.
MAX_ROOM_FRAC = 0.75
EXTERIOR = 1


@dataclass
class RoomRegion:
    room_id: str
    mask_label: int
    polygon_px: list[list[float]]
    area_px: float
    centroid_px: tuple[float, float]
    label: str | None = None
    label_text: str | None = None
    label_confidence: float = 0.0
    ocr_dims_m: tuple[float, float] | None = None
    ocr_area_m2: float | None = None
    seeded_by: str = "caption"          # caption | distance_peak
    confidence: float = 0.5
    qa_flags: list[str] = field(default_factory=list)


@dataclass
class Vectorisation:
    rooms: list[RoomRegion]
    labels: np.ndarray                  # H x W int32 watershed result
    free: np.ndarray                    # H x W bool, ink-free space used for the DT
    footprint: np.ndarray | None = None  # H x W bool, gross internal area
    inside: np.ndarray | None = None     # H x W bool, everything the building encloses
    n_outlines: int = 1
    text_erased: int = 0
    directions_deg: list[float] = field(default_factory=list)
    non_manhattan: bool = False
    qa_flags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# text erasure


def erase_text(ink: np.ndarray, text: PlanText) -> tuple[np.ndarray, int]:
    """Delete the ink components OCR identified as lettering.

    Bounded by the word box's own size so that a letter overlapping a wall does
    not take the wall with it — the check is "is this blob no bigger than the word
    that reported it", which a wall never is.
    """
    if not text.words:
        return ink, 0
    lab, n = ndi.label(ink)
    if n == 0:
        return ink, 0
    objs = ndi.find_objects(lab)
    img_area = float(ink.size)
    kill: set[int] = set()
    h, w = ink.shape
    for word in text.words:
        x0, y0 = max(0, int(word.x)), max(0, int(word.y))
        x1, y1 = min(w, int(word.x + word.w) + 1), min(h, int(word.y + word.h) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        for c in np.unique(lab[y0:y1, x0:x1]):
            if c == 0 or c in kill:
                continue
            sl = objs[c - 1]
            bh = sl[0].stop - sl[0].start
            bw = sl[1].stop - sl[1].start
            if bh <= word.h * 2.2 and bw <= word.w * 2.2 and bh * bw < img_area * 0.02:
                kill.add(int(c))
    if not kill:
        return ink, 0
    out = ink.copy()
    out[np.isin(lab, list(kill))] = 0
    return out, len(kill)


def structure_hull(clean_ink: np.ndarray) -> np.ndarray:
    """Convex hull of the big ink components — i.e. of the building.

    Used to reject captions printed *outside* the plan ("FIRST FLOOR",
    "APPROXIMATE GROSS INTERNAL AREA"), which would otherwise seed phantom rooms
    in the margin.
    """
    lab, n = ndi.label(clean_ink)
    if n == 0:
        return np.ones_like(clean_ink, dtype=np.uint8)
    sizes = ndi.sum(clean_ink, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes > sizes.max() * 0.05)[0])
    pts = np.column_stack(np.where(keep))[:, ::-1].astype(np.int32)
    if len(pts) < 3:
        return np.ones_like(clean_ink, dtype=np.uint8)
    hull = cv2.convexHull(pts)
    m = np.zeros(clean_ink.shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    return m


# --------------------------------------------------------------------------
# segmentation


def _caption_seeds(text: PlanText, hull: np.ndarray) -> list[TextBlock]:
    h, w = hull.shape
    out = []
    for b in text.blocks:
        if not b.is_room_caption:
            continue
        cx, cy = int(b.cx), int(b.cy)
        if not (0 <= cy < h and 0 <= cx < w and hull[cy, cx]):
            continue
        out.append(b)
    return out


def segment_learned(pi: PlanImage, text: PlanText) -> Vectorisation | None:
    """Rooms from the trained plan reader, when one is installed.

    Returns ``None`` if there is no checkpoint on this box or if it produced nothing
    usable, so the caller falls back to the classical engine rather than failing.
    Labels still come from OCR: the model finds *where* the rooms are, the plan's own
    text says *what* they are, and the text is the better source for that.
    """
    from . import learned
    if not learned.available():
        return None
    try:
        room_mask, wall_mask = learned.predict(pi.rgb)
    except Exception as e:  # noqa: BLE001
        log.warning("learned vectoriser failed, falling back to the classical engine: %s", e)
        return None
    masks = learned.rooms_from_prediction(room_mask)
    if not masks:
        return None

    labels = np.zeros(room_mask.shape, np.int32)
    labels[room_mask == 0] = EXTERIOR
    seeds: list[tuple[int, TextBlock | None, tuple[float, float]]] = []
    caption_at = {}
    for b in text.blocks:
        if b.is_room_caption:
            caption_at[(int(b.cy), int(b.cx))] = b
    for i, m in enumerate(masks):
        mid = EXTERIOR + 1 + i
        labels[m > 0] = mid
        # Attach the caption whose centre falls inside this room, if any.
        block = next((b for (y, x), b in caption_at.items()
                      if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]), None)
        ys, xs = np.nonzero(m)
        seeds.append((mid, block, (float(xs.mean()), float(ys.mean()))))

    rooms = _regions_from_labels(labels, seeds)
    footprint = _footprint(labels, pi.wall_half_px)
    flags = ["learned_vectoriser"]
    if wall_mask.mean() < 0.005:
        flags.append("learned_vectoriser_found_few_walls")
    return Vectorisation(rooms=rooms, labels=labels, free=(room_mask > 0),
                         footprint=footprint, inside=(room_mask > 0),
                         n_outlines=count_outlines(footprint), text_erased=0,
                         qa_flags=flags)


def segment(pi: PlanImage, text: PlanText) -> Vectorisation:
    """Read the plan with whichever engine reads *this* plan better.

    Three engines, in order of preference:

    1. A full trained plan reader, if one is installed (``learned``). It predicts
       room polygons directly and needs no help.
    2. The classical watershed over the ink mask.
    3. The same watershed over a *wall map from a pretrained net* (``wallnet``)
       instead of the ink mask.

    2 and 3 differ in one thing only -- what counts as a wall -- and they fail on
    opposite kinds of plan. Across the golden set, on the twelve plans the ink
    mask was failing the net wins eleven (median outline-on-wall 0.40 to 0.79);
    on the twelve the ink mask was handling, the ink mask wins eleven (0.89 to
    0.84). Picking one engine for all plans throws half of that away, so run both
    and keep the reading that puts more of its outline on real drawn lines --
    a judgement made from the plan alone, with no ground truth.
    """
    v = segment_learned(pi, text)
    if v is not None and v.rooms:
        return v

    classical = segment_classical(pi, text)
    barrier = (wallnet.barrier(pi.rgb, pi.ink, text.words, pi.wall_half_px)
               if wallnet.available() else None)
    net = segment_with_wallnet(pi, text, barrier)
    if net is None or not net.rooms:
        return classical
    if not classical.rooms:
        return net

    # Score against the *wall* map, not the ink. Scoring against ink cannot
    # adjudicate this at all: a cabinet run is ink, a door swing is ink, a bed is
    # ink, so an outline that traces the furniture scores as well as one that
    # traces the wall. On the golden set the two references differ by 8 points of
    # median outline fit, and that gap is precisely the furniture error.
    def fit(v: Vectorisation) -> float:
        scores = wallnet.outline_on_wall([r.polygon_px for r in v.rooms], barrier)
        return float(np.median(scores)) if scores else 0.0

    c_fit, n_fit = fit(classical), fit(net)
    # Floor accounted for, in the same pixels for both, so the two are comparable.
    # Each reading's own footprint is not a denominator: a reading that loses the
    # flat loses its footprint with it and still scores full coverage.
    c_px = float(sum(r.area_px for r in classical.rooms))
    n_px = float(sum(r.area_px for r in net.rooms))

    # Outline fit alone is not safe to choose on, and one plan showed exactly why:
    # a leak in the wall map let the exterior basin flood the flat, leaving a
    # ring-shaped polygon traced along the outer wall and a few pockets. That ring
    # scores near 1.0 -- it *is* the wall -- while covering a fifth of the floor.
    # A reading that loses most of the building is not the better reading however
    # neatly its outlines sit, so coverage gates the comparison and fit decides
    # inside it.
    most = max(c_px, n_px) or 1.0
    c_cov, n_cov = c_px / most, n_px / most
    c_ok = c_cov >= COVERAGE_TOLERANCE
    n_ok = n_cov >= COVERAGE_TOLERANCE
    if c_ok and not n_ok:
        pick_net = False
    elif n_ok and not c_ok:
        pick_net = True
    else:
        pick_net = n_fit > c_fit

    winner, tag = (net, "wallnet") if pick_net else (classical, "ink")
    winner.qa_flags = list(winner.qa_flags) + [
        f"wall_source_{tag}",
        f"wall_source_fit_{c_fit:.2f}ink_vs_{n_fit:.2f}net",
        f"wall_source_coverage_{c_cov:.2f}ink_vs_{n_cov:.2f}net",
    ]
    if c_ok and n_ok and abs(c_fit - n_fit) < 0.05:
        winner.qa_flags.append("wall_source_close_call")
    if c_ok != n_ok:
        winner.qa_flags.append("wall_source_decided_on_coverage")
    return winner


#: A reading may be preferred on outline fit only while it still accounts for
#: this much of the floor the other reading found. Below it, it has lost rooms.
COVERAGE_TOLERANCE = 0.8


def segment_with_wallnet(pi: PlanImage, text: PlanText,
                         mask: np.ndarray | None = None) -> Vectorisation | None:
    """The classical watershed, but told what a wall is by the pretrained net.

    Everything downstream of the wall map is identical -- this is a swap of one
    input, not a second pipeline.
    """
    if mask is None:
        if not wallnet.available():
            return None
        mask = wallnet.barrier(pi.rgb, pi.ink, text.words, pi.wall_half_px)
    if mask is None or not mask.any():
        return None
    swapped = replace(pi, ink=mask, walls=mask)
    try:
        v = segment_classical(swapped, text)
    except Exception as e:                                       # noqa: BLE001
        log.warning("wallnet vectorisation failed, keeping the ink reading: %s", e)
        return None
    v.qa_flags = list(v.qa_flags) + ["walls_from_pretrained_net"]
    return v


def segment_classical(pi: PlanImage, text: PlanText) -> Vectorisation:
    """Watershed the free space, seeded by captions and by unclaimed distance peaks."""
    clean, n_erased = erase_text(pi.ink, text)
    hull = structure_hull(clean)
    free = (clean == 0) & (hull > 0)
    free = ndi.binary_opening(free, np.ones((3, 3)))
    h, w = free.shape
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    markers = np.zeros((h, w), np.int32)
    markers[0, :] = markers[-1, :] = EXTERIOR
    markers[:, 0] = markers[:, -1] = EXTERIOR
    # Everything outside the building's hull is exterior too, so the margin text
    # cannot be reached by a room basin creeping through a window gap.
    markers[(hull == 0)] = EXTERIOR

    blocks = _caption_seeds(text, hull)
    seeds: list[tuple[int, TextBlock | None, tuple[float, float]]] = []
    next_id = EXTERIOR + 1
    for b in blocks:
        x0, y0, x1, y1 = (int(v) for v in b.box)
        markers[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = next_id
        seeds.append((next_id, b, (b.cx, b.cy)))
        next_id += 1

    # Pass 1: captions against the exterior. The exterior basin this produces is
    # our working definition of "outside the building" — better than the convex
    # hull, which includes the margin the title block sits in, and better than a
    # flood fill, which leaks through every window gap.
    m1 = markers.copy()
    m1[~free] = 0
    m1[0, :] = m1[-1, :] = EXTERIOR
    m1[:, 0] = m1[:, -1] = EXTERIOR
    first = watershed(-dist, m1, mask=free) if len(seeds) else np.zeros_like(markers)
    inside = first > EXTERIOR if len(seeds) else (hull > 0) & free

    # Pass 2 seeds: rooms nobody named. A peak of the free-space distance
    # transform that sits inside the building and that no caption has claimed is a
    # hallway, a cupboard or a WC — the spaces UK plans routinely leave unlabelled.
    claimed = markers > EXTERIOR
    interior_dist = np.where(inside, dist, 0)
    peak_min_dist = max(8, int(0.02 * max(h, w)))
    peaks = peak_local_max(interior_dist, min_distance=peak_min_dist,
                           threshold_abs=max(4.0, float(interior_dist.max()) * 0.30),
                           labels=inside.astype(np.uint8), exclude_border=False)
    for py, px in peaks:
        if not inside[py, px]:
            continue
        r = max(3, int(dist[py, px] * 0.45))
        win = claimed[max(0, py - r * 3):py + r * 3, max(0, px - r * 3):px + r * 3]
        if win.any():
            continue
        markers[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = next_id
        claimed[max(0, py - r * 2):py + r * 2, max(0, px - r * 2):px + r * 2] = True
        seeds.append((next_id, None, (float(px), float(py))))
        next_id += 1

    markers[~free] = 0
    markers[0, :] = markers[-1, :] = EXTERIOR
    markers[:, 0] = markers[:, -1] = EXTERIOR
    labels = watershed(-dist, markers, mask=free)
    # Nothing outside the building is a room, whatever the watershed decided.
    labels[(~inside) & (labels > EXTERIOR)] = EXTERIOR

    rooms = _regions_from_labels(labels, seeds)
    footprint = _footprint(labels, pi.wall_half_px)
    n_outlines = count_outlines(footprint)
    flags: list[str] = []
    if n_outlines > 1:
        # A maisonette's sheet draws one outline per storey. Phase 1 builds a single
        # storey, and building all of them puts two floors on top of each other in
        # the same plane. Keep the largest and say so.
        flags.append("multiple_plan_outlines")
        rooms, footprint = _keep_largest_outline(rooms, footprint, labels)
    covered = float((labels > EXTERIOR).sum())
    if footprint.any() and covered < 0.55 * float(footprint.sum()):
        # Most of the building is not in any room. Usually unlabelled circulation
        # space; occasionally the vectoriser missing half the plan. Either way a
        # human should look before this listing is trusted.
        flags.append("unclaimed_interior_area")
    if not blocks:
        flags.append("no_room_captions")
    if not rooms:
        flags.append("no_rooms_found")
    return Vectorisation(rooms=rooms, labels=labels, free=free, footprint=footprint,
                         inside=inside, n_outlines=n_outlines, text_erased=n_erased,
                         qa_flags=flags)


def _keep_largest_outline(rooms, footprint, labels):
    """Restrict to the biggest connected outline on the sheet."""
    lab, n = ndi.label(footprint)
    if n <= 1:
        return rooms, footprint
    sizes = ndi.sum(footprint, lab, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    mask = lab == keep
    out = []
    for r in rooms:
        cx, cy = int(r.centroid_px[0]), int(r.centroid_px[1])
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
            out.append(r)
    return (out or rooms), mask


def _footprint(labels: np.ndarray, wall_half_px: float) -> np.ndarray:
    """Gross internal footprint: the rooms, grown by a wall, holes filled.

    Derived from the rooms rather than from a flood fill, because a flood fill
    leaks through every window gap. A printed "gross internal area" is measured to
    the outside of the internal walls, which is what growing by one wall thickness
    reproduces.
    """
    rooms_mask = (labels > EXTERIOR)
    if not rooms_mask.any():
        return np.zeros_like(rooms_mask)
    r = max(2, int(round(wall_half_px * 2.0)))
    grown = cv2.dilate(rooms_mask.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)))
    filled = ndi.binary_fill_holes(grown.astype(bool))
    return cv2.erode(filled.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r + 1, r + 1))).astype(bool)


def _regions_from_labels(labels: np.ndarray,
                         seeds: list[tuple[int, TextBlock | None, tuple[float, float]]]
                         ) -> list[RoomRegion]:
    room_pixels = float((labels > EXTERIOR).sum())
    if room_pixels <= 0:
        return []
    out: list[RoomRegion] = []
    for idx, (mid, block, seed_xy) in enumerate(seeds):
        mask = (labels == mid).astype(np.uint8)
        area = float(mask.sum())
        floor = MIN_ROOM_FRAC if block is not None else MIN_UNLABELLED_ROOM_FRAC
        if area < room_pixels * floor:
            continue
        poly = _largest_contour(mask)
        if poly is None or len(poly) < 3:
            continue
        cx, cy = geom.centroid(poly)
        flags: list[str] = []
        if area > room_pixels * MAX_ROOM_FRAC and len(seeds) > 1:
            flags.append("room_dominates_plan")
        out.append(RoomRegion(
            room_id=f"p{idx}", mask_label=mid,
            polygon_px=[[float(x), float(y)] for x, y in poly],
            area_px=area, centroid_px=(cx, cy),
            label=block.label if block else None,
            label_text=block.label_text if block else None,
            label_confidence=(float(np.mean([w.conf for w in block.words])) / 100.0
                              if block else 0.0),
            ocr_dims_m=block.dims_m if block else None,
            ocr_area_m2=block.area_m2 if block else None,
            seeded_by="caption" if block else "distance_peak",
            qa_flags=flags,
        ))
    return out


def _largest_contour(mask: np.ndarray) -> list[list[float]] | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 9:
        return None
    eps = 0.006 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return geom.ensure_ccw(approx.astype(float))


# --------------------------------------------------------------------------
# Atlanta-world regularisation


def dominant_directions(polys: list[list[list[float]]], *, max_dirs: int = 4,
                        min_share: float = 0.06) -> list[float]:
    """Wall directions the plan actually uses, in degrees, modulo 180.

    Manhattan (exactly two perpendicular directions) is the usual case and the
    wrong assumption often enough to matter: UK period stock has bay windows and
    angled returns, and the roadmap calls that out as the non-Manhattan
    requirement for Sprint 4. So we find the *set* of directions — an Atlanta
    world — and always include the perpendicular of the strongest one, because a
    room with three walls on one axis still has a fourth.
    """
    acc = np.zeros(180, dtype=float)
    for poly in polys:
        a = np.asarray(poly, dtype=float)
        d = np.roll(a, -1, axis=0) - a
        lengths = np.linalg.norm(d, axis=1)
        angles = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
        for ang, L in zip(angles, lengths):
            lo = int(np.floor(ang))
            frac = ang - lo
            acc[lo % 180] += L * (1 - frac)
            acc[(lo + 1) % 180] += L * frac
    if acc.sum() <= 0:
        return [0.0, 90.0]
    # Smooth circularly so a wall drawn 1 degree off does not start a new direction.
    k = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    sm = np.convolve(np.r_[acc[-4:], acc, acc[:4]], k / k.sum(), mode="same")[4:-4]
    primary = float(np.argmax(sm))
    dirs = [primary, (primary + 90.0) % 180.0]
    total = sm.sum()
    order = np.argsort(-sm)
    for i in order:
        ang = float(i)
        if len(dirs) >= max_dirs:
            break
        if sm[i] < total * min_share:
            break
        if all(abs(geom.wrap_deg(ang - d, 180.0)) > 12.0 for d in dirs):
            dirs.append(ang)
            perp = (ang + 90.0) % 180.0
            if len(dirs) < max_dirs and all(abs(geom.wrap_deg(perp - d, 180.0)) > 12.0
                                            for d in dirs):
                dirs.append(perp)
    return sorted(dirs)


def regularise(poly: list[list[float]], directions: list[float],
               tol_deg: float = 22.0) -> tuple[list[list[float]], float, bool]:
    """Snap edges onto the plan's wall directions and rebuild the corners.

    Returns ``(polygon, mean_snap_degrees, used_a_non_manhattan_direction)``.
    Edges further than ``tol_deg`` from every direction are left alone — a curved
    bay is better represented badly than squared off into a lie.
    """
    a = np.asarray(poly, dtype=float)
    n = len(a)
    if n < 3 or not directions:
        return geom.ensure_ccw(a), 0.0, False
    lines: list[tuple[np.ndarray, float] | None] = []
    snaps: list[float] = []
    used_extra = False
    prim = {directions[0], (directions[0] + 90.0) % 180.0}
    for i in range(n):
        p, q = a[i], a[(i + 1) % n]
        d = q - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            lines.append(None)
            continue
        ang = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
        best = min(directions, key=lambda t: abs(geom.wrap_deg(ang - t, 180.0)))
        delta = abs(geom.wrap_deg(ang - best, 180.0))
        if delta > tol_deg:
            lines.append(None)
            continue
        snaps.append(delta)
        if best not in prim:
            used_extra = True
        th = np.radians(best)
        nvec = np.array([-np.sin(th), np.cos(th)])
        mid = (p + q) / 2.0
        lines.append((nvec, float(nvec @ mid)))

    # Rebuild vertices by intersecting consecutive snapped lines. Where a line was
    # left unsnapped, the original vertex is kept — that is what preserves bays.
    out: list[list[float]] = []
    for i in range(n):
        prev, cur = lines[i - 1], lines[i]
        if prev is None or cur is None:
            out.append([float(a[i][0]), float(a[i][1])])
            continue
        m = np.array([prev[0], cur[0]])
        rhs = np.array([prev[1], cur[1]])
        det = float(np.linalg.det(m))
        if abs(det) < 1e-8:
            out.append([float(a[i][0]), float(a[i][1])])
            continue
        v = np.linalg.solve(m, rhs)
        # A snapped corner that has run away is worse than no snap at all.
        if np.linalg.norm(v - a[i]) > 0.25 * geom.perimeter(a):
            out.append([float(a[i][0]), float(a[i][1])])
        else:
            out.append([float(v[0]), float(v[1])])
    reg = geom.ensure_ccw(out)
    if geom.area(reg) < 0.4 * geom.area(a):
        return geom.ensure_ccw(a), 0.0, used_extra
    return reg, (float(np.mean(snaps)) if snaps else 0.0), used_extra


def regularise_all(v: Vectorisation) -> Vectorisation:
    """Snap every room to the plan's shared direction set, in place."""
    polys = [r.polygon_px for r in v.rooms]
    if not polys:
        return v
    dirs = dominant_directions(polys)
    v.directions_deg = [round(d, 2) for d in dirs]
    non_manhattan = False
    for r in v.rooms:
        reg, snap, extra = regularise(r.polygon_px, dirs)
        r.polygon_px = [[round(x, 2), round(y, 2)] for x, y in reg]
        r.area_px = geom.area(reg)
        r.centroid_px = geom.centroid(reg)
        non_manhattan = non_manhattan or extra
        if snap > 8.0:
            r.qa_flags.append("large_regularisation_snap")
    v.non_manhattan = non_manhattan or len(dirs) > 2
    return v
