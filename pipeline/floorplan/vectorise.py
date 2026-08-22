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
from . import roomfinder, wallnet
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


def building_footprint(walls: np.ndarray, wall_half_px: float = 2.0) -> np.ndarray:
    """What the walls actually enclose, concave bits and all.

    ``structure_hull`` returns a *convex* hull, and the hull of an L-shaped or
    bay-fronted plan includes the garden in the crook of the L -- measured across
    the golden set at seven times the real footprint. Rooms are free to grow into
    that, and they do, as long thin spikes out into the page margin.

    Two ways of finding the real outline, because neither alone survives every
    plan. Sealing the drawing shut and taking its silhouette works on a plan drawn
    with heavy walls and fails on a thin-line one, where closing does not connect
    the outline. Flooding in from the page edge works on a plan whose outer wall is
    unbroken and fails where an external door or a bay leaves a gap, because the
    flood pours through it and swallows the building. Take whichever gives the more
    plausible answer, and fall back to the hull when both look wrong -- being
    generous is only a missed improvement, being wrong deletes rooms.
    """
    h, w = walls.shape
    page = float(h * w)
    hull = structure_hull(walls)
    candidates = []

    k = max(3, int(round(wall_half_px * 6)))
    closed = cv2.morphologyEx(walls.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((k, k), np.uint8))
    lab, n = ndi.label(closed)
    if n:
        sizes = ndi.sum(closed, lab, range(1, n + 1))
        biggest = (lab == (1 + int(np.argmax(sizes))))
        candidates.append(ndi.binary_fill_holes(biggest).astype(np.uint8))

    seal = max(2, int(round(wall_half_px * 3)))
    sealed = cv2.morphologyEx(walls.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((seal, seal), np.uint8))
    lab2, n2 = ndi.label(sealed == 0)
    if n2:
        border = np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])
        outside = set(np.unique(border).tolist()) - {0}
        inside = (~np.isin(lab2, list(outside))) & (lab2 > 0)
        candidates.append(ndi.binary_fill_holes(inside | (sealed > 0)).astype(np.uint8))

    # A footprint has to be smaller than the hull -- that is the whole point -- but
    # not so small that it has clearly lost the building.
    hull_area = float(hull.sum()) or page
    usable = [c for c in candidates
              if 0.25 * hull_area <= float(c.sum()) <= 1.01 * hull_area
              and float(c.sum()) / page > 0.02]
    if not usable:
        return hull
    return min(usable, key=lambda c: float(c.sum()))


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

    Four engines, in order of preference:

    1. A full trained plan reader, if one is installed (``learned``).
    2. Room polygons predicted elsewhere by a GPU model (``roomfinder``), reported
       as it drew them and named from the plan's own printed captions. Measured on
       the golden set the predictions find the rooms -- open-plan spaces whole,
       unlabelled WCs included -- and the plan's own text says what they are.
    3. The classical watershed over the ink mask.
    4. The same watershed over a *wall map from a pretrained net* (``wallnet``)
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

    v = segment_from_room_finder(pi, text)
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


def segment_from_room_finder(pi: PlanImage, text: PlanText) -> Vectorisation | None:
    """Rooms as the GPU model drew them, named from what the plan prints.

    The division of labour, settled by measurement (see
    ``docs/PLAN-READING-REPORT.md`` section 8): the model finds the rooms and draws
    them, and we read the plan to find out what they are called. Their geometry,
    our words.

    Returns ``None`` when there is no prediction for this listing, which is the
    normal case until ``tools.import_room_predictions`` has been run -- so a fresh
    checkout behaves exactly as it did before.
    """
    listing = getattr(pi, "listing_id", None) or _LISTING.get("id")
    if not listing or not roomfinder.available(listing):
        return None
    seeds, names, meta = roomfinder.seeds_for(listing, pi)
    if not seeds:
        return None
    mask = (wallnet.barrier(pi.rgb, pi.ink, text.words, pi.wall_half_px)
            if wallnet.available() else None)
    try:
        v = segment_from_room_seeds(pi, text, seeds, mask=mask, fallback_labels=names)
    except Exception as e:                                       # noqa: BLE001
        log.warning("room-finder vectorisation failed, falling back: %s", e)
        return None
    if v is None or not v.rooms:
        return None
    printed = sum(1 for r in v.rooms if r.seeded_by == "caption")
    unnamed = sum(1 for r in v.rooms if "no_printed_name" in r.qa_flags)
    v.qa_flags = list(v.qa_flags) + [
        "rooms_from_room_finder",
        f"room_finder_kept_{len(v.rooms)}_of_{meta.get('predicted_rooms', '?')}",
        f"room_names_printed_{printed}_of_{len(v.rooms)}",
    ]
    if KEEP_PREDICTED_OUTLINES:
        v.qa_flags.append("outlines_as_predicted")
    if unnamed:
        # Worth seeing in the QA sheet: these are the rooms the plan never named,
        # and under the plan-only policy they reach the shell with no name at all.
        v.qa_flags.append(f"rooms_with_no_printed_name_{unnamed}")
    if mask is None:
        v.qa_flags.append("room_finder_without_wall_model")
    return v


#: Stage 5 tells the vectoriser which listing it is reading, so the room-finder can
#: find that listing's predictions. Set by ``stage.build_plan``; a module-level
#: handoff rather than a signature change so no other engine has to care.
_LISTING: dict = {}


def set_listing(listing_id: str | None) -> None:
    _LISTING["id"] = listing_id


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


#: Report the outlines the model drew, rather than our watershed's opinion of where
#: those rooms end. Growing them out to the walls does put more of each edge on a
#: wall -- 81% to 91% on the 25 test plans -- but it is our reading of the room, not
#: the model's, and it drags the outlines around to get there. Measured and chosen in
#: ``notebooks/plan_reading_modal.ipynb``.
#:
#: The rooms still grow *internally*: two rooms share a doorway only where their free
#: space meets, so adjacency and apertures are computed from the grown regions even
#: though the outlines reported are the model's own.
KEEP_PREDICTED_OUTLINES = True

#: Name rooms only from what the plan prints. A room-polygon model's types come from
#: whatever collection it was trained on -- CubiCasa5K answers "Undefined" for
#: anything it is unsure of, and has no word for an airing cupboard at all -- while
#: the plan prints "ENSUITE" and "AIRING CUPBOARD" in plain English and we read it.
#: Where the plan says nothing the room stays unnamed and carries the model's guess
#: as a QA flag, rather than wearing it as a name.
NAME_FROM_PLAN_ONLY = True

#: How far inside a predicted room its starting point is pulled, as a fraction of
#: the room's own inscribed radius. A coarse polygon's edge routinely lands on -- or
#: past -- the wall, and a starting point touching a wall lets the room grow straight
#: through it into its neighbour. Swept in ``notebooks/plan_reading_modal.ipynb``.
SEED_CORE_FRACTION = 0.45

#: Keep rooms inside what the walls actually enclose, rather than inside the convex
#: hull of the drawing. The hull of an L-shaped or bay-fronted plan includes the
#: garden in the crook of the L -- measured at 7x the real footprint -- and rooms
#: grow into it as long spikes. Off by default until the notebook says it helps.
CONFINE_TO_FOOTPRINT = False


def segment_from_room_seeds(pi: PlanImage, text: PlanText,
                            room_seeds: list, mask: np.ndarray | None = None,
                            fallback_labels: list | None = None, *,
                            seed_core: float | None = None,
                            confine: bool | None = None,
                            keep_outlines: bool | None = None,
                            name_from_plan_only: bool | None = None
                            ) -> Vectorisation | None:
    """Grow rooms somebody else found out to the walls we can see.

    Two models, each good at the half the other is bad at. A room-polygon model
    knows *which* rooms exist, what type each is, and that an open-plan kitchen and
    living room are one space -- all things our caption seeding has to guess. What
    it cannot do is put a corner on a wall: it works on a small copy of the plan
    and rounds every coordinate to a coarse grid.

    The wall map is the opposite: exact about where a wall is, silent about which
    enclosed region is a room.

    So take the other model's rooms as starting points and let each one grow until
    it meets a wall. Same watershed as ``segment_classical``, better starting
    points -- one per real room instead of one per caption.

    ``room_seeds`` are polygons in **this** ``PlanImage``'s pixels, which is the
    deskewed, size-capped copy rather than the file on disk. Getting that wrong
    puts every outline out by a couple of percent.
    """
    if not room_seeds:
        return None
    if mask is None:
        mask = wallnet.barrier(pi.rgb, pi.ink, text.words, pi.wall_half_px)
    if mask is None or not mask.any():
        mask, _ = erase_text(pi.ink, text)

    want_confine = CONFINE_TO_FOOTPRINT if confine is None else confine
    hull = building_footprint(mask, pi.wall_half_px) if want_confine else structure_hull(mask)
    free = (mask == 0) & (hull > 0)
    free = ndi.binary_opening(free, np.ones((3, 3)))
    h, w = free.shape
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    markers = np.zeros((h, w), np.int32)
    markers[0, :] = markers[-1, :] = EXTERIOR
    markers[:, 0] = markers[:, -1] = EXTERIOR
    markers[hull == 0] = EXTERIOR

    seeds: list[tuple[int, TextBlock | None, tuple[float, float]]] = []
    order: list[int] = []          # which incoming seed each marker came from
    next_id = EXTERIOR + 1
    for idx, poly in enumerate(room_seeds):
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        filled = np.zeros((h, w), np.uint8)
        cv2.fillPoly(filled, [pts], 1)
        # Pull the marker well inside the room. A coarse polygon's edge routinely
        # lands on -- or past -- the wall, and a marker touching the wall lets the
        # room grow straight through it into its neighbour.
        keep = filled & free.astype(np.uint8)
        if keep.any():
            room_dist = cv2.distanceTransform(keep, cv2.DIST_L2, 5)
            frac = SEED_CORE_FRACTION if seed_core is None else seed_core
            core = (room_dist >= max(2.0, frac * float(room_dist.max()))).astype(np.uint8)
        else:
            core = np.zeros((h, w), np.uint8)
        if not core.any():
            # The polygon missed the free space entirely. Fall back to its centre
            # if that at least lands somewhere a room could be.
            cx, cy = pts.mean(axis=0).astype(int)
            if 0 <= cy < h and 0 <= cx < w and free[cy, cx]:
                r = max(2, int(pi.wall_half_px))
                core[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = 1
            else:
                continue
        ys, xs = np.nonzero(core)
        markers[core > 0] = next_id
        seeds.append((next_id, None, (float(xs.mean()), float(ys.mean()))))
        order.append(idx)
        next_id += 1

    if not seeds:
        return None
    fallback = [(fallback_labels[i] if fallback_labels and i < len(fallback_labels) else None)
                for i in order]

    markers[~free] = 0
    markers[0, :] = markers[-1, :] = EXTERIOR
    markers[:, 0] = markers[:, -1] = EXTERIOR
    labels = watershed(-dist, markers, mask=free)
    labels[(hull == 0) & (labels > EXTERIOR)] = EXTERIOR

    # Their rooms, our names. The seeds knew *where* the rooms are; the plan prints
    # what they are called and we read it, which is the half we are better at. Do
    # this after growing rather than before: a coarse seed's edge lands anywhere,
    # so a caption that falls outside it still falls inside the grown room.
    seeds = _name_from_captions(labels, seeds, text, fallback)

    rooms = _regions_from_labels(labels, seeds, trusted=True)
    if not rooms:
        return None

    # The growing above decided which room is which and where the doorways are.
    # What it reports as the room's shape is a separate question: hand back the
    # outline the model drew, unless asked to hand back the grown one.
    keep = KEEP_PREDICTED_OUTLINES if keep_outlines is None else keep_outlines
    if keep:
        came_from = {mid: i for (mid, _b, _xy), i in zip(seeds, order)}
        for room in rooms:
            idx = came_from.get(room.mask_label)
            if idx is None:
                continue
            pts = np.asarray(room_seeds[idx], dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                continue
            poly = geom.ensure_ccw(pts)
            room.polygon_px = [[float(x), float(y)] for x, y in poly]
            room.area_px = float(geom.area(poly))
            room.centroid_px = geom.centroid(poly)
            room.qa_flags = list(room.qa_flags) + ["outline_as_predicted"]

    # Where the plan printed no name, the model has a guess. Whether that guess
    # becomes the room's name is a policy decision, not a technical one: its
    # vocabulary is its training set's, and ours is the plan's own words.
    plan_only = NAME_FROM_PLAN_ONLY if name_from_plan_only is None else name_from_plan_only
    predicted = {mid: name for (mid, _b, _xy), name in zip(seeds, fallback) if name}
    for room in rooms:
        if room.label:
            continue
        name = predicted.get(room.mask_label)
        # Whatever happens to the name, the room itself came from the model, not
        # from a distance peak -- which is what `_regions_from_labels` assumes for
        # anything without a caption.
        room.seeded_by = "room_finder"
        if not name:
            room.qa_flags = list(room.qa_flags) + ["no_printed_name"]
        elif plan_only:
            slug = "".join(ch if ch.isalnum() else "_" for ch in str(name).lower()).strip("_")
            room.qa_flags = list(room.qa_flags) + ["no_printed_name", f"model_says_{slug}"]
        else:
            room.label = name
            room.qa_flags = list(room.qa_flags) + ["label_predicted_not_printed"]
    footprint = _footprint(labels, pi.wall_half_px)
    flags = ["rooms_from_external_seeds"]
    n_outlines = count_outlines(footprint)
    if n_outlines > 1:
        flags.append("multiple_plan_outlines")
        rooms, footprint = _keep_largest_outline(rooms, footprint, labels)
    return Vectorisation(rooms=rooms, labels=labels, free=free, footprint=footprint,
                         inside=(hull > 0), n_outlines=n_outlines, text_erased=0,
                         qa_flags=flags)


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


def _name_from_captions(labels: np.ndarray,
                        seeds: list, text: PlanText,
                        fallback: list) -> list:
    """Attach each printed room name to the room it was printed inside.

    A room can hold more than one caption -- "RECEPTION" and "DINING ROOM" printed
    in one open-plan space -- and the biggest one wins, since a room's own name is
    set in larger type than the dimensions beneath it. Where no caption landed in
    a room we fall back to the type the seeds came with, which is a guess from
    pixels rather than something the plan actually says, so it is marked as such.
    """
    h, w = labels.shape
    best: dict[int, TextBlock] = {}
    for block in text.blocks:
        if not block.is_room_caption:
            continue
        cx, cy = int(block.cx), int(block.cy)
        if not (0 <= cy < h and 0 <= cx < w):
            continue
        mid = int(labels[cy, cx])
        if mid <= EXTERIOR:
            continue
        x0, y0, x1, y1 = block.box
        size = abs(x1 - x0) * abs(y1 - y0)
        prev = best.get(mid)
        if prev is None:
            best[mid] = block
        else:
            px0, py0, px1, py1 = prev.box
            if size > abs(px1 - px0) * abs(py1 - py0):
                best[mid] = block
    named = []
    for i, (mid, _old, xy) in enumerate(seeds):
        named.append((mid, best.get(mid), xy))
    return named


def _regions_from_labels(labels: np.ndarray,
                         seeds: list[tuple[int, TextBlock | None, tuple[float, float]]],
                         trusted: bool = False) -> list[RoomRegion]:
    room_pixels = float((labels > EXTERIOR).sum())
    if room_pixels <= 0:
        return []
    out: list[RoomRegion] = []
    for idx, (mid, block, seed_xy) in enumerate(seeds):
        mask = (labels == mid).astype(np.uint8)
        area = float(mask.sum())
        # The larger floor exists to throw away rooms *guessed* from a distance
        # peak. A room a model asserted is not that kind of guess, and holding it
        # to the strict floor drops exactly the WCs and cupboards it is good at.
        floor = (MIN_ROOM_FRAC if (block is not None or trusted)
                 else MIN_UNLABELLED_ROOM_FRAC)
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
