"""Pixels to metres, from what the plan prints on itself.

Phase 0's most useful data point for Phase 1: **room dimensions are printed on
54% of UK plans (62% of high-resolution ones)**. That single fact does two jobs.
It gives the plan channel a metric scale with no external measurement, and it
gives gate G1 its self-consistency check — the listing supplies both the
reconstruction and the number it has to agree with (ROADMAP §0b).

Candidates are ranked, all of them are kept, and disagreement between them is a
QA flag rather than a tie-break we quietly win.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..core import geom

log = logging.getLogger("floorplan.scale")

#: Ranked worst-to-best. Printed dimensions beat printed totals because they are
#: per-room and there are several of them; a stated area from the listing text is
#: last because Phase 0 found portal metadata unreliable.
SOURCE_RANK = {"stated_area": 1, "scale_ratio": 2, "printed_area": 3, "printed_dimensions": 4}


@dataclass
class ScaleCandidate:
    source: str
    px_per_metre: float
    weight: float
    detail: str

    def to_dict(self) -> dict:
        return {"source": self.source, "px_per_metre": round(self.px_per_metre, 4),
                "weight": round(self.weight, 3), "detail": self.detail}


def from_printed_dimensions(rooms) -> tuple[ScaleCandidate | None, list[dict]]:
    """One estimate per room that prints its size; the median wins.

    A room drawn 420 px on its long side and captioned "3.96 m" is 106 px/m. Both
    sides are used, long matched to long, and the spread across rooms is reported:
    on a plan that is honestly to scale the estimates agree to a few percent, and
    when they do not that is exactly what the 'not to scale' disclaimer was
    warning about.
    """
    per_room: list[dict] = []
    for r in rooms:
        if not r.ocr_dims_m:
            continue
        long_m, short_m = r.ocr_dims_m
        long_px, short_px, _ang = geom.oriented_extent(r.polygon_px)
        if short_m < 0.3 or short_px < 4:
            continue
        e_long = long_px / long_m
        e_short = short_px / short_m
        per_room.append({
            "room_id": r.room_id, "label": r.label,
            "printed_m": [round(long_m, 2), round(short_m, 2)],
            "drawn_px": [round(long_px, 1), round(short_px, 1)],
            "px_per_metre": round(float(np.mean([e_long, e_short])), 3),
            "axis_disagreement_pct": round(100 * abs(e_long - e_short) / max(e_long, e_short), 1),
        })
    if not per_room:
        return None, []
    vals = np.array([p["px_per_metre"] for p in per_room])
    med = float(np.median(vals))
    spread = float(np.max(np.abs(vals - med)) / med) if med > 0 else 1.0
    # More rooms agreeing is worth more; wide spread is worth less. Both capped so
    # a single lucky room cannot outweigh a careful stated area.
    weight = min(4.0, len(per_room) * 1.2) * max(0.25, 1.0 - spread)
    return ScaleCandidate(
        "printed_dimensions", med, weight,
        f"{len(per_room)} room(s), spread {100 * spread:.0f}%",
    ), per_room


def from_area(total_px: float, area_m2: float | None, source: str,
              weight: float) -> ScaleCandidate | None:
    """sqrt(pixels / m²). Areas scale as the square, hence the root."""
    if not area_m2 or area_m2 <= 0 or total_px <= 0:
        return None
    ppm = float(np.sqrt(total_px / area_m2))
    if not (1.0 < ppm < 5000.0):
        return None
    return ScaleCandidate(source, ppm, weight, f"{area_m2:.1f} m² over {total_px:.0f} px²")


#: Above this, two candidates are not disagreeing about a detail — one of them is
#: reading something that is not a room dimension.
GROSS_DISAGREEMENT = 1.8


def choose(candidates: list[ScaleCandidate]) -> tuple[ScaleCandidate | None, float | None]:
    """Pick by rank, then by weight. Returns ``(winner, max_disagreement_pct)``.

    We deliberately do *not* average the candidates. They measure different
    things — printed dimensions measure the drawing, a stated area measures the
    flat including its walls — and averaging them hides which one we trusted.

    But rank is a tiebreak among *plausible* candidates, not a licence to ignore
    the evidence. When the top-ranked candidate disagrees with a whole-flat area
    by more than a factor of 1.8, it is not measuring the same building: a stated
    floor area can be a little wrong, it cannot be 3x wrong. In that case the
    area-derived candidate wins, and the QA flag says the plan's printed
    dimensions were not believed.
    """
    if not candidates:
        return None, None
    ranked = sorted(candidates, key=lambda c: (-SOURCE_RANK.get(c.source, 0), -c.weight))
    best = ranked[0]
    area_based = [c for c in candidates if c.source in ("printed_area", "stated_area")]
    if best.source == "printed_dimensions" and area_based:
        sanity = max(area_based, key=lambda c: (SOURCE_RANK.get(c.source, 0), c.weight))
        ratio = max(best.px_per_metre / sanity.px_per_metre,
                    sanity.px_per_metre / best.px_per_metre)
        if ratio > GROSS_DISAGREEMENT:
            best = sanity
    vals = [c.px_per_metre for c in candidates]
    disagreement = (100 * (max(vals) - min(vals)) / max(vals)) if len(vals) > 1 else None
    return best, disagreement


def solve(rooms, footprint_px: float, plan_total_m2: float | None,
          stated_m2: float | None, scale_ratio: int | None = None
          ) -> tuple[ScaleCandidate | None, list[ScaleCandidate], list[dict], float | None]:
    """All candidates, the winner, and the per-room evidence behind the best one."""
    cands: list[ScaleCandidate] = []
    dim_c, per_room = from_printed_dimensions(rooms)
    if dim_c:
        cands.append(dim_c)
    room_px = sum(geom.area(r.polygon_px) for r in rooms)
    # A printed "gross internal area" includes the wall thicknesses, so it is
    # matched against the footprint; per-room polygons are internal and are
    # matched against the sum of rooms.
    c = from_area(footprint_px or room_px, plan_total_m2, "printed_area", 2.5)
    if c:
        cands.append(c)
    c = from_area(footprint_px or room_px, stated_m2, "stated_area", 1.5)
    if c:
        cands.append(c)
    best, disagreement = choose(cands)
    return best, cands, per_room, disagreement
