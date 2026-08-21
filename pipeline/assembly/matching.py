"""Stage 6's cost model and assignment.

The feasibility report calls assembly *the wrong-but-confident stage*, and it is
right: when this is wrong the output is still a plausible-looking flat, so nothing
downstream notices. Two consequences shape the whole module.

**Every cost is broken out and stored.** Not a scalar — the named terms that
produced it. When a bedroom lands in the kitchen you need to know whether it was
the area term or the label term that did it.

**Every match records its margin.** The cost gap to the room's next-best polygon
is the honest measure of whether this assignment is a decision or a coin flip, and
it is what routes a listing to the review console.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core import geom

log = logging.getLogger("assembly.matching")

#: How much each term contributes at its worst. They sum to 1.0 so a total cost is
#: readable as "how wrong is this match, 0 to 1".
WEIGHTS = {"room_type": 0.34, "area_ratio": 0.30, "aspect": 0.18, "apertures": 0.10,
           "confidence": 0.08}

#: Labels that get confused with each other on real listings, and how much that
#: confusion should cost. The report predicts hall/bedroom and dining/living as
#: the leaders; both are here, and both cost much less than an outright mismatch.
LABEL_AFFINITY: dict[frozenset, float] = {
    frozenset({"living_room", "dining_room"}): 0.15,
    frozenset({"living_room", "reception"}): 0.05,
    frozenset({"dining_room", "reception"}): 0.10,
    frozenset({"living_room", "kitchen"}): 0.35,
    frozenset({"kitchen", "dining_room"}): 0.30,
    frozenset({"hallway", "bedroom"}): 0.55,
    frozenset({"hallway", "storage"}): 0.25,
    frozenset({"bathroom", "wc"}): 0.10,
    frozenset({"bathroom", "utility"}): 0.35,
    frozenset({"study", "bedroom"}): 0.30,
    frozenset({"other_room", "storage"}): 0.20,
    frozenset({"balcony", "garden"}): 0.15,
}
UNKNOWN_LABEL_COST = 0.5


@dataclass
class Match:
    room_id: str
    plan_room_id: str
    cost: float
    breakdown: dict[str, float]
    margin: float | None = None
    pose: dict | None = None
    fit_iou: float | None = None
    confidence: float = 0.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)


def label_cost(a: str | None, b: str | None) -> float:
    """0 for the same room type, 1 for an outright mismatch, in between for the
    confusions we expect to see."""
    if a is None or b is None:
        return UNKNOWN_LABEL_COST
    if a == b:
        return 0.0
    return LABEL_AFFINITY.get(frozenset({a, b}), 1.0)


def area_cost(a_m2: float, b_m2: float) -> float:
    """Symmetric in log space, so 'half the size' and 'twice the size' cost the
    same. A plain ratio does not, and would bias every assignment towards the
    smaller polygon."""
    if a_m2 <= 0 or b_m2 <= 0:
        return 1.0
    r = abs(math.log(a_m2 / b_m2))
    return float(min(1.0, r / math.log(2.2)))     # 1.0 at a factor of ~2.2


def aspect_cost(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return 0.5
    return float(min(1.0, abs(math.log(a / b)) / math.log(2.0)))


def aperture_cost(n_a: int, n_b: int) -> float:
    if n_a == 0 and n_b == 0:
        return 0.3                                # no evidence either way
    return float(min(1.0, abs(n_a - n_b) / 3.0))


def normalise_areas(rooms: list[dict], plan_rooms: list[dict]) -> None:
    """Put both sides on a common scale by dividing through by their own totals.

    Needed for the plans that print no dimensions: their polygons are in pixels,
    the reconstructed rooms are in metres, and an absolute area comparison between
    the two is meaningless. Room *proportions* survive the change of units, and the
    matcher only ever needed proportions — a kitchen is a third of this flat whether
    you measure the flat in metres or in pixels.
    """
    for side in (rooms, plan_rooms):
        total = sum(r.get("area_m2") or 0.0 for r in side)
        if total <= 0:
            continue
        ref = 60.0            # a typical flat, so the log-ratio costs keep their scale
        for r in side:
            if r.get("area_m2"):
                r["area_m2"] = r["area_m2"] / total * ref


def build_cost_matrix(rooms: list[dict], plan_rooms: list[dict]
                      ) -> tuple[np.ndarray, list[list[dict]]]:
    """``(cost, breakdown)`` for every reconstructed room against every polygon."""
    n, m = len(rooms), len(plan_rooms)
    cost = np.zeros((n, m))
    parts: list[list[dict]] = [[{} for _ in range(m)] for _ in range(n)]
    for i, r in enumerate(rooms):
        r_area = r.get("area_m2") or 0.0
        r_aspect = r.get("aspect") or 1.0
        r_ap = len(r.get("apertures") or [])
        for j, p in enumerate(plan_rooms):
            b = {
                "room_type": WEIGHTS["room_type"] * label_cost(r.get("label"), p.get("label")),
                "area_ratio": WEIGHTS["area_ratio"] * area_cost(r_area, p.get("area_m2") or 0.0),
                "aspect": WEIGHTS["aspect"] * aspect_cost(r_aspect, p.get("aspect") or 1.0),
                "apertures": WEIGHTS["apertures"] * aperture_cost(r_ap, len(p.get("aperture_ids") or [])),
                # A room the geometry is unsure about should be easier to move.
                "confidence": WEIGHTS["confidence"] * (1.0 - min(1.0, (r.get("confidence") or 0.5))),
            }
            parts[i][j] = {k: round(v, 5) for k, v in b.items()}
            cost[i, j] = sum(b.values())
    return cost, parts


def assign(rooms: list[dict], plan_rooms: list[dict], *, max_cost: float = 0.72
           ) -> tuple[list[Match], list[str], list[str], np.ndarray]:
    """Hungarian assignment, then drop the matches that are worse than nothing.

    ``max_cost`` is a refusal threshold, not a tuning knob: a match above it is
    saying "this room does not resemble this polygon in type, size, shape or
    openings", and shipping it would place a room somewhere it visibly is not.
    Refusing leaves the room unmatched, which the viewer can render honestly.
    """
    if not rooms or not plan_rooms:
        return [], [r["room_id"] for r in rooms], [p["room_id"] for p in plan_rooms], np.zeros((0, 0))
    cost, parts = build_cost_matrix(rooms, plan_rooms)
    ri, pj = linear_sum_assignment(cost)
    matches: list[Match] = []
    used_rooms: set[str] = set()
    used_plans: set[str] = set()
    for i, j in zip(ri, pj):
        c = float(cost[i, j])
        row = cost[i].copy()
        row[j] = np.inf
        runner_up = float(row.min()) if np.isfinite(row).any() else None
        margin = (runner_up - c) if runner_up is not None else None
        alts = sorted(((plan_rooms[k]["room_id"], round(float(cost[i, k]), 4))
                       for k in range(len(plan_rooms)) if k != j),
                      key=lambda t: t[1])[:3]
        if c > max_cost:
            continue
        matches.append(Match(
            room_id=rooms[i]["room_id"], plan_room_id=plan_rooms[j]["room_id"],
            cost=round(c, 4), breakdown=parts[i][j],
            margin=round(margin, 4) if margin is not None else None,
            confidence=round(_match_confidence(c, margin), 3),
            alternatives=alts,
        ))
        used_rooms.add(rooms[i]["room_id"])
        used_plans.add(plan_rooms[j]["room_id"])
    unmatched_rooms = [r["room_id"] for r in rooms if r["room_id"] not in used_rooms]
    unmatched_plans = [p["room_id"] for p in plan_rooms if p["room_id"] not in used_plans]
    return matches, unmatched_rooms, unmatched_plans, cost


def _match_confidence(cost: float, margin: float | None) -> float:
    c = max(0.0, 1.0 - cost / 0.72)
    if margin is not None:
        # A clear winner is worth as much as a low absolute cost. A 0.05 margin
        # over the runner-up means the assignment could have gone either way.
        c = 0.6 * c + 0.4 * min(1.0, margin / 0.25)
    return min(1.0, max(0.02, c))
