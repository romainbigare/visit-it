"""Service profiles — AD-16/17/18, amendment A.

A profile is *config*, not a second pipeline: per-stage engine binding, per-stage
latency budget, and an SLO. ``pipeline run <listing> --profile instant``.

Phase 1 only has quality bindings for some stages, so `instant` currently falls
back to the same engine as `standard` in places. The budgets are still asserted
from day one, which is the point of AD-17: a stage that quietly grows past its
budget is found by CI, not by a customer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Per-stage wall-clock budget in seconds. `instant` must total <= 10 s (VARIANTS
#: §6). Phase 1 stages are CPU-cheap; the budgets that will actually bite are
#: stage 3 and stage 8, which arrive properly in Phase 2.
_INSTANT_BUDGETS = {
    "0-triage": 1.0, "1-conditioning": 0.8, "2-grouping": 1.0, "3-geometry": 2.5,
    "4-layout": 1.0, "5-plan": 1.5, "6-assembly": 0.3, "7-scale": 0.2,
    "8-shell": 0.8, "9-package": 0.4,
}
_STANDARD_BUDGETS = {k: v * 20 for k, v in _INSTANT_BUDGETS.items()}
_PREMIUM_BUDGETS = {k: v * 120 for k, v in _INSTANT_BUDGETS.items()}


@dataclass(frozen=True)
class Profile:
    name: str
    engines: dict[str, str]
    budgets_s: dict[str, float]
    slo_s: float
    cogs_target_gbp: float
    allow_vlm: bool
    allow_human_gate: bool
    notes: str = ""
    qa_policy: str = "review"          # review | degrade
    extras: dict = field(default_factory=dict)

    def budget(self, stage: str) -> float | None:
        return self.budgets_s.get(stage)

    def engine(self, stage: str, default: str) -> str:
        return self.engines.get(stage, default)


PROFILES: dict[str, Profile] = {
    "instant": Profile(
        name="instant",
        engines={"3-geometry": "moge2", "4-layout": "fast", "5-plan": "raster",
                 "8-shell": "extrude"},
        budgets_s=_INSTANT_BUDGETS,
        slo_s=10.0,
        cogs_target_gbp=0.03,
        allow_vlm=False,
        allow_human_gate=False,
        qa_policy="degrade",
        notes=("Hot-path bans (AD-17): no frontier-VLM calls, no per-scene optimisation, "
               "no human gates, no cold model loads. A listing that fails checks degrades "
               "or refuses with a reason — it never waits for a person."),
    ),
    "standard": Profile(
        name="standard",
        engines={"3-geometry": "mapanything", "4-layout": "full", "5-plan": "raster",
                 "8-shell": "extrude"},
        budgets_s=_STANDARD_BUDGETS,
        slo_s=300.0,
        cogs_target_gbp=0.15,
        allow_vlm=True,
        allow_human_gate=True,
        qa_policy="review",
        notes="Minutes are fine; fidelity is the product.",
    ),
    "premium": Profile(
        name="premium",
        engines={"3-geometry": "mapanything", "4-layout": "full", "5-plan": "raster",
                 "8-shell": "extrude"},
        budgets_s=_PREMIUM_BUDGETS,
        slo_s=1800.0,
        cogs_target_gbp=1.00,
        allow_vlm=True,
        allow_human_gate=True,
        qa_policy="review",
        notes="Everything standard does, plus whatever quality passes exist.",
    ),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; have {', '.join(PROFILES)}") from None
