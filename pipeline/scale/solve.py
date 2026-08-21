"""AD-7: one global scale scalar, by weighted least squares in log space.

Why log space. The constraints are a mixture of lengths (door heights, ceiling
heights) and areas (the stated floor area, printed room areas), which respond to
scale as ``s`` and ``s²``. In log space both become linear —
``log target = log observed + power · log s`` — so one weighted least-squares fit
handles them together, and a 10% error costs the same whether the model is 10%
too big or 10% too small. In linear space it would not, and the fit would be
biased towards making everything smaller.

Why a single scalar at all. Per-room scales would fit better and mean less: the
flat is one object, and a reconstruction where the kitchen and the bedroom
disagree about what a metre is cannot be assembled into a shell.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("scale.solve")

#: Weight per constraint kind, from ARCHITECTURE §5's rationale column. These are
#: relative confidences, not tuned parameters: a dimension printed on the plan is
#: worth more than a ceiling-height prior, and both are worth more than raw metric
#: depth with its honest ±10-15%.
KIND_WEIGHT = {
    "plan_room_area": 3.0,
    "plan_total_area": 2.5,
    "stated_area": 2.0,
    "door_height": 1.5,
    "ceiling_height": 1.0,
    "fixture": 0.4,
    "metric_depth": 0.6,
}

#: UK ceiling prior. Modern stock clusters at 2.4-2.6 m, period stock runs higher;
#: the constraint is deliberately weak and exists mainly to catch a reconstruction
#: that has gone badly wrong.
CEILING_PRIOR_M = 2.55
#: EU/UK internal door leaf, the standard 1981 mm plus frame.
DOOR_HEIGHT_M = 2.04


@dataclass
class Constraint:
    kind: str
    target: float           # what the constraint says the quantity is
    observed: float         # what the unscaled reconstruction says it is
    power: int = 1          # 1 for a length, 2 for an area
    detail: str = ""
    source: str = ""
    weight: float | None = None
    used: bool = True
    residual_log: float = 0.0
    residual_pct: float = 0.0

    def w(self) -> float:
        return self.weight if self.weight is not None else KIND_WEIGHT.get(self.kind, 1.0)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "target": round(self.target, 5),
                "observed": round(self.observed, 5), "power": self.power,
                "weight": round(self.w(), 4), "residual_log": round(self.residual_log, 5),
                "residual_pct": round(self.residual_pct, 3), "source": self.source,
                "used": self.used}


@dataclass
class Solution:
    scale: float = 1.0
    log_scale: float = 0.0
    sigma_log: float = 0.0
    residual_rms_pct: float = 0.0
    constraints: list[Constraint] = field(default_factory=list)
    n_used: int = 0
    rejected: list[str] = field(default_factory=list)


def solve(constraints: list[Constraint], *, robust_rounds: int = 2,
          reject_sigma: float = 2.5) -> Solution:
    """Weighted least squares on ``log s``, with a couple of robust re-fits.

    The re-fit matters more than it looks. One badly-OCR'd room dimension — "9.91"
    where the plan printed "5.91" — is a 68% outlier that would drag the global
    scale by several percent. Rejecting past 2.5 sigma removes it; keeping the
    rejection *recorded* means we can tell the difference between a clean solve
    and one that had to throw evidence away.
    """
    usable = [c for c in constraints
              if c.target > 0 and c.observed > 0 and math.isfinite(c.target)
              and math.isfinite(c.observed)]
    sol = Solution(constraints=list(constraints))
    if not usable:
        return sol

    active = list(usable)
    rejected: list[str] = []
    for _ in range(robust_rounds + 1):
        y = np.array([math.log(c.target) - math.log(c.observed) for c in active])
        a = np.array([float(c.power) for c in active])
        w = np.array([c.w() for c in active])
        denom = float((w * a * a).sum())
        if denom <= 0:
            break
        ls = float((w * a * y).sum() / denom)
        resid = y - a * ls
        if len(active) > 2:
            # Median absolute deviation, not RMS. RMS is inflated by the very
            # outlier we are trying to find — a single mis-OCR'd "9.91" for
            # "5.91" raises the threshold enough to protect itself.
            med = float(np.median(resid))
            mad = 1.4826 * float(np.median(np.abs(resid - med)))
            if mad > 1e-6:
                keep = np.abs(resid - med) <= reject_sigma * mad
                if keep.sum() >= 2 and keep.sum() < len(active):
                    for c, k in zip(active, keep):
                        if not k:
                            c.used = False
                            rejected.append(f"{c.kind}:{c.detail or c.source}")
                    active = [c for c, k in zip(active, keep) if k]
                    continue
        break

    y = np.array([math.log(c.target) - math.log(c.observed) for c in active])
    a = np.array([float(c.power) for c in active])
    w = np.array([c.w() for c in active])
    ls = float((w * a * y).sum() / max(float((w * a * a).sum()), 1e-12))

    for c in constraints:
        r = (math.log(c.target) - math.log(c.observed) - c.power * ls
             if c.target > 0 and c.observed > 0 else float("nan"))
        c.residual_log = 0.0 if math.isnan(r) else r
        c.residual_pct = 0.0 if math.isnan(r) else 100.0 * (math.exp(abs(r) / max(c.power, 1)) - 1.0)

    resid = y - a * ls
    dof = max(1, len(active) - 1)
    sigma = float(np.sqrt((w * resid ** 2).sum() / (w.sum() * dof))) if w.sum() > 0 else 0.0
    rms_pct = (100.0 * (math.exp(float(np.sqrt((resid ** 2).mean()))) - 1.0)
               if len(resid) else 0.0)

    sol.scale = float(math.exp(ls))
    sol.log_scale = ls
    sol.sigma_log = sigma
    sol.residual_rms_pct = rms_pct
    sol.n_used = len(active)
    sol.rejected = rejected
    return sol


def quality(sol: Solution, n_independent_kinds: int) -> float:
    """How much to trust this scale, 0 to 1.

    Rewards agreeing evidence from *different kinds* of constraint, not just more
    of the same: three room areas from one plan are one measurement made three
    times, whereas a room area agreeing with a ceiling height is two.
    """
    if sol.n_used == 0:
        return 0.0
    q = 0.25 + 0.15 * min(3, n_independent_kinds)
    q += 0.2 * max(0.0, 1.0 - sol.residual_rms_pct / 25.0)
    q += 0.1 * min(1.0, sol.n_used / 4.0)
    if sol.rejected:
        q -= 0.1
    return round(min(1.0, max(0.02, q)), 3)
