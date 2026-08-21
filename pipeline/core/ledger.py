"""The run ledger — one JSON record per pipeline run, per listing.

This is where M12 (latency, p50/p95 per stage) comes from, and where "why did
this listing regress last night" gets answered. Kept as flat files rather than
Postgres for now: at 30 listings a directory of JSON is faster to grep than a
database is to query, and AD-13 says do not start with the heavy thing.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import data_root


@dataclass
class StageRecord:
    stage: str
    status: str                        # ok | failed | skipped | cached
    attempts: int = 1
    seconds: float = 0.0
    budget_s: float | None = None
    over_budget: bool = False
    artifact: str | None = None
    artifact_sha256: str | None = None
    confidence: float | None = None
    qa_flags: list[str] = field(default_factory=list)
    error: str | None = None
    engine: str | None = None


@dataclass
class RunRecord:
    run_id: str
    listing_id: str
    profile: str
    started_at: str
    finished_at: str | None = None
    code_version: str | None = None
    from_stage: str | None = None
    stages: list[StageRecord] = field(default_factory=list)
    status: str = "ok"
    total_seconds: float = 0.0
    qa_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = [asdict(s) if not isinstance(s, dict) else s for s in self.stages]
        return d


class Ledger:
    """Append-only run history for one listing."""

    def __init__(self, listing_id: str, root: Path | None = None):
        self.dir = (root or data_root()) / listing_id / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, rec: RunRecord) -> Path:
        p = self.dir / f"{rec.run_id}.json"
        p.write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
        return p

    def runs(self) -> list[dict]:
        return sorted((json.loads(p.read_text()) for p in self.dir.glob("*.json")),
                      key=lambda r: r["started_at"])

    def latest(self) -> dict | None:
        rs = self.runs()
        return rs[-1] if rs else None


def new_run_id() -> str:
    """Sortable, human-readable, unique enough for a single box."""
    import uuid
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def latency_report(root: Path | None = None, profile: str | None = None) -> dict:
    """M12. p50/p95 per stage and end to end, across every recorded run.

    Reports the *number of runs* alongside every percentile: a p95 over four runs
    is the maximum wearing a suit, and the report should say so.
    """
    root = root or data_root()
    per_stage: dict[str, list[float]] = {}
    totals: list[float] = []
    runs = 0
    for listing_dir in sorted(root.glob("*/runs")) if root.exists() else []:
        for p in listing_dir.glob("*.json"):
            r = json.loads(p.read_text())
            if profile and r.get("profile") != profile:
                continue
            runs += 1
            totals.append(r.get("total_seconds", 0.0))
            for s in r.get("stages", []):
                if s.get("status") in ("ok", "cached"):
                    per_stage.setdefault(s["stage"], []).append(s.get("seconds", 0.0))

    def pct(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        return round(xs[i], 3)

    return {
        "metric": "M12",
        "profile": profile or "all",
        "n_runs": runs,
        "caveat": ("percentiles over fewer than 20 runs are indicative only"
                   if runs < 20 else None),
        "end_to_end": {"p50": pct(totals, 0.5), "p95": pct(totals, 0.95),
                       "mean": round(statistics.fmean(totals), 3) if totals else None},
        "stages": {
            s: {"n": len(v), "p50": pct(v, 0.5), "p95": pct(v, 0.95),
                "max": round(max(v), 3)}
            for s, v in sorted(per_stage.items())
        },
    }
