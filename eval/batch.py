"""The nightly batch runner and its regression alerts (ROADMAP S5, stream F).

Reprocess the whole golden set, score it, and — the part that matters — **compare
against the last run and shout when a metric drops**. The roadmap sets the bar at
2 sigma, which is the right shape of rule and needs one honest caveat: with a
handful of history points, sigma is barely estimated. So the alert fires on the
larger of "2 sigma" and an absolute floor, and the report says which rule fired.

    python -m eval.batch --split dev              # run, score, compare, write history
    python -m eval.batch --score-only             # just score what is already there
    python -m eval.batch --check                  # exit non-zero if anything regressed
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

from pipeline.core import run_listing

from . import harness
from .holdout import load as load_split

log = logging.getLogger("batch")

HISTORY = Path("eval/results/history.jsonl")

#: Direction each metric improves in, and the absolute drop that is worth waking
#: someone for even when the variance estimate says it is fine.
METRIC_RULES = {
    "M1": {"lower_is_better": True, "floor": 3.0},     # percentage points
    "M2": {"lower_is_better": True, "floor": 2.5},
    "M3": {"lower_is_better": False, "floor": 0.08},   # fraction
    "M4": {"lower_is_better": False, "floor": 0.05},   # IoU
    "M5": {"lower_is_better": False, "floor": 5.0},    # percentage points
}
SIGMA = 2.0


def append_history(report: dict, path: Path = HISTORY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "at": report["generated_at"], "split": report["split"],
        "channel": report["channel"],
        "complete": report["summary"]["complete"],
        "metrics": {k: v["median"] for k, v in report["summary"].get("metrics", {}).items()},
        "g1": {k: v["pass_rate"] for k, v in report["summary"].get("g1", {}).items()},
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def history(path: Path = HISTORY, split: str | None = None,
            channel: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if split:
        rows = [r for r in rows if r["split"] == split]
    if channel:
        rows = [r for r in rows if r["channel"] == channel]
    return rows


def regressions(report: dict, path: Path = HISTORY) -> list[dict]:
    """What got worse, and by which rule we decided it counts."""
    past = history(path, report["split"], report["channel"])
    if len(past) < 1:
        return []
    now = {k: v["median"] for k, v in report["summary"].get("metrics", {}).items()}
    out: list[dict] = []
    for mid, rule in METRIC_RULES.items():
        cur = now.get(mid)
        series = [r["metrics"].get(mid) for r in past]
        series = [v for v in series if v is not None]
        if cur is None or not series:
            continue
        prev = series[-1]
        sigma = statistics.pstdev(series) if len(series) >= 3 else 0.0
        threshold = max(SIGMA * sigma, rule["floor"])
        drop = (cur - prev) if rule["lower_is_better"] else (prev - cur)
        if drop > threshold:
            out.append({
                "metric": mid, "previous": prev, "current": cur,
                "drop": round(drop, 3), "threshold": round(threshold, 3),
                "rule": ("2σ" if SIGMA * sigma >= rule["floor"] else "absolute floor"),
                "history_points": len(series),
                "caveat": ("σ over fewer than 3 runs is not an estimate; the floor "
                           "rule is doing the work" if len(series) < 3 else None),
            })
    return out


def run(golden: Path, split: str = "dev", store_root: Path | None = None,
        channel: str = "full", reprocess: bool = True, profile: str = "standard",
        max_rooms: int = 0) -> dict:
    p = load_split(golden / "holdout_split.json")
    ids = p[split]
    listings = harness.__dict__  # noqa: F841  (kept explicit below for clarity)
    payload = json.loads((golden / "golden_set.json").read_text())
    by_id = {l["listing_id"]: l for l in payload["listings"]}

    runs = []
    if reprocess:
        for lid in ids:
            rec = run_listing(by_id[lid], profile=profile, golden_root=golden,
                              store_root=store_root,
                              options={"max_rooms": max_rooms} if max_rooms else {})
            runs.append({"listing_id": lid, "status": rec.status,
                         "seconds": rec.total_seconds,
                         "stages_ok": sum(1 for s in rec.stages if s.status == "ok")})
            log.info("%s %s %.1fs", lid, rec.status, rec.total_seconds)

    report = harness.run(golden, split, store_root, channel)
    report["batch"] = {
        "reprocessed": len(runs),
        "runs": runs,
        "failed": [r["listing_id"] for r in runs if r["status"] == "failed"],
        "partial": [r["listing_id"] for r in runs if r["status"] == "partial"],
        "total_seconds": round(sum(r["seconds"] for r in runs), 1),
    }
    report["regressions"] = regressions(report)
    return report


def render(report: dict) -> str:
    lines = [harness.render(report)]
    b = report.get("batch") or {}
    if b:
        lines.append(f"\n## Batch\n\n{b['reprocessed']} reprocessed in "
                     f"{b['total_seconds']}s · {len(b['failed'])} failed, "
                     f"{len(b['partial'])} partial\n")
        if b["failed"]:
            lines.append("Failed: " + ", ".join(b["failed"]) + "\n")
        if b["partial"]:
            lines.append("Partial: " + ", ".join(b["partial"]) + "\n")
    regs = report.get("regressions") or []
    lines.append("\n## Regression check\n")
    if not regs:
        lines.append("\nNo metric regressed against the previous run.\n")
    else:
        lines.append("\n| metric | previous | now | drop | threshold | rule |\n"
                     "|---|---|---|---|---|---|\n")
        for r in regs:
            lines.append(f"| {r['metric']} | {r['previous']} | {r['current']} | "
                         f"{r['drop']} | {r['threshold']} | {r['rule']} |\n")
        caveats = {r["caveat"] for r in regs if r.get("caveat")}
        for c in caveats:
            lines.append(f"\n> {c}\n")
    return "".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--channel", default="full", choices=["full", "plan"])
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--max-rooms", type=int, default=0)
    ap.add_argument("--score-only", action="store_true",
                    help="do not reprocess; just score what is on disk")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if a metric regressed (for CI)")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rep = run(a.golden, a.split, a.store, a.channel, not a.score_only, a.profile,
              a.max_rooms)
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"batch_{a.channel}_{a.split}"
    (a.out / f"{name}.json").write_text(json.dumps(rep, indent=2) + "\n")
    (a.out / f"{name}.md").write_text(render(rep))
    print(render(rep))
    if not a.no_history:
        append_history(rep)
    if a.check and rep.get("regressions"):
        log.error("%d metric(s) regressed", len(rep["regressions"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
