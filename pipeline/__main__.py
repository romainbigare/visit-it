"""The stage-runner CLI (ARCHITECTURE §3, ROADMAP S1 stream F).

    python -m pipeline run <listing_id> [--from 6] [--profile instant]
    python -m pipeline run --all --profile standard
    python -m pipeline stages                 # the DAG and what each stage reads
    python -m pipeline show <listing_id>      # which artifacts exist, and their QA state
    python -m pipeline latency                # M12: p50/p95 per stage, across all runs

One code path drives this locally and in workers, which is the point: a bug found
on a laptop is the same bug that runs in the queue.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .core import (ArtifactStore, Ledger, STAGE_DEPS, STAGE_ORDER, get_profile,
                   latency_report, normalise_stage, registered, run_listing)

log = logging.getLogger("pipeline")


def load_listings(golden: Path) -> dict[str, dict]:
    payload = json.loads((golden / "golden_set.json").read_text())
    return {l["listing_id"]: l for l in payload["listings"]}


def cmd_run(a) -> int:
    listings = load_listings(a.golden)
    ids = list(listings) if a.all else a.listing_id
    if not ids:
        print("give a listing id, or --all")
        return 2
    if a.split:
        from eval.holdout import load as load_split
        split = load_split(a.golden / "holdout_split.json")
        ids = [i for i in ids if i in split[a.split]]
        print(f"{a.split} split: {len(ids)} listings")
    failures = 0
    for lid in ids:
        if lid not in listings:
            print(f"unknown listing {lid}")
            failures += 1
            continue
        rec = run_listing(listings[lid], profile=a.profile, from_stage=a.from_stage,
                          only=a.only, golden_root=a.golden, store_root=a.store,
                          retries=a.retries, options=_options(a))
        mark = {"ok": "OK ", "partial": "PART", "failed": "FAIL"}[rec.status]
        print(f"{mark} {lid} {rec.total_seconds:6.1f}s  " +
              " ".join(f"{s.stage.split('-')[0]}:{s.status[0]}" for s in rec.stages) +
              (f"  flags={','.join(rec.qa_flags)}" if rec.qa_flags else ""))
        failures += rec.status == "failed"
    return 1 if failures else 0


def _options(a) -> dict:
    o = {}
    if a.no_triage_model:
        o["triage_model"] = False
    if a.max_rooms:
        o["max_rooms"] = a.max_rooms
    return o


def cmd_stages(a) -> int:
    impl = registered()
    prof = get_profile(a.profile)
    print(f"{'stage':<16}{'budget':>8}  reads")
    for s in STAGE_ORDER:
        b = prof.budget(s)
        mark = "" if s in impl else "  (not implemented)"
        print(f"{s:<16}{b if b else '-':>8}  {', '.join(STAGE_DEPS[s]) or '(listing)'}{mark}")
        if s in impl and impl[s].description:
            print(f"{'':16}{'':8}  {impl[s].description}")
    return 0


def cmd_show(a) -> int:
    store = ArtifactStore(a.listing_id, a.store)
    present = store.stages_present()
    if not present:
        print(f"{a.listing_id}: nothing has been run")
        return 1
    print(f"{a.listing_id}: {len(present)} stage(s) with artifacts")
    for s in STAGE_ORDER:
        hist = store.history(s)
        if not hist:
            continue
        p = store.read(s) or {}
        flags = p.get("qa_flags", [])
        print(f"  {s:<16} v{hist[-1]['version']} conf={p.get('confidence', '-'):<6} "
              f"{hist[-1]['sha256'][:10]}  {', '.join(flags[:4])}"
              f"{' …' if len(flags) > 4 else ''}")
    last = Ledger(a.listing_id, a.store).latest()
    if last:
        print(f"  last run {last['run_id']} {last['status']} "
              f"{last['total_seconds']}s profile={last['profile']}")
    return 0


def cmd_latency(a) -> int:
    print(json.dumps(latency_report(a.store, a.profile if a.profile != "all" else None),
                     indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None,
                    help="artifact root (default $VISITIT_RUN_HOME or data/runs)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline for one or more listings")
    r.add_argument("listing_id", nargs="*")
    r.add_argument("--all", action="store_true")
    r.add_argument("--split", choices=["dev", "holdout"],
                   help="restrict to one side of the frozen split")
    r.add_argument("--profile", default="standard", choices=["instant", "standard", "premium"])
    r.add_argument("--from", dest="from_stage", default=None,
                   help="resume from this stage (number, name or both)")
    r.add_argument("--only", nargs="*", default=None, help="run just these stages")
    r.add_argument("--retries", type=int, default=1)
    r.add_argument("--no-triage-model", action="store_true",
                   help="skip the classifier; fall back to portal metadata")
    r.add_argument("--max-rooms", type=int, default=0,
                   help="cap rooms reconstructed per listing (stage 3 is the slow one)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("stages", help="show the DAG, budgets and implementations")
    s.add_argument("--profile", default="standard")
    s.set_defaults(fn=cmd_stages)

    sh = sub.add_parser("show", help="what has been produced for one listing")
    sh.add_argument("listing_id")
    sh.set_defaults(fn=cmd_show)

    la = sub.add_parser("latency", help="M12 latency report over the run ledger")
    la.add_argument("--profile", default="all")
    la.set_defaults(fn=cmd_latency)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    if a.cmd == "run" and a.from_stage:
        a.from_stage = normalise_stage(a.from_stage)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
