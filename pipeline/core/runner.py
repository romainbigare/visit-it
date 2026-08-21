"""The DAG runner: execute stages in order, with retries, budgets and a ledger.

Design notes worth keeping:

* **A failed stage skips its dependents, it does not feed them garbage.** The
  expensive kind of bug in this system is a stage that produces a plausible
  artifact from broken input, because the failure then surfaces three stages
  later wearing a disguise.
* **Latency is measured from day one** (AD-17, M12). Every stage is timed
  against its profile's budget and the overrun is recorded, not just logged.
* **Re-runs are cheap.** ``--from 6`` reads stages 0-5 from the store.
"""
from __future__ import annotations

import logging
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .artifacts import STAGE_ARTIFACTS, ArtifactStore, SchemaError
from .ledger import Ledger, RunRecord, StageRecord, new_run_id
from .profiles import Profile, get_profile
from .stages import STAGE_DEPS, STAGE_ORDER, StageContext, get_stage, normalise_stage

log = logging.getLogger("runner")


def code_version() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def run_listing(listing: dict, *, profile: str = "standard", from_stage: str | None = None,
                only: list[str] | None = None, golden_root: Path | None = None,
                store_root: Path | None = None, retries: int = 1,
                options: dict | None = None, stop_on_error: bool = False) -> RunRecord:
    """Run the pipeline for one listing. Returns the ledger record."""
    prof: Profile = get_profile(profile)
    listing_id = listing["listing_id"]
    store = ArtifactStore(listing_id, store_root)
    ledger = Ledger(listing_id, store_root)
    run_id = new_run_id()

    if only:
        wanted = [normalise_stage(s) for s in only]
    else:
        start = STAGE_ORDER.index(normalise_stage(from_stage)) if from_stage else 0
        wanted = STAGE_ORDER[start:]

    rec = RunRecord(run_id=run_id, listing_id=listing_id, profile=prof.name,
                    started_at=_now(), code_version=code_version(),
                    from_stage=wanted[0] if wanted else None)
    opts = dict(options or {})
    # Human corrections survive a re-run, so a listing a reviewer fixed stays fixed
    # when the nightly batch reprocesses it.
    from .overrides import load as load_overrides
    stored = load_overrides(listing_id, store_root)
    if stored:
        opts.setdefault("overrides", stored)
        rec_flags = ["override_applied"]
    else:
        rec_flags = []
    ctx = StageContext(listing_id=listing_id, listing=listing, store=store, profile=prof,
                       golden_root=golden_root, options=opts)
    rec.qa_flags = rec_flags

    blocked: set[str] = set()
    t_run = time.perf_counter()
    for name in wanted:
        deps = STAGE_DEPS.get(name, ())
        if any(d in blocked for d in deps):
            reason = f"upstream {', '.join(d for d in deps if d in blocked)} unavailable"
            rec.stages.append(StageRecord(stage=name, status="skipped", error=reason))
            blocked.add(name)
            log.info("%s %s skipped (%s)", listing_id, name, reason)
            continue
        missing = [d for d in deps if store.read(d) is None]
        if missing:
            reason = f"missing upstream artifact(s): {', '.join(missing)}"
            rec.stages.append(StageRecord(stage=name, status="skipped", error=reason))
            blocked.add(name)
            log.info("%s %s skipped (%s)", listing_id, name, reason)
            continue

        sr = _run_one(ctx, name, prof, run_id, retries)
        rec.stages.append(sr)
        if sr.status == "failed":
            blocked.add(name)
            if stop_on_error:
                break
        elif sr.status == "skipped":
            blocked.add(name)
        rec.qa_flags = sorted(set(rec.qa_flags) | set(sr.qa_flags))

    rec.total_seconds = round(time.perf_counter() - t_run, 3)
    rec.finished_at = _now()
    statuses = {s.status for s in rec.stages}
    rec.status = ("failed" if statuses == {"failed"} else
                  "partial" if ("failed" in statuses or "skipped" in statuses) else "ok")
    if rec.total_seconds > prof.slo_s:
        rec.qa_flags = sorted(set(rec.qa_flags) | {"slo_breach"})
    ledger.write(rec)
    log.info("%s run %s: %s in %.2fs (%d stages ok)", listing_id, run_id, rec.status,
             rec.total_seconds, sum(1 for s in rec.stages if s.status == "ok"))
    return rec


def _run_one(ctx: StageContext, name: str, prof: Profile, run_id: str,
             retries: int) -> StageRecord:
    budget = prof.budget(name)
    try:
        stage = get_stage(name)
    except KeyError as e:
        return StageRecord(stage=name, status="skipped", error=str(e), budget_s=budget)

    last_err: str | None = None
    for attempt in range(1, retries + 2):
        t0 = time.perf_counter()
        try:
            result = stage.run(ctx)
            secs = round(time.perf_counter() - t0, 3)
            if result.skipped:
                return StageRecord(stage=name, status="skipped", attempts=attempt,
                                   seconds=secs, budget_s=budget,
                                   error=result.skip_reason, engine=result.engine)
            payload = dict(result.payload)
            payload.setdefault("timing_s", secs)
            art = ctx.store.write(name, payload, run_id=run_id)
            for bname, data in result.binaries.items():
                ctx.store.write_binary(bname, data, run_id=run_id)
            over = bool(budget and secs > budget)
            if over:
                # AD-17 / cross-cutting rule 7: an overrun is data, not a shrug.
                log.warning("%s %s took %.2fs, over the %s budget of %.2fs",
                            ctx.listing_id, name, secs, prof.name, budget)
            return StageRecord(
                stage=name, status="ok", attempts=attempt, seconds=secs,
                budget_s=budget, over_budget=over,
                artifact=STAGE_ARTIFACTS.get(name, ("", f"{name}.json"))[1],
                artifact_sha256=art.sha256,
                confidence=payload.get("confidence"),
                qa_flags=list(payload.get("qa_flags", [])),
                engine=result.engine or payload.get("engine"),
            )
        except SchemaError as e:
            # A schema failure is a code bug; retrying it wastes the operator's time.
            secs = round(time.perf_counter() - t0, 3)
            log.error("%s %s emitted an invalid artifact: %s", ctx.listing_id, name, e)
            return StageRecord(stage=name, status="failed", attempts=attempt, seconds=secs,
                               budget_s=budget, error=f"SchemaError: {e}")
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            log.warning("%s %s attempt %d failed: %s", ctx.listing_id, name, attempt, last_err)
            log.debug("%s", traceback.format_exc())
    return StageRecord(stage=name, status="failed", attempts=retries + 1,
                       budget_s=budget, error=last_err)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
