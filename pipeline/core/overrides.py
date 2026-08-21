"""Human corrections, stored beside the artifacts and applied by the stages.

The report's sharpest operational finding is that **human review time rivals all
compute**, so the review console is not a nicety — reducing the review *rate* is
worth more than any GPU optimisation, and making each review fast is worth more
than making the model slightly better. Which means corrections have to be:

* **Cheap to apply.** An override is a small JSON file, not a re-annotation.
* **Surgical.** Nudging one assignment re-runs stages 6-9 in seconds, not the
  whole pipeline including the 80-second geometry pass.
* **Durable.** Overrides survive a re-run, so a corrected listing stays corrected
  when the nightly batch reprocesses it.
* **Visible.** Every artifact an override touched carries ``edited_by_human`` and
  an ``override_applied`` QA flag, so a hand-fixed listing is never mistaken for a
  clean automatic one in the metrics.

    {"assembly": {"pin": {"kitchen": "p0"}, "reject": ["bathroom"]},
     "plan":     {"rooms": {"p2": {"label": "living_room"}}},
     "scale":    {"px_per_metre": 152.0}}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import data_root

log = logging.getLogger("overrides")


def path_for(listing_id: str, root: Path | None = None) -> Path:
    return (root or data_root()) / listing_id / "overrides.json"


def load(listing_id: str, root: Path | None = None) -> dict:
    p = path_for(listing_id, root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:  # noqa: BLE001
        log.warning("%s: unreadable overrides (%s); ignoring", listing_id, e)
        return {}


def save(listing_id: str, payload: dict, root: Path | None = None) -> Path:
    p = path_for(listing_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def merge(listing_id: str, patch: dict, root: Path | None = None) -> dict:
    """Deep-merge a patch into the stored overrides and persist it."""
    cur = load(listing_id, root)
    for section, value in patch.items():
        if isinstance(value, dict) and isinstance(cur.get(section), dict):
            for k, v in value.items():
                if isinstance(v, dict) and isinstance(cur[section].get(k), dict):
                    cur[section][k].update(v)
                else:
                    cur[section][k] = v
        else:
            cur[section] = value
    save(listing_id, cur, root)
    return cur


def clear(listing_id: str, section: str | None = None, root: Path | None = None) -> dict:
    cur = load(listing_id, root)
    if section is None:
        cur = {}
    else:
        cur.pop(section, None)
    save(listing_id, cur, root)
    return cur


#: Which stages an override in each section invalidates. Nudging an assignment
#: does not need the geometry re-run, and re-running it would cost 80 seconds to
#: reach the same answer.
RERUN_FROM = {"plan": "5-plan", "assembly": "6-assembly", "scale": "7-scale",
              "layout": "4-layout", "triage": "0-triage"}


def rerun_stage_for(sections: list[str]) -> str:
    """Earliest stage that has to re-run, given which sections changed."""
    from .stages import STAGE_ORDER
    stages = [RERUN_FROM[s] for s in sections if s in RERUN_FROM]
    if not stages:
        return STAGE_ORDER[0]
    return min(stages, key=STAGE_ORDER.index)
