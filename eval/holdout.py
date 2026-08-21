"""Freeze the dev/holdout split — Sprint 1 carry-over, ROADMAP §2 and §9 of PHASE-0-REPORT.

    python -m eval.holdout freeze          # writes data/golden/holdout_split.json (once)
    python -m eval.holdout verify          # re-derives the seal, fails if tampered
    python -m eval.holdout show            # print the split

A split chosen after you have seen how the pipeline behaves is not a split. So:

* the assignment is a **pure function** of the golden set — a keyed hash of the
  listing id, not a PRNG whose state depends on iteration order;
* the result is **sealed** with a checksum over the listing ids, and `verify`
  recomputes it, so an accidental edit is loud;
* `freeze` **refuses to overwrite** an existing split. Re-freezing needs
  ``--force --reason "..."`` and the reason is written into the file, where code
  review will see it.

Stratification keeps both sides comparable: every (has_floorplan × price band)
cell is filled in the same proportion, so the holdout is not accidentally all
cheap plan-less flats. G1 is measured on holdout listings **that have a plan**
(ROADMAP §3), which is why the plan-bearing axis is the one we stratify hardest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("holdout")

SPLIT_VERSION = "v1"
#: Salt for the keyed hash. Changing it reshuffles the split, which is exactly the
#: thing we are promising not to do — treat it as immutable.
SPLIT_KEY = "visit-it/holdout/v1"
DEFAULT_HOLDOUT_N = 20

#: Same bands as the collector used (pipeline/ingest/collect.py), so the split's
#: strata line up with the set's own stratification.
PRICE_BANDS: tuple[tuple[int | None, int | None], ...] = (
    (None, 250_000),
    (250_000, 450_000),
    (450_000, 750_000),
    (750_000, None),
)


def price_band(price_gbp: int | None) -> str:
    if price_gbp is None:
        return "unknown"
    for lo, hi in PRICE_BANDS:
        if (lo is None or price_gbp >= lo) and (hi is None or price_gbp < hi):
            lo_s = "0" if lo is None else f"{lo // 1000}"
            hi_s = "inf" if hi is None else f"{hi // 1000}"
            return f"{lo_s}-{hi_s}k"
    return "unknown"


def stratum_of(listing: dict) -> str:
    """The cell a listing belongs to. Plan-bearing first — G1 is measured on plans."""
    plan = "plan" if listing.get("has_floorplan") else "noplan"
    return f"{plan}/{price_band(listing.get('price_gbp'))}"


def _rank(listing_id: str) -> str:
    """Deterministic, order-independent rank. Not a PRNG: a keyed digest of the id.

    This matters more than it looks. A ``random.shuffle`` seeded once and applied
    to a list gives a different answer if the golden set is ever re-ordered or a
    listing is inserted. A per-id digest gives the same answer forever.
    """
    return hashlib.sha256(f"{SPLIT_KEY}:{listing_id}".encode()).hexdigest()


def seal(dev: list[str], holdout: list[str]) -> str:
    """Checksum over the split itself, so tampering is detectable."""
    payload = json.dumps(
        {"version": SPLIT_VERSION, "dev": sorted(dev), "holdout": sorted(holdout)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_split(listings: list[dict], holdout_n: int = DEFAULT_HOLDOUT_N) -> dict:
    """Stratified deterministic split. Returns the payload that gets written."""
    if holdout_n >= len(listings):
        raise ValueError(f"holdout_n={holdout_n} leaves no dev listings out of {len(listings)}")

    by_stratum: dict[str, list[dict]] = {}
    for lst in listings:
        by_stratum.setdefault(stratum_of(lst), []).append(lst)
    for cell in by_stratum.values():
        cell.sort(key=lambda l: _rank(l["listing_id"]))

    total = len(listings)
    holdout: list[str] = []
    dev: list[str] = []
    # Largest-remainder apportionment over strata: each cell contributes its fair
    # share of holdout slots, and the leftover slots go to the cells with the
    # biggest fractional claim. Plain rounding would drift by several listings on
    # a set this small.
    quotas: list[tuple[float, str, int]] = []
    for name, cell in sorted(by_stratum.items()):
        exact = len(cell) * holdout_n / total
        quotas.append((exact - int(exact), name, int(exact)))
    allocated = {name: base for _, name, base in quotas}
    leftover = holdout_n - sum(allocated.values())
    for _frac, name, _base in sorted(quotas, key=lambda q: (-q[0], q[1]))[:leftover]:
        allocated[name] += 1

    for name, cell in sorted(by_stratum.items()):
        take = min(allocated[name], len(cell))
        holdout += [l["listing_id"] for l in cell[:take]]
        dev += [l["listing_id"] for l in cell[take:]]

    strata_table = {
        name: {
            "total": len(cell),
            "holdout": sum(1 for l in cell if l["listing_id"] in set(holdout)),
        }
        for name, cell in sorted(by_stratum.items())
    }
    plan_ids = {l["listing_id"] for l in listings if l.get("has_floorplan")}
    return {
        "version": SPLIT_VERSION,
        "key": SPLIT_KEY,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_listings": total,
        "n_dev": len(dev),
        "n_holdout": len(holdout),
        "holdout_with_plan": sum(1 for i in holdout if i in plan_ids),
        "dev_with_plan": sum(1 for i in dev if i in plan_ids),
        "strata": strata_table,
        "dev": sorted(dev),
        "holdout": sorted(holdout),
        "seal": seal(dev, holdout),
        "rule": (
            "The holdout is measured at gates only. No tuning, no threshold picking, "
            "no failure inspection on these listings. ROADMAP §2, §3."
        ),
    }


def load(split_path: Path) -> dict:
    payload = json.loads(split_path.read_text())
    expected = seal(payload["dev"], payload["holdout"])
    if payload["seal"] != expected:
        raise ValueError(
            f"holdout split seal mismatch in {split_path}: the file has been edited "
            f"since it was frozen (expected {expected[:12]}, found {payload['seal'][:12]})"
        )
    return payload


def split_for(split_path: Path, listing_id: str) -> str:
    p = load(split_path)
    if listing_id in p["holdout"]:
        return "holdout"
    if listing_id in p["dev"]:
        return "dev"
    return "unknown"


def freeze(golden: Path, holdout_n: int = DEFAULT_HOLDOUT_N, force: bool = False,
           reason: str | None = None) -> dict:
    out = golden / "holdout_split.json"
    if out.exists() and not force:
        raise SystemExit(
            f"{out} already exists. The split is frozen; that is the point. "
            f"Re-freeze with --force --reason '...' only if you can defend it."
        )
    listings = json.loads((golden / "golden_set.json").read_text())["listings"]
    payload = make_split(listings, holdout_n)
    if force and out.exists():
        payload["refrozen_from"] = json.loads(out.read_text()).get("seal")
        payload["refreeze_reason"] = reason or "(none given)"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("froze %d holdout / %d dev to %s", payload["n_holdout"], payload["n_dev"], out)
    return payload


def _render(p: dict) -> str:
    rows = "\n".join(
        f"| `{name}` | {v['total']} | {v['holdout']} | {v['total'] - v['holdout']} |"
        for name, v in p["strata"].items()
    )
    return (
        f"split {p['version']} · sealed {p['seal'][:16]} · frozen {p['frozen_at']}\n"
        f"holdout {p['n_holdout']} ({p['holdout_with_plan']} with a plan) · "
        f"dev {p['n_dev']} ({p['dev_with_plan']} with a plan)\n\n"
        f"| stratum | total | holdout | dev |\n|---|---|---|---|\n{rows}\n\n"
        f"holdout: {', '.join(p['holdout'])}\n"
        f"dev:     {', '.join(p['dev'])}\n"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["freeze", "verify", "show"])
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--holdout-n", type=int, default=DEFAULT_HOLDOUT_N)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    ap.add_argument("--reason", default=None, help="why the re-freeze is defensible")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    path = a.golden / "holdout_split.json"
    if a.cmd == "freeze":
        print(_render(freeze(a.golden, a.holdout_n, a.force, a.reason)))
        return 0
    if not path.exists():
        print(f"no split at {path} — run `python -m eval.holdout freeze` first")
        return 1
    try:
        p = load(path)
    except ValueError as e:
        print(f"FAIL {e}")
        return 1
    if a.cmd == "verify":
        print(f"OK seal {p['seal'][:16]} · {p['n_holdout']} holdout / {p['n_dev']} dev")
        return 0
    print(_render(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
