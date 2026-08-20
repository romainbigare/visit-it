"""Dataset CLI.

    python -m pipeline.datasets list
    python -m pipeline.datasets status
    python -m pipeline.datasets ensure swiss_dwellings
    python -m pipeline.datasets bootstrap --tags floorplan
    python -m pipeline.datasets verify
"""
from __future__ import annotations

import argparse
import logging
import sys

from .fetch import (ManualAcquisitionRequired, data_home, ensure, ensure_all,
                    status, verify)
from .registry import REGISTRY


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.datasets", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    e = sub.add_parser("ensure"); e.add_argument("name")
    e.add_argument("--full", action="store_true"); e.add_argument("--force", action="store_true")
    b = sub.add_parser("bootstrap", help="fetch everything auto-fetchable")
    b.add_argument("--tags", default="", help="comma-separated tag filter")
    b.add_argument("--full", action="store_true")
    v = sub.add_parser("verify"); v.add_argument("name", nargs="?")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.cmd == "list":
        print(f"cache: {data_home()}\n")
        print(f"{'name':17}{'acq':8}{'ess.MB':>9}  {'comm':6}{'licence':34}tags")
        print("-" * 108)
        for n, s in REGISTRY.items():
            comm = {True: "yes", False: "no", None: "?"}[s.commercial_ok]
            print(f"{n:17}{s.acquisition:8}{s.total_mb():>9,.0f}  {comm:6}{s.licence[:32]:34}{','.join(s.tags)}")
        return 0

    if a.cmd == "status":
        print(f"cache: {data_home()}\n")
        for n in REGISTRY:
            st = status(n)
            mark = "PRESENT" if st["present"] else "absent "
            print(f"  {mark}  {n:17}{st['local_mb']:>9,.1f} MB  {st['path']}")
        return 0

    if a.cmd == "bootstrap":
        tags = tuple(t.strip() for t in a.tags.split(",") if t.strip())
        res = ensure_all(tags, full=a.full)
        print()
        for n, outcome in res.items():
            print(f"  {n:17} {outcome}")
        return 0 if all(v == "ok" or v.startswith("skipped") for v in res.values()) else 1

    if a.cmd == "verify":
        res = verify(a.name)
        if not res:
            print("nothing in the lockfile yet")
            return 0
        for k, v in res.items():
            print(f"  {v:20} {k}")
        return 0 if all(v == "ok" for v in res.values()) else 1

    try:
        p = ensure(a.name, full=a.full, force=a.force)
    except ManualAcquisitionRequired as ex:
        print(f"\nMANUAL STEP REQUIRED\n\n{ex}\n", file=sys.stderr)
        return 3
    print(f"ready: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
