"""CLI:  python -m pipeline.datasets list | status | ensure <name>"""
from __future__ import annotations

import argparse
import logging
import sys

from .fetch import LicenceRefused, ManualAcquisitionRequired, ensure, status, data_home
from .registry import REGISTRY


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.datasets", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show the registry")
    sub.add_parser("status", help="show what is present locally")
    e = sub.add_parser("ensure", help="download a dataset if absent")
    e.add_argument("name")
    e.add_argument("--accept-research-licence", action="store_true")
    e.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.cmd == "list":
        print(f"cache: {data_home()}\n")
        print(f"{'name':18}{'verdict':18}{'acq':8}{'urls':6}licence")
        print("-" * 96)
        for n, s in REGISTRY.items():
            print(f"{n:18}{s.verdict:18}{s.acquisition:8}{('yes' if s.urls else 'no'):6}{s.licence[:44]}")
        return 0

    if a.cmd == "status":
        print(f"cache: {data_home()}\n")
        for n in REGISTRY:
            st = status(n)
            print(f"  {'PRESENT' if st['present'] else 'absent ':8} {n:18} {st['path']}")
        return 0

    try:
        p = ensure(a.name, accept_research_licence=a.accept_research_licence or None,
                   force=a.force)
    except LicenceRefused as ex:
        print(f"\nREFUSED\n{ex}\n", file=sys.stderr)
        return 2
    except ManualAcquisitionRequired as ex:
        print(f"\nMANUAL STEP REQUIRED\n{ex}\n", file=sys.stderr)
        return 3
    print(f"ready: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
