"""Write a contact sheet for every listing that has been run, plus an index.

    python -m tools.build_sheets --out data/runs/_sheets

Static HTML with everything inlined, so the whole directory can be zipped and sent
to someone who does not have the repo. That has been the fastest way to get a
second pair of eyes on a bad reconstruction.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core.artifacts import ArtifactStore   # noqa: E402
from services.review.contact_sheet import render    # noqa: E402
from services.review.server import listing_rows     # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/runs/_sheets"))
    a = ap.parse_args(argv)
    listings = {l["listing_id"]: l
                for l in json.loads((a.golden / "golden_set.json").read_text())["listings"]}
    have = set(ArtifactStore.list_listings(a.store))
    a.out.mkdir(parents=True, exist_ok=True)
    made = 0
    for lid, listing in listings.items():
        if lid not in have:
            continue
        (a.out / f"{lid}.html").write_text(render(listing, a.store, a.golden))
        made += 1
    rows = "".join(
        f"<tr><td><a href='{r['listing_id']}.html'>{r['listing_id']}</a></td>"
        f"<td>{html.escape(r['address'][:46])}</td><td>{r['status']}</td>"
        f"<td>{r['stages_ok']}/10</td><td>{r['rooms']}</td>"
        f"<td>{r['confidence'] if r['confidence'] is not None else '—'}</td>"
        f"<td>{' '.join(r['blocking']) or '—'}</td></tr>"
        for r in listing_rows({k: v for k, v in listings.items() if k in have}, a.store))
    (a.out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>contact sheets</title>"
        "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ddd;padding:4px 8px;font-size:12px}</style>"
        f"<h1>Contact sheets — {made} listings</h1><table>"
        "<tr><th>listing</th><th>address</th><th>status</th><th>stages</th>"
        f"<th>rooms</th><th>conf</th><th>blocking</th></tr>{rows}</table>")
    print(f"wrote {made} contact sheets to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
