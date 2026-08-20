"""Download the images for an existing golden_set.json, in place.

    python -m pipeline.ingest.fetch_media --set data/golden/golden_set.json

Separated from collect.py so media can be re-fetched (or fetched later, on a
machine with more disk) without re-hitting the portal's HTML pages.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .http import PoliteSession

log = logging.getLogger("fetch_media")


def fetch(set_path: Path, min_interval: float = 0.4, limit_per_listing: int | None = None) -> dict:
    payload = json.loads(set_path.read_text())
    root = set_path.parent
    session = PoliteSession(min_interval=min_interval)
    ok = failed = 0
    total_bytes = 0
    for lst in payload["listings"]:
        refs = lst["photos"] + lst["floorplans"]
        if limit_per_listing:
            refs = lst["photos"][:limit_per_listing] + lst["floorplans"]
        for i, ref in enumerate(refs):
            ext = Path(ref["url"].split("?")[0]).suffix or ".jpg"
            dest = root / "media" / lst["listing_id"] / f"{ref['kind']}_{i:03d}{ext}"
            if dest.exists() and dest.stat().st_size > 1000:
                ref["local_path"] = str(dest.relative_to(root))
                ok += 1
                continue
            try:
                total_bytes += session.download(ref["url"], dest)
                ref["local_path"] = str(dest.relative_to(root))
                ok += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                log.warning("failed %s: %s", ref["url"], e)
        log.info("%s: %d/%d media", lst["listing_id"], ok, ok + failed)
    payload["media"] = {"downloaded": ok, "failed": failed,
                        "bytes": total_bytes, "root": "media/"}
    set_path.write_text(json.dumps(payload, indent=2))
    return payload["media"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, default=Path("data/golden/golden_set.json"))
    ap.add_argument("--min-interval", type=float, default=0.4)
    ap.add_argument("--limit-per-listing", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    r = fetch(a.set, a.min_interval, a.limit_per_listing)
    print(f"media: {r['downloaded']} ok, {r['failed']} failed, {r['bytes']/1e6:.1f} MB")
    return 0 if r["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
