"""Fetch the pretrained wall segmentation weights stage 5 uses.

One 98 MB file, MIT-licensed, from Hugging Face. It is not committed -- weights
are not source (see ``.gitignore``) -- so this is how a fresh checkout gets one.

    python -m tools.fetch_wallnet          # download if missing
    python -m tools.fetch_wallnet --force  # re-download

Without it stage 5 falls back to the ink-mask reading and says so in its QA flags,
which is the pre-existing behaviour rather than a failure.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.floorplan.wallnet import MODEL_PATH, MODEL_URL   # noqa: E402

EXPECTED_BYTES = 97_851_168


def fetch(force: bool = False) -> int:
    if MODEL_PATH.exists() and not force:
        size = MODEL_PATH.stat().st_size
        print(f"already here: {MODEL_PATH} ({size / 1e6:.0f} MB)")
        return 0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".partial")
    print(f"fetching {MODEL_URL}")
    try:
        with urllib.request.urlopen(MODEL_URL) as r, tmp.open("wb") as f:
            total = int(r.headers.get("content-length") or 0)
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="")
        print()
    except Exception as exc:                                    # noqa: BLE001
        tmp.unlink(missing_ok=True)
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    size = tmp.stat().st_size
    if size < EXPECTED_BYTES * 0.9:
        tmp.unlink()
        print(f"got {size} bytes, expected about {EXPECTED_BYTES} — not keeping it",
              file=sys.stderr)
        return 1
    tmp.replace(MODEL_PATH)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:16]
    print(f"wrote {MODEL_PATH} ({size / 1e6:.0f} MB, sha256:{digest}…)")

    from pipeline.floorplan import wallnet
    if wallnet.available():
        print("stage 5 will use it on the next run")
    else:
        print("weights are in place, but torch or segmentation-models-pytorch is "
              "missing — pip install segmentation-models-pytorch safetensors")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    return fetch(ap.parse_args(argv).force)


if __name__ == "__main__":
    raise SystemExit(main())
