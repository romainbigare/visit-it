"""Bring the GPU model's room predictions into the repo so stage 5 can use them.

``notebooks/plan_reading_modal.ipynb`` runs the room-polygon model on a GPU and
hands back a zip. This unpacks whichever reading you pick into
``data/room_predictions/<listing_id>.json``, in the source image's pixels, which
is where ``pipeline.floorplan.roomfinder`` looks for it.

    python -m tools.import_room_predictions results.zip
    python -m tools.import_room_predictions results.zip --reading both_models_together
    python -m tools.import_room_predictions --list results.zip
    python -m tools.import_room_predictions --clear

Pick the reading the notebook's table said was best at *finding rooms* -- that is
what these are used for. Their corners get replaced by ours either way.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.floorplan.roomfinder import PREDICTION_DIR   # noqa: E402

#: Readings the notebook writes that are ours, not the model's. Importing one of
#: these would feed our own rooms back in as if they were a second opinion.
OURS = {"what_we_have_today", "both_models_together"}


def readings_in(archive: Path) -> dict[str, str]:
    with zipfile.ZipFile(archive) as z:
        return {Path(n).stem: n for n in z.namelist()
                if n.endswith(".json") and Path(n).stem != "ladder"}


def load_reading(archive: Path, name: str) -> dict:
    with zipfile.ZipFile(archive) as z:
        return json.loads(z.read(readings_in(archive)[name]))


def write(reading: dict, source: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for listing_id, rec in reading.items():
        rooms = [r for r in (rec.get("rooms") or []) if len(r.get("polygon_px") or []) >= 3]
        if not rooms:
            continue
        (out_dir / f"{listing_id}.json").write_text(json.dumps({
            "listing_id": listing_id,
            "source": "raster2seq",
            "checkpoint": source,
            "space": "source_image_pixels",
            "rooms": [{"polygon_px": r["polygon_px"], "label": r.get("label", "")}
                      for r in rooms],
        }, indent=1))
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", nargs="?", type=Path)
    ap.add_argument("--reading", help="which reading in the zip to import")
    ap.add_argument("--list", action="store_true", help="show what is in the zip")
    ap.add_argument("--clear", action="store_true", help="delete every imported prediction")
    ap.add_argument("--out", type=Path, default=PREDICTION_DIR)
    args = ap.parse_args(argv)

    if args.clear:
        if args.out.exists():
            shutil.rmtree(args.out)
            print(f"removed {args.out}")
        else:
            print("nothing to remove")
        return 0

    if not args.archive or not args.archive.exists():
        ap.error("give me the zip the notebook produced")

    available = readings_in(args.archive)
    if args.list or not args.reading:
        print(f"readings in {args.archive.name}:")
        for name in sorted(available):
            note = "  (ours, not the model's — importing this feeds our own rooms back in)" \
                if name in OURS else ""
            print(f"  {name}{note}")
        if args.list:
            return 0
        print("\npick one with --reading")
        return 1

    if args.reading not in available:
        print(f"no reading called {args.reading!r} — see --list", file=sys.stderr)
        return 1
    if args.reading in OURS:
        print(f"'{args.reading}' is our own reading, not the model's. Importing it would "
              f"feed our rooms back in as their own seeds.", file=sys.stderr)
        return 1

    n = write(load_reading(args.archive, args.reading), args.reading, args.out)
    print(f"imported {n} listings to {args.out}")
    print("stage 5 will use them on the next run — re-run with: "
          "python -m pipeline run <ids> --from 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
