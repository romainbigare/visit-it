"""Draw a plan with its vectorised polygons labelled, for the annotation drive.

Arrangement truth is the one fact the plan cannot supply on its own (a plan says
where the *rooms* are, not which *photograph* is of which room), so a person has
to look. This makes looking cheap: one image per listing with every polygon
outlined and named, next to the list of reconstructed rooms waiting to be placed.

    python -m tools.annotation_aid 87977241 --out /tmp/aid
    python -m tools.annotation_aid --split holdout --limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
from PIL import Image, ImageDraw, ImageFont          # noqa: E402

from pipeline.core.artifacts import ArtifactStore    # noqa: E402
from pipeline.floorplan.preprocess import load_rgb   # noqa: E402

COLOURS = [(70, 130, 200), (220, 140, 70), (90, 180, 110), (210, 100, 140),
           (150, 110, 200), (200, 180, 70), (90, 180, 175), (190, 120, 100)]


def draw(listing_id: str, store_root: Path | None, out_dir: Path,
         max_side: int = 1100) -> tuple[Path, dict] | None:
    store = ArtifactStore(listing_id, store_root)
    plan = store.read("5-plan")
    layouts = store.read("4-layout")
    if not plan or not plan.get("source_image"):
        return None
    src = Path(plan["source_image"])
    if not src.exists():
        return None
    rgb = load_rgb(src)
    # plan.json's polygons are in the *geometry* resolution the preprocessor used,
    # so the aid has to be drawn at that same size or nothing lines up.
    gw, gh = plan["image_size_px"]
    im = Image.fromarray(rgb).resize((gw, gh), Image.LANCZOS)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  max(16, gw // 45))
    except OSError:
        font = ImageFont.load_default()
    for i, r in enumerate(plan["rooms"]):
        c = COLOURS[i % len(COLOURS)]
        pts = [tuple(p) for p in r["polygon_px"]]
        d.polygon(pts, fill=c + (70,), outline=c + (255,), width=4)
        cx, cy = r["centroid_px"]
        tag = f"{r['room_id']} {r.get('label') or '?'}"
        box = d.textbbox((0, 0), tag, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        d.rectangle([cx - w / 2 - 6, cy - h / 2 - 4, cx + w / 2 + 6, cy + h / 2 + 6],
                    fill=(255, 255, 255, 225))
        d.text((cx - w / 2, cy - h / 2), tag, font=font, fill=c + (255,))
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{listing_id}_plan.png"
    im.save(p)

    facts = {
        "listing_id": listing_id,
        "plan_rooms": [{"plan_room_id": r["room_id"], "label": r.get("label"),
                        "area_m2": r.get("area_m2"), "printed": r.get("ocr_dims_m")}
                       for r in plan["rooms"]],
        "reconstructed_rooms": [{"room_id": r["room_id"], "label": r.get("room_label"),
                                 "area_m2": r["area_m2"],
                                 "height_m": r.get("room_height_m")}
                                for r in (layouts or {}).get("rooms", [])],
    }
    (out_dir / f"{listing_id}_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    return p, facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("listing_id", nargs="*")
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--split", default=None, choices=["dev", "holdout"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/runs/_annotation_aid"))
    a = ap.parse_args(argv)
    ids = list(a.listing_id)
    if a.split:
        from eval.holdout import load as load_split
        ids = load_split(a.golden / "holdout_split.json")[a.split]
    if a.limit:
        ids = ids[:a.limit]
    made = []
    for lid in ids:
        r = draw(lid, a.store, a.out)
        if r:
            made.append(str(r[0]))
    print(json.dumps({"aids": made, "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
