"""Stage 5 validation: what can we actually read off a real UK floor plan?

The scale solve (ARCHITECTURE §5) wants a metric anchor. In France that is the
loi Carrez area, printed on the plan. In the UK there is no such requirement, so
this measures what UK agency plans actually carry:

  * room labels      -> feeds the plan/photo assignment in stage 6
  * room dimensions  -> "3.73m x 3.20m" or "12'3\" x 10'6\"", a direct scale anchor
  * total area       -> "68 sq m" / "732 sq ft"
  * scale notation   -> "1:100"
  * a "not to scale" disclaimer, which invalidates geometric measurement

Resolution matters a lot, so the image is upscaled before OCR and the native
size is recorded alongside the result.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import pytesseract
from PIL import Image

log = logging.getLogger("plan_ocr")

ROOM_WORDS = ("bedroom", "bed ", "lounge", "living", "kitchen", "bathroom", "ensuite",
              "en-suite", "reception", "dining", "diner", "hall", "landing", "wc",
              "cloakroom", "utility", "study", "conservatory", "balcony", "terrace",
              "storage", "cupboard", "shower", "garage", "porch")

RX_METRIC = re.compile(r"(\d{1,2}[.,]\d{1,2})\s*m?\s*[xX×]\s*(\d{1,2}[.,]\d{1,2})\s*m", re.I)
RX_IMPERIAL = re.compile(r"(\d{1,2})\s*['’]\s*(\d{1,2})?\s*[\"”]?\s*[xX×]\s*"
                         r"(\d{1,2})\s*['’]\s*(\d{1,2})?\s*[\"”]?")
RX_AREA = re.compile(r"(\d[\d,]{1,6}(?:\.\d+)?)\s*(sq\.?\s*m|sqm|m²|sq\.?\s*ft|sqft|ft²)", re.I)
RX_SCALE = re.compile(r"\b1\s*[:：]\s*(\d{2,4})\b")
RX_NOT_TO_SCALE = re.compile(r"not\s+to\s+scale|approximate|illustrative|"
                             r"no\s+responsibility|for\s+illustrative\s+purposes", re.I)


def ocr(path: Path, min_side: int = 1800) -> tuple[str, tuple[int, int]]:
    im = Image.open(path).convert("L")
    w, h = im.size
    scale = max(1.0, min_side / max(w, h))
    if scale > 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pytesseract.image_to_string(im), (w, h)


def parse(text: str) -> dict:
    low = text.lower()
    rooms = sorted({wd.strip() for wd in ROOM_WORDS if wd in low})
    metric = [(float(a.replace(",", ".")), float(b.replace(",", ".")))
              for a, b in RX_METRIC.findall(text)]
    imperial = []
    for ft1, in1, ft2, in2 in RX_IMPERIAL.findall(text):
        try:
            imperial.append((round(int(ft1) * 0.3048 + int(in1 or 0) * 0.0254, 2),
                             round(int(ft2) * 0.3048 + int(in2 or 0) * 0.0254, 2)))
        except ValueError:
            pass
    areas = []
    for val, unit in RX_AREA.findall(text):
        try:
            v = float(val.replace(",", ""))
        except ValueError:
            continue
        u = unit.lower().replace(" ", "").replace(".", "")
        areas.append(round(v * 0.09290304, 1) if u in ("sqft", "ft²") else round(v, 1))
    scale_m = RX_SCALE.search(text)
    return {
        "room_labels": rooms,
        "n_room_labels": len(rooms),
        "metric_dims": metric[:20],
        "imperial_dims": imperial[:20],
        "n_dims": len(metric) + len(imperial),
        "areas_sqm": areas[:10],
        "scale_ratio": int(scale_m.group(1)) if scale_m else None,
        "not_to_scale_disclaimer": bool(RX_NOT_TO_SCALE.search(text)),
        "n_chars": len(text.strip()),
    }


def run(golden: Path, out_dir: Path) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = json.loads((golden / "golden_set.json").read_text())
    plans = [(golden / r["local_path"], lst["listing_id"], lst.get("floor_area_sqm"))
             for lst in data["listings"] for r in lst["floorplans"] if r.get("local_path")]
    log.info("OCR over %d floor plans", len(plans))

    rows, t0 = [], time.time()
    for p, lid, stated in plans:
        text, size = ocr(p)
        parsed = parse(text)
        parsed.update({"listing_id": lid, "path": str(p.relative_to(golden)),
                       "native_px": f"{size[0]}x{size[1]}",
                       "native_max_side": max(size),
                       "stated_area_sqm": stated,
                       "raw_text": text.strip()[:800]})
        rows.append(parsed)
        log.info("  %s %s labels=%d dims=%d areas=%s scale=%s nts=%s",
                 lid, f"{size[0]}x{size[1]}", parsed["n_room_labels"], parsed["n_dims"],
                 parsed["areas_sqm"][:2], parsed["scale_ratio"],
                 parsed["not_to_scale_disclaimer"])
    dt = time.time() - t0

    n = len(rows)
    def frac(pred) -> float:
        return round(sum(1 for r in rows if pred(r)) / n, 3) if n else 0.0

    hi = [r for r in rows if r["native_max_side"] >= 1000]
    summary = {
        "stage": "5-plan-ocr", "engine": f"tesseract {pytesseract.get_tesseract_version()}",
        "n_plans": n, "seconds_total": round(dt, 1),
        "seconds_per_plan": round(dt / max(n, 1), 2),
        "with_any_text": frac(lambda r: r["n_chars"] > 20),
        "with_room_labels": frac(lambda r: r["n_room_labels"] >= 2),
        "with_dimensions": frac(lambda r: r["n_dims"] >= 1),
        "with_area": frac(lambda r: bool(r["areas_sqm"])),
        "with_scale_ratio": frac(lambda r: r["scale_ratio"] is not None),
        "with_not_to_scale_disclaimer": frac(lambda r: r["not_to_scale_disclaimer"]),
        "median_room_labels": sorted(r["n_room_labels"] for r in rows)[n // 2] if n else 0,
        "n_high_res": len(hi),
        "with_dimensions_high_res": (round(sum(1 for r in hi if r["n_dims"] >= 1) / len(hi), 3)
                                     if hi else None),
        "plans": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage5_plan_ocr.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    a = ap.parse_args()
    s = run(a.golden, a.out)
    print(f"\n{s['n_plans']} plans, {s['seconds_per_plan']}s each")
    for k in ("with_any_text", "with_room_labels", "with_dimensions", "with_area",
              "with_scale_ratio", "with_not_to_scale_disclaimer"):
        print(f"  {k:32} {s[k]:.0%}")
    print(f"  high-res subset (>=1000px): {s['n_high_res']} plans, "
          f"dims on {s['with_dimensions_high_res']}")
