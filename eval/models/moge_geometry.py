"""Stage 3-4 validation with MoGe-2, which estimates its own intrinsics.

This is the counterpart to geometry.py (Depth Anything V2). The difference is
the point of the exercise: DA-V2 gives metric depth but NO camera intrinsics,
so unprojecting it requires guessing the field of view. MoGe-2 predicts the
intrinsics as part of its forward pass, which is exactly why AD-5 picks it.

Outputs per image: predicted FOV, metric point map, floor/ceiling planes from
the point map, and the resulting room height.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .geometry import (PLAUSIBLE_CEILING, TYPICAL_CEILING, dominant_up_axis,
                       estimate_normals, floor_ceiling)

log = logging.getLogger("moge")
MODEL_ID = "Ruicheng/moge-2-vitl-normal"


def load_model(threads: int = 4):
    from moge.model.v2 import MoGeModel  # noqa: PLC0415 - needs vendored path
    torch.set_num_threads(threads)
    return MoGeModel.from_pretrained(MODEL_ID).eval().float()


@torch.no_grad()
def infer(model, image: Image.Image) -> dict:
    x = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1)
    out = model.infer(x, use_fp16=False)
    pts = out["points"].numpy()
    depth = out["depth"].numpy()
    intr = out["intrinsics"].numpy()
    mask = out["mask"].numpy() if "mask" in out else np.ones(depth.shape, bool)
    fov_x = float(2 * np.degrees(np.arctan(0.5 / intr[0, 0])))
    fov_y = float(2 * np.degrees(np.arctan(0.5 / intr[1, 1])))
    return {"points": pts, "depth": depth, "mask": mask,
            "fov_x_deg": fov_x, "fov_y_deg": fov_y, "intrinsics": intr}


def room_from_points(points: np.ndarray, mask: np.ndarray, stride: int = 3) -> dict:
    p = points[::stride, ::stride]
    m = mask[::stride, ::stride]
    h, w, _ = p.shape
    flat = p.reshape(-1, 3)
    normals = estimate_normals(flat, (h, w))
    good = m.reshape(-1) & np.isfinite(flat).all(1)
    if good.sum() < 200:
        return {"room_height_m": None, "up": None, "n_points": int(good.sum())}
    up = dominant_up_axis(normals[good])
    lo, hi, _, _ = floor_ceiling(flat[good], up)
    height = None if (lo is None or hi is None) else abs(hi - lo)
    return {"room_height_m": None if height is None else round(float(height), 3),
            "up": up.tolist(), "n_points": int(good.sum()),
            "floor_h": lo, "ceiling_h": hi}


def run(golden: Path, out_dir: Path, triage_json: Path, limit: int = 20,
        max_side: int = 512) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tri = json.loads(triage_json.read_text())
    interiors = [p for p in tri["predictions"] if p["pred_type"] == "interior"]
    by_listing: dict[str, list] = {}
    for p in interiors:
        by_listing.setdefault(p["listing_id"], []).append(p)
    picked, i = [], 0
    while len(picked) < limit and any(len(v) > i for v in by_listing.values()):
        for v in by_listing.values():
            if len(v) > i and len(picked) < limit:
                picked.append(v[i])
        i += 1

    model = load_model()
    rows, t_total = [], 0.0
    for k, p in enumerate(picked, 1):
        im = Image.open(golden / p["path"]).convert("RGB")
        im.thumbnail((max_side, max_side))
        t0 = time.time()
        try:
            o = infer(model, im)
        except Exception as e:  # noqa: BLE001
            log.warning("  [%d/%d] failed: %s", k, len(picked), e)
            continue
        dt = time.time() - t0
        t_total += dt
        geo = room_from_points(o["points"], o["mask"])
        rows.append({"path": p["path"], "listing_id": p["listing_id"],
                     "pred_room": p.get("pred_room"),
                     "fov_x_deg": round(o["fov_x_deg"], 1),
                     "fov_y_deg": round(o["fov_y_deg"], 1),
                     "depth_median_m": round(float(np.nanmedian(o["depth"])), 2),
                     "room_height_m": geo["room_height_m"],
                     "seconds": round(dt, 1)})
        log.info("  [%d/%d] FOV=%.0f deg  depth_med=%.2fm  height=%s  [%.0fs]",
                 k, len(picked), o["fov_x_deg"],
                 float(np.nanmedian(o["depth"])), geo["room_height_m"], dt)

    fovs = np.array([r["fov_x_deg"] for r in rows]) if rows else np.array([])
    hs = np.array([r["room_height_m"] for r in rows if r["room_height_m"] is not None])
    out = {
        "stage": "3-4-geometry-moge", "model": MODEL_ID, "device": "cpu",
        "n_images": len(rows), "max_side_px": max_side,
        "seconds_per_image": round(t_total / max(len(rows), 1), 1),
        "fov": {"median_deg": round(float(np.median(fovs)), 1) if fovs.size else None,
                "p10_deg": round(float(np.percentile(fovs, 10)), 1) if fovs.size else None,
                "p90_deg": round(float(np.percentile(fovs, 90)), 1) if fovs.size else None},
        "room_height": {
            "n": int(hs.size),
            "median_m": round(float(np.median(hs)), 2) if hs.size else None,
            "p10_m": round(float(np.percentile(hs, 10)), 2) if hs.size else None,
            "p90_m": round(float(np.percentile(hs, 90)), 2) if hs.size else None,
            "frac_plausible": round(float(((hs >= PLAUSIBLE_CEILING[0]) &
                                           (hs <= PLAUSIBLE_CEILING[1])).mean()), 3) if hs.size else None,
            "frac_typical": round(float(((hs >= TYPICAL_CEILING[0]) &
                                         (hs <= TYPICAL_CEILING[1])).mean()), 3) if hs.size else None,
        },
        "results": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage34_moge.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument("--triage", type=Path, default=Path("eval/results/stage0_triage.json"))
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()
    r = run(a.golden, a.out, a.triage, a.limit)
    print(f"\nMoGe-2: {r['n_images']} images, {r['seconds_per_image']}s/img (CPU)")
    print(f"  predicted FOV: median {r['fov']['median_deg']}deg "
          f"(p10 {r['fov']['p10_deg']} / p90 {r['fov']['p90_deg']})")
    rh = r["room_height"]
    print(f"  room height:   median {rh['median_m']}m  plausible {rh['frac_plausible']}  "
          f"typical {rh['frac_typical']}")
