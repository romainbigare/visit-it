"""Stages 3-4 validation: monocular depth -> point cloud -> floor/ceiling planes.

What this actually tests
------------------------
Whether metric monocular depth on REAL estate-agent photography (ultra-wide,
HDR-composited, furnished) yields a point cloud from which the architectural
shell can be recovered. The measurable output is floor-to-ceiling height, which
is the one number we can sanity-check without ground truth: UK residential
ceilings cluster at 2.3-2.7 m in modern stock and up to ~3.5 m in period
conversions. Anything outside 1.8-4.5 m is wrong.

The honest caveat: we have no camera intrinsics (EXIF is stripped), so focal
length is ASSUMED. The report (§4.2) says this matters enormously, and this
script quantifies exactly how much by sweeping the assumption.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("geometry")

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"

# Estate agents shoot ultra-wide. 14-24mm full-frame equivalent is roughly
# 84-104 deg horizontal FOV; 60 deg is a "normal" lens, included to show the
# sensitivity.
FOV_SWEEP_DEG = (60.0, 75.0, 90.0, 104.0)
DEFAULT_FOV_DEG = 90.0

PLAUSIBLE_CEILING = (1.8, 4.5)
TYPICAL_CEILING = (2.3, 3.2)


def load_depth_model(threads: int = 4):
    from transformers import pipeline
    torch.set_num_threads(threads)
    return pipeline("depth-estimation", model=DEPTH_MODEL, device=-1)


def depth_of(pipe, image: Image.Image) -> np.ndarray:
    out = pipe(image)
    d = out.get("predicted_depth", out.get("depth"))
    return np.asarray(d, dtype=np.float32)


def unproject(depth: np.ndarray, fov_deg: float, stride: int = 4) -> np.ndarray:
    """Depth map -> Nx3 camera-space points, assuming a pinhole with `fov_deg`."""
    h, w = depth.shape
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    x = (xs - w / 2) * z / f
    y = (ys - h / 2) * z / f
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def estimate_normals(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Cheap normals from the structured grid (cross product of local gradients)."""
    h, w = shape
    grid = pts.reshape(h, w, 3)
    du = np.zeros_like(grid); dv = np.zeros_like(grid)
    du[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
    dv[1:-1, :] = grid[2:, :] - grid[:-2, :]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    return np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-9).reshape(-1, 3)


def dominant_up_axis(normals: np.ndarray) -> np.ndarray:
    """Indoor scenes are dominated by floor+ceiling, whose normals are antiparallel.

    We take the principal axis of the outer-product scatter (which is sign
    agnostic, so antiparallel normals reinforce rather than cancel).
    """
    valid = normals[np.linalg.norm(normals, axis=1) > 0.5]
    if len(valid) < 50:
        return np.array([0.0, 1.0, 0.0])
    m = valid.T @ valid
    w, v = np.linalg.eigh(m)
    up = v[:, np.argmax(w)]
    return up / np.linalg.norm(up)


def floor_ceiling(pts: np.ndarray, up: np.ndarray,
                  bins: int = 220) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """Height histogram along `up`; floor and ceiling are the outer dense peaks."""
    h = pts @ up
    finite = np.isfinite(h)
    h = h[finite]
    if h.size < 200:
        return None, None, np.array([]), np.array([])
    lo, hi = np.percentile(h, [1, 99])
    if not np.isfinite(lo) or hi - lo < 1e-3:
        return None, None, np.array([]), np.array([])
    counts, edges = np.histogram(h, bins=bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    thresh = counts.max() * 0.12
    dense = np.flatnonzero(counts > thresh)
    if dense.size < 2:
        return None, None, counts, centres
    return float(centres[dense[0]]), float(centres[dense[-1]]), counts, centres


def analyse(depth: np.ndarray, fov_deg: float, stride: int = 4) -> dict:
    pts = unproject(depth, fov_deg, stride)
    h, w = depth[::stride, ::stride].shape
    normals = estimate_normals(pts, (h, w))
    up = dominant_up_axis(normals)
    lo, hi, counts, centres = floor_ceiling(pts, up)
    height = None if (lo is None or hi is None) else abs(hi - lo)
    return {"fov_deg": fov_deg, "up": up.tolist(), "floor_h": lo, "ceiling_h": hi,
            "room_height_m": None if height is None else round(float(height), 3),
            "n_points": int(len(pts)),
            "depth_median_m": round(float(np.median(depth)), 2),
            "hist_counts": counts.tolist(), "hist_centres": centres.tolist()}


def run(golden: Path, out_dir: Path, triage_json: Path, limit: int = 40,
        sweep: bool = True) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tri = json.loads(triage_json.read_text())
    interiors = [p for p in tri["predictions"] if p["pred_type"] == "interior"]
    # Spread across listings rather than taking 40 photos of one flat.
    by_listing: dict[str, list] = {}
    for p in interiors:
        by_listing.setdefault(p["listing_id"], []).append(p)
    picked, i = [], 0
    while len(picked) < limit and any(v[i:] for v in by_listing.values() if len(v) > i):
        for v in by_listing.values():
            if len(v) > i and len(picked) < limit:
                picked.append(v[i])
        i += 1

    pipe = load_depth_model()
    results, t_depth = [], 0.0
    for k, p in enumerate(picked, 1):
        img = Image.open(golden / p["path"]).convert("RGB")
        img.thumbnail((768, 768))
        t0 = time.time()
        d = depth_of(pipe, img)
        t_depth += time.time() - t0
        fovs = FOV_SWEEP_DEG if sweep else (DEFAULT_FOV_DEG,)
        per_fov = {}
        for fov in fovs:
            a = analyse(d, fov)
            per_fov[str(fov)] = {kk: vv for kk, vv in a.items()
                                 if kk not in ("hist_counts", "hist_centres")}
        results.append({"path": p["path"], "listing_id": p["listing_id"],
                        "pred_room": p.get("pred_room"),
                        "depth_min": round(float(d.min()), 2),
                        "depth_max": round(float(d.max()), 2),
                        "depth_median": round(float(np.median(d)), 2),
                        "by_fov": per_fov})
        log.info("  [%d/%d] %s h@%.0f=%s", k, len(picked), p["path"].split("/")[-1],
                 DEFAULT_FOV_DEG, per_fov[str(DEFAULT_FOV_DEG)]["room_height_m"])

    def stats(fov: float) -> dict:
        hs = [r["by_fov"][str(fov)]["room_height_m"] for r in results
              if r["by_fov"][str(fov)]["room_height_m"] is not None]
        if not hs:
            return {"n": 0}
        a = np.array(hs)
        return {"n": len(hs), "median_m": round(float(np.median(a)), 2),
                "p10_m": round(float(np.percentile(a, 10)), 2),
                "p90_m": round(float(np.percentile(a, 90)), 2),
                "frac_plausible": round(float(((a >= PLAUSIBLE_CEILING[0]) &
                                               (a <= PLAUSIBLE_CEILING[1])).mean()), 3),
                "frac_typical": round(float(((a >= TYPICAL_CEILING[0]) &
                                             (a <= TYPICAL_CEILING[1])).mean()), 3)}

    out = {"stage": "3-4-geometry", "model": DEPTH_MODEL, "device": "cpu",
           "n_images": len(results),
           "seconds_depth_total": round(t_depth, 1),
           "seconds_per_image": round(t_depth / max(len(results), 1), 2),
           "fov_sweep": {str(f): stats(f) for f in (FOV_SWEEP_DEG if sweep else (DEFAULT_FOV_DEG,))},
           "results": results}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage34_geometry.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument("--triage", type=Path, default=Path("eval/results/stage0_triage.json"))
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    r = run(a.golden, a.out, a.triage, a.limit)
    print(f"\n{r['n_images']} images, {r['seconds_per_image']:.2f}s/img on CPU")
    for fov, s in r["fov_sweep"].items():
        if s.get("n"):
            print(f"  FOV {fov:>5}deg: median height {s['median_m']}m "
                  f"(p10 {s['p10_m']} p90 {s['p90_m']})  "
                  f"plausible {s['frac_plausible']:.0%}  typical {s['frac_typical']:.0%}")
