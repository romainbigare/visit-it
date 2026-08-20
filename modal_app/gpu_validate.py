"""GPU validation of the stages the CPU box could not reach.

    modal run modal_app/gpu_validate.py::upload          # push golden set + groups
    modal run modal_app/gpu_validate.py::moge_speed      # GPU timing baseline
    modal run modal_app/gpu_validate.py::mapanything     # stage 3 multi-view
    modal run modal_app/gpu_validate.py::gsplat_train    # stage 8 appearance
    modal run modal_app/gpu_validate.py::fetch_results   # pull JSON + images back

Each function writes JSON into the shared volume in the same shape as the CPU
results in eval/results/, so the two merge into one report.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from .common import (CACHE_DIR, DATA_DIR, app, data_volume, full_image,
                     gpu_image, model_cache)

LOCAL_GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden"
LOCAL_RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"

# L40S: 48 GB, ~$1.95/hr on Modal at time of writing. Enough for MapAnything at
# 8 views and gsplat training, without paying for an H100 we cannot saturate.
GPU_TYPE = os.environ.get("VISITIT_GPU", "L40S")


# --------------------------------------------------------------- data upload
@app.local_entrypoint()
def upload() -> None:
    """Push the golden-set images and the room groups into the Modal volume."""
    groups = LOCAL_RESULTS / "room_groups.json"
    if not groups.exists():
        raise SystemExit("run `python -m eval.models.grouping` first")
    media = LOCAL_GOLDEN / "media"
    files = sorted(p for p in media.rglob("*") if p.is_file())
    print(f"uploading {len(files)} images "
          f"({sum(f.stat().st_size for f in files)/1e6:.0f} MB) + metadata")
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(groups, "/room_groups.json")
        batch.put_file(LOCAL_GOLDEN / "golden_set.json", "/golden_set.json")
        for f in files:
            batch.put_file(f, f"/media/{f.relative_to(media)}")
    print("upload complete")


@app.function(image=gpu_image, volumes={DATA_DIR: data_volume}, timeout=600)
def inventory() -> dict:
    """Sanity-check what actually landed in the volume."""
    root = Path(DATA_DIR)
    imgs = list((root / "media").rglob("*")) if (root / "media").exists() else []
    groups = json.loads((root / "room_groups.json").read_text()) if (root / "room_groups.json").exists() else {}
    return {"images": sum(1 for p in imgs if p.is_file()),
            "groups": groups.get("n_groups", 0),
            "has_golden": (root / "golden_set.json").exists()}


# ------------------------------------------------------- MoGe GPU speed test
@app.function(image=full_image, gpu=GPU_TYPE, volumes={DATA_DIR: data_volume,
              CACHE_DIR: model_cache}, timeout=1800)
def moge_speed(n: int = 20, max_side: int = 512) -> dict:
    """Re-run the CPU MoGe-2 test on a GPU purely to get the speed factor.

    This is the number the whole cost model hangs on: 14.4 s/image on 4 CPU
    cores has to come down to fractions of a second for the `instant` profile
    to be viable at all.
    """
    os.environ["HF_HOME"] = CACHE_DIR
    import subprocess
    import sys
    import numpy as np
    import torch
    from PIL import Image

    # MoGe is not on PyPI; vendor it the same way tools/vendor_moge.py does.
    vend = Path("/tmp/moge")
    if not (vend / "moge").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "git+https://github.com/microsoft/MoGe.git"], check=True)
    from moge.model.v2 import MoGeModel

    dev = "cuda"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval().to(dev)
    groups = json.loads((Path(DATA_DIR) / "room_groups.json").read_text())["groups"]
    paths = [p for g in groups for p in g["paths"]][:n]

    times, fovs, results = [], [], []
    for i, rel in enumerate(paths):
        im = Image.open(Path(DATA_DIR) / rel).convert("RGB")
        im.thumbnail((max_side, max_side))
        x = torch.tensor(np.array(im) / 255.0, dtype=torch.float32).permute(2, 0, 1).to(dev)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.infer(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if i > 0:  # first call includes CUDA warm-up
            times.append(dt)
        intr = out["intrinsics"].float().cpu().numpy()
        fov = float(2 * np.degrees(np.arctan(0.5 / intr[0, 0])))
        fovs.append(fov)
        results.append({"path": rel, "seconds": round(dt, 3), "fov_x_deg": round(fov, 1),
                        "depth_median_m": round(float(np.nanmedian(out["depth"].float().cpu().numpy())), 2)})

    res = {"stage": "3-moge-gpu", "gpu": GPU_TYPE, "n_images": len(paths),
           "max_side_px": max_side,
           "seconds_per_image": round(float(np.mean(times)), 3) if times else None,
           "seconds_first_call": round(results[0]["seconds"], 2) if results else None,
           "fov_median_deg": round(float(np.median(fovs)), 1) if fovs else None,
           "results": results}
    (Path(DATA_DIR) / "results_moge_gpu.json").write_text(json.dumps(res, indent=2))
    data_volume.commit()
    return res


# ------------------------------------------------------------ fetch results
@app.function(image=gpu_image, volumes={DATA_DIR: data_volume}, timeout=600)
def _read_results() -> dict:
    root = Path(DATA_DIR)
    out = {}
    for f in sorted(root.glob("results_*.json")):
        out[f.name] = json.loads(f.read_text())
    return out


@app.local_entrypoint()
def fetch_results() -> None:
    """Copy the JSON results out of the volume into eval/results/."""
    got = _read_results.remote()
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    for name, payload in got.items():
        (LOCAL_RESULTS / name).write_text(json.dumps(payload, indent=2))
        print(f"  {name}")
    if not got:
        print("no results in the volume yet")
