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


# ------------------------------------------------- stage 3: MapAnything
@app.function(image=full_image, gpu=GPU_TYPE,
              volumes={DATA_DIR: data_volume, CACHE_DIR: model_cache},
              timeout=3600)
def mapanything(max_groups: int = 12) -> dict:
    """Multi-view reconstruction per room group (AD-4).

    Uses the Apache checkpoint specifically - the default one is CC-BY-NC.
    """
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    root = Path(DATA_DIR)
    groups = json.loads((root / "room_groups.json").read_text())["groups"][:max_groups]
    model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()

    out_rows = []
    for gi, g in enumerate(groups, 1):
        paths = [str(root / p) for p in g["paths"]]
        try:
            views = load_images(paths)
            torch.cuda.synchronize()
            t0 = time.time()
            preds = model.infer(views, memory_efficient_inference=True,
                                use_amp=True, amp_dtype="bf16")
            torch.cuda.synchronize()
            dt = time.time() - t0
        except Exception as e:  # noqa: BLE001
            out_rows.append({"group_id": g["group_id"], "error": f"{type(e).__name__}: {e}"})
            print(f"  [{gi}/{len(groups)}] {g['group_id']} FAILED: {e}")
            continue

        pts = torch.cat([p["pts3d"].reshape(-1, 3).float().cpu() for p in preds]).numpy()
        conf = torch.cat([p["confidence"].reshape(-1).float().cpu() for p in preds]).numpy()
        poses = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
        Ks = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds])
        keep = np.isfinite(pts).all(1) & (conf > np.percentile(conf, 40))
        good = pts[keep]

        # Baseline spread tells us whether the cameras were actually placed apart
        # or collapsed onto each other, which is the classic failure mode.
        cams = poses[:, :3, 3]
        baseline = float(np.linalg.norm(cams[:, None] - cams[None], axis=-1).max())
        extent = (good.max(0) - good.min(0)).tolist() if len(good) else None

        # Save the point cloud + poses so gsplat can start from them.
        np.savez_compressed(root / f"recon_{g['group_id']}.npz",
                            points=pts.astype(np.float32), conf=conf.astype(np.float32),
                            poses=poses.astype(np.float32), Ks=Ks.astype(np.float32))
        row = {"group_id": g["group_id"], "room_type": g["room_type"],
               "n_views": len(paths), "seconds": round(dt, 2),
               "n_points": int(len(pts)), "n_points_conf": int(keep.sum()),
               "max_camera_baseline_m": round(baseline, 3),
               "scene_extent_m": [round(float(v), 2) for v in extent] if extent else None,
               "mean_confidence": round(float(conf.mean()), 3)}
        out_rows.append(row)
        print(f"  [{gi}/{len(groups)}] {g['group_id']:26} {len(paths)}v "
              f"{dt:5.1f}s  baseline {baseline:.2f}m  extent {row['scene_extent_m']}")

    ok = [r for r in out_rows if "error" not in r]
    res = {"stage": "3-mapanything", "gpu": GPU_TYPE, "checkpoint": "facebook/map-anything-apache",
           "n_groups": len(groups), "n_ok": len(ok), "n_failed": len(out_rows) - len(ok),
           "seconds_per_group_median": (round(float(np.median([r["seconds"] for r in ok])), 2)
                                        if ok else None),
           "results": out_rows}
    (root / "results_mapanything.json").write_text(json.dumps(res, indent=2))
    data_volume.commit()
    return res


# ------------------------------------------------------ stage 8: gsplat
@app.function(image=full_image, gpu=GPU_TYPE,
              volumes={DATA_DIR: data_volume, CACHE_DIR: model_cache},
              timeout=3600)
def gsplat_train(max_groups: int = 4, iters: int = 1500, max_side: int = 512) -> dict:
    """Train a Gaussian splat per room, initialised from MapAnything (AD-6).

    This is the real stage 3 -> stage 8 chain: no COLMAP anywhere, poses come
    from the pointmap model. Holds out one view to get an honest PSNR.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from gsplat import rasterization

    root = Path(DATA_DIR)
    groups = json.loads((root / "room_groups.json").read_text())["groups"]
    dev = "cuda"
    rows = []

    for g in groups[:max_groups]:
        npz = root / f"recon_{g['group_id']}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        poses, Ks = d["poses"], d["Ks"]
        pts, conf = d["points"], d["conf"]

        imgs = []
        for p in g["paths"]:
            im = Image.open(root / p).convert("RGB")
            im.thumbnail((max_side, max_side))
            imgs.append(torch.tensor(np.array(im) / 255.0, dtype=torch.float32))
        H, W = imgs[0].shape[:2]
        gt = torch.stack(imgs).to(dev)

        # Rescale intrinsics to the working resolution.
        n_views = min(len(imgs), len(poses))
        Ks_t = torch.tensor(Ks[:n_views], dtype=torch.float32, device=dev).clone()
        # MapAnything K is for its own processing size; normalise via the pts3d grid.
        sx = W / (2 * Ks_t[:, 0, 2]); sy = H / (2 * Ks_t[:, 1, 2])
        Ks_t[:, 0, :] *= sx[:, None]; Ks_t[:, 1, :] *= sy[:, None]

        c2w = torch.tensor(poses[:n_views], dtype=torch.float32, device=dev)
        viewmats = torch.inverse(c2w)

        keep = np.isfinite(pts).all(1) & (conf > np.percentile(conf, 50))
        init = pts[keep]
        if len(init) > 120_000:
            init = init[np.random.choice(len(init), 120_000, replace=False)]
        N = len(init)
        if N < 1000:
            rows.append({"group_id": g["group_id"], "error": "too few init points"})
            continue

        means = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32, device=dev))
        scale0 = float(np.percentile(np.linalg.norm(init - init.mean(0), axis=1), 5)) or 0.02
        scales = torch.nn.Parameter(torch.full((N, 3), np.log(scale0 * 0.5), device=dev))
        quats = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1))
        opac = torch.nn.Parameter(torch.full((N,), 0.1, device=dev).logit())
        colors = torch.nn.Parameter(torch.full((N, 3), 0.5, device=dev))

        opt = torch.optim.Adam([
            {"params": [means], "lr": 1.6e-4}, {"params": [scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3}, {"params": [opac], "lr": 5e-2},
            {"params": [colors], "lr": 2.5e-3}])

        train_idx = list(range(n_views))[:-1] or [0]
        held = n_views - 1
        torch.cuda.synchronize(); t0 = time.time()
        for it in range(iters):
            i = train_idx[it % len(train_idx)]
            rc, _, _ = rasterization(
                means=means, quats=F.normalize(quats, dim=-1), scales=scales.exp(),
                opacities=opac.sigmoid(), colors=colors.clamp(0, 1),
                viewmats=viewmats[i:i + 1], Ks=Ks_t[i:i + 1], width=W, height=H,
                sh_degree=None, render_mode="RGB")
            loss = F.l1_loss(rc[0], gt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); train_s = time.time() - t0

        with torch.no_grad():
            rc, _, _ = rasterization(
                means=means, quats=F.normalize(quats, dim=-1), scales=scales.exp(),
                opacities=opac.sigmoid(), colors=colors.clamp(0, 1),
                viewmats=viewmats[held:held + 1], Ks=Ks_t[held:held + 1],
                width=W, height=H, sh_degree=None, render_mode="RGB")
            mse = F.mse_loss(rc[0], gt[held]).item()
            psnr = float(-10 * np.log10(max(mse, 1e-10)))
            Image.fromarray((rc[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
                root / f"splat_{g['group_id']}_render.png")
            Image.fromarray((gt[held].cpu().numpy() * 255).astype(np.uint8)).save(
                root / f"splat_{g['group_id']}_gt.png")

        rows.append({"group_id": g["group_id"], "room_type": g["room_type"],
                     "n_gaussians": N, "n_views": n_views, "iters": iters,
                     "train_seconds": round(train_s, 1),
                     "heldout_psnr_db": round(psnr, 2),
                     "resolution": f"{W}x{H}"})
        print(f"  {g['group_id']:26} {N:6} gaussians  {train_s:5.1f}s  "
              f"held-out PSNR {psnr:.2f} dB")

    ok = [r for r in rows if "error" not in r]
    res = {"stage": "8-gsplat", "gpu": GPU_TYPE, "n_scenes": len(rows), "n_ok": len(ok),
           "median_psnr_db": (round(float(np.median([r["heldout_psnr_db"] for r in ok])), 2)
                              if ok else None),
           "median_train_seconds": (round(float(np.median([r["train_seconds"] for r in ok])), 1)
                                    if ok else None),
           "results": rows}
    (root / "results_gsplat.json").write_text(json.dumps(res, indent=2))
    data_volume.commit()
    return res


@app.local_entrypoint()
def run_all() -> None:
    """Upload, then run every GPU stage in order, then pull results back."""
    upload()
    print("\ninventory:", inventory.remote())
    print("\n== MoGe GPU timing =="); print(json.dumps(
        {k: v for k, v in moge_speed.remote().items() if k != "results"}, indent=2))
    print("\n== MapAnything =="); print(json.dumps(
        {k: v for k, v in mapanything.remote().items() if k != "results"}, indent=2))
    print("\n== gsplat =="); print(json.dumps(
        {k: v for k, v in gsplat_train.remote().items() if k != "results"}, indent=2))
    fetch_results()
