"""Run the GPU validation anywhere there is a CUDA GPU.

No Modal, no volumes, no upload step: the golden set's image URLs are already
in data/golden/golden_set.json, so this re-fetches them itself. That makes it
runnable on Colab, Kaggle, RunPod, or any borrowed machine.

    # Colab / Kaggle, in one cell:
    !git clone -q https://github.com/romainbigare/visit-it && cd visit-it && \
        python tools/gpu_validate_standalone.py --all

    # Anywhere else:
    python tools/gpu_validate_standalone.py --all --out eval/results

Writes results_*.json and splat_*.png into --out, matching the shape of the CPU
results so the two merge into one report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "eval" / "results"
DEFAULT_MEDIA = REPO / "data" / "golden" / "media"


def sh(*args: str) -> None:
    print("  $", " ".join(args), flush=True)
    subprocess.run(list(args), check=True)


def ensure_deps(need_gsplat: bool = True) -> None:
    """Install what is missing. Colab and Kaggle already have torch + CUDA."""
    import importlib.util as iu
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    if iu.find_spec("utils3d") is None:
        sh(*pip, "utils3d", "einops", "safetensors", "huggingface_hub")
    if iu.find_spec("moge") is None:
        sh(*pip, "git+https://github.com/microsoft/MoGe.git")
    if iu.find_spec("mapanything") is None:
        sh(*pip, "git+https://github.com/facebookresearch/map-anything.git")
    if need_gsplat and iu.find_spec("gsplat") is None:
        install_gsplat(pip)


def install_gsplat(pip: list[str]) -> None:
    """Prefer a prebuilt wheel; fall back to compiling.

    Building gsplat's CUDA kernels from source takes several minutes on a Colab
    T4 and is the most likely step to fail, so try the matching prebuilt wheel
    first. The index is keyed by torch and CUDA version, e.g. pt24cu124.
    """
    try:
        import torch
        tv = "".join(torch.__version__.split("+")[0].split(".")[:2])   # 2.4.1 -> 24
        cv = (torch.version.cuda or "").replace(".", "")               # 12.4  -> 124
        if cv:
            idx = f"https://docs.gsplat.studio/whl/pt{tv}cu{cv}"
            print(f"  trying prebuilt gsplat wheel: {idx}")
            try:
                sh(*pip, "gsplat", "--index-url", idx)
                import importlib
                importlib.invalidate_caches()
                if importlib.util.find_spec("gsplat"):
                    return
            except subprocess.CalledProcessError:
                print("  no prebuilt wheel for this torch/CUDA pair; compiling instead")
    except Exception as e:  # noqa: BLE001
        print(f"  wheel probe failed ({e}); compiling instead")
    sh(*pip, "gsplat")


def fetch_media(golden_json: Path, media_root: Path, groups: list[dict],
                workers: int = 8) -> int:
    """Download only the images the groups actually reference."""
    import requests
    data = json.loads(golden_json.read_text())
    url_by_path = {}
    for lst in data["listings"]:
        for ref in lst["photos"] + lst["floorplans"]:
            if ref.get("local_path"):
                url_by_path[ref["local_path"]] = ref["url"]
    wanted = {p for g in groups for p in g["paths"]}
    todo = [(p, url_by_path[p]) for p in sorted(wanted)
            if p in url_by_path and not (media_root.parent / p).exists()]
    if not todo:
        return 0
    print(f"  fetching {len(todo)} images")
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0"

    def one(item):
        rel, url = item
        dest = media_root.parent / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = sess.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    failed {rel}: {e}")
            return False

    with ThreadPoolExecutor(workers) as ex:
        ok = sum(ex.map(one, todo))
    print(f"  {ok}/{len(todo)} downloaded")
    return ok


# ------------------------------------------------------------------ stages
def _amp_dtype() -> str:
    """T4/V100 (compute capability < 8.0) have no native bf16 - use fp16 there."""
    import torch
    major = torch.cuda.get_device_capability()[0]
    return "bf16" if major >= 8 else "fp16"


def _vram_gb() -> float:
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _auto_max_views(explicit: int | None) -> int:
    """MapAnything's backbone is a ViT-g; 8 views at full size OOMs a 16 GB T4.

    Scale the view count to the card rather than failing on every group.
    """
    if explicit:
        return explicit
    gb = _vram_gb()
    return 3 if gb < 20 else (6 if gb < 40 else 8)


def _pick(d: dict, *names, required: bool = True):
    """MapAnything's output key names are not stable across versions, and an
    earlier run died on a hard-coded 'confidence'. Try the known aliases and
    fail with the actual keys listed rather than a bare KeyError."""
    for n in names:
        if n in d:
            return d[n]
    if required:
        raise KeyError(f"none of {names} in prediction; available keys: {sorted(d)}")
    return None


def _resized_copies(paths: list[str], max_side: int, tmp: Path) -> list[str]:
    """Shrink inputs before load_images. Memory scales with pixel count, and
    listing photos are far larger than the model needs."""
    from PIL import Image
    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        q = tmp / f"v{i:02d}.jpg"
        im.save(q, quality=92)
        out.append(str(q))
    return out


def stage_mapanything(groups, media_parent: Path, out: Path, max_groups: int,
                      max_views: int | None = None, max_side: int = 384) -> dict:
    import gc
    import tempfile
    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    dev = "cuda"
    n_views_cap = _auto_max_views(max_views)
    dtype = _amp_dtype()
    print(f"  VRAM {_vram_gb():.1f} GB -> max {n_views_cap} views/group, "
          f"{max_side}px, amp {dtype}", flush=True)

    model = MapAnything.from_pretrained("facebook/map-anything-apache").to(dev).eval()
    rows, logged_keys = [], False
    for i, g in enumerate(groups[:max_groups], 1):
        paths = [str(media_parent / p) for p in g["paths"]][:n_views_cap]
        if not all(Path(p).exists() for p in paths):
            rows.append({"group_id": g["group_id"], "error": "missing images"}); continue
        try:
            with tempfile.TemporaryDirectory() as td:
                small = _resized_copies(paths, max_side, Path(td))
                views = load_images(small)
                torch.cuda.synchronize(); t0 = time.time()
                preds = model.infer(views, memory_efficient_inference=True,
                                    use_amp=True, amp_dtype=dtype)
                torch.cuda.synchronize(); dt = time.time() - t0

            if not logged_keys:
                print(f"  prediction keys: {sorted(preds[0])}", flush=True)
                logged_keys = True

            pts_l = [_pick(p, "pts3d", "points_3d", "points", "pts3d_in_other_view")
                     for p in preds]
            pts = torch.cat([t.reshape(-1, 3).float().cpu() for t in pts_l]).numpy()
            conf_l = [_pick(p, "confidence", "conf", "conf_mask", required=False)
                      for p in preds]
            if all(c is not None for c in conf_l):
                conf = torch.cat([c.reshape(-1).float().cpu() for c in conf_l]).numpy()
            else:
                conf = np.ones(len(pts), dtype=np.float32)
            poses = np.stack([_pick(p, "camera_poses", "camera_pose", "cam2world")[0]
                              .float().cpu().numpy() for p in preds])
            Ks = np.stack([_pick(p, "intrinsics", "camera_intrinsics", "K")[0]
                           .float().cpu().numpy() for p in preds])
            shapes = np.array([list(t.shape[-3:-1]) for t in pts_l], dtype=np.int32)

            cams = poses[:, :3, 3]
            baseline = float(np.linalg.norm(cams[:, None] - cams[None], axis=-1).max())
            keep = np.isfinite(pts).all(1) & (conf > np.percentile(conf, 40))
            good = pts[keep]
            extent = [round(float(v), 2) for v in (good.max(0) - good.min(0))] if len(good) else None
            np.savez_compressed(out / f"recon_{g['group_id']}.npz",
                                points=pts.astype(np.float32), conf=conf.astype(np.float32),
                                poses=poses.astype(np.float32), Ks=Ks.astype(np.float32),
                                shapes=shapes)
            rows.append({"group_id": g["group_id"], "room_type": g["room_type"],
                         "n_views": len(paths), "seconds": round(dt, 2),
                         "n_points": int(len(pts)), "max_camera_baseline_m": round(baseline, 3),
                         "scene_extent_m": extent,
                         "mean_confidence": round(float(conf.mean()), 3)})
            print(f"  [{i}] {g['group_id']:26} {len(paths)}v {dt:5.1f}s "
                  f"baseline {baseline:.2f}m extent {extent}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad group must not sink the run
            rows.append({"group_id": g["group_id"], "error": f"{type(e).__name__}: {e}"})
            print(f"  [{i}] {g['group_id']} FAILED: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
        finally:
            # Without this the first OOM poisons every later group: the failed
            # allocation stays resident and they all fail identically, which is
            # exactly what happened on the first T4 run.
            preds = views = pts_l = conf_l = None
            gc.collect(); torch.cuda.empty_cache()

    ok = [r for r in rows if "error" not in r]
    res = {"stage": "3-mapanything", "gpu": _gpu_name(),
           "vram_gb": round(_vram_gb(), 1), "max_views": n_views_cap,
           "max_side_px": max_side, "amp_dtype": dtype,
           "n_groups": min(len(groups), max_groups),
           "n_ok": len(ok), "n_failed": len(rows) - len(ok),
           "seconds_per_group_median": (round(float(np.median([r["seconds"] for r in ok])), 2)
                                        if ok else None), "results": rows}
    (out / "results_mapanything.json").write_text(json.dumps(res, indent=2))
    return res


def stage_gsplat(groups, media_parent: Path, out: Path, max_groups: int, iters: int) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from gsplat import rasterization

    dev = "cuda"
    rows = []
    for g in groups[:max_groups]:
        npz = out / f"recon_{g['group_id']}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        poses, Ks, pts, conf = d["poses"], d["Ks"], d["points"], d["conf"]
        H, W = int(d["shapes"][0][0]), int(d["shapes"][0][1])
        imgs = [torch.tensor(np.array(Image.open(media_parent / p).convert("RGB")
                                      .resize((W, H), Image.LANCZOS)) / 255.0,
                             dtype=torch.float32) for p in g["paths"]]
        gt = torch.stack(imgs).to(dev)
        n = min(len(imgs), len(poses))
        Ks_t = torch.tensor(Ks[:n], dtype=torch.float32, device=dev)
        viewmats = torch.inverse(torch.tensor(poses[:n], dtype=torch.float32, device=dev))

        keep = np.isfinite(pts).all(1) & (conf > np.percentile(conf, 50))
        init = pts[keep]
        if len(init) > 120_000:
            init = init[np.random.choice(len(init), 120_000, replace=False)]
        if len(init) < 1000:
            rows.append({"group_id": g["group_id"], "error": "too few init points"}); continue
        N = len(init)
        means = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32, device=dev))
        s0 = float(np.percentile(np.linalg.norm(init - init.mean(0), axis=1), 5)) or 0.02
        scales = torch.nn.Parameter(torch.full((N, 3), float(np.log(s0 * 0.5)), device=dev))
        quats = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1))
        opac = torch.nn.Parameter(torch.full((N,), 0.1, device=dev).logit())
        colors = torch.nn.Parameter(torch.full((N, 3), 0.5, device=dev))
        opt = torch.optim.Adam([{"params": [means], "lr": 1.6e-4},
                                {"params": [scales], "lr": 5e-3},
                                {"params": [quats], "lr": 1e-3},
                                {"params": [opac], "lr": 5e-2},
                                {"params": [colors], "lr": 2.5e-3}])
        train = list(range(n))[:-1] or [0]
        held = n - 1
        torch.cuda.synchronize(); t0 = time.time()
        for it in range(iters):
            i = train[it % len(train)]
            rc, _, _ = rasterization(means=means, quats=F.normalize(quats, dim=-1),
                                     scales=scales.exp(), opacities=opac.sigmoid(),
                                     colors=colors.clamp(0, 1), viewmats=viewmats[i:i+1],
                                     Ks=Ks_t[i:i+1], width=W, height=H,
                                     sh_degree=None, render_mode="RGB")
            loss = F.l1_loss(rc[0], gt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); train_s = time.time() - t0
        with torch.no_grad():
            rc, _, _ = rasterization(means=means, quats=F.normalize(quats, dim=-1),
                                     scales=scales.exp(), opacities=opac.sigmoid(),
                                     colors=colors.clamp(0, 1), viewmats=viewmats[held:held+1],
                                     Ks=Ks_t[held:held+1], width=W, height=H,
                                     sh_degree=None, render_mode="RGB")
            psnr = float(-10 * np.log10(max(F.mse_loss(rc[0], gt[held]).item(), 1e-10)))
            Image.fromarray((rc[0].clamp(0, 1).cpu().numpy()*255).astype(np.uint8)).save(
                out / f"splat_{g['group_id']}_render.png")
            Image.fromarray((gt[held].cpu().numpy()*255).astype(np.uint8)).save(
                out / f"splat_{g['group_id']}_gt.png")
        rows.append({"group_id": g["group_id"], "room_type": g["room_type"],
                     "n_gaussians": N, "n_views": n, "iters": iters,
                     "train_seconds": round(train_s, 1), "heldout_psnr_db": round(psnr, 2),
                     "resolution": f"{W}x{H}"})
        print(f"  {g['group_id']:26} {N:6} gaussians {train_s:5.1f}s PSNR {psnr:.2f} dB",
              flush=True)

    ok = [r for r in rows if "error" not in r]
    res = {"stage": "8-gsplat", "gpu": _gpu_name(), "n_scenes": len(rows), "n_ok": len(ok),
           "median_psnr_db": (round(float(np.median([r["heldout_psnr_db"] for r in ok])), 2)
                              if ok else None),
           "median_train_seconds": (round(float(np.median([r["train_seconds"] for r in ok])), 1)
                                    if ok else None), "results": rows}
    (out / "results_gsplat.json").write_text(json.dumps(res, indent=2))
    return res


def stage_moge(groups, media_parent: Path, out: Path, n_images: int) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from moge.model.v2 import MoGeModel
    dev = "cuda"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval().to(dev)
    paths = [p for g in groups for p in g["paths"]][:n_images]
    times, rows = [], []
    for i, rel in enumerate(paths):
        f = media_parent / rel
        if not f.exists():
            continue
        im = Image.open(f).convert("RGB"); im.thumbnail((512, 512))
        x = torch.tensor(np.array(im)/255.0, dtype=torch.float32).permute(2, 0, 1).to(dev)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            o = model.infer(x)
        torch.cuda.synchronize(); dt = time.time() - t0
        if i:
            times.append(dt)
        intr = o["intrinsics"].float().cpu().numpy()
        rows.append({"path": rel, "seconds": round(dt, 3),
                     "fov_x_deg": round(float(2*np.degrees(np.arctan(0.5/intr[0, 0]))), 1)})
    res = {"stage": "3-moge-gpu", "gpu": _gpu_name(), "n_images": len(rows),
           "seconds_per_image": round(float(np.mean(times)), 3) if times else None,
           "seconds_first_call": rows[0]["seconds"] if rows else None,
           "fov_median_deg": round(float(np.median([r["fov_x_deg"] for r in rows])), 1) if rows else None,
           "results": rows}
    (out / "results_moge_gpu.json").write_text(json.dumps(res, indent=2))
    return res


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--groups", type=Path, default=DEFAULT_OUT / "room_groups.json")
    ap.add_argument("--golden", type=Path, default=REPO / "data" / "golden" / "golden_set.json")
    ap.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--moge", action="store_true"); ap.add_argument("--mapanything", action="store_true")
    ap.add_argument("--gsplat", action="store_true")
    ap.add_argument("--max-groups", type=int, default=12)
    ap.add_argument("--max-views", type=int, default=None,
                    help="views per group; default scales with VRAM")
    ap.add_argument("--ma-max-side", type=int, default=384)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--skip-install", action="store_true")
    a = ap.parse_args(argv)
    if a.all:
        a.moge = a.mapanything = a.gsplat = True
    a.out.mkdir(parents=True, exist_ok=True)

    if not a.skip_install:
        print("== dependencies"); ensure_deps(need_gsplat=a.gsplat)
    import torch
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU visible. On Colab: Runtime > Change runtime type > GPU.",
              file=sys.stderr)
        return 2
    print(f"== GPU: {_gpu_name()}  torch {torch.__version__}")

    if not a.groups.exists():
        sh(sys.executable, "-m", "eval.models.grouping")
    groups = json.loads(a.groups.read_text())["groups"]
    print(f"== {len(groups)} room groups")
    fetch_media(a.golden, a.media, groups[:a.max_groups])
    media_parent = a.media.parent

    summary = {}
    if a.moge:
        print("== MoGe-2 (GPU timing)"); summary["moge"] = stage_moge(groups, media_parent, a.out, a.n_images)
    if a.mapanything:
        print("== MapAnything"); summary["mapanything"] = stage_mapanything(groups, media_parent, a.out, a.max_groups,
                                                                  a.max_views, a.ma_max_side)
    if a.gsplat:
        print("== gsplat"); summary["gsplat"] = stage_gsplat(groups, media_parent, a.out, min(a.max_groups, 4), a.iters)

    print("\n==== SUMMARY ====")
    for k, v in summary.items():
        print(f"{k}: " + json.dumps({kk: vv for kk, vv in v.items() if kk != "results"}))
    print(f"\nresults in {a.out}  (send the results_*.json and splat_*.png back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
