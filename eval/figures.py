"""Render the validation figures.

Palette is the validated categorical set (blue/orange/aqua/yellow); the light
mode carries a contrast WARN on aqua and yellow, so every bar is directly
labelled rather than relying on fill alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, BAD = "#0ca30c", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _thumb(path: Path, size=(320, 240)) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    return np.asarray(im)


def fig_triage(res: dict, golden: Path, out: Path) -> None:
    """Show what floor-plan detection got right and wrong - the failures matter most."""
    preds = res["predictions"]
    tp = [p for p in preds if p["truth_type"] == "floorplan" and p["pred_type"] == "floorplan"]
    fn = [p for p in preds if p["truth_type"] == "floorplan" and p["pred_type"] != "floorplan"]
    fp = [p for p in preds if p["truth_type"] == "photo" and p["pred_type"] == "floorplan"]
    show = ([("correct", p, GOOD) for p in tp[:4]]
            + [("MISSED", p, BAD) for p in fn]
            + [("false alarm", p, BAD) for p in fp[:5]])
    n = len(show)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.5))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (tag, p, col) in zip(axes, show):
        ax.imshow(_thumb(golden / p["path"]))
        ax.set_title(f"{tag}\n{p['pred_type']} {p['type_score']:.2f}",
                     fontsize=8, color=col)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    d = res["floorplan_detection"]
    fig.suptitle(f"Stage 0 — floor-plan detection on {res['n_images']} real images   "
                 f"P {d['precision']:.2f} · R {d['recall']:.2f} · F1 {d['f1']:.2f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=110); plt.close(fig)


def fig_fov(geo: dict, moge: dict, out: Path) -> None:
    """The headline: assumed FOV drives room height; MoGe measures FOV instead."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    fovs = sorted(float(k) for k in geo["fov_sweep"])
    med = [geo["fov_sweep"][str(f)]["median_m"] for f in fovs]
    p10 = [geo["fov_sweep"][str(f)]["p10_m"] for f in fovs]
    p90 = [geo["fov_sweep"][str(f)]["p90_m"] for f in fovs]
    x = np.arange(len(fovs))
    ax1.axhspan(2.3, 3.2, color=S3, alpha=0.14, zorder=0)
    ax1.text(-0.42, 2.75, "typical UK\nceiling", fontsize=8, color=INK2, va="center")
    ax1.vlines(x, p10, p90, color=S1, lw=2, alpha=0.45)
    ax1.plot(x, med, "o-", color=S1, lw=2, ms=8, label="Depth Anything V2 (assumed FOV)")
    for xi, m in zip(x, med):
        ax1.annotate(f"{m:.2f} m", (xi, m), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=INK)
    mf = moge.get("fov", {}).get("median_deg")
    if mf:
        ax1.axvline(np.interp(mf, fovs, x), color=S2, lw=2, ls="--")
        ax1.text(np.interp(mf, fovs, x) + 0.06, max(p90) * 0.96,
                 f"MoGe-2 measures\n{mf:.0f}° here", fontsize=9, color=S2, va="top")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{f:.0f}°" for f in fovs])
    ax1.set_xlabel("assumed horizontal field of view")
    ax1.set_ylabel("estimated room height (m)")
    ax1.set_title("Guessing the lens guesses the room")
    ax1.grid(axis="y", lw=0.6); ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")

    hs = [r["room_height_m"] for r in moge.get("results", []) if r.get("room_height_m")]
    dv = geo["fov_sweep"][str(90.0)]
    ax2.axhspan(2.3, 3.2, color=S3, alpha=0.14, zorder=0)
    if hs:
        ax2.boxplot([hs], positions=[1], widths=0.45, patch_artist=True,
                    boxprops=dict(facecolor=S2, alpha=0.55, edgecolor=S2),
                    medianprops=dict(color=INK, lw=2), showfliers=False,
                    whiskerprops=dict(color=S2), capprops=dict(color=S2))
        ax2.scatter(np.random.normal(1, 0.05, len(hs)), hs, s=18, color=S2,
                    edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax2.annotate(f"median {np.median(hs):.2f} m", (1, np.median(hs)),
                     textcoords="offset points", xytext=(28, 0), fontsize=9, color=INK)
    ax2.scatter([0], [dv["median_m"]], s=90, marker="_", color=S1, linewidth=3)
    ax2.annotate(f"{dv['median_m']:.2f} m", (0, dv["median_m"]),
                 textcoords="offset points", xytext=(20, 0), fontsize=9, color=INK)
    ax2.set_xlim(-0.6, 1.8); ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["DA-V2\n@ assumed 90°", "MoGe-2\n(own intrinsics)"])
    ax2.set_ylabel("room height (m)")
    ax2.set_title("Measured beats assumed")
    ax2.grid(axis="y", lw=0.6); ax2.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_depth(golden: Path, moge: dict, out: Path, n: int = 4) -> None:
    import torch  # noqa: PLC0415
    from moge.model.v2 import MoGeModel  # noqa: PLC0415
    torch.set_num_threads(4)
    model = MoGeModel.from_pretrained(moge["model"]).eval().float()
    picks = [r for r in moge["results"] if r.get("room_height_m")][:n]
    fig, axes = plt.subplots(3, len(picks), figsize=(3.1 * len(picks), 8.2))
    for j, r in enumerate(picks):
        im = Image.open(golden / r["path"]).convert("RGB"); im.thumbnail((512, 512))
        with torch.no_grad():
            x = torch.tensor(np.array(im) / 255.0, dtype=torch.float32).permute(2, 0, 1)
            o = model.infer(x, use_fp16=False)
        depth = o["depth"].numpy()
        pts = o["points"].numpy()
        mask = o["mask"].numpy() if "mask" in o else np.ones(depth.shape, bool)
        axes[0, j].imshow(np.asarray(im)); axes[0, j].axis("off")
        axes[0, j].set_title(f"{r.get('pred_room','?')}\nFOV {r['fov_x_deg']:.0f}°", fontsize=9)
        dm = np.where(mask, depth, np.nan)
        axes[1, j].imshow(dm, cmap="viridis"); axes[1, j].axis("off")
        axes[1, j].set_title(f"depth · median {np.nanmedian(dm):.1f} m", fontsize=9)
        p = pts[::3, ::3].reshape(-1, 3); m = mask[::3, ::3].reshape(-1)
        p = p[m & np.isfinite(p).all(1)]
        if len(p):
            axes[2, j].scatter(p[:, 0], -p[:, 1], s=0.4, c=p[:, 2], cmap="viridis")
            axes[2, j].set_aspect("equal")
            axes[2, j].set_title(f"point cloud · height {r['room_height_m']:.2f} m", fontsize=9)
        axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
        for s in axes[2, j].spines.values():
            s.set_color(GRID)
    fig.suptitle("Stage 3 — MoGe-2 on real listing photos: image, metric depth, point cloud",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=110); plt.close(fig)


def fig_plan_ocr(ocr: dict, golden: Path, out: Path) -> None:
    keys = [("with_room_labels", "room labels"), ("with_dimensions", "room dimensions"),
            ("with_area", "total area"), ("with_scale_ratio", "scale ratio"),
            ("with_not_to_scale_disclaimer", '"not to scale"')]
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[0, 0])
    vals = [ocr[k] * 100 for k, _ in keys]
    cols = [S1, S1, S1, S1, S2]
    y = np.arange(len(keys))
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([lbl for _, lbl in keys])
    ax.invert_yaxis(); ax.set_xlim(0, 108); ax.set_xlabel("% of 24 UK plans")
    ax.set_title("What a UK floor plan actually carries")
    ax.grid(axis="x", lw=0.6); ax.set_axisbelow(True)
    best = sorted(ocr["plans"], key=lambda p: -p["n_dims"])[:2]
    for i, p in enumerate(best):
        a = fig.add_subplot(gs[0, i + 1])
        a.imshow(_thumb(golden / p["path"], (520, 520)), cmap="gray")
        a.axis("off")
        a.set_title(f"{p['native_px']} · {p['n_room_labels']} labels · "
                    f"{p['n_dims']} dims\narea {p['areas_sqm'][:1] or '—'}", fontsize=9)
    fig.suptitle("Stage 5 — OCR over the 24 real floor plans in the golden set",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=110); plt.close(fig)


def main(results: Path, golden: Path, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    tri = json.loads((results / "stage0_triage.json").read_text())
    fig_triage(tri, golden, figdir / "fig1_triage.png"); print("  fig1_triage.png")
    ocr = json.loads((results / "stage5_plan_ocr.json").read_text())
    fig_plan_ocr(ocr, golden, figdir / "fig4_plan_ocr.png"); print("  fig4_plan_ocr.png")
    gp, mp = results / "stage34_geometry.json", results / "stage34_moge.json"
    if gp.exists() and mp.exists():
        geo, moge = json.loads(gp.read_text()), json.loads(mp.read_text())
        fig_fov(geo, moge, figdir / "fig2_fov.png"); print("  fig2_fov.png")
        try:
            fig_depth(golden, moge, figdir / "fig3_depth.png"); print("  fig3_depth.png")
        except Exception as e:  # noqa: BLE001
            print(f"  fig3_depth skipped: {e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("eval/results"))
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--figdir", type=Path, default=Path("eval/results/figures"))
    a = ap.parse_args()
    main(a.results, a.golden, a.figdir)
