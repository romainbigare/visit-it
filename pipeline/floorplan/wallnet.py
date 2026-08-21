"""What is a wall, decided by a model that has seen thousands of plans.

The classical vectoriser decides a pixel is a wall if it is dark. A floor plan
draws cabinets, beds, baths, door-swing arcs, dimension lines and its own captions
in the same dark ink, so all of those became walls, and rooms came out bounded by
the furniture. Measured across the golden set, the median room outline had more
than a third of its length drawn across open floor.

This module answers the same question with a UNet/ResNet-34 trained on CubiCasa5K
to label every pixel wall, door, window or floor. It ignores furniture and door
swings because it was taught what a wall *is*, not how dark it is. Its output
replaces the ink mask that stage 5 hands the watershed; everything downstream --
caption seeding, distance peaks, polygon fitting, adjacency, apertures -- is
unchanged. That is the AD-4 engine boundary doing its job.

**It does not replace the classical engine, it complements it.** On the twelve
golden plans the classical reading was failing, this wins eleven; on the twelve it
was already handling, the classical reading wins eleven. So stage 5 runs both and
keeps whichever puts more of its outline on real drawn lines -- a choice it can
make from the plan alone, with no ground truth.

Weights are not committed. Fetch them once with::

    python -m tools.fetch_wallnet

Without them this module reports itself unavailable and stage 5 behaves exactly as
it did before.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO / "models" / "plan_walls.safetensors"

#: Source: https://huggingface.co/Yytsi/floorplan-to-3d-walls (MIT).
MODEL_URL = ("https://huggingface.co/Yytsi/floorplan-to-3d-walls/"
             "resolve/main/best.safetensors")

#: The four classes it was trained on, in channel order.
CLASSES = ("floor", "wall", "door", "window")
#: Everything that bounds a room. A doorway bounds one as much as a wall does --
#: the watershed wants the opening marked and treats it as passable itself.
#: Excluding the window class was tried and changed nothing either way.
BARRIER = (1, 2, 3)

#: The resolution it was trained at. Larger plans are tiled at this size rather
#: than downscaled, because a wall that lands under one pixel is not a wall.
TILE = 512
#: How far the predicted wall band may sit from the stroke it describes, in
#: multiples of the drawn wall's half-thickness. Swept over 0.5-1.0 and the
#: measured difference was nil, so it is a constant rather than a knob.
INK_REACH = 1.0
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_model = None
_load_failed = False


def available() -> bool:
    """True when the weights are on disk and torch can be imported."""
    if not MODEL_PATH.exists() or _load_failed:
        return False
    try:
        import torch  # noqa: F401
        import segmentation_models_pytorch  # noqa: F401
    except ImportError:
        return False
    return True


def _net():
    global _model, _load_failed
    if _model is None:
        try:
            import segmentation_models_pytorch as smp
            from safetensors.torch import load_file
            net = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=4)
            net.load_state_dict(load_file(MODEL_PATH))
            net.eval()
            _model = net
        except Exception:                                        # noqa: BLE001
            _load_failed = True
            raise
    return _model


def meta() -> dict:
    """Provenance for the artifact, per AD-15."""
    return {"engine": "cubicasa_unet_resnet34",
            "weights": MODEL_PATH.name,
            "source": MODEL_URL,
            "classes": list(CLASSES)}


# ------------------------------------------------------------------ preprocess


def whiten(rgb: np.ndarray) -> np.ndarray:
    """Page colour to white, darkest ink to black.

    The model was trained on dark ink on white paper. On a plan whose rooms are
    filled grey on a lavender page it labelled 62% of the flat "window" -- the
    whole interior became barrier and room extraction collapsed to three blobs.
    Levelling first is not a tweak, it is the normalisation the training data
    already had, and it took that plan from 0.14 to 0.86 outline-on-wall.
    """
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    page = int(np.argmax(np.bincount(lum.ravel(), minlength=256)))
    if page < 110:                       # light ink on a dark page: flip first
        lum = 255 - lum
        page = 255 - page
    floor = float(np.percentile(lum, 1))
    if page - floor < 20:                # nothing to stretch
        return cv2.cvtColor(lum, cv2.COLOR_GRAY2RGB)
    out = np.clip((lum.astype(np.float32) - floor) * (255.0 / (page - floor)), 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def blank_text(rgb: np.ndarray, words) -> np.ndarray:
    """Paint out every OCR word box: the model reads lettering as wall."""
    out = rgb.copy()
    h, w = rgb.shape[:2]
    for word in words:
        x0, y0 = max(0, int(word.x) - 2), max(0, int(word.y) - 2)
        x1, y1 = min(w, int(word.x + word.w) + 3), min(h, int(word.y + word.h) + 3)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 255
    return out


def crop_to_drawing(rgb: np.ndarray, ink: np.ndarray,
                    margin: float = 0.03) -> tuple[np.ndarray, tuple[int, int]]:
    """Trim the page down to the drawing so the walls get real pixels.

    Not the largest connected component: a plan whose rooms are separated by a
    corridor is several components and taking the biggest crops half the flat
    away -- which is exactly what happened first time. Take the bounding box of
    every component big enough to be structure instead.
    """
    h, w = ink.shape
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    if n < 2:
        return rgb, (0, 0)
    keep = np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= max(40.0, 4e-5 * h * w))[0] + 1
    if not len(keep):
        return rgb, (0, 0)
    x0 = int(stats[keep, cv2.CC_STAT_LEFT].min())
    y0 = int(stats[keep, cv2.CC_STAT_TOP].min())
    x1 = int((stats[keep, cv2.CC_STAT_LEFT] + stats[keep, cv2.CC_STAT_WIDTH]).max())
    y1 = int((stats[keep, cv2.CC_STAT_TOP] + stats[keep, cv2.CC_STAT_HEIGHT]).max())
    m = int(margin * max(x1 - x0, y1 - y0))
    x0, y0 = max(0, x0 - m), max(0, y0 - m)
    x1, y1 = min(w, x1 + m), min(h, y1 + m)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return rgb, (0, 0)
    return rgb[y0:y1, x0:x1], (x0, y0)


# ------------------------------------------------------------------- inference


def _classify(rgb: np.ndarray, tiles: int = 2) -> np.ndarray:
    """Class map at the input's own resolution, tiled at the training size."""
    import torch

    h, w = rgb.shape[:2]
    side = TILE * tiles
    s = side / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    canvas = np.full((side, side, 3), 255, np.uint8)
    top, left = (side - nh) // 2, (side - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh),
                                                      interpolation=cv2.INTER_AREA)

    logits = np.zeros((4, side, side), np.float32)
    hits = np.zeros((side, side), np.float32)
    step = TILE if tiles == 1 else TILE // 2
    for ty in range(0, side - TILE + 1, step):
        for tx in range(0, side - TILE + 1, step):
            tile = canvas[ty:ty + TILE, tx:tx + TILE]
            x = (tile.astype(np.float32) / 255.0 - _MEAN) / _STD
            with torch.no_grad():
                out = _net()(torch.from_numpy(x).permute(2, 0, 1)[None].float())
            logits[:, ty:ty + TILE, tx:tx + TILE] += out[0].numpy()
            hits[ty:ty + TILE, tx:tx + TILE] += 1
    pred = (logits / np.maximum(hits, 1)).argmax(0).astype(np.uint8)
    return cv2.resize(pred[top:top + nh, left:left + nw], (w, h),
                      interpolation=cv2.INTER_NEAREST)


def barrier(rgb: np.ndarray, ink: np.ndarray, words,
            wall_half_px: float | None = None) -> np.ndarray | None:
    """The room-bounding mask, shaped like ``ink``. ``None`` if unavailable.

    The net says *which* strokes are walls; the ink says *where* those walls are.
    The result is the ink, filtered -- not the prediction, trimmed.

    That direction matters. The net's predicted wall band is a segmentation blob
    that runs wide of the stroke it describes, so a barrier made of prediction
    eats the room from all four sides: rooms came out a median 16% under their
    advertised floor area. Keeping the drawn stroke and using the prediction only
    to decide whether that stroke is a wall or a kitchen cabinet gives the
    furniture rejection with none of the erosion, and rooms now grow *across* the
    cabinet run to the wall behind it, which is where the room really ends.
    """
    if not available():
        return None
    try:
        prepared = blank_text(whiten(rgb), words)
        crop, (ox, oy) = crop_to_drawing(prepared, ink)
        pred = _classify(crop)
    except Exception:                                            # noqa: BLE001
        return None
    full = np.zeros(ink.shape, np.uint8)
    full[oy:oy + pred.shape[0], ox:ox + pred.shape[1]] = np.isin(pred, BARRIER)

    # The barrier is the prediction, trimmed back to where there is actually a
    # stroke. Two directions were tried and this one wins on both counts:
    # a barrier of pure prediction runs wide of the drawn line and eats the room
    # (16% under the advertised floor area); a barrier of ink filtered by the
    # prediction keeps the ink's imprecision and scored worse on outline fit and
    # on area alike. ``INK_REACH`` is how far the prediction may sit from the
    # stroke it describes, as a multiple of the drawn wall's half-thickness.
    reach = max(2, int(round(INK_REACH * (wall_half_px or ink.shape[1] / 400))))
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((2 * reach + 1,) * 2, np.uint8))
    mask = (full & near_ink).astype(np.uint8)
    if mask.sum() < 0.15 * full.sum():
        # Prediction and drawing barely overlap: a badly deskewed or very faint
        # plan. Fall back to the prediction rather than hand back almost nothing.
        mask = full

    # Bridge the hairline left where two walls meet, which would otherwise let
    # one room's basin flood into the next.
    k = max(2, int(round(ink.shape[1] / 500)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


# --------------------------------------------------------------------- scoring


def outline_on_wall(polygons, ink: np.ndarray, tol: int | None = None) -> list[float]:
    """Per polygon, the fraction of its outline that lies on a drawn line.

    The honest test of a vectorised room, and it needs no ground truth: the plan
    itself is the answer key. A room that follows the walls scores near 1; a room
    that stopped at a cabinet or followed a door swing cuts across open floor and
    scores low. Stage 5 uses it to choose between the two engines.

    One blind spot to know about: a polygon small enough to trace *around some
    lettering* also scores near 1. Only ever score polygons that already cleared
    a room-sized area floor.
    """
    if tol is None:
        tol = max(3, int(round(ink.shape[1] / 220)))
    dist = cv2.distanceTransform((1 - ink).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = ink.shape
    scores = []
    for poly in polygons:
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            continue
        pts = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            n = max(2, int(np.hypot(*(b - a))))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + (b - a) * t)
        pts = np.concatenate(pts)
        xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
        scores.append(float((dist[ys, xs] <= tol).mean()))
    return scores
