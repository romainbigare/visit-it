"""Turn a scraped floor-plan image into a wall mask and a building footprint.

Everything here is deliberately style-agnostic. Phase 0's sample alone contains
black-on-white line plans, navy-on-white plans, greyscale PNGs with an alpha
channel, and at least one plan drawn light-on-dark. So we never assume "walls are
black" — we assume *walls are the thickest strokes on the page*, which held on
every plan we have looked at.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

log = logging.getLogger("floorplan.preprocess")

#: Plans below this get upscaled before OCR; Phase 0 measured that dimension
#: extraction is strongly resolution-dependent (54% overall, 62% on high-res).
OCR_MIN_SIDE = 1800
#: Anything past this is downscaled for the *geometry* pass — the morphology cost
#: is quadratic in resolution and nothing is gained past ~2000 px.
GEOM_MAX_SIDE = 2000


@dataclass
class PlanImage:
    rgb: np.ndarray                  # H x W x 3, uint8, alpha already flattened onto white
    ink: np.ndarray                  # H x W uint8, 1 = drawn on
    walls: np.ndarray                # H x W uint8, 1 = thick stroke (a wall)
    wall_half_px: float
    deskew_deg: float
    scale_from_source: float         # geometry px -> source px
    source_size: tuple[int, int]
    qa_flags: list[str] = field(default_factory=list)


def load_rgb(path: Path) -> np.ndarray:
    """Load any plan asset as RGB with transparency composited onto white.

    Not cosmetic: a PNG in ``LA`` mode read straight through OpenCV comes out with
    a *black* background, and every downstream threshold then inverts.
    """
    im = Image.open(path)
    if im.mode in ("LA", "RGBA", "PA", "P"):
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)),
                                   im.convert("RGBA"))
    return np.array(im.convert("RGB"))


def ink_mask(rgb: np.ndarray, threshold: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Ink = far from the page colour. Returns ``(mask, background_rgb)``.

    The page colour is the modal colour over a coarse grid, not the corner pixel:
    plans arrive with coloured borders, watermarks and letterboxing.
    """
    small = rgb[::4, ::4].reshape(-1, 3)
    q = (small // 16).astype(np.int32)
    keys, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
    k = int(keys[counts.argmax()])
    bg = np.array([(k // 256) * 16 + 8, ((k // 16) % 16) * 16 + 8, (k % 16) * 16 + 8],
                  dtype=np.float32)
    mask = (np.linalg.norm(rgb.astype(np.float32) - bg, axis=2) > threshold).astype(np.uint8)
    if mask.mean() > 0.55:
        # Light ink on a dark page. Rare but real, and silently unrecoverable if missed.
        mask = 1 - mask
    return mask, bg


def estimate_skew(walls: np.ndarray, max_deg: float = 8.0) -> float:
    """Dominant wall direction, in degrees, wrapped to (-45, 45].

    Estimated from wall edges rather than from the whole ink mask, because text
    baselines and dimension arrows outvote the architecture otherwise.
    """
    edges = cv2.Canny((walls * 255).astype(np.uint8), 50, 150)
    if edges.sum() == 0:
        return 0.0
    min_len = max(30, int(0.06 * max(walls.shape)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60,
                            minLineLength=min_len, maxLineGap=6)
    if lines is None:
        return 0.0
    acc: dict[int, float] = {}
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 90.0
        if ang > 45.0:
            ang -= 90.0
        acc[int(round(ang * 4))] = acc.get(int(round(ang * 4)), 0.0) + length
    if not acc:
        return 0.0
    best = max(acc.items(), key=lambda kv: kv[1])[0] / 4.0
    return best if abs(best) <= max_deg else 0.0


def rotate(img: np.ndarray, deg: float, fill: int | tuple = 255) -> np.ndarray:
    if abs(deg) < 1e-6:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    flags = cv2.INTER_NEAREST if img.dtype == np.uint8 and img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=fill)


def wall_half_thickness(ink: np.ndarray) -> float:
    """Half the wall stroke width, in pixels, from the distance transform's ridge.

    Text and fixture lines put a peak at 1-3 px; walls put one higher up. We take
    the *highest* peak carrying real support, which is the wall.
    """
    d = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    ridge = (d >= cv2.dilate(d, np.ones((3, 3), np.uint8)) - 1e-6) & (d > 0.9)
    v = d[ridge]
    if v.size < 20:
        return 2.0
    hi = float(np.percentile(v, 99.5))
    hist, edges = np.histogram(v, bins=np.arange(1.0, max(6.0, hi) + 2.0))
    if hist.max() == 0:
        return 2.0
    supported = np.where(hist >= hist.max() * 0.10)[0]
    return float(edges[supported[-1]] + 0.5)


def extract_walls(ink: np.ndarray, half_px: float, keep_frac: float = 0.90) -> tuple[np.ndarray, int]:
    """Open the ink with a disc slightly narrower than a wall, keep the big pieces.

    The opening is what deletes text, furniture symbols and dimension arrows. The
    component filter is what stops a solid compass rose or a filled bathtub from
    being treated as architecture. Returns ``(walls, n_components_kept)``.
    """
    k = max(3, int(round(half_px * 1.5)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, se)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if n <= 1:
        return opened, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas) + 1
    total = areas.sum()
    keep = np.zeros_like(opened)
    acc, kept = 0, 0
    for i in order:
        if stats[i, cv2.CC_STAT_AREA] < total * 0.01 and kept >= 1:
            break
        keep[lab == i] = 1
        acc += stats[i, cv2.CC_STAT_AREA]
        kept += 1
        if acc >= keep_frac * total:
            break
    return keep, kept


def prepare(path: Path, *, geom_max_side: int = GEOM_MAX_SIDE) -> PlanImage:
    """Full preprocessing pass. This is what stage 5 calls."""
    src = load_rgb(path)
    src_h, src_w = src.shape[:2]
    scale = min(1.0, geom_max_side / max(src_h, src_w))
    rgb = cv2.resize(src, (int(src_w * scale), int(src_h * scale)),
                     interpolation=cv2.INTER_AREA) if scale < 1.0 else src

    ink, _bg = ink_mask(rgb)
    half = wall_half_thickness(ink)
    walls0, _ = extract_walls(ink, half)
    skew = estimate_skew(walls0)
    flags: list[str] = []
    if abs(skew) > 0.25:
        rgb = rotate(rgb, skew)
        ink, _bg = ink_mask(rgb)
        half = wall_half_thickness(ink)
        flags.append(f"deskewed_{skew:.2f}deg")

    walls, _n_kept = extract_walls(ink, half)
    if half < 2.0:
        flags.append("low_resolution_plan")

    return PlanImage(rgb=rgb, ink=ink, walls=walls, wall_half_px=half, deskew_deg=skew,
                     scale_from_source=scale, source_size=(src_w, src_h), qa_flags=flags)


def count_outlines(footprint: np.ndarray) -> int:
    """How many separate buildings the sheet draws.

    Plans for maisonettes carry one outline per storey. Worth knowing about:
    Phase 1's shell is single-storey, so a second outline is a QA flag rather than
    a feature, and the largest one is the one we build.
    """
    lab, n = ndi.label(footprint)
    if n <= 1:
        return int(n)
    sizes = ndi.sum(footprint, lab, range(1, n + 1))
    return int((sizes > 0.25 * sizes.max()).sum())
