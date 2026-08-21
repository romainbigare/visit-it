"""The `learned` vectoriser binding — the plan reader trained by the Colab notebook.

`notebooks/train_plan_vectoriser_colab.ipynb` produces `models/plan_vectoriser.pt`.
Drop it in and stage 5 uses it; leave it out and stage 5 uses the classical engine.
Nothing else changes — both emit the same room masks, both are scored the same way
by the harness, and the classical engine stays as the baseline the learned one has
to beat (AD-4).

The model predicts three classes per pixel — outside the flat, a room's interior,
a wall — which is exactly the shape the rest of stage 5 already consumes. Rooms
come out as connected components of the room class; the OCR, the scale solve, the
adjacency graph and the aperture detector are untouched.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("floorplan.learned")

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "plan_vectoriser.pt"
#: The classes the notebook trains, in order.
CLASSES = ("outside", "room", "wall")
_MODEL = None
_META: dict = {}


def available(path: Path = MODEL_PATH) -> bool:
    """Is there a trained plan reader on this box?"""
    if not path.exists():
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def load(path: Path = MODEL_PATH):
    """Load once and keep it. A cold load per listing would dominate the stage."""
    global _MODEL, _META
    if _MODEL is not None:
        return _MODEL
    import torch
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise RuntimeError(
            "the learned vectoriser needs segmentation-models-pytorch "
            "(pip install segmentation-models-pytorch)") from e
    ck = torch.load(path, map_location="cpu")
    _META = {k: v for k, v in ck.items() if k != "model"}
    arch = _META.get("arch", "unet_resnet34")
    encoder = arch.split("_", 1)[1] if "_" in arch else "resnet34"
    m = smp.Unet(encoder, encoder_weights=None, classes=len(CLASSES))
    m.load_state_dict(ck["model"])
    _MODEL = m.eval()
    log.info("loaded learned vectoriser %s (room IoU %s on its own val split)",
             arch, _META.get("room_iou"))
    return _MODEL


def meta() -> dict:
    return dict(_META)


def predict(rgb: np.ndarray, size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(room_mask, wall_mask)`` at the input image's resolution.

    Inference is at the size the model was trained on and the masks are resized
    back, because the rest of stage 5 works in the plan's own pixel frame and every
    polygon it emits has to line up with the OCR word boxes.
    """
    import torch
    model = load()
    n = size or int(_META.get("size", 512))
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
    with torch.no_grad():
        pred = model(t).argmax(1)[0].numpy().astype(np.uint8)
    up = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return (up == 1).astype(np.uint8), (up == 2).astype(np.uint8)


def rooms_from_prediction(room_mask: np.ndarray, min_frac: float = 0.004
                          ) -> list[np.ndarray]:
    """Connected components of the room class, largest first, slivers dropped."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(room_mask, 8)
    total = float(room_mask.size)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] / total < min_frac:
            continue
        out.append((lab == i).astype(np.uint8))
    out.sort(key=lambda m: -int(m.sum()))
    return out
