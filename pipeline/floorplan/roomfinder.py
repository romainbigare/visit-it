"""Room polygons predicted by a model that needs a GPU, read back here.

Measured on the golden set, the two readers are good at opposite halves of the
job. A room-polygon model (Raster2Seq) finds *which* rooms exist, keeps an
open-plan kitchen-diner as one room, and finds a WC nobody labelled -- all things
our caption seeding has to guess at. What it cannot do is put a corner on a wall:
it works on a 256-pixel copy and rounds every coordinate to a 32-step grid, about
30 cm on a real flat. Our own reading is the mirror image: exact about walls,
guessing about rooms, and *good at names*, because the plan prints them and we
read them.

So: their rooms, our walls, our names.

**The model needs a GPU and this pipeline runs on four CPU cores**, so predictions
are made once, elsewhere -- ``notebooks/plan_reading_modal.ipynb`` -- and imported:

    python -m tools.import_room_predictions results.zip

They land in ``data/room_predictions/<listing_id>.json`` in the *source* image's
pixels. Without them stage 5 behaves exactly as it did before and says so in its
QA flags.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .preprocess import PlanImage

log = logging.getLogger("floorplan.roomfinder")

REPO = Path(__file__).resolve().parents[2]
PREDICTION_DIR = REPO / "data" / "room_predictions"

#: Predictions carry the room type the model inferred. We prefer the name printed
#: on the plan when there is one, so these are a fallback rather than the answer.
FALLBACK_LABELS = {
    "kitchen": "kitchen", "living room": "living_room", "living_room": "living_room",
    "bed room": "bedroom", "bedroom": "bedroom", "bath": "bathroom",
    "bathroom": "bathroom", "restroom": "bathroom", "washing room": "utility",
    "entry": "hall", "corridor": "hall", "storage": "storage", "closet": "storage",
    "garage": "garage", "balcony": "balcony",
}
#: Classes that are not rooms in a flat, however the model labels them.
NOT_A_ROOM = {"outside", "outdoor", "unknown", "ps"}


def available(listing_id: str, root: Path | None = None) -> bool:
    return _path(listing_id, root).exists()


def _path(listing_id: str, root: Path | None = None) -> Path:
    return (root or PREDICTION_DIR) / f"{listing_id}.json"


def load(listing_id: str, root: Path | None = None) -> dict | None:
    path = _path(listing_id, root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:                                     # noqa: BLE001
        log.warning("room predictions for %s are unreadable: %s", listing_id, exc)
        return None


def to_geometry(polygons: list, pi: PlanImage) -> list[np.ndarray]:
    """Move polygons from the source file's pixels into the ones stage 5 works in.

    Two steps, in this order, because that is the order ``preprocess.prepare``
    applies them: scale the whole image down to the working size, then rotate
    about the *scaled* image's centre to take the skew out. Applying them the
    other way round, or skipping the rotation, puts every outline out by a couple
    of percent -- which looks exactly like the model being wrong when it is not.
    """
    h, w = pi.ink.shape
    scale = pi.scale_from_source or 1.0
    out = []
    rot = (cv2.getRotationMatrix2D((w / 2.0, h / 2.0), pi.deskew_deg, 1.0)
           if abs(pi.deskew_deg) > 1e-6 else None)
    for poly in polygons:
        p = np.asarray(poly, dtype=float).reshape(-1, 2) * scale
        if rot is not None:
            p = (np.hstack([p, np.ones((len(p), 1))]) @ rot.T)
        out.append(p)
    return out


def seeds_for(listing_id: str, pi: PlanImage,
              root: Path | None = None) -> tuple[list[np.ndarray], list[str], dict]:
    """Room polygons for this plan, in ``pi``'s pixels, plus their predicted types."""
    payload = load(listing_id, root)
    if not payload:
        return [], [], {}
    rooms = payload.get("rooms") or []
    keep, labels = [], []
    for room in rooms:
        poly = room.get("polygon_px") or []
        if len(poly) < 3:
            continue
        name = str(room.get("label") or "").strip().lower()
        if name in NOT_A_ROOM:
            continue
        keep.append(poly)
        labels.append(FALLBACK_LABELS.get(name, name or ""))
    if not keep:
        return [], [], {}
    meta = {"source": payload.get("source", "raster2seq"),
            "checkpoint": payload.get("checkpoint"),
            "predicted_rooms": len(rooms),
            "used_rooms": len(keep)}
    return to_geometry(keep, pi), labels, meta
