"""Stage 3 engine plugins — one interface, several backends (AD-4).

    reconstruct(images, priors) -> Reconstruction

Engines are swappable on purpose: this is the fastest-moving corner of the stack
and we expect to replace the model inside twelve months. Phase 1 ships two:

``moge2``
    The monocular workhorse. Chosen because it predicts **camera intrinsics**,
    which Phase 0 proved is not optional — estate agents shoot at a median 98.6°
    horizontal field of view, and a depth model that makes you assume a lens puts
    the ceiling anywhere between 2.8 m and 5.9 m depending on what you assumed.

``synthetic``
    A deterministic fake that returns a plausible room box. Not a toy: the
    roadmap's decoupling contract says streams develop against synthetic
    counterparts of each other, and it is what lets stages 4-9 be tested on a box
    with no GPU and no weights.

``mapanything`` is the Phase 2 multi-view engine (1.35 s/group measured in Phase
0); its binding is declared here so profiles can name it, and it reports itself
unavailable until the weights are wired in.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("room_geometry.engines")


@dataclass
class Reconstruction:
    """What every engine returns, whatever it did inside."""
    points: np.ndarray                    # (N, 3) camera/world points, metres
    confidence: np.ndarray                # (N,) per-point, 0..1
    poses: list[list[list[float]]] = field(default_factory=list)   # 4x4 cam2world
    intrinsics: list[list[list[float]]] = field(default_factory=list)
    fov_x_deg: float | None = None
    fov_y_deg: float | None = None
    engine: str = "unknown"
    up_prior: np.ndarray | None = None    # what the camera says "up" probably is
    n_views: int = 0
    metric: bool = True                   # False when the scale is arbitrary
    notes: dict = field(default_factory=dict)


class EngineUnavailable(RuntimeError):
    """The engine's weights or source are not on this box. Not a bug — a routing signal."""


# --------------------------------------------------------------------------


class SyntheticEngine:
    """A deterministic room box, seeded by the image path.

    Deliberately *plausible* rather than random: 2.4-3.0 m ceilings, 2.2-6.5 m
    sides, a scattering of wall noise. That means a synthetic listing exercises
    the same plausibility checks as a real one, and a bug in stage 4's plane
    fitting shows up here first, where it is cheap to find.
    """

    name = "synthetic"

    def available(self) -> bool:
        return True

    def reconstruct(self, images: list[Path], priors: dict | None = None) -> Reconstruction:
        seed = int(hashlib.sha256("|".join(str(p) for p in images).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        w = float(rng.uniform(2.4, 6.5))
        d = float(rng.uniform(2.2, 5.5))
        h = float(rng.uniform(2.4, 3.0))
        # Faces are NOT sampled uniformly. A 98-degree lens held at chest height
        # sees a great deal of floor and ceiling and only part of each wall, and
        # a uniformly-sampled box would make "up" genuinely ambiguous — which is
        # a property of the fake, not of real rooms.
        face_p = np.array([0.30, 0.22, 0.12, 0.12, 0.12, 0.12])
        pts = []
        for _ in range(9000):
            face = int(rng.choice(6, p=face_p))
            u, v = rng.random(2)
            if face == 0:
                pts.append([u * w, v * d, 0.0])                      # floor
            elif face == 1:
                pts.append([u * w, v * d, h])                        # ceiling
            elif face == 2:
                pts.append([0.0, v * d, u * h])
            elif face == 3:
                pts.append([w, v * d, u * h])
            elif face == 4:
                pts.append([u * w, 0.0, v * h])
            else:
                pts.append([u * w, d, v * h])
        arr = np.asarray(pts) + rng.normal(0, 0.012, (9000, 3))
        conf = np.clip(rng.normal(0.8, 0.1, len(arr)), 0.05, 1.0)
        return Reconstruction(points=arr, confidence=conf, engine=self.name,
                              up_prior=np.array([0.0, 0.0, 1.0]),
                              n_views=len(images), fov_x_deg=98.0, fov_y_deg=74.0,
                              notes={"synthetic_box_m": [round(w, 2), round(d, 2), round(h, 2)]})


class MoGe2Engine:
    """MoGe-2 monocular point maps with predicted intrinsics (AD-5).

    Loaded lazily and cached on the class: the model is ~1.5 GB and a cold load
    per room would dominate the stage's whole budget, which AD-17 bans outright on
    the hot path.
    """

    name = "moge2"
    model_id = "Ruicheng/moge-2-vitl-normal"
    _model = None

    def __init__(self, max_side: int = 512, threads: int = 4):
        self.max_side = max_side
        self.threads = threads

    def available(self) -> bool:
        vendor = Path(__file__).resolve().parents[2] / "vendor" / "moge"
        return vendor.exists() and (vendor / "moge").exists()

    def _load(self):
        if MoGe2Engine._model is not None:
            return MoGe2Engine._model
        import sys
        import torch
        vendor = Path(__file__).resolve().parents[2] / "vendor" / "moge"
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))
        torch.set_num_threads(self.threads)
        try:
            from moge.model.v2 import MoGeModel
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"MoGe-2 source not importable: {e}") from e
        # .float() is not optional: the checkpoint is fp16 and CPU convolutions
        # refuse a half-precision input against float bias. Phase 0 hit the same
        # wall on the Colab CPU path.
        MoGe2Engine._model = MoGeModel.from_pretrained(self.model_id).eval().float()
        return MoGe2Engine._model

    def reconstruct(self, images: list[Path], priors: dict | None = None) -> Reconstruction:
        import torch
        from PIL import Image
        if not images:
            raise ValueError("no images")
        model = self._load()
        # Monocular era (ROADMAP S3): one view per room. Multi-view lands in P2
        # with MapAnything, and this interface does not change when it does.
        path = images[0]
        im = Image.open(path).convert("RGB")
        im.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(t, use_fp16=False)
        pts = out["points"].cpu().numpy()
        mask = (out["mask"].cpu().numpy().astype(bool) if "mask" in out
                else np.ones(pts.shape[:2], bool))
        good = mask & np.isfinite(pts).all(axis=-1)
        xyz = pts[good]
        conf_map = out.get("mask_prob")
        conf_vals = (conf_map.cpu().numpy()[good] if conf_map is not None else None)
        k = out.get("intrinsics")
        fov_x = fov_y = None
        intr: list[list[list[float]]] = []
        if k is not None:
            kk = k.cpu().numpy()
            fov_x = float(np.degrees(2 * np.arctan(0.5 / kk[0, 0])))
            fov_y = float(np.degrees(2 * np.arctan(0.5 / kk[1, 1])))
            intr = [kk.tolist()]
        conf = np.full(len(xyz), 0.8, dtype=float)
        # MoGe returns camera-frame points with +y down, and estate agents hold the
        # camera roughly level. That makes -y a genuine prior on "up", not a guess.
        return Reconstruction(points=xyz, confidence=conf, intrinsics=intr,
                              fov_x_deg=fov_x, fov_y_deg=fov_y, engine=self.name,
                              up_prior=np.array([0.0, -1.0, 0.0]),
                              n_views=1, notes={"image": str(path), "px": list(im.size)})


class MapAnythingEngine:
    """Multi-view, 3-8 images per room. Phase 2's primary engine (AD-4).

    Declared now so profiles can bind to it and so the interface is exercised;
    reports itself unavailable until the weights are wired in, which routes
    listings to MoGe-2 instead of failing them.
    """

    name = "mapanything"

    def available(self) -> bool:
        return False

    def reconstruct(self, images: list[Path], priors: dict | None = None) -> Reconstruction:
        raise EngineUnavailable(
            "MapAnything is a Phase 2 deliverable — measured at 1.35 s/group in "
            "Phase 0 but not yet wired into the pipeline. Falling back to MoGe-2.")


ENGINES = {"synthetic": SyntheticEngine, "moge2": MoGe2Engine, "mapanything": MapAnythingEngine}


def get_engine(name: str, **kw):
    if name not in ENGINES:
        raise KeyError(f"unknown geometry engine {name!r}; have {', '.join(ENGINES)}")
    return ENGINES[name](**kw) if name != "synthetic" else ENGINES[name]()


def resolve(preferred: str, fallbacks: tuple[str, ...] = ("moge2", "synthetic")):
    """First available engine, preferred first. Returns ``(engine, chain_note)``."""
    tried: list[str] = []
    for name in (preferred, *fallbacks):
        if name in tried:
            continue
        tried.append(name)
        try:
            e = get_engine(name)
        except KeyError:
            continue
        if e.available():
            return e, ("preferred" if name == preferred
                       else f"fell back from {preferred} to {name}")
    raise EngineUnavailable(f"no geometry engine available (tried {', '.join(tried)})")
