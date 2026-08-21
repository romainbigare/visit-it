"""Stage 3 — per-room geometry, engine-pluggable (AD-4)."""
from .engines import ENGINES, EngineUnavailable, Reconstruction, get_engine, resolve  # noqa: F401
from .stage import run  # noqa: F401
