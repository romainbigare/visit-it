"""Stage 8 (Phase 1) — the shell builder."""
from .mesh import Mesh, PROVENANCE, build_room, to_glb  # noqa: F401
from .stage import SHELL_BYTES_BUDGET, build, run  # noqa: F401
