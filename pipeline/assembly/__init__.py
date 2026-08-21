"""Stage 6 — assembly."""
from .matching import assign, build_cost_matrix  # noqa: F401
from .pose import apply_pose, refine  # noqa: F401
from .stage import assemble, run  # noqa: F401
