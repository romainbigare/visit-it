"""Pipeline plumbing: artifacts, stage contracts, the DAG runner, the run ledger.

Nothing in here knows anything about floor plans or Gaussians. That separation is
what lets a stage be re-run, re-scored and swapped without touching the others
(ARCHITECTURE AD-3).
"""
from .artifacts import Artifact, ArtifactStore, SchemaError, data_root, validate
from .stages import (STAGE_DEPS, STAGE_ORDER, Stage, StageContext, StageResult,
                     get_stage, normalise_stage, register_stage, registered)
from .profiles import PROFILES, Profile, get_profile
from .ledger import Ledger, RunRecord, StageRecord, latency_report, new_run_id
from .runner import run_listing

__all__ = [
    "Artifact", "ArtifactStore", "SchemaError", "data_root", "validate",
    "STAGE_DEPS", "STAGE_ORDER", "Stage", "StageContext", "StageResult",
    "get_stage", "normalise_stage", "register_stage", "registered",
    "PROFILES", "Profile", "get_profile",
    "Ledger", "RunRecord", "StageRecord", "latency_report", "new_run_id",
    "run_listing",
]
