"""On-demand dataset acquisition for pipeline stages that need external corpora."""
from .fetch import (  # noqa: F401
    ensure, ensure_all, status, verify, data_home, sha256_of,
    ManualAcquisitionRequired,
)
from .registry import REGISTRY, DatasetSpec, DatasetFile  # noqa: F401
