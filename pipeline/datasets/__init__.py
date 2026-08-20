"""On-demand dataset acquisition for pipeline stages that need external corpora."""
from .fetch import (  # noqa: F401
    ensure, status, data_home, LicenceRefused, ManualAcquisitionRequired,
)
from .registry import REGISTRY, DatasetSpec  # noqa: F401
