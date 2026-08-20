"""Shared Modal image and volume definitions.

Why this exists: the CPU validation in eval/ proved what runs without a GPU.
Three things could not be tested there - MapAnything (stage 3, multi-view),
gsplat (stage 8, appearance) and GPU timings for the cost model. This runs them.

Note the image installs MapAnything and gsplat straight from GitHub. That works
here even though it fails in our dev sandbox: Modal's builders have unrestricted
network access, so the raw.githubusercontent vendoring hack in tools/vendor_moge.py
is only needed locally.
"""
from __future__ import annotations

import modal

APP_NAME = "visit-it-gpu-validation"

# CUDA 12.4 + devel headers: gsplat compiles CUDA kernels at install time and
# needs nvcc, which the runtime-only images do not carry.
CUDA_TAG = "12.4.1-devel-ubuntu22.04"

gpu_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2", "pillow", "scipy", "opencv-python-headless", "matplotlib",
        "huggingface_hub", "safetensors", "transformers", "einops",
        "utils3d", "plyfile", "tqdm",
    )
    # Include the older architectures: T4 (7.5) and V100 (7.0) are what the
    # free tiers hand out, and gsplat compiles its CUDA kernels against this
    # list - omitting 7.5 makes it fail to run on exactly the GPUs we may end
    # up using. P100 (6.0) is included for Kaggle.
    .env({"TORCH_CUDA_ARCH_LIST": "6.0;7.0;7.5;8.0;8.6;8.9;9.0+PTX"})
)

# gsplat and MapAnything are layered separately so a failure in one does not
# invalidate the cached build of the other.
gsplat_image = gpu_image.pip_install("gsplat")

mapanything_image = gpu_image.pip_install(
    "git+https://github.com/facebookresearch/map-anything.git"
)

full_image = (
    gpu_image
    .pip_install("gsplat")
    .pip_install("git+https://github.com/facebookresearch/map-anything.git")
)

# Persisted across runs: the golden set goes up once, results come back down.
data_volume = modal.Volume.from_name("visit-it-data", create_if_missing=True)
DATA_DIR = "/data"

# Model weights cached so repeat runs do not re-download several GB.
model_cache = modal.Volume.from_name("visit-it-models", create_if_missing=True)
CACHE_DIR = "/cache"

# Cheapest first. The deploying account may not have every tier enabled -
# Modal gates GPUs behind a payment method - so this is ordered so a fallback
# is a one-word change rather than a redesign.
GPU_CHOICES = ("T4", "L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100")

app = modal.App(APP_NAME)
