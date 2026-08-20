"""Vendor MoGe-2 source into a local directory so it can run anywhere.

MoGe is not on PyPI and installs from GitHub, which repo-scoped sandboxes and
locked-down CI cannot reach. raw.githubusercontent.com usually *is* reachable,
so this pulls the ~30 source files directly and drops in a small shim for
`utils3d_moge` (a git-pinned dependency that is likewise unreachable and of
which MoGe uses only a handful of geometry helpers).

    python tools/vendor_moge.py --dest vendor/moge
    PYTHONPATH=vendor/moge python -m eval.models.moge_geometry
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/microsoft/MoGe/main"

FILES = [
    "moge/__init__.py",
    "moge/model/__init__.py",
    "moge/model/v2.py",
    "moge/model/utils.py",
    "moge/model/modules/__init__.py",
    "moge/model/modules/conv_stack.py",
    "moge/model/modules/dinov2_encoder.py",
    "moge/model/modules/mlp.py",
    "moge/utils/__init__.py",
    "moge/utils/tools.py",
    "moge/utils/geometry_torch.py",
    "moge/utils/geometry_numpy.py",
]
DINOV2 = [
    "__init__.py", "hub/__init__.py", "hub/backbones.py", "hub/utils.py",
    "layers/__init__.py", "layers/attention.py", "layers/block.py",
    "layers/dino_head.py", "layers/drop_path.py", "layers/layer_scale.py",
    "layers/mlp.py", "layers/patch_embed.py", "layers/swiglu_ffn.py",
    "models/__init__.py", "models/vision_transformer.py",
]
FILES += [f"moge/model/modules/dinov2/{f}" for f in DINOV2]


def fetch(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for rel in FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{BASE}/{rel}", timeout=40)
        if r.status_code == 200:
            target.write_bytes(r.content)
        else:
            failed.append(f"{rel} (HTTP {r.status_code})")
        print(f"  {'ok  ' if r.status_code == 200 else 'FAIL'} {rel}")
    shim_src = Path(__file__).resolve().parents[1] / "vendor" / "utils3d_moge_shim.py"
    shim_dir = dest / "utils3d_moge"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shim_src, shim_dir / "__init__.py")
    print(f"  ok   utils3d_moge/__init__.py (shim from {shim_src.name})")
    if failed:
        print("\nFAILED:", *failed, sep="\n  ")
    return len(failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("vendor/moge"))
    a = ap.parse_args()
    n = fetch(a.dest)
    if n:
        print(f"\n{n} file(s) failed - MoGe will not import", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nvendored MoGe into {a.dest}")
    print(f"use with: PYTHONPATH={a.dest} python -m eval.models.moge_geometry")
