"""Check whether this box can run the pipeline, and say what is missing.

    make doctor

Written because "it doesn't work" is almost always one of six things, and finding
out which one takes longer than it should. Every failure line says what to run.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, BAD = "  ok ", " warn", " MISS"
rows: list[tuple[str, str, str, str]] = []


def check(name: str, ok: bool, detail: str, fix: str = "", warn_only: bool = False):
    rows.append((OK if ok else (WARN if warn_only else BAD), name, detail,
                 "" if ok else fix))
    return ok


def main() -> int:
    check("python >= 3.11", sys.version_info >= (3, 11),
          ".".join(map(str, sys.version_info[:3])), "install python 3.11+")

    for mod, pkg in (("numpy", "numpy"), ("scipy", "scipy"), ("cv2", "opencv-python-headless"),
                     ("PIL", "pillow"), ("skimage", "scikit-image"), ("shapely", "shapely"),
                     ("jsonschema", "jsonschema"), ("referencing", "referencing"),
                     ("matplotlib", "matplotlib"), ("pytesseract", "pytesseract")):
        try:
            importlib.import_module(mod)
            check(f"python: {mod}", True, "importable")
        except ImportError:
            check(f"python: {mod}", False, "not importable",
                  f"pip install {pkg}   (or: make setup)")

    # Heavy, and only stage 0 and stage 3 need them.
    for mod, what in (("torch", "stage 3 geometry"), ("transformers", "stage 0 triage")):
        try:
            m = importlib.import_module(mod)
            check(f"python: {mod}", True, f"{getattr(m, '__version__', '?')} — {what}")
        except ImportError:
            check(f"python: {mod}", False, f"needed for {what}",
                  "pip install torch torchvision "
                  "--index-url https://download.pytorch.org/whl/cpu", warn_only=True)

    tess = shutil.which("tesseract")
    check("tesseract binary", bool(tess), tess or "not on PATH",
          "apt-get install tesseract-ocr   (mac: brew install tesseract)")

    moge = ROOT / "vendor" / "moge" / "moge"
    check("MoGe-2 source vendored", moge.exists(), str(moge.parent),
          "make vendor")

    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf"))
    weights = list(hf.rglob("*moge*")) if hf.exists() else []
    check("MoGe-2 weights cached", bool(weights),
          f"{hf}" if weights else "not downloaded (~1.3 GB, fetched on first run)",
          "first `pipeline run` downloads them", warn_only=True)

    golden = ROOT / "data" / "golden" / "golden_set.json"
    if check("golden set manifest", golden.exists(), str(golden), "git checkout data/golden"):
        media = ROOT / "data" / "golden" / "media"
        n = len(list(media.glob("*"))) if media.exists() else 0
        check("golden set images", n >= 30, f"{n} listing folders (~87 MB)",
              "python -m pipeline.ingest.fetch_media --set data/golden/golden_set.json")

    split = ROOT / "data" / "golden" / "holdout_split.json"
    if check("frozen holdout split", split.exists(), str(split), "make holdout"):
        try:
            sys.path.insert(0, str(ROOT))
            from eval.holdout import load
            p = load(split)
            check("holdout seal", True, f"{p['seal'][:12]} · {p['n_holdout']}/{p['n_dev']}")
        except Exception as e:  # noqa: BLE001
            check("holdout seal", False, str(e), "the split file has been edited")

    node = shutil.which("node")
    check("node (viewer only)", bool(node),
          subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
          if node else "not on PATH", "install node 20+ — only needed for `make viewer`",
          warn_only=True)

    width = max(len(r[1]) for r in rows) + 2
    print()
    for status, name, detail, fix in rows:
        print(f"[{status}] {name:<{width}} {detail}")
        if fix:
            print(f"{'':>8}{'':<{width}} -> {fix}")
    missing = [r for r in rows if r[0] == BAD]
    print()
    if missing:
        print(f"{len(missing)} thing(s) missing. Fix the '->' lines above, then re-run.")
        return 1
    warns = [r for r in rows if r[0] == WARN]
    print("ready to run: python -m pipeline run 87977241"
          + (f"   ({len(warns)} optional item(s) absent)" if warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
