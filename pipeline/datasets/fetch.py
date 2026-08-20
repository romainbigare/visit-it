"""Lazy, cached, resumable dataset acquisition.

    from pipeline.datasets import ensure
    root = ensure("swiss_dwellings")     # downloads once, then returns instantly

Design for replicability across environments (we move boxes often):

* Cache root is $VISITIT_DATA_HOME (default ~/.cache/visit-it/datasets), so a
  new machine is configured with one env var.
* Downloads resume: interrupted 2 GB fetches continue via HTTP Range rather
  than starting again.
* Every completed file is recorded in `datasets.lock.json` (committed to the
  repo) with its URL, size and sha256. `verify` re-checks local files against
  the lock, so "did this box get the same bytes?" is a one-command question.
* Files with the same filename are treated as MIRRORS of each other and tried
  in order.
* Licences are recorded but do not block. This project is internal R&D; the
  metadata is kept because it becomes load-bearing the day anything ships.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests

from .registry import REGISTRY, DatasetFile, DatasetSpec

log = logging.getLogger("datasets")

LOCKFILE = Path(__file__).resolve().parents[2] / "datasets.lock.json"
CHUNK = 1 << 20
UA = "visit-it-dataset-fetcher/0.2"


class ManualAcquisitionRequired(RuntimeError):
    """The dataset cannot be fetched automatically; instructions are attached."""


def data_home() -> Path:
    p = os.environ.get("VISITIT_DATA_HOME")
    root = Path(p).expanduser() if p else Path.home() / ".cache" / "visit-it" / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------- lockfile
def _load_lock() -> dict:
    if LOCKFILE.exists():
        try:
            return json.loads(LOCKFILE.read_text())
        except json.JSONDecodeError:
            log.warning("lockfile corrupt, starting fresh")
    return {"version": 1, "files": {}}


def _save_lock(lock: dict) -> None:
    LOCKFILE.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- download
def _download_resumable(url: str, dest: Path, expect_size: int | None = None) -> Path:
    """GET `url` to `dest`, resuming from a partial file if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": UA}
    if have:
        headers["Range"] = f"bytes={have}-"
        log.info("resuming %s at %.0f MB", dest.name, have / 1e6)

    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        if have and r.status_code == 200:
            have = 0  # server ignored Range; restart
            part.unlink(missing_ok=True)
        elif have and r.status_code == 416:
            part.rename(dest)  # already complete
            return dest
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0) + have
        mode = "ab" if have else "wb"
        done = have
        t0 = time.time()
        last_log = 0.0
        with open(part, mode) as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if time.time() - last_log > 20:
                    last_log = time.time()
                    rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                    pct = f"{100*done/total:.0f}%" if total else "?"
                    log.info("  %s %s (%.0f MB, %.1f MB/s)", dest.name, pct, done / 1e6, rate)
    if expect_size and abs(part.stat().st_size - expect_size) > max(1024, expect_size * 0.02):
        log.warning("%s: size %.0f MB differs from expected %.0f MB",
                    dest.name, part.stat().st_size / 1e6, expect_size / 1e6)
    part.rename(dest)
    return dest


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    n = archive.name.lower()
    if n.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2")):
        with tarfile.open(archive) as tf:
            safe = [m for m in tf.getmembers()
                    if not (m.name.startswith("/") or ".." in Path(m.name).parts)]
            tf.extractall(dest, members=safe)
    elif n.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for m in zf.namelist():
                if m.startswith("/") or ".." in Path(m).parts:
                    continue
                zf.extract(m, dest)
    else:
        target = dest / archive.name
        if not target.exists():
            shutil.copy2(archive, target)


# --------------------------------------------------------------- public API
def status(name: str) -> dict:
    spec = REGISTRY[name]
    root = data_home() / name
    present = root.exists() and any(root.iterdir())
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if present else 0
    return {"name": name, "present": present, "path": str(root),
            "local_mb": round(size / 1e6, 1), "licence": spec.licence,
            "commercial_ok": spec.commercial_ok, "acquisition": spec.acquisition,
            "essential_mb": round(spec.total_mb(), 1)}


def ensure(name: str, *, full: bool = False, force: bool = False,
           update_lock: bool = True) -> Path:
    """Return a local path to the dataset, fetching it if needed."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    spec: DatasetSpec = REGISTRY[name]
    root = data_home() / name
    if root.exists() and any(root.iterdir()) and not force:
        return root

    # A manually-supplied archive always wins - it may be the only copy.
    incoming = data_home() / "_incoming"
    manual = [p for p in sorted(incoming.glob(f"{name}.*"))
              if not p.name.endswith(".part")] if incoming.exists() else []
    if manual:
        log.info("using manually-supplied %s", manual[0])
        _extract(manual[0], root)
        return root

    wanted = spec.files if full else spec.essential_files
    if spec.acquisition == "manual" or not wanted:
        raise ManualAcquisitionRequired(
            spec.manual_instructions
            or f"No automatic source for '{name}'. Homepage: {spec.homepage}")

    # Group by filename: same filename == mirrors of one another.
    groups: dict[str, list[DatasetFile]] = {}
    for f in wanted:
        groups.setdefault(f.filename, []).append(f)

    lock = _load_lock()
    cache = data_home() / "_archives" / name
    errors: list[str] = []
    fetched: list[Path] = []
    for filename, mirrors in groups.items():
        archive = cache / filename
        if not archive.exists():
            for m in mirrors:
                try:
                    _download_resumable(m.url, archive, m.size_bytes)
                    break
                except Exception as e:  # noqa: BLE001 - try the next mirror
                    errors.append(f"{m.url}: {type(e).__name__}: {e}")
                    log.warning("mirror failed %s (%s)", m.url, e)
            else:
                continue
        digest = sha256_of(archive)
        declared = next((m.sha256 for m in mirrors if m.sha256), None)
        if declared and digest != declared:
            archive.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {filename}: {digest} != {declared}")
        if update_lock:
            lock["files"][f"{name}/{filename}"] = {
                "url": mirrors[0].url, "sha256": digest,
                "size_bytes": archive.stat().st_size,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        if any(m.extract for m in mirrors):
            _extract(archive, root)
        else:
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, root / filename)
        fetched.append(archive)

    if not fetched:
        msg = f"all sources failed for '{name}':\n  " + "\n  ".join(errors)
        if spec.manual_instructions:
            raise ManualAcquisitionRequired(msg + "\n\n" + spec.manual_instructions)
        raise RuntimeError(msg)

    if update_lock:
        _save_lock(lock)
    log.info("%s ready at %s", name, root)
    return root


def ensure_all(tags: tuple[str, ...] = (), *, full: bool = False,
               skip_manual: bool = True) -> dict[str, str]:
    """Fetch every dataset (optionally filtered by tag). Returns name -> outcome."""
    out: dict[str, str] = {}
    for name, spec in REGISTRY.items():
        if tags and not set(tags) & set(spec.tags):
            continue
        if skip_manual and spec.acquisition == "manual":
            out[name] = "skipped (manual)"
            continue
        try:
            ensure(name, full=full)
            out[name] = "ok"
        except ManualAcquisitionRequired:
            out[name] = "manual step required"
        except Exception as e:  # noqa: BLE001
            out[name] = f"failed: {type(e).__name__}: {e}"
    return out


def verify(name: str | None = None) -> dict[str, str]:
    """Re-check cached archives against datasets.lock.json."""
    lock = _load_lock()
    results: dict[str, str] = {}
    for key, rec in lock.get("files", {}).items():
        ds, filename = key.split("/", 1)
        if name and ds != name:
            continue
        path = data_home() / "_archives" / ds / filename
        if not path.exists():
            results[key] = "absent"
        elif sha256_of(path) == rec["sha256"]:
            results[key] = "ok"
        else:
            results[key] = "CHECKSUM MISMATCH"
    return results
