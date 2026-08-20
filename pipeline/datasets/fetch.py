"""Lazy, cached dataset acquisition.

Call `ensure("rent3d")` from any pipeline stage that needs the data. The first
call downloads and extracts it; later calls are a no-op returning the path.

    from pipeline.datasets import ensure
    root = ensure("rent3d")          # -> Path, or raises with instructions

Cache location: $VISITIT_DATA_HOME, else ~/.cache/visit-it/datasets.

Licence gate: datasets whose ledger verdict is not `prod_ok` refuse to download
unless the caller acknowledges the restriction, either by passing
`accept_research_licence=True` or by setting VISITIT_ACCEPT_RESEARCH_LICENCE=1.
This is deliberate friction - it is what stops encumbered data drifting into a
training run. See docs/LICENSING.md.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests

from .registry import REGISTRY, DatasetSpec

log = logging.getLogger("datasets")

ACK_ENV = "VISITIT_ACCEPT_RESEARCH_LICENCE"


class LicenceRefused(RuntimeError):
    """Raised when a restricted dataset is requested without acknowledgement."""


class ManualAcquisitionRequired(RuntimeError):
    """Raised when a dataset cannot be fetched automatically."""


def data_home() -> Path:
    p = os.environ.get("VISITIT_DATA_HOME")
    root = Path(p) if p else Path.home() / ".cache" / "visit-it" / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=timeout,
                      headers={"User-Agent": "visit-it-dataset-fetcher/0.1"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done % (32 << 20) < (1 << 20):
                    log.info("  %.0f%% (%.0f/%.0f MB)", 100 * done / total,
                             done / 1e6, total / 1e6)
    tmp.rename(dest)
    return dest


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers()
                       if not (m.name.startswith("/") or ".." in Path(m.name).parts)]
            tf.extractall(dest, members=members)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for n in zf.namelist():
                if n.startswith("/") or ".." in Path(n).parts:
                    continue
                zf.extract(n, dest)
    else:
        shutil.copy2(archive, dest / archive.name)


def status(name: str) -> dict:
    spec = REGISTRY[name]
    root = data_home() / name
    present = root.exists() and any(root.iterdir()) if root.exists() else False
    return {
        "name": name,
        "present": present,
        "path": str(root),
        "licence": spec.licence,
        "verdict": spec.verdict,
        "acquisition": spec.acquisition,
        "urls_known": bool(spec.urls),
    }


def ensure(name: str, *, accept_research_licence: bool | None = None,
           force: bool = False) -> Path:
    """Return a local path to the dataset, downloading it if needed."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    spec: DatasetSpec = REGISTRY[name]
    root = data_home() / name

    if root.exists() and any(root.iterdir()) and not force:
        log.debug("%s already present at %s", name, root)
        return root

    if spec.needs_ack:
        ack = accept_research_licence
        if ack is None:
            ack = os.environ.get(ACK_ENV, "").strip() in ("1", "true", "yes")
        if not ack:
            raise LicenceRefused(
                f"'{name}' is marked {spec.verdict}: {spec.licence}.\n"
                f"It must not be used to train anything we ship (docs/LICENSING.md).\n"
                f"To download it for evaluation/research use, either set "
                f"{ACK_ENV}=1 or pass --accept-research-licence."
            )

    # A manually-supplied archive always wins: the user went and got it, and it
    # may be the only copy that exists (dead upstream links are common here).
    incoming = data_home() / "_incoming"
    candidates = sorted(incoming.glob(f"{name}.*")) if incoming.exists() else []
    candidates = [c for c in candidates if not c.name.endswith(".part")]
    if candidates:
        log.info("using manually-supplied archive %s", candidates[0])
        _extract(candidates[0], root)
        return root

    if spec.acquisition == "manual" or not spec.urls:
        msg = spec.manual_instructions or (
            f"No download URL is known for '{name}'.\n"
            f"Homepage: {spec.homepage}\n"
            f"Place the archive at {incoming / (name + '.zip')} and re-run.")
        raise ManualAcquisitionRequired(msg)

    cache = data_home() / "_archives"
    last_err: Exception | None = None
    for url in spec.urls:
        archive = cache / Path(url.split("?")[0]).name
        try:
            if not archive.exists():
                _download(url, archive)
            expected = spec.sha256.get(archive.name)
            if expected:
                got = _sha256(archive)
                if got != expected:
                    archive.unlink(missing_ok=True)
                    raise RuntimeError(f"checksum mismatch for {archive.name}: {got}")
            _extract(archive, root)
            log.info("%s ready at %s", name, root)
            return root
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
            log.warning("source failed (%s): %s", url, e)
    # Every mirror failed. If we know how a human can obtain it, say so rather
    # than dying with a stack trace - for these datasets a dead link is the
    # normal case, not an exceptional one.
    if spec.manual_instructions:
        raise ManualAcquisitionRequired(
            f"Automatic download of '{name}' failed ({last_err}).\n\n"
            + spec.manual_instructions)
    raise RuntimeError(f"all sources failed for '{name}': {last_err}")
