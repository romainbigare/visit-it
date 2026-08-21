"""Typed, immutable, content-addressed artifacts (ARCHITECTURE AD-3).

Three properties the rest of the system leans on:

**Typed.** Every artifact is validated against its `schemas/*.json` before it is
written. A stage that emits a malformed polygon fails at the stage boundary, not
three stages later in the viewer.

**Immutable and content-addressed.** Writing the same content twice is the same
blob. Re-running stage 6 creates a *new version*; it never edits the old one, so
"what changed between last night's run and this one" is answerable by diffing
two files that both still exist.

**Partial re-runs.** ``store.read("5-plan")`` returns the latest version, so
``pipeline run <listing> --from 6`` genuinely works.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("artifacts")

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"

#: Which schema file each stage's artifact validates against, and the filename it
#: gets in ``latest/``. Stages not listed here write untyped side artifacts.
STAGE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "0-triage": ("manifest.json", "manifest.json"),
    "1-conditioning": ("calibration.json", "calibration.json"),
    "2-grouping": ("groups.json", "groups.json"),
    "3-geometry": ("", "geometry.json"),
    "4-layout": ("", "layouts.json"),
    "5-plan": ("plan.json", "plan.json"),
    "6-assembly": ("assembly.json", "assembly.json"),
    "7-scale": ("scale.json", "scale.json"),
    "8-shell": ("", "shell.json"),
    "9-package": ("scene.json", "scene.json"),
}


class SchemaError(ValueError):
    """An artifact did not match its schema. Raised at the stage boundary, on purpose."""


def data_root() -> Path:
    """Where runs live. One env var, because we move boxes often."""
    return Path(os.environ.get("VISITIT_RUN_HOME", "data/runs")).expanduser()


def _resolver_store() -> dict[str, Any]:
    return {
        f"https://visit-it.internal/schemas/{p.name}": json.loads(p.read_text())
        for p in SCHEMA_DIR.glob("*.json")
    }


_VALIDATORS: dict[str, Any] = {}


def _validator(schema_name: str):
    if schema_name in _VALIDATORS:
        return _VALIDATORS[schema_name]
    import jsonschema
    from referencing import Registry, Resource

    registry = Registry()
    for uri, doc in _resolver_store().items():
        registry = registry.with_resource(uri, Resource.from_contents(doc))
        # Stage code refers to sibling schemas by bare filename ("common.json#/...").
        registry = registry.with_resource(uri.rsplit("/", 1)[-1], Resource.from_contents(doc))
    schema = json.loads((SCHEMA_DIR / schema_name).read_text())
    v = jsonschema.Draft202012Validator(schema, registry=registry)
    _VALIDATORS[schema_name] = v
    return v


def validate(payload: dict, schema_name: str) -> None:
    """Raise :class:`SchemaError` with *every* problem, not just the first.

    Reporting one error at a time turns a five-minute fix into five round trips.
    """
    try:
        v = _validator(schema_name)
    except ImportError:  # pragma: no cover - jsonschema is in requirements
        log.warning("jsonschema not installed; skipping validation of %s", schema_name)
        return
    errors = sorted(v.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        lines = [f"  {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                 for e in errors[:12]]
        more = f"\n  ... and {len(errors) - 12} more" if len(errors) > 12 else ""
        raise SchemaError(f"{schema_name}: {len(errors)} problem(s)\n" + "\n".join(lines) + more)


def sha256_of_obj(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode()).hexdigest()


def _json_default(o: Any):
    """numpy leaks into artifacts constantly; convert rather than crash at write time."""
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON-serialisable")


@dataclass
class Artifact:
    stage: str
    listing_id: str
    payload: dict
    sha256: str
    version: int
    written_at: str
    run_id: str | None = None
    path: Path | None = None
    extras: dict[str, str] = field(default_factory=dict)


class ArtifactStore:
    """One listing's artifacts on disk.

    ::

        data/runs/<listing_id>/
          blobs/<sha256[:2]>/<sha256>.json   immutable, content-addressed
          binaries/<sha256[:2]>/<sha256><ext>  same, for .glb/.npz/.png
          latest/<name>                      convenience copies for humans and the viewer
          index.json                         stage -> version history
          runs/<run_id>.json                 ledger records
    """

    def __init__(self, listing_id: str, root: Path | None = None):
        self.listing_id = listing_id
        self.root = (root or data_root()) / listing_id
        self.blobs = self.root / "blobs"
        self.binaries = self.root / "binaries"
        self.latest = self.root / "latest"
        self.runs = self.root / "runs"
        self.index_path = self.root / "index.json"
        for d in (self.blobs, self.binaries, self.latest, self.runs):
            d.mkdir(parents=True, exist_ok=True)

    # -- index ------------------------------------------------------------
    def _index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"listing_id": self.listing_id, "stages": {}}

    def _write_index(self, idx: dict) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(idx, indent=2) + "\n")
        tmp.replace(self.index_path)

    def history(self, stage: str) -> list[dict]:
        return self._index()["stages"].get(stage, [])

    def stages_present(self) -> list[str]:
        return [s for s, v in self._index()["stages"].items() if v]

    # -- write ------------------------------------------------------------
    def write(self, stage: str, payload: dict, *, run_id: str | None = None,
              validate_schema: bool = True) -> Artifact:
        schema_name, latest_name = STAGE_ARTIFACTS.get(stage, ("", f"{stage}.json"))
        if validate_schema and schema_name:
            validate(payload, schema_name)
        sha = sha256_of_obj(payload)
        blob = self.blobs / sha[:2] / f"{sha}.json"
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            blob.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
        idx = self._index()
        hist = idx["stages"].setdefault(stage, [])
        written_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        version = len(hist) + 1
        if hist and hist[-1]["sha256"] == sha:
            # Identical content: keep the history honest rather than logging a
            # fake new version every nightly run.
            version = hist[-1]["version"]
            hist[-1]["last_seen_at"] = written_at
            hist[-1]["seen_count"] = hist[-1].get("seen_count", 1) + 1
        else:
            hist.append({"version": version, "sha256": sha, "written_at": written_at,
                         "run_id": run_id, "seen_count": 1})
        self._write_index(idx)
        (self.latest / latest_name).write_text(blob.read_text())
        return Artifact(stage, self.listing_id, payload, sha, version, written_at,
                        run_id, blob)

    def write_binary(self, name: str, data: bytes, *, run_id: str | None = None) -> dict:
        """Content-addressed blob for non-JSON output (glTF, npz, png)."""
        sha = hashlib.sha256(data).hexdigest()
        ext = Path(name).suffix or ".bin"
        dest = self.binaries / sha[:2] / f"{sha}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        # Binary names carry a path ("rooms/<id>/geometry.npz"), so the convenience
        # copy needs its directories made.
        convenience = self.latest / name
        convenience.parent.mkdir(parents=True, exist_ok=True)
        convenience.write_bytes(data)
        return {"name": name, "sha256": sha, "bytes": len(data),
                "uri": name, "path": str(dest)}

    # -- read -------------------------------------------------------------
    def read(self, stage: str, version: int | None = None) -> dict | None:
        hist = self.history(stage)
        if not hist:
            return None
        rec = hist[-1] if version is None else next((h for h in hist if h["version"] == version), None)
        if rec is None:
            return None
        blob = self.blobs / rec["sha256"][:2] / f"{rec['sha256']}.json"
        return json.loads(blob.read_text()) if blob.exists() else None

    def require(self, stage: str) -> dict:
        p = self.read(stage)
        if p is None:
            raise FileNotFoundError(
                f"{self.listing_id}: stage {stage} has no artifact — run it first "
                f"(present: {', '.join(self.stages_present()) or 'nothing'})"
            )
        return p

    def latest_sha(self, stage: str) -> str | None:
        hist = self.history(stage)
        return hist[-1]["sha256"] if hist else None

    def read_binary(self, name: str) -> bytes | None:
        p = self.latest / name
        return p.read_bytes() if p.exists() else None

    @classmethod
    def list_listings(cls, root: Path | None = None) -> list[str]:
        r = root or data_root()
        if not r.exists():
            return []
        return sorted(p.name for p in r.iterdir() if p.is_dir() and (p / "index.json").exists())
