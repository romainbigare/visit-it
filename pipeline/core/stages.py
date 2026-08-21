"""The stage contract, and the registry that makes the DAG discoverable.

A stage is a small object that declares what it needs, what it produces, and how
long it is allowed to take. It does *not* know how it will be scheduled, retried
or scored — which is what lets stream B and stream C develop against each other's
synthetic counterparts and only meet at stage 6.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("stages")

#: Execution order. Stage names are stable identifiers used in CLIs, the ledger
#: and the contact sheet, so they carry their number.
STAGE_ORDER: list[str] = [
    "0-triage",
    "1-conditioning",
    "2-grouping",
    "3-geometry",
    "4-layout",
    "5-plan",
    "6-assembly",
    "7-scale",
    "8-shell",
    "9-package",
]

#: What each stage reads. Stage 6 is the only join: it is where the photo channel
#: (B) and the plan channel (C) finally meet, exactly as the dependency graph in
#: ROADMAP §1 says.
STAGE_DEPS: dict[str, tuple[str, ...]] = {
    "0-triage": (),
    "1-conditioning": ("0-triage",),
    "2-grouping": ("0-triage",),
    "3-geometry": ("1-conditioning", "2-grouping"),
    "4-layout": ("3-geometry",),
    "5-plan": ("0-triage",),
    "6-assembly": ("4-layout", "5-plan"),
    "7-scale": ("4-layout", "5-plan", "6-assembly"),
    "8-shell": ("5-plan", "6-assembly", "7-scale"),
    "9-package": ("8-shell",),
}


def normalise_stage(name: str) -> str:
    """Accept '4', '4-layout', 'layout' — all the ways a person actually types it."""
    if name in STAGE_ORDER:
        return name
    for s in STAGE_ORDER:
        num, _, word = s.partition("-")
        if name == num or name == word:
            return s
    raise KeyError(f"unknown stage {name!r}; have {', '.join(STAGE_ORDER)}")


@dataclass
class StageContext:
    """Everything a stage is allowed to see."""
    listing_id: str
    listing: dict                      # the golden-set / submitted listing record
    store: Any                         # ArtifactStore
    profile: Any                       # Profile
    golden_root: Any = None            # Path — where media/ lives
    options: dict = field(default_factory=dict)

    def read(self, stage: str) -> dict | None:
        return self.store.read(normalise_stage(stage))

    def require(self, stage: str) -> dict:
        return self.store.require(normalise_stage(stage))

    def media_path(self, rel: str):
        from pathlib import Path
        return Path(self.golden_root) / rel if self.golden_root else Path(rel)


@dataclass
class StageResult:
    payload: dict
    binaries: dict[str, bytes] = field(default_factory=dict)
    engine: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class Stage:
    name: str
    run: Callable[[StageContext], StageResult]
    description: str = ""
    optional: bool = False
    #: A stage marked required-for gates the rest of the DAG: if it cannot run,
    #: downstream stages are skipped rather than fed garbage.
    deps: tuple[str, ...] = ()


_REGISTRY: dict[str, Stage] = {}


def register_stage(name: str, *, description: str = "", optional: bool = False):
    """Decorator. ``@register_stage("5-plan")`` over a ``(ctx) -> StageResult``."""
    name = normalise_stage(name)

    def deco(fn: Callable[[StageContext], StageResult]) -> Callable[[StageContext], StageResult]:
        _REGISTRY[name] = Stage(name=name, run=fn, description=description or (fn.__doc__ or "").strip().split("\n")[0],
                                optional=optional, deps=STAGE_DEPS.get(name, ()))
        return fn
    return deco


def get_stage(name: str) -> Stage:
    name = normalise_stage(name)
    if name not in _REGISTRY:
        load_all_stages()
    if name not in _REGISTRY:
        raise KeyError(f"stage {name!r} is declared in STAGE_ORDER but no implementation is registered")
    return _REGISTRY[name]


def registered() -> dict[str, Stage]:
    load_all_stages()
    return dict(_REGISTRY)


_LOADED = False


def load_all_stages() -> None:
    """Import the stage packages so their decorators fire.

    Kept lazy and forgiving: a stage whose optional dependency is missing (torch,
    tesseract) should not stop the other nine from being runnable.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import importlib
    for mod in ("pipeline.triage", "pipeline.conditioning", "pipeline.grouping",
                "pipeline.room_geometry", "pipeline.layout", "pipeline.floorplan",
                "pipeline.assembly", "pipeline.scale", "pipeline.shell",
                "pipeline.packaging"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            log.debug("stage package %s not loadable: %s", mod, e)
