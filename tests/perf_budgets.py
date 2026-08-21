"""Per-stage latency budgets, asserted (ROADMAP §6 rule 7, AD-17).

Every stage ships its fast binding first and CI asserts each stage stays inside
its profile's budget. Two honest caveats, both stated in the output rather than
hidden:

* **The reference machine matters.** A GitHub runner is not the L40S the budgets
  were written for, so this checks the CPU-cheap stages (4, 6, 7, 8, 9) against
  the *standard* profile and reports stage 3 without failing on it — stage 3 is
  GPU work and 40× slower on CPU, which Phase 0 measured rather than guessed.
* **A budget overrun is a signal, not a crash.** It fails the build so that going
  over needs a deliberate waiver in the PR description, which is the point.

    python -m tests.perf_budgets
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from pipeline.core import geom, get_profile
from pipeline.layout.stage import layout_from_points
from pipeline.assembly.stage import assemble
from pipeline.scale.stage import build_scale
from pipeline.shell.stage import build as build_shell
from pipeline.packaging.stage import build_scene
from pipeline.room_geometry.engines import get_engine

#: Stages whose cost is CPU-bound and therefore comparable on any runner.
CPU_STAGES = ("4-layout", "6-assembly", "7-scale", "8-shell", "9-package")


def _fixture(n_rooms: int = 5):
    """A whole listing's worth of artifacts, from the synthetic engine."""
    layouts = {"schema": "layouts/v1", "listing_id": "perf", "engine": "synthetic",
               "rooms": [], "summary": {}, "confidence": 0.7, "qa_flags": []}
    for i in range(n_rooms):
        rec = get_engine("synthetic").reconstruct([Path(f"perf{i}.jpg")])
        lay = layout_from_points(rec.points, rec.confidence, rec.up_prior,
                                 room_id=f"r{i}", room_label="bedroom",
                                 engine="synthetic")
        lay["listing_id"] = "perf"
        layouts["rooms"].append(lay)
    plan = {"schema": "plan/v1", "listing_id": "perf", "px_per_metre": 100.0,
            "scale_source": "printed_dimensions", "scale_candidates": [],
            "rooms": [{"room_id": f"p{i}", "label": "bedroom",
                       "polygon_px": geom.rectangle(400, 300, i * 500, 0),
                       "polygon_m": geom.rectangle(4.0, 3.0, i * 5.0, 0),
                       "area_px": 120000.0, "area_m2": 12.0,
                       "ocr_dims_m": [4.0, 3.0], "ocr_area_m2": None,
                       "centroid_px": [i * 500, 0], "centroid_m": [i * 5.0, 0.0],
                       "aperture_ids": [], "confidence": 0.8, "qa_flags": []}
                      for i in range(n_rooms)],
            "adjacency": [], "apertures": [],
            "ocr": {"total_area_m2": 60.0}, "totals": {"plan_area_m2": 60.0},
            "confidence": 0.8, "qa_flags": []}
    manifest = {"schema": "manifest/v1", "listing_id": "perf", "images": [], "plans": [],
                "listing": {"advertised_area_m2": 60.0}, "confidence": 0.9,
                "qa_flags": []}
    return layouts, plan, manifest


def main() -> int:
    prof = get_profile("standard")
    layouts, plan, manifest = _fixture()
    timings: dict[str, float] = {}

    t = time.perf_counter()
    rec = get_engine("synthetic").reconstruct([Path("perf0.jpg")])
    layout_from_points(rec.points, rec.confidence, rec.up_prior, room_id="r0")
    timings["4-layout"] = time.perf_counter() - t

    t = time.perf_counter()
    assembly = assemble(layouts, plan)
    timings["6-assembly"] = time.perf_counter() - t

    t = time.perf_counter()
    scale = build_scale(layouts, plan, manifest, assembly)
    timings["7-scale"] = time.perf_counter() - t

    t = time.perf_counter()
    shell, glb = build_shell(layouts, plan, assembly, scale)
    timings["8-shell"] = time.perf_counter() - t

    t = time.perf_counter()
    build_scene(shell, plan, assembly, scale, manifest, {"display_address": "perf"})
    timings["9-package"] = time.perf_counter() - t

    failures = []
    print(f"{'stage':<14}{'seconds':>9}{'budget':>9}  verdict")
    for s in CPU_STAGES:
        b = prof.budget(s) or float("inf")
        v = timings[s]
        ok = v <= b
        print(f"{s:<14}{v:>9.3f}{b:>9.2f}  {'ok' if ok else 'OVER BUDGET'}")
        if not ok:
            failures.append((s, v, b))
    print(f"\nshell: {len(glb)} bytes of glTF, {shell['glb']['triangles']} triangles "
          f"({'within' if shell['glb']['within_budget'] else 'OVER'} the 1 MB budget)")
    if not shell["glb"]["within_budget"]:
        failures.append(("8-shell payload", len(glb), 1_048_576))
    print("\nstage 3 is not asserted here: it is GPU work, and Phase 0 measured "
          "MoGe-2 at 0.364 s/image on a T4 against 14.4 s on 4 CPU cores. A CI "
          "runner would only ever tell us about the runner.")
    if failures:
        print(f"\n{len(failures)} budget breach(es). A PR that does this needs an "
              f"explicit waiver in its description (ROADMAP §6 rule 7).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
