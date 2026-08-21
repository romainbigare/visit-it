"""Put a scene where the viewer can load it, and make the fixture the viewer needs.

Two jobs:

``fixture``
    Write a hand-authored ``scene.json`` + ``shell.glb`` into ``viewer/public/``.
    This is the roadmap's decoupling contract made real (§1): stream E works
    against the *contract*, not the pipeline, so the viewer is developable and
    testable before — and independently of — any listing running successfully.

``export``
    Copy a real run's scene and shell into ``viewer/public/scenes/<listing>/`` and
    refresh the index the picker reads.

    python -m tools.export_scene fixture
    python -m tools.export_scene export 87977241
    python -m tools.export_scene export --all
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import geom                      # noqa: E402
from pipeline.core.artifacts import ArtifactStore   # noqa: E402
from pipeline.packaging.stage import (DISPLAY_NAMES, EYE_HEIGHT_M,   # noqa: E402
                                      PROVENANCE_LEGEND)
from pipeline.shell.mesh import (Mesh, PROVENANCE_RGBA,  # noqa: E402
                                 build_room, to_glb)

VIEWER_PUBLIC = Path("viewer/public")


def make_fixture() -> tuple[dict, bytes]:
    """A small, deliberately imperfect flat.

    It carries one *inferred* room on purpose. A fixture where everything is
    perfect lets the honesty rendering rot unnoticed, and the honesty rendering is
    the thing that makes a bad reconstruction visible rather than plausible.
    """
    spec = [
        ("living_room", geom.rectangle(4.6, 3.8, 2.3, 1.9), 2.62, "reconstructed", 0.81),
        ("kitchen", geom.rectangle(3.0, 2.6, 6.2, 1.3), 2.62, "reconstructed", 0.74),
        ("hallway", [[4.6, 2.6], [5.4, 2.6], [5.4, 6.0], [4.6, 6.0]], 2.55, "inferred", 0.33),
        ("bedroom", geom.rectangle(3.9, 3.1, 2.35, 5.35), 2.58, "reconstructed", 0.77),
        ("bathroom", geom.rectangle(2.2, 2.1, 6.5, 4.95), 2.45, "reconstructed", 0.62),
    ]
    mesh = Mesh()
    rooms, waypoints = [], []
    for label, poly, h, prov, conf in spec:
        rid = label
        build_room(mesh, poly, 0.0, h, rid, prov)
        centre = geom.representative_point(poly)
        rooms.append({
            "room_id": rid, "label": label,
            "display_name": DISPLAY_NAMES.get(label, "Room"),
            "polygon_m": [[round(x, 3), round(y, 3)] for x, y in poly],
            "centroid_m": [round(centre[0], 3), round(centre[1], 3)],
            "height_m": h, "area_m2": round(geom.area(poly), 2),
            "provenance": prov, "confidence": conf, "photo_ids": [], "splats": None,
        })
        far = max(poly, key=lambda v: math.dist(v, centre))
        waypoints.append({
            "waypoint_id": f"w_{rid}", "room_id": rid,
            "position_m": [round(centre[0], 3), round(centre[1], 3), EYE_HEIGHT_M],
            "look_deg": round(math.degrees(math.atan2(far[1] - centre[1],
                                                      far[0] - centre[0])), 2),
            "kind": "room_centre", "label": DISPLAY_NAMES.get(label, "Room"),
        })
    edges = [["w_living_room", "w_hallway"], ["w_hallway", "w_bedroom"],
             ["w_hallway", "w_bathroom"], ["w_living_room", "w_kitchen"]]
    polys = [r["polygon_m"] for r in rooms]
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    scene = {
        "schema": "scene/v1", "listing_id": "fixture-0001",
        "generated_at": "2026-08-21T00:00:00+00:00", "tier": "A", "units": "metres",
        "profile": "standard", "advertised_area_m2": 58.0,
        "address": "Fixture flat (hand-authored, not a real listing)",
        "shell": {"uri": "shell.glb", "bytes": 0, "triangles": mesh.n_triangles},
        "rooms": rooms, "waypoints": waypoints,
        "waypoint_edges": [sorted(e) for e in edges],
        "minimap": {"bounds_m": [[min(xs), min(ys)], [max(xs), max(ys)]],
                    "footprint_m": geom.union_polygon(polys)},
        "provenance_legend": PROVENANCE_LEGEND,
        "provenance_colours": {k: list(v) for k, v in PROVENANCE_RGBA.items()},
        "confidence": 0.66,
        "qa_flags": ["fixture_not_a_real_listing", "hallway_inferred"],
    }
    glb = to_glb(mesh)
    scene["shell"]["bytes"] = len(glb)
    return scene, glb


def write_fixture(public: Path = VIEWER_PUBLIC) -> Path:
    d = public / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    scene, glb = make_fixture()
    (d / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    (d / "shell.glb").write_bytes(glb)
    return d


def export(listing_id: str, store_root: Path | None = None,
           public: Path = VIEWER_PUBLIC) -> Path | None:
    store = ArtifactStore(listing_id, store_root)
    scene = store.read("9-package")
    if not scene:
        return None
    dest = public / "scenes" / listing_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    glb = store.read_binary("shell.glb")
    if glb:
        (dest / "shell.glb").write_bytes(glb)
    return dest


def write_index(public: Path = VIEWER_PUBLIC) -> Path:
    root = public / "scenes"
    entries = []
    for d in sorted(root.glob("*/scene.json")) if root.exists() else []:
        s = json.loads(d.read_text())
        entries.append({"listing_id": s["listing_id"], "address": s.get("address"),
                        "rooms": len(s.get("rooms", [])), "tier": s.get("tier"),
                        "confidence": s.get("confidence"),
                        "qa_flags": len(s.get("qa_flags", [])),
                        "url": f"./scenes/{s['listing_id']}/scene.json"})
    root.mkdir(parents=True, exist_ok=True)
    p = root / "index.json"
    p.write_text(json.dumps({"scenes": entries}, indent=2) + "\n")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fixture", "export"])
    ap.add_argument("listing_id", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--public", type=Path, default=VIEWER_PUBLIC)
    a = ap.parse_args(argv)
    if a.cmd == "fixture":
        print(f"fixture written to {write_fixture(a.public)}")
        return 0
    ids = a.listing_id
    if a.all:
        ids = ArtifactStore.list_listings(a.store)
    ok = 0
    for lid in ids:
        d = export(lid, a.store, a.public)
        if d:
            ok += 1
        else:
            print(f"{lid}: no scene.json — has stage 9 run?")
    write_index(a.public)
    print(f"exported {ok}/{len(ids)} scene(s) to {a.public / 'scenes'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
