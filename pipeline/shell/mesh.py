"""Extrude assembled room polygons into a glTF shell.

Phase 1's deliverable is "deliberately unglamorous and right": floors, ceilings
and walls, flat-shaded, with door and window apertures cut out, and **a provenance
tag on every face** (AD-15). The provenance tag is not decoration — Phase 2 hangs
splats off these faces, and when a room looks wrong the first question is always
"is that surface real?".

Budget: ARCHITECTURE §6 caps the shell at 1 MB. A flat is a few hundred triangles,
so the cap is generous; the CI check exists because it is the kind of budget that
gets eaten silently.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field

import numpy as np

from ..core import geom

log = logging.getLogger("shell.mesh")

#: Provenance classes, in the order the viewer's legend lists them.
PROVENANCE = ("photographed", "reconstructed", "inferred", "generated")

#: Flat, honest colours. Inferred surfaces are desaturated on purpose
#: (ARCHITECTURE §10): a viewer that renders invented geometry identically to
#: measured geometry is one we cannot debug.
PROVENANCE_RGBA = {
    "photographed": (0.82, 0.80, 0.76, 1.0),
    "reconstructed": (0.76, 0.78, 0.80, 1.0),
    "inferred": (0.62, 0.62, 0.64, 1.0),
    "generated": (0.58, 0.54, 0.62, 1.0),
}


@dataclass
class Mesh:
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    #: Parallel to indices/3 — which provenance class each triangle belongs to.
    face_provenance: list[str] = field(default_factory=list)
    face_room: list[str] = field(default_factory=list)

    @property
    def n_triangles(self) -> int:
        return len(self.indices) // 3

    def add_quad(self, a, b, c, d, provenance: str, room_id: str) -> None:
        base = len(self.positions)
        n = _normal(a, b, c)
        self.positions += [a, b, c, d]
        self.normals += [n, n, n, n]
        self.indices += [base, base + 1, base + 2, base, base + 2, base + 3]
        self.face_provenance += [provenance, provenance]
        self.face_room += [room_id, room_id]

    def add_polygon(self, poly2, z: float, provenance: str, room_id: str,
                    flip: bool = False) -> None:
        """Fan-triangulate a horizontal polygon at height ``z``."""
        pts = geom.ensure_ccw(poly2)
        tris = _fan(pts)
        base = len(self.positions)
        n = (0.0, 0.0, -1.0) if flip else (0.0, 0.0, 1.0)
        for x, y in pts:
            self.positions.append((float(x), float(y), float(z)))
            self.normals.append(n)
        for t in tris:
            tri = (t[0], t[2], t[1]) if flip else t
            self.indices += [base + tri[0], base + tri[1], base + tri[2]]
            self.face_provenance.append(provenance)
            self.face_room.append(room_id)


def _normal(a, b, c):
    u = np.subtract(b, a)
    v = np.subtract(c, a)
    n = np.cross(u, v)
    L = float(np.linalg.norm(n))
    return tuple((n / L).tolist()) if L > 1e-9 else (0.0, 0.0, 1.0)


def _fan(pts) -> list[tuple[int, int, int]]:
    """Ear clipping. A fan alone is wrong for the L-shaped rooms real flats have."""
    n = len(pts)
    if n < 3:
        return []
    idx = list(range(n))
    out: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:
                continue
            tri = [a, b, c]
            if any(geom.point_in_polygon(pts[j], tri)
                   for j in idx if j not in (i0, i1, i2)):
                continue
            out.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        out.append((idx[0], idx[1], idx[2]))
    return out


def build_room(mesh: Mesh, poly2, floor_z: float, height: float, room_id: str,
               provenance: str, apertures: list[dict] | None = None) -> None:
    """Floor, ceiling and walls for one room, with apertures cut from the walls."""
    top = floor_z + height
    mesh.add_polygon(poly2, floor_z, provenance, room_id)
    mesh.add_polygon(poly2, top, provenance, room_id, flip=True)
    pts = geom.ensure_ccw(poly2)
    n = len(pts)
    for i in range(n):
        p = pts[i]
        q = pts[(i + 1) % n]
        cuts = _cuts_on_edge(p, q, apertures or [])
        _wall_with_cuts(mesh, p, q, floor_z, top, cuts, provenance, room_id)


def _cuts_on_edge(p, q, apertures: list[dict]) -> list[tuple[float, float, float, float]]:
    """Apertures that lie on this wall, as ``(t0, t1, z0, z1)`` along the edge."""
    px, py = p
    qx, qy = q
    dx, dy = qx - px, qy - py
    L = float(np.hypot(dx, dy))
    if L < 1e-6:
        return []
    out = []
    for ap in apertures:
        pos = ap.get("position_m") or ap.get("centre_m")
        if not pos or len(pos) < 2:
            continue
        t = ((pos[0] - px) * dx + (pos[1] - py) * dy) / (L * L)
        perp = abs((pos[0] - px) * (-dy / L) + (pos[1] - py) * (dx / L))
        if not (0.0 <= t <= 1.0) or perp > 0.45:
            continue
        w = float(ap.get("width_m") or 0.85)
        half = (w / 2.0) / L
        z0 = 0.0 if ap.get("type") in ("door", "opening") else 0.9
        z1 = float(ap.get("height_m") or (2.04 if ap.get("type") != "window" else 2.1))
        out.append((max(0.0, t - half), min(1.0, t + half), z0, z1))
    return sorted(out)


def _wall_with_cuts(mesh: Mesh, p, q, z0: float, z1: float,
                    cuts: list[tuple[float, float, float, float]],
                    provenance: str, room_id: str) -> None:
    """One wall panel, split around each aperture: below, above, and either side."""
    def at(t: float, z: float):
        return (float(p[0] + (q[0] - p[0]) * t), float(p[1] + (q[1] - p[1]) * t), float(z))

    if not cuts:
        mesh.add_quad(at(0, z0), at(1, z0), at(1, z1), at(0, z1), provenance, room_id)
        return
    cursor = 0.0
    for t0, t1, cz0, cz1 in cuts:
        t0, t1 = max(cursor, t0), max(cursor, t1)
        if t0 > cursor:
            mesh.add_quad(at(cursor, z0), at(t0, z0), at(t0, z1), at(cursor, z1),
                          provenance, room_id)
        lo = z0 + cz0
        hi = min(z1, z0 + cz1)
        if lo > z0:
            mesh.add_quad(at(t0, z0), at(t1, z0), at(t1, lo), at(t0, lo), provenance, room_id)
        if hi < z1:
            mesh.add_quad(at(t0, hi), at(t1, hi), at(t1, z1), at(t0, z1), provenance, room_id)
        cursor = t1
    if cursor < 1.0:
        mesh.add_quad(at(cursor, z0), at(1, z0), at(1, z1), at(cursor, z1), provenance, room_id)


# --------------------------------------------------------------------------
# glTF export


def to_glb(mesh: Mesh) -> bytes:
    """A self-contained binary glTF, one primitive per provenance class.

    Splitting by provenance rather than by room is what lets the viewer shade
    inferred surfaces differently without a per-face attribute lookup, and it keeps
    the primitive count at four however many rooms the flat has.
    """
    import json

    groups: dict[str, list[int]] = {}
    for tri_i, prov in enumerate(mesh.face_provenance):
        groups.setdefault(prov, []).append(tri_i)

    pos = np.asarray(mesh.positions, dtype=np.float32)
    nrm = np.asarray(mesh.normals, dtype=np.float32)
    if len(pos) == 0:
        pos = np.zeros((0, 3), np.float32)
        nrm = np.zeros((0, 3), np.float32)
    idx_all = np.asarray(mesh.indices, dtype=np.uint32)

    buffers: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    offset = 0

    def add_view(data: bytes, target: int) -> int:
        nonlocal offset
        pad = (-len(data)) % 4
        buffers.append(data + b"\x00" * pad)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data),
                      "target": target})
        offset += len(data) + pad
        return len(views) - 1

    pos_view = add_view(pos.tobytes(), 34962)
    nrm_view = add_view(nrm.tobytes(), 34962)
    accessors.append({"bufferView": pos_view, "componentType": 5126, "count": len(pos),
                      "type": "VEC3",
                      "min": pos.min(axis=0).tolist() if len(pos) else [0, 0, 0],
                      "max": pos.max(axis=0).tolist() if len(pos) else [0, 0, 0]})
    accessors.append({"bufferView": nrm_view, "componentType": 5126, "count": len(nrm),
                      "type": "VEC3"})

    primitives, materials = [], []
    for prov, tris in sorted(groups.items()):
        sub = np.concatenate([idx_all[3 * t: 3 * t + 3] for t in tris]) if tris else \
            np.zeros(0, np.uint32)
        v = add_view(sub.astype(np.uint32).tobytes(), 34963)
        accessors.append({"bufferView": v, "componentType": 5125, "count": len(sub),
                          "type": "SCALAR"})
        r, g, b, a = PROVENANCE_RGBA.get(prov, (0.7, 0.7, 0.7, 1.0))
        materials.append({
            "name": prov,
            "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, a],
                                     "metallicFactor": 0.0, "roughnessFactor": 0.95},
            "doubleSided": True,
        })
        primitives.append({"attributes": {"POSITION": 0, "NORMAL": 1},
                           "indices": len(accessors) - 1,
                           "material": len(materials) - 1})

    blob = b"".join(buffers)
    gltf = {
        "asset": {"version": "2.0", "generator": "visit-it shell builder (Phase 1)"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # glTF is y-up, our metric frame is z-up. The rotation lives in the node so
        # every consumer gets it, rather than each viewer inventing its own.
        "nodes": [{"mesh": 0, "rotation": [-0.7071067811865476, 0.0, 0.0,
                                           0.7071067811865476]}],
        "meshes": [{"name": "shell", "primitives": primitives}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blob += b"\x00" * ((-len(blob)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(blob), 0x004E4942) + blob
    return out
