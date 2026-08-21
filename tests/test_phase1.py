"""Unit tests for the Phase 1 stages.

Two principles behind what is tested and what is not:

* **Test against ground truth we control.** The golden set has no measured
  reference, so the geometry tests build a room of known size, run the real
  pipeline code over it, and assert the recovered size. That is the only place in
  this project where "accuracy" is a word we are entitled to use.
* **Test the decisions, not the numbers.** A test that pins the median room-area
  error to 4.7% breaks every time the vectoriser improves. A test that asserts a
  mis-OCR'd dimension gets rejected by the scale solve keeps working, and it is
  the property we actually care about.
"""
from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.core import geom
from pipeline.core.artifacts import ArtifactStore, SchemaError, validate
from pipeline.core.overrides import clear, load, merge, rerun_stage_for
from pipeline.core.stages import STAGE_DEPS, STAGE_ORDER, normalise_stage, registered
from pipeline.layout import apertures, planes
from pipeline.layout.stage import layout_from_points
from pipeline.room_geometry.engines import get_engine, resolve
from pipeline.scale.solve import Constraint, quality, solve
from pipeline.shell.mesh import Mesh, build_room, to_glb


class TestGeom(unittest.TestCase):
    def test_area_and_centroid(self):
        r = geom.rectangle(4, 3)
        self.assertAlmostEqual(geom.area(r), 12.0)
        self.assertAlmostEqual(geom.centroid(r)[0], 0.0, places=6)

    def test_l_shape_centroid_is_outside_but_representative_point_is_not(self):
        # The reason waypoints use representative_point: an L-shaped room's area
        # centroid sits in the notch, and a waypoint there is inside a wall.
        L = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
        self.assertFalse(geom.point_in_polygon(geom.centroid(L), L))
        self.assertTrue(geom.point_in_polygon(geom.representative_point(L), L))

    def test_oriented_extent_ignores_rotation(self):
        r = geom.rectangle(5, 2)
        rot = geom.se2(r, 0, 0, 37.0)
        a = geom.oriented_extent(r)
        b = geom.oriented_extent(rot)
        self.assertAlmostEqual(a[0], b[0], places=3)
        self.assertAlmostEqual(a[1], b[1], places=3)

    def test_iou_and_shared_edge(self):
        a = geom.rectangle(4, 3)
        self.assertAlmostEqual(geom.iou(a, a), 1.0)
        b = geom.se2(a, 4.0, 0.0, 0.0)
        self.assertAlmostEqual(geom.iou(a, b), 0.0, places=6)
        self.assertGreater(geom.shared_edge_length(a, b), 2.5)

    def test_ccw_is_enforced(self):
        cw = [[0, 0], [0, 3], [4, 3], [4, 0]]
        self.assertGreater(geom.signed_area(geom.ensure_ccw(cw)), 0)


class TestStageGraph(unittest.TestCase):
    def test_every_stage_is_implemented(self):
        impl = registered()
        missing = [s for s in STAGE_ORDER if s not in impl]
        self.assertEqual(missing, [], f"unimplemented stages: {missing}")

    def test_dependencies_are_acyclic_and_backwards_only(self):
        for stage, deps in STAGE_DEPS.items():
            for d in deps:
                self.assertLess(STAGE_ORDER.index(d), STAGE_ORDER.index(stage),
                                f"{stage} depends on later stage {d}")

    def test_stage_names_are_forgiving(self):
        for name in ("4", "layout", "4-layout"):
            self.assertEqual(normalise_stage(name), "4-layout")
        with self.assertRaises(KeyError):
            normalise_stage("nope")

    def test_assembly_is_the_only_join_of_both_channels(self):
        # ROADMAP §1: B and C meet only at stage 6. If that stops being true the
        # streams stop being parallelisable and the plan needs rewriting.
        self.assertIn("4-layout", STAGE_DEPS["6-assembly"])
        self.assertIn("5-plan", STAGE_DEPS["6-assembly"])
        self.assertNotIn("5-plan", STAGE_DEPS["4-layout"])
        self.assertNotIn("4-layout", STAGE_DEPS["5-plan"])


class TestArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_schema_error_lists_every_problem(self):
        with self.assertRaises(SchemaError) as cm:
            validate({"schema": "plan/v1"}, "plan.json")
        msg = str(cm.exception)
        self.assertIn("listing_id", msg)
        self.assertIn("rooms", msg)

    def test_write_is_content_addressed_and_versioned(self):
        store = ArtifactStore("x1", self.tmp)
        p = {"schema": "groups/v1", "listing_id": "x1", "groups": [], "qa_flags": []}
        a = store.write("2-grouping", p)
        b = store.write("2-grouping", p)
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(len(store.history("2-grouping")), 1, "identical content is one version")
        p2 = dict(p, qa_flags=["changed"])
        c = store.write("2-grouping", p2)
        self.assertNotEqual(a.sha256, c.sha256)
        self.assertEqual(len(store.history("2-grouping")), 2)
        self.assertEqual(store.read("2-grouping")["qa_flags"], ["changed"])
        self.assertEqual(store.read("2-grouping", version=1)["qa_flags"], [])

    def test_binaries_with_nested_names(self):
        store = ArtifactStore("x2", self.tmp)
        rec = store.write_binary("rooms/kitchen/geometry.npz", b"abc")
        self.assertEqual(rec["bytes"], 3)
        self.assertEqual(store.read_binary("rooms/kitchen/geometry.npz"), b"abc")

    def test_require_names_what_is_missing(self):
        store = ArtifactStore("x3", self.tmp)
        with self.assertRaises(FileNotFoundError) as cm:
            store.require("5-plan")
        self.assertIn("5-plan", str(cm.exception))


class TestOverrides(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_merge_is_deep_and_clear_is_targeted(self):
        merge("l1", {"assembly": {"pin": {"a": "p0"}}}, self.tmp)
        merge("l1", {"assembly": {"pin": {"b": "p1"}}}, self.tmp)
        self.assertEqual(load("l1", self.tmp)["assembly"]["pin"], {"a": "p0", "b": "p1"})
        merge("l1", {"plan": {"px_per_metre": 100.0}}, self.tmp)
        clear("l1", "assembly", self.tmp)
        self.assertNotIn("assembly", load("l1", self.tmp))
        self.assertIn("plan", load("l1", self.tmp))

    def test_rerun_starts_at_the_earliest_affected_stage(self):
        # The whole point of the override mechanism: nudging an assignment must not
        # re-run the 80-second geometry pass.
        self.assertEqual(rerun_stage_for(["assembly"]), "6-assembly")
        self.assertEqual(rerun_stage_for(["assembly", "plan"]), "5-plan")


class TestPlanes(unittest.TestCase):
    """Ground truth we control: a box of known size, through the real code."""

    def _room(self, w, d, h, n=40000, seed=1):
        rng = np.random.default_rng(seed)
        faces = rng.choice(6, size=n, p=[0.30, 0.22, 0.12, 0.12, 0.12, 0.12])
        u, v = rng.random(n), rng.random(n)
        pts = np.zeros((n, 3))
        pts[faces == 0] = np.stack([u, v, np.zeros(n)], 1)[faces == 0] * [w, d, 1]
        pts[faces == 1] = np.stack([u, v, np.full(n, 1.0)], 1)[faces == 1] * [w, d, h]
        pts[faces == 2] = np.stack([np.zeros(n), v, u], 1)[faces == 2] * [1, d, h]
        pts[faces == 3] = np.stack([np.full(n, 1.0), v, u], 1)[faces == 3] * [w, d, h]
        pts[faces == 4] = np.stack([u, np.zeros(n), v], 1)[faces == 4] * [w, 1, h]
        pts[faces == 5] = np.stack([u, np.full(n, 1.0), v], 1)[faces == 5] * [w, d, h]
        return pts + rng.normal(0, 0.01, (n, 3))

    def test_recovers_a_known_room_to_a_few_percent(self):
        for w, d, h in ((4.2, 3.1, 2.55), (5.8, 2.4, 2.72), (3.0, 3.0, 2.40)):
            pts = self._room(w, d, h)
            lay = layout_from_points(pts, None, np.array([0, 0, 1.0]),
                                     room_id="r", engine="test")
            self.assertLess(abs(lay["area_m2"] - w * d) / (w * d), 0.06,
                            f"area for {w}x{d}: got {lay['area_m2']}")
            self.assertLess(abs(lay["room_height_m"] - h) / h, 0.06,
                            f"height for {h}: got {lay['room_height_m']}")

    def test_window_content_is_clipped(self):
        pts = self._room(4.0, 3.0, 2.6)
        rng = np.random.default_rng(9)
        # A building 20 m away, seen through the window. Phase 0 saw exactly this
        # as 10-16 m "rooms" on glazed city apartments.
        far = np.stack([rng.uniform(-8, 12, 3000), np.full(3000, 22.0),
                        rng.uniform(0, 14, 3000)], 1)
        rot, _ = planes.orient(np.vstack([pts, far]), np.array([0, 0, 1.0]))
        kept, _, rep = planes.clip_outliers(rot, None)
        self.assertIn("window_content_or_far_field", rep["reasons"])
        self.assertLess(kept[:, :2].max() - kept[:, :2].min(), 8.0)

    def test_implausible_ceiling_is_flagged_not_hidden(self):
        pts = self._room(4.0, 3.0, 5.4)          # a 5.4 m ceiling is not a UK flat
        lay = layout_from_points(pts, None, np.array([0, 0, 1.0]), room_id="r")
        self.assertIn("ceiling_outside_2.3_3.2m", lay["qa_flags"])
        self.assertLess(lay["confidence"], 0.6)

    def test_layout_validates_against_its_schema(self):
        lay = layout_from_points(self._room(4, 3, 2.6), None, np.array([0, 0, 1.0]),
                                 room_id="r")
        lay["listing_id"] = "t"
        validate(lay, "layout.json")


class TestApertures(unittest.TestCase):
    def test_finds_a_door_and_a_window_at_the_right_size(self):
        rng = np.random.default_rng(3)
        W, D, H = 4.0, 3.0, 2.6
        pts = []
        for _ in range(60000):
            f = int(rng.integers(0, 6))
            u, v = rng.random(2)
            if f == 0:
                pts.append([u * W, v * D, 0.0])
            elif f == 1:
                pts.append([u * W, v * D, H])
            elif f == 2:
                pts.append([0.0, v * D, u * H])
            elif f == 3:
                pts.append([W, v * D, u * H])
            elif f == 4:
                x, z = u * W, v * H
                if 1.6 < x < 2.5 and z < 2.05:
                    continue
                pts.append([x, 0.0, z])
            else:
                x, z = u * W, v * H
                if 1.0 < x < 3.0 and 0.95 < z < 2.15:
                    continue
                pts.append([x, D, z])
        P = np.asarray(pts) + rng.normal(0, 0.006, (len(pts), 3))
        found = apertures.detect(P, [[0, 0], [W, 0], [W, D], [0, D]], 0.0, H)
        kinds = {a.kind for a in found}
        self.assertIn("door", kinds)
        self.assertIn("window", kinds)
        door = next(a for a in found if a.kind == "door")
        self.assertAlmostEqual(door.width_m, 0.9, delta=0.25)
        self.assertAlmostEqual(door.height_m, 2.05, delta=0.3)

    def test_basin_contacts_survive_large_label_values(self):
        # The pair encoding used to assume labels below 100000; a noisy plan that
        # produced more basins than that decoded to the wrong pair and silently
        # dropped every aperture.
        from pipeline.floorplan.topology import _basin_contacts
        lab = np.zeros((10, 20), dtype=np.int32)
        lab[:, :10] = 200_000
        lab[:, 10:] = 200_001
        contacts = _basin_contacts(lab)
        pairs = {pair for pair, _pts in contacts}
        self.assertIn((200_000, 200_001), pairs)

    def test_a_solid_wall_yields_nothing(self):
        rng = np.random.default_rng(4)
        n = 40000
        u, v = rng.random(n), rng.random(n)
        faces = rng.choice(6, size=n)
        pts = []
        for f, a, b in zip(faces, u, v):
            if f == 0:
                pts.append([a * 4, b * 3, 0])
            elif f == 1:
                pts.append([a * 4, b * 3, 2.6])
            elif f == 2:
                pts.append([0, b * 3, a * 2.6])
            elif f == 3:
                pts.append([4, b * 3, a * 2.6])
            elif f == 4:
                pts.append([a * 4, 0, b * 2.6])
            else:
                pts.append([a * 4, 3, b * 2.6])
        found = apertures.detect(np.asarray(pts), [[0, 0], [4, 0], [4, 3], [0, 3]], 0.0, 2.6)
        self.assertEqual(found, [], "invented an opening in a solid wall")


class TestScaleSolve(unittest.TestCase):
    def test_recovers_a_known_scale(self):
        s = 1.18
        cs = [Constraint("plan_room_area", 14.0 * s ** 2, 14.0, 2, "k"),
              Constraint("plan_room_area", 10.0 * s ** 2, 10.0, 2, "b"),
              Constraint("ceiling_height", 2.5 * s, 2.5, 1, "h")]
        sol = solve(cs)
        self.assertAlmostEqual(sol.scale, s, places=3)
        self.assertLess(sol.residual_rms_pct, 0.5)

    def test_areas_and_lengths_solve_together(self):
        # The reason for log space: areas go as s², lengths as s. A solver that
        # treated them alike would land between the two answers.
        cs = [Constraint("plan_room_area", 20.0, 5.0, 2, "area"),   # implies s=2
              Constraint("door_height", 4.0, 2.0, 1, "door")]       # implies s=2
        self.assertAlmostEqual(solve(cs).scale, 2.0, places=6)

    def test_a_mis_ocred_dimension_is_rejected(self):
        # The real failure this guards: "5.91" read as "9.91" on one caption.
        good = [Constraint("plan_room_area", 14.5, 14.0, 2, "kitchen"),
                Constraint("plan_room_area", 10.5, 10.1, 2, "bed1"),
                Constraint("plan_room_area", 13.2, 12.8, 2, "bed2"),
                Constraint("stated_area", 62.0, 60.0, 2, "listing")]
        clean = solve([Constraint(*[c.kind, c.target, c.observed, c.power, c.detail])
                       for c in good])
        poisoned = solve([Constraint(*[c.kind, c.target, c.observed, c.power, c.detail])
                          for c in good] +
                         [Constraint("plan_room_area", 27.7, 10.0, 2, "bad_ocr")])
        self.assertTrue(any("bad_ocr" in r for r in poisoned.rejected))
        self.assertAlmostEqual(clean.scale, poisoned.scale, delta=0.02)

    def test_quality_rewards_independent_evidence(self):
        one_kind = solve([Constraint("plan_room_area", 14.0, 13.6, 2, f"r{i}")
                          for i in range(4)])
        mixed = solve([Constraint("plan_room_area", 14.0, 13.6, 2, "r0"),
                       Constraint("stated_area", 60.0, 58.3, 2, "listing"),
                       Constraint("ceiling_height", 2.55, 2.51, 1, "h")])
        self.assertGreater(quality(mixed, 3), quality(one_kind, 1))

    def test_no_constraints_is_scale_one_not_a_crash(self):
        sol = solve([])
        self.assertEqual(sol.scale, 1.0)
        self.assertEqual(sol.n_used, 0)


class TestAssembly(unittest.TestCase):
    def test_matches_rooms_to_the_right_polygons(self):
        from pipeline.assembly.matching import assign
        rooms = [{"room_id": "kitchen", "label": "kitchen", "area_m2": 14.0,
                  "aspect": 1.1, "confidence": 0.8},
                 {"room_id": "bed", "label": "bedroom", "area_m2": 10.4,
                  "aspect": 1.6, "confidence": 0.8}]
        plan = [{"room_id": "p0", "label": "bedroom", "area_m2": 10.5, "aspect": 1.56},
                {"room_id": "p1", "label": "kitchen", "area_m2": 14.5, "aspect": 1.08}]
        ms, _, _, _ = assign(rooms, plan)
        got = {m.room_id: m.plan_room_id for m in ms}
        self.assertEqual(got, {"kitchen": "p1", "bed": "p0"})

    def test_identical_rooms_report_no_margin(self):
        # Two identical bedrooms are genuinely interchangeable. Reporting a
        # confident assignment there would be a lie the viewer would render.
        from pipeline.assembly.matching import assign
        rooms = [{"room_id": f"b{i}", "label": "bedroom", "area_m2": 10.5,
                  "aspect": 1.5, "confidence": 0.8} for i in range(2)]
        plan = [{"room_id": f"p{i}", "label": "bedroom", "area_m2": 10.5,
                 "aspect": 1.5} for i in range(2)]
        ms, _, _, _ = assign(rooms, plan)
        self.assertTrue(all(m.margin is not None and m.margin < 0.01 for m in ms))
        self.assertTrue(all(m.confidence < 0.7 for m in ms))

    def test_a_hopeless_match_is_refused(self):
        from pipeline.assembly.matching import assign
        rooms = [{"room_id": "bath", "label": "bathroom", "area_m2": 3.0,
                  "aspect": 1.2, "confidence": 0.4}]
        plan = [{"room_id": "p0", "label": "living_room", "area_m2": 34.0, "aspect": 3.4}]
        ms, unmatched, _, _ = assign(rooms, plan)
        self.assertEqual(ms, [])
        self.assertEqual(unmatched, ["bath"])

    def test_a_pin_beats_a_reject(self):
        # Both come from a hand-edited override file, so a room really can appear
        # in each. Without the guard it comes out matched *and* unmatched, and the
        # shell builder and the scoreboard then disagree about the listing.
        from pipeline.assembly.stage import assemble
        layouts = {"listing_id": "t", "rooms": [
            {"room_id": "kitchen", "room_label": "kitchen", "area_m2": 14.0,
             "polygon_m": geom.rectangle(4, 3.5), "confidence": 0.8,
             "room_height_m": 2.5, "qa_flags": [], "apertures": []}]}
        plan = {"rooms": [
            {"room_id": "p0", "label": "kitchen", "area_m2": 14.2,
             "polygon_m": geom.rectangle(4, 3.55), "aperture_ids": [],
             "confidence": 0.8}]}
        out = assemble(layouts, plan, {"pin": {"kitchen": "p0"},
                                       "reject": ["kitchen"]})
        matched = {m["room_id"] for m in out["matches"]}
        self.assertIn("kitchen", matched)
        self.assertNotIn("kitchen", out["unmatched_rooms"])

    def test_pose_refinement_finds_the_rotation(self):
        from pipeline.assembly.pose import refine
        room = geom.rectangle(4.0, 2.6)
        target = geom.se2(geom.rectangle(4.05, 2.55), 7.0, 3.0, 90.0)
        pose, iou = refine(room, target)
        self.assertGreater(iou, 0.9)


class TestShell(unittest.TestCase):
    def test_glb_is_valid_and_splits_by_provenance(self):
        m = Mesh()
        build_room(m, geom.rectangle(4, 3), 0.0, 2.6, "r0", "reconstructed")
        build_room(m, geom.rectangle(3, 3, 5.5, 0), 0.0, 2.6, "r1", "inferred")
        glb = to_glb(m)
        self.assertEqual(glb[:4], b"glTF")
        import struct
        jl = struct.unpack("<I", glb[12:16])[0]
        doc = json.loads(glb[20:20 + jl])
        names = {mat["name"] for mat in doc["materials"]}
        self.assertEqual(names, {"reconstructed", "inferred"})
        self.assertEqual(len(doc["meshes"][0]["primitives"]), 2)

    def test_shell_fits_the_1mb_budget(self):
        from pipeline.shell.stage import SHELL_BYTES_BUDGET
        m = Mesh()
        for i in range(12):
            build_room(m, geom.rectangle(4, 3, i * 5, 0), 0.0, 2.6, f"r{i}", "reconstructed")
        self.assertLess(len(to_glb(m)), SHELL_BYTES_BUDGET)

    def test_a_door_puts_a_hole_in_the_wall(self):
        plain = Mesh()
        build_room(plain, geom.rectangle(4, 3), 0.0, 2.6, "r", "reconstructed")
        holed = Mesh()
        build_room(holed, geom.rectangle(4, 3), 0.0, 2.6, "r", "reconstructed",
                   [{"type": "door", "position_m": [0.0, -1.5], "width_m": 0.85,
                     "height_m": 2.04}])
        self.assertGreater(holed.n_triangles, plain.n_triangles,
                           "cutting an aperture should split the wall panel")

    def test_l_shaped_rooms_triangulate_without_self_overlap(self):
        m = Mesh()
        L = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
        build_room(m, L, 0.0, 2.6, "r", "reconstructed")
        self.assertGreaterEqual(m.n_triangles, 6)
        self.assertEqual(len(m.indices) % 3, 0)


class TestEngines(unittest.TestCase):
    def test_synthetic_engine_is_deterministic(self):
        a = get_engine("synthetic").reconstruct([Path("a.jpg")])
        b = get_engine("synthetic").reconstruct([Path("a.jpg")])
        self.assertTrue(np.allclose(a.points, b.points))

    def test_unavailable_engine_falls_back_rather_than_failing(self):
        engine, note = resolve("mapanything", fallbacks=("synthetic",))
        self.assertNotEqual(engine.name, "mapanything")
        self.assertIn("fell back", note)


class TestHoldout(unittest.TestCase):
    def test_split_is_deterministic_and_sealed(self):
        from eval.holdout import make_split, seal
        listings = [{"listing_id": f"L{i}", "has_floorplan": i % 5 != 0,
                     "price_gbp": 200_000 + i * 31_000} for i in range(30)]
        a = make_split(listings, 20)
        b = make_split(list(reversed(listings)), 20)
        self.assertEqual(a["holdout"], b["holdout"], "split must not depend on input order")
        self.assertEqual(a["seal"], seal(a["dev"], a["holdout"]))
        self.assertEqual(len(set(a["dev"]) & set(a["holdout"])), 0)
        self.assertEqual(len(a["dev"]) + len(a["holdout"]), 30)

    def test_real_split_file_still_verifies(self):
        p = Path("data/golden/holdout_split.json")
        if not p.exists():
            self.skipTest("no frozen split on this checkout")
        from eval.holdout import load as load_split
        payload = load_split(p)          # raises if the seal does not match
        self.assertGreaterEqual(payload["n_holdout"], 20)


class TestMetrics(unittest.TestCase):
    def test_unjudged_criteria_are_not_counted_as_passes(self):
        from eval.metrics import summarise_g1
        rows = [{"arrangement": {"passed": None, "value": None, "threshold": 0.7}},
                {"arrangement": {"passed": True, "value": 0.9, "threshold": 0.7}}]
        s = summarise_g1(rows)
        self.assertEqual(s["arrangement"]["listings_judged"], 1)
        self.assertEqual(s["arrangement"]["listings_unjudged"], 1)
        self.assertEqual(s["arrangement"]["pass_rate"], 1.0)

    def test_m5_refuses_to_score_without_an_annotation(self):
        from eval.metrics import m5_assignment
        m = m5_assignment({"matches": [{"room_id": "a", "plan_room_id": "p0"}]}, None)
        self.assertIsNone(m.value, "M5 must not invent a reference")


if __name__ == "__main__":
    unittest.main()
