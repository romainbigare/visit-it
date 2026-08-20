"""Tests for the ingestion and dataset layers.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from pipeline.ingest.robots import RobotsRules
from pipeline.ingest.models import CoverageStats, Listing, MediaRef
from pipeline.ingest import rightmove
from pipeline.ingest.collect import is_single_dwelling, _select

FIXTURES = Path(__file__).parent / "fixtures"


class TestRobots(unittest.TestCase):
    """The stdlib robotparser silently ignores '*', which would have let us
    fetch disallowed paths. These lock in the wildcard behaviour."""

    BODY = """
    User-agent: *
    Disallow: /api/*
    Disallow: /property-for-sale/draw-a-search.html?*
    Disallow: /*/fullscreen/image-gallery.html*
    Allow: /api/public/$

    User-agent: GPTBot
    Disallow: /
    """

    def setUp(self):
        self.r = RobotsRules(self.BODY, "Mozilla/5.0 Chrome/126")

    def test_plain_path_allowed(self):
        self.assertTrue(self.r.can_fetch("https://x/properties/12345"))

    def test_wildcard_prefix_blocked(self):
        self.assertFalse(self.r.can_fetch("https://x/api/foo"))

    def test_wildcard_with_query_blocked(self):
        self.assertFalse(self.r.can_fetch("https://x/property-for-sale/draw-a-search.html?a=1"))

    def test_wildcard_in_middle_blocked(self):
        self.assertFalse(self.r.can_fetch("https://x/anything/fullscreen/image-gallery.html?z"))

    def test_dollar_anchor_allow_beats_shorter_disallow(self):
        self.assertTrue(self.r.can_fetch("https://x/api/public/"))

    def test_named_agent_group_wins(self):
        g = RobotsRules(self.BODY, "GPTBot")
        self.assertFalse(g.can_fetch("https://x/properties/1"))

    def test_empty_disallow_means_allow_all(self):
        r = RobotsRules("User-agent: *\nDisallow:", "any")
        self.assertTrue(r.can_fetch("https://x/anything"))

    def test_crawl_delay_parsed(self):
        r = RobotsRules("User-agent: *\nCrawl-delay: 2.5", "any")
        self.assertEqual(r.crawl_delay, 2.5)


class TestUnflatten(unittest.TestCase):
    """Rightmove detail pages use a devalue-style flat array."""

    def test_resolves_indices(self):
        arr = [{"propertyData": 1}, {"id": 2, "beds": 3}, 163598753, 2]
        self.assertEqual(rightmove._unflatten(arr),
                         {"propertyData": {"id": 163598753, "beds": 2}})

    def test_handles_lists_and_shared_refs(self):
        arr = [{"a": 1, "b": 1}, [2, 3], "x", "y"]
        out = rightmove._unflatten(arr)
        self.assertEqual(out, {"a": ["x", "y"], "b": ["x", "y"]})

    def test_handles_cycles_without_recursion_error(self):
        arr = [{"self": 0, "v": 1}, "leaf"]
        out = rightmove._unflatten(arr)
        self.assertIs(out["self"], out)
        self.assertEqual(out["v"], "leaf")

    def test_negative_is_undefined(self):
        self.assertEqual(rightmove._unflatten([{"x": -1}]), {"x": None})


class TestRightmoveParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detail = FIXTURES / "rightmove_property.html"
        if not cls.detail.exists():
            raise unittest.SkipTest("fixture not present")

    def test_parses_property_into_listing(self):
        pd = rightmove.parse_property(self.detail.read_text(encoding="utf-8", errors="replace"))
        lst = rightmove.to_listing(pd)
        self.assertEqual(lst.source, "rightmove")
        self.assertTrue(lst.listing_id.isdigit())
        self.assertGreater(lst.n_photos, 0)
        self.assertIsNotNone(lst.display_address)
        self.assertTrue(all(p.url.startswith("http") for p in lst.photos))

    def test_area_from_description_when_sizings_empty(self):
        area, src = rightmove._area_sqm(
            {"sizings": [], "text": {"description": "A superb flat of 520 sq ft in total."}})
        self.assertAlmostEqual(area, 48.31, places=1)
        self.assertEqual(src, "description:sqft")

    def test_area_prefers_structured_sizings(self):
        area, src = rightmove._area_sqm(
            {"sizings": [{"unit": "sqm", "minimumSize": 74}], "text": {"description": "900 sq ft"}})
        self.assertEqual((area, src), (74.0, "sizings:sqm"))

    def test_area_absent_is_none(self):
        self.assertEqual(rightmove._area_sqm({"sizings": [], "text": {}}), (None, None))


class TestCoverageStats(unittest.TestCase):
    def test_counts_and_rates(self):
        s = CoverageStats(source="t")
        for n_fp, n_ph in [(1, 10), (0, 5), (2, 20), (1, 15)]:
            s.observe(n_floorplans=n_fp, n_photos=n_ph, has_area=n_fp > 0)
        self.assertEqual((s.seen, s.with_floorplan, s.without_floorplan), (4, 3, 1))
        self.assertEqual(s.with_multiple_floorplans, 1)
        self.assertAlmostEqual(s.floorplan_rate, 0.75)
        self.assertEqual(s.median_photos, 12.5)

    def test_empty_is_safe(self):
        s = CoverageStats(source="t")
        self.assertEqual(s.floorplan_rate, 0.0)
        self.assertIsNone(s.median_photos)


class TestDwellingFilter(unittest.TestCase):
    def test_rejects_too_many_bedrooms(self):
        self.assertFalse(is_single_dwelling({"bedrooms": 19})[0])

    def test_rejects_marketing_keywords(self):
        for blob in ("Superb investment opportunity", "Off Plan Manchester",
                     "6% Rental Yields | Great", "Residential Portfolio"):
            self.assertFalse(is_single_dwelling({"bedrooms": 2, "displayAddress": blob})[0], blob)

    def test_accepts_normal_flat(self):
        self.assertTrue(is_single_dwelling(
            {"bedrooms": 2, "displayAddress": "Knowle Road, Totterdown, Bristol",
             "propertySubType": "Flat"})[0])


class TestSelection(unittest.TestCase):
    def _cands(self, n_with, n_without):
        return ([{"id": f"w{i}", "numberOfFloorplans": 1, "_stratum": f"s{i%3}"} for i in range(n_with)]
                + [{"id": f"n{i}", "numberOfFloorplans": 0, "_stratum": f"s{i%3}"} for i in range(n_without)])

    def test_meets_quota_and_keeps_some_planless(self):
        w, wo = _select(self._cands(60, 40), target=30, fp_frac=0.75)
        self.assertEqual(len(w) + len(wo), 30)
        self.assertGreaterEqual(len(w) / 30, 0.65)
        self.assertGreater(len(wo), 0, "no-plan listings are the Tier B test cases")

    def test_backfills_when_few_planless(self):
        w, wo = _select(self._cands(60, 2), target=30, fp_frac=0.75)
        self.assertEqual(len(w) + len(wo), 30)
        self.assertGreaterEqual(len(w) / 30, 0.65)

    def test_spreads_across_strata(self):
        w, _ = _select(self._cands(60, 40), target=30, fp_frac=0.75)
        self.assertGreater(len({c["_stratum"] for c in w}), 1)


class TestListingModel(unittest.TestCase):
    def test_has_floorplan_and_serialisation(self):
        l = Listing(listing_id="1", source="rightmove", url="u",
                    photos=[MediaRef(url="p", kind="photo")])
        self.assertFalse(l.has_floorplan)
        l.floorplans.append(MediaRef(url="f", kind="floorplan"))
        d = l.to_dict()
        self.assertTrue(d["has_floorplan"])
        self.assertEqual(d["n_photos"], 1)
        self.assertEqual(d["provenance"], "scraped")
        json.dumps(d)  # must be JSON-serialisable


class TestDatasets(unittest.TestCase):
    """The dataset layer must be reproducible across machines: same cache root
    from one env var, manual archives win, downloads are checksummed."""

    def setUp(self):
        import os
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["VISITIT_DATA_HOME"] = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_unknown_dataset_raises(self):
        from pipeline.datasets import ensure
        with self.assertRaises(KeyError):
            ensure("nope")

    def test_registry_covers_the_corpora_we_planned(self):
        from pipeline.datasets import REGISTRY
        for name in ("swiss_dwellings", "cubicasa5k", "resplan", "msd", "c3po",
                     "zind", "structured3d", "rent3d", "rent3dpp", "hypersim"):
            self.assertIn(name, REGISTRY)

    def test_every_direct_dataset_has_urls(self):
        from pipeline.datasets import REGISTRY
        for name, spec in REGISTRY.items():
            if spec.acquisition == "direct":
                self.assertTrue(spec.files, f"{name} is direct but has no files")
                for f in spec.files:
                    self.assertTrue(f.url.startswith("https://"), f"{name}: {f.url}")

    def test_manual_datasets_explain_themselves(self):
        from pipeline.datasets import REGISTRY
        for name, spec in REGISTRY.items():
            if spec.acquisition == "manual":
                self.assertTrue(spec.manual_instructions,
                                f"{name} is manual but gives no instructions")

    def test_incoming_archive_wins_and_caches(self):
        from pipeline.datasets import ensure, data_home
        inc = data_home() / "_incoming"
        inc.mkdir(parents=True, exist_ok=True)
        src = Path(self.tmp.name) / "src" / "rent3d"
        src.mkdir(parents=True)
        (src / "meta.json").write_text("{}")
        with tarfile.open(inc / "rent3d.tar.gz", "w:gz") as tf:
            tf.add(src.parent, arcname=".")
        root = ensure("rent3d")
        self.assertTrue((root / "rent3d" / "meta.json").exists())
        self.assertTrue(ensure("rent3d").exists())  # cached second call

    def test_manual_dataset_without_archive_gives_instructions(self):
        from pipeline.datasets import ensure, ManualAcquisitionRequired
        with self.assertRaises(ManualAcquisitionRequired) as cm:
            ensure("zind")
        self.assertIn("zind", str(cm.exception).lower())

    def test_extract_rejects_path_traversal(self):
        from pipeline.datasets.fetch import _extract
        evil = Path(self.tmp.name) / "evil.tar.gz"
        payload = Path(self.tmp.name) / "ok.txt"
        payload.write_text("x")
        with tarfile.open(evil, "w:gz") as tf:
            tf.add(payload, arcname="../escaped.txt")
        dest = Path(self.tmp.name) / "out"
        _extract(evil, dest)
        self.assertFalse((Path(self.tmp.name) / "escaped.txt").exists())

    def test_sha256_is_stable(self):
        from pipeline.datasets import sha256_of
        f = Path(self.tmp.name) / "a.bin"
        f.write_bytes(b"visit-it")
        self.assertEqual(sha256_of(f), sha256_of(f))
        self.assertEqual(len(sha256_of(f)), 64)


if __name__ == "__main__":
    unittest.main()


class TestLiveRightmove(unittest.TestCase):
    """Opt-in smoke test against the real site.

    The committed fixture is synthetic (see fixtures/make_fixture.py), so it
    proves the parser handles the *shape* but not that the shape is still
    current. This test is the thing that catches Rightmove changing their
    payload. Run it deliberately:

        VISITIT_LIVE_TESTS=1 python -m unittest tests.test_ingest.TestLiveRightmove
    """

    @classmethod
    def setUpClass(cls):
        import os
        if os.environ.get("VISITIT_LIVE_TESTS") != "1":
            raise unittest.SkipTest("set VISITIT_LIVE_TESTS=1 to run live network tests")

    def test_search_and_detail_still_parse(self):
        from pipeline.ingest.http import PoliteSession
        s = PoliteSession(min_interval=1.5)
        self.assertTrue(s.allowed("https://www.rightmove.co.uk/property-for-sale/find.html?x=1"))
        stats = CoverageStats(source="rightmove")
        recs = list(rightmove.iter_search_results(s, "london", max_pages=1, stats=stats))
        self.assertGreater(len(recs), 5, "search returned no results")
        self.assertIn("numberOfFloorplans", recs[0])
        self.assertGreater(stats.seen, 0)
        lst = rightmove.fetch_listing(s, recs[0]["id"])
        self.assertEqual(lst.source, "rightmove")
        self.assertGreater(lst.n_photos, 0)
        self.assertTrue(lst.display_address)

    def test_robots_still_permits_what_we_fetch(self):
        from pipeline.ingest.http import PoliteSession
        s = PoliteSession()
        self.assertTrue(s.allowed("https://www.rightmove.co.uk/properties/163598753"))
        self.assertFalse(s.allowed("https://www.rightmove.co.uk/api/anything"))
