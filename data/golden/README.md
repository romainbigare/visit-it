# UK golden set (Phase 0)

30 real UK listings scraped from Rightmove, used as the Phase 0 development and
regression set. See `ROADMAP.md` P0 and `docs/DATA-SOURCES.md`.

## What is here

| File | Committed | Contents |
|---|---|---|
| `golden_set.json` | yes | listings: metadata, photo/floorplan URLs, local paths |
| `coverage_report.md` | yes | floor-plan coverage over everything seen, plus the selected set |
| `media/` | **no** (gitignored) | ~553 images, ~87 MB — the photographs themselves |

`media/` is deliberately not committed — it keeps the repo small, and the
images are re-fetchable from the URLs in the manifest at any time (which is also
how the GPU validation script runs on Colab with no upload step). Every record
carries `"provenance": "scraped"` so inputs stay traceable when a reconstruction
misbehaves.

## Reproduce

```bash
# 1. metadata (~3 min, polite rate limiting, ~110 search+detail requests)
python -m pipeline.ingest.collect \
    --target 30 --min-floorplan-frac 0.65 \
    --regions london,manchester,birmingham,leeds,glasgow,bristol \
    --max-pages 3 --out-dir data/golden

# 2. images (~87 MB)
python -m pipeline.ingest.fetch_media --set data/golden/golden_set.json
```

Exit code is non-zero if the floor-plan quota is missed or the target is not
reached, so this is safe to run in CI.

## Known limitations

- **Rightmove only.** Zoopla is behind Cloudflare and returns 403 to any
  datacentre IP, including for `robots.txt`. See `pipeline/ingest/zoopla.py`.
- **No measured reference.** Nothing here has been measured with a tape. The
  reference is the listing's own floor plan, so this set supports
  *self-consistency* checks (does the reconstruction agree with the dimensions
  printed on the plan?) and *plausibility* checks, not absolute accuracy —
  see `ROADMAP.md` §0b.
- **Area figures are indicative.** Where `floor_area_sqm` is set, it comes from
  Rightmove's `sizings` or from parsing the description. Dimensions printed on
  the plan (54% of ours) are the stronger signal (`docs/DATA-SOURCES.md` §5).
