# UK golden set (Phase 0)

30 real UK listings scraped from Rightmove, used as the Phase 0 development and
regression set. See `ROADMAP.md` P0 and `docs/DATA-SOURCES.md`.

## What is here

| File | Committed | Contents |
|---|---|---|
| `golden_set.json` | yes | listings: metadata, photo/floorplan URLs, local paths |
| `coverage_report.md` | yes | floor-plan coverage over everything seen, plus the selected set |
| `media/` | **no** (gitignored) | ~553 images, ~87 MB — the photographs themselves |

`media/` is deliberately not committed: the photographs are the photographers'
copyright (`docs/DATA-SOURCES.md` §3.7), so republishing them in a public repo
is not ours to do. Every record carries `"provenance": "scraped"`, and CI must
keep scraped assets out of any training corpus.

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
- **No verified measurements yet.** This set gives layout and arrangement
  ground truth at best. Gate G0 additionally requires laser measurements for
  ≥10 listings and scans for ≥5; neither is in here.
- **Area figures are indicative.** Where `floor_area_sqm` is set, it comes from
  Rightmove's `sizings` or from parsing the description — not from a measured
  survey. The UK has no loi Carrez equivalent (`docs/DATA-SOURCES.md` §5).
