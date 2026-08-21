# Runbook — running the Phase 1 pipeline

Everything here runs CPU-only. A GPU makes stage 3 about 40× faster (Phase 0
measured MoGe-2 at 0.364 s/image on a T4 against 14.4 s on 4 CPU cores) and
changes nothing else.

## First run on a fresh box

```bash
make setup                 # python deps + tesseract
make vendor                # MoGe-2 source (not on PyPI)
python -m pipeline.ingest.fetch_media --set data/golden/golden_set.json   # ~87 MB
make holdout               # freeze the split, or verify the frozen one
```

The MoGe-2 weights (~1.3 GB) download on first use into `$HF_HOME`.

## One listing, end to end

```bash
python -m pipeline run 87977241
python -m pipeline show 87977241          # what exists, with its QA flags
```

Expect **90–110 s** per listing on 4 CPU cores, of which 80–95 s is stage 3.
Everything else totals under 10 s.

Useful flags:

| flag | what it does | when |
|---|---|---|
| `--from 6` | resume from a stage, reusing what is on disk | after changing assembly, scale or the shell — saves the 90 s geometry pass |
| `--only 5-plan` | run exactly one stage | iterating on the vectoriser |
| `--max-rooms 4` | cap rooms reconstructed | a quick check on a 30-photo listing |
| `--no-triage-model` | skip SigLIP, fall back to portal metadata | no weights on the box; the artifact says so in its flags |
| `--profile instant` | the hot-path bindings and budgets (AD-17) | checking latency, not quality |
| `--split dev` | with `--all`, restrict to one side of the frozen split | always, for development |

## Scoring

```bash
make score            # M1-M5 and the G1 criteria, dev split
make score-plan       # the plan channel alone — isolates C from B
make batch            # reprocess + score + regression check
python -m pipeline latency        # M12, p50/p95 per stage
```

**Never score the holdout during development.** `--split holdout` is a gate
measurement and the harness logs a warning when you ask for it.

## Looking at what went wrong

```bash
make console          # http://127.0.0.1:8080 — queue, contact sheets, fix actions
make sheets           # static contact sheets, zippable
```

The contact sheet is the first thing to open, always. Phase 0's lesson stands: a
flat grey render scored 12 dB and read as *merely poor* in a results table; only
opening the PNG showed it was nothing at all.

Debug order, which matches where things actually break:

1. **Plan channel** — are the room polygons on the plan roughly where a person
   would draw them, and is `px_per_metre` from `printed_dimensions`? If not, the
   arrangement cannot be right no matter what the photos say.
2. **Layout** — are the ceilings 2.3–3.2 m? A 5 m ceiling means "up" went wrong
   and the room is on its side.
3. **Assembly** — look at the cost matrix, not the answer. A small `margin` means
   the assignment was a coin flip.
4. **Scale** — look at which constraints were *rejected*. One is normal; three
   means the plan and the photos disagree about the flat.

## Fixing a listing by hand

In the console: relabel or drop a room in the plan overlay editor, or drag a room
onto a different polygon in the assignment nudge. Both write to
`data/runs/<listing>/overrides.json` and re-run only from the stage they
invalidate — an assignment nudge takes about a second.

Corrections survive re-runs, so the nightly batch will not undo them. Everything
they touch is flagged `override_applied`, so a hand-fixed listing never gets
counted as a clean automatic one.

To undo: `Clear all overrides and re-run` in the console, or delete the file.

## The viewer

```bash
make viewer                        # dev server against the hand-authored fixture
python -m tools.export_scene export --all && (cd viewer && npm run build)
```

`?dev=1` shows per-room areas, confidences and QA flags. Without it the only area
shown is the advertised one (ARCHITECTURE §5).

## When it breaks

| symptom | likely cause | fix |
|---|---|---|
| stage 3 fails immediately | MoGe source not vendored | `make vendor` |
| `no_floorplan` on a listing that has one | image not downloaded | re-run `fetch_media` |
| `portal_metadata_fallback` in the manifest | transformers or sentencepiece missing | `pip install -r requirements.txt` |
| stage 5 `no_plan_scale` | the plan prints no dimensions, no total, and the listing states no area | genuinely unscalable from the plan; the photo channel's ceiling prior carries it, at much lower confidence |
| `multiple_plan_outlines` | maisonette, two storeys on one sheet | expected; the largest outline is used |
| every ceiling implausible | "up" is wrong for this listing | look at the layout polygons in the contact sheet — a room on its side is obvious there |
| stage 3 over budget | it is; 80–95 s against a 50 s standard-profile budget on CPU | expected without a GPU. The overrun is recorded rather than hidden |
