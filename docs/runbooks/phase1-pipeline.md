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
python -m tools.fetch_wallnet   # 98 MB wall segmenter for stage 5
```

The MoGe-2 weights (~1.3 GB) download on first use into `$HF_HOME`.

`fetch_wallnet` is optional but strongly recommended: without it stage 5 decides
what a wall is by asking whether the pixel is dark, which makes kitchen cabinets
and door swings into room boundaries. With it, stage 5 reads each plan both ways
and keeps the better reading. See [PLAN-READING-REPORT.md](../PLAN-READING-REPORT.md).

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
python -m tools.plan_vs_shell build --out out/review   # plan beside shell, per listing
```

## Improving the plan reading

```bash
python -m tools.annotate_walls              # correct the model's wall reading, in a browser
python -m tools.annotate_walls --status     # how many plans are done
python -m tools.annotate_walls --export     # write data/golden/wall_training/
```

Each plan opens with the wall model's own reading painted on, so the work is scrubbing
off what is not a wall -- door swings, cabinet runs, dimension lines -- rather than
tracing an outline cold. A minute or two a plan; twenty is a useful fine-tuning set.
Then run `notebooks/finetune_wallnet_colab.ipynb` on a Colab GPU and drop the resulting
`plan_walls.safetensors` into `models/`. `python -m tools.fetch_wallnet --force` puts
the original back if it turns out worse.

### Using the room-finder

```bash
python -m tools.import_room_predictions results.zip --list
python -m tools.import_room_predictions results.zip --reading <name>
python -m pipeline run <ids> --from 5     # stage 5 picks them up automatically
python -m tools.import_room_predictions --clear   # back to the ink/wall reading
```

`results.zip` comes from `notebooks/plan_reading_modal.ipynb`. With predictions in place
stage 5 reports `method: room_finder_vectorise/v1` and flags how many of the rooms got
their name from the plan rather than from the model.

Before doing any of that, run `notebooks/plan_reading_modal.ipynb` on a Modal GPU. It
starts from the room-finder as published and tunes it: all five checkpoints the authors
publish, the picture it is shown (cleaned, cropped, thresholded, strokes thickened), the
plan cut into quarters, four views merged, and its own answer tidied. The last step pulls
the corners onto our wall map, which is what stage 5 does anyway.

It is not automatic and is not meant to be. Every step is measured on the same 25 plans,
draws all of them before and after the change, and then stops at a `True`/`False` cell. What
you keep is what the next step builds on, and the notebook ends by printing the recipe you
kept -- checkpoint, picture steps and reading -- along with the name to import.

Two numbers, because either alone can be gamed: the share of the room names printed on the
plan that land inside exactly one predicted room, and the share of each room edge that sits
on a wall. A reading that deletes an awkward room loses on the first. Import a **room-finder**
row, not the grown one; stage 5 does the growing itself.

The contact sheet is the first thing to open, always. Phase 0's lesson stands: a
flat grey render scored 12 dB and read as *merely poor* in a results table; only
opening the PNG showed it was nothing at all.

Debug order, which matches where things actually break:

1. **Plan channel** — are the room polygons on the plan roughly where a person
   would draw them, and is `px_per_metre` from `printed_dimensions`? If not, the
   arrangement cannot be right no matter what the photos say. `plan_vs_shell`
   answers the first half by eye; the `wall_source_*` QA flags say which engine
   read the plan and how the two compared. An outline that stops at the kitchen
   cabinets means the ink reading won and should not have.
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
