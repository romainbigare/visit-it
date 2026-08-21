# Eval — full channel, holdout split

17/20 listings scored (3 incomplete) · 2026-08-21T13:21:18+00:00

| metric | median | p90 | coverage | reference | n obs |
|---|---|---|---|---|---|
| M1 total area error | 16.118 % | 25.444 | 13/17 | stated | 13 |
| M2 per-room area error | 15.21 % | 28.938 | 5/17 | printed | 15 |
| M3 adjacency accuracy | 0.0 fraction | 0.0 | 17/17 | printed | 136 |
| M4 layout IoU | 1.0 IoU | 1.0 | 14/17 | printed | 127 |
| M5 assignment accuracy | 50.0 % | 77.5 | 6/17 | annotated | 33 |

## G1 criteria

| criterion | pass rate | median | threshold | judged | unjudged |
|---|---|---|---|---|---|
| self_consistency | 0.2 | 15.214 | 10.0 | 5 | 12 |
| plausibility | 0.294 | 0.633 | 0.8 | 17 | 0 |
| arrangement | 0.333 | 0.5 | 0.7 | 6 | 11 |
| cross_model_scale | 0.385 | 17.99 | 15.0 | 13 | 4 |
| shell_budget | 1.0 | 23912.0 | 1048576 | 17 | 0 |

## QA flags

| flag | listings |
|---|---|
| `low_assignment_margin` | 16 |
| `wall_source_wallnet` | 14 |
| `walls_from_pretrained_net` | 14 |
| `unmatched_plan_polygons` | 12 |
| `under_80pct_plausible_ceilings` | 11 |
| `cross_model_scale_disagreement` | 8 |
| `room_area_disagrees_with_printed` | 8 |
| `not_to_scale_disclaimer` | 7 |
| `dropped_1_implausible_regions` | 7 |
| `wall_source_close_call` | 6 |
| `majority_of_rooms_unphotographed` | 5 |
| `scale_constraints_disagree` | 4 |
| `scale_constraint_rejected` | 4 |
| `unmatched_reconstructed_rooms` | 4 |
| `scale_candidates_disagree` | 4 |

> No tape-measure ground truth exists for this set. Every number here is plausibility and self-consistency, not accuracy — ROADMAP §0b.

## Batch

0 reprocessed in 0s · 0 failed, 0 partial

## Regression check

| metric | previous | now | drop | threshold | rule |
|---|---|---|---|---|---|
| M2 | 9.01 | 15.21 | 6.2 | 2.5 | absolute floor |

> σ over fewer than 3 runs is not an estimate; the floor rule is doing the work
