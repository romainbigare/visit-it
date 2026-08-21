# Eval — full channel, holdout split

17/20 listings scored (3 incomplete) · 2026-08-21T11:03:23+00:00

| metric | median | p90 | coverage | reference | n obs |
|---|---|---|---|---|---|
| M1 total area error | 16.646 % | 31.331 | 13/17 | stated | 13 |
| M2 per-room area error | 9.01 % | 17.514 | 5/17 | printed | 14 |
| M3 adjacency accuracy | 0.0 fraction | 0.0 | 17/17 | printed | 127 |
| M4 layout IoU | 1.0 IoU | 1.0 | 14/17 | printed | 123 |
| M5 assignment accuracy | 50.0 % | 87.5 | 6/17 | annotated | 33 |

## G1 criteria

| criterion | pass rate | median | threshold | judged | unjudged |
|---|---|---|---|---|---|
| self_consistency | 0.6 | 9.008 | 10.0 | 5 | 12 |
| plausibility | 0.294 | 0.633 | 0.8 | 17 | 0 |
| arrangement | 0.333 | 0.5 | 0.7 | 6 | 11 |
| cross_model_scale | 0.462 | 17.74 | 15.0 | 13 | 4 |
| shell_budget | 1.0 | 24168.0 | 1048576 | 17 | 0 |

## QA flags

| flag | listings |
|---|---|
| `low_assignment_margin` | 17 |
| `unmatched_plan_polygons` | 12 |
| `under_80pct_plausible_ceilings` | 11 |
| `cross_model_scale_disagreement` | 7 |
| `not_to_scale_disclaimer` | 7 |
| `room_area_disagrees_with_printed` | 7 |
| `no_doors_detected` | 5 |
| `majority_of_rooms_unphotographed` | 5 |
| `large_regularisation_snap` | 4 |
| `scale_constraints_disagree` | 4 |
| `dropped_1_implausible_regions` | 4 |
| `scale_constraint_rejected` | 4 |
| `unmatched_reconstructed_rooms` | 4 |
| `scale_candidates_disagree` | 4 |
| `matched_without_a_plan_scale` | 3 |

> No tape-measure ground truth exists for this set. Every number here is plausibility and self-consistency, not accuracy — ROADMAP §0b.

## Batch

0 reprocessed in 0s · 0 failed, 0 partial

## Regression check

No metric regressed against the previous run.
