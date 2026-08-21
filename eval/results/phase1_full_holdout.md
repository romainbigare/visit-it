# Eval — full channel, holdout split

14/20 listings scored (6 incomplete) · 2026-08-21T10:24:27+00:00

| metric | median | p90 | coverage | reference | n obs |
|---|---|---|---|---|---|
| M1 total area error | 30.211 % | 61.925 | 13/14 | stated | 13 |
| M2 per-room area error | 14.3 % | 137.442 | 7/14 | printed | 13 |
| M3 adjacency accuracy | 0.081 fraction | 0.85 | 10/14 | printed | 78 |
| M4 layout IoU | 0.37 IoU | 0.582 | 14/14 | printed | 61 |
| M5 assignment accuracy | 66.667 % | 95.0 | 4/14 | annotated | 21 |

## G1 criteria

| criterion | pass rate | median | threshold | judged | unjudged |
|---|---|---|---|---|---|
| self_consistency | 0.143 | 14.299 | 10.0 | 7 | 7 |
| plausibility | 0.214 | 0.667 | 0.8 | 14 | 0 |
| arrangement | 0.5 | 0.667 | 0.7 | 4 | 10 |
| cross_model_scale | 0.462 | 16.72 | 15.0 | 13 | 1 |
| shell_budget | 1.0 | 5526.0 | 1048576 | 14 | 0 |

## QA flags

| flag | listings |
|---|---|
| `low_assignment_margin` | 12 |
| `under_80pct_plausible_ceilings` | 10 |
| `unmatched_plan_polygons` | 9 |
| `cross_model_scale_disagreement` | 7 |
| `not_to_scale_disclaimer` | 7 |
| `rooms_omitted_from_shell` | 7 |
| `unmatched_reconstructed_rooms` | 7 |
| `scale_candidates_disagree` | 6 |
| `room_area_disagrees_with_printed` | 6 |
| `no_doors_detected` | 5 |
| `scale_constraints_disagree` | 5 |
| `scale_constraint_rejected` | 5 |
| `poor_polygon_fit` | 4 |
| `large_regularisation_snap` | 4 |
| `self_consistency_outside_10pct` | 4 |

> No tape-measure ground truth exists for this set. Every number here is plausibility and self-consistency, not accuracy — ROADMAP §0b.
