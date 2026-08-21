# Eval — full channel, dev split

4/10 listings scored (6 incomplete) · 2026-08-21T00:52:41+00:00

| metric | median | p90 | coverage | reference | n obs |
|---|---|---|---|---|---|
| M1 total area error | 31.509 % | 62.722 | 4/4 | stated | 4 |
| M2 per-room area error | 9.02 % | 14.868 | 3/4 | printed | 6 |
| M3 adjacency accuracy | 0.267 fraction | 0.8 | 4/4 | annotated | 19 |
| M4 layout IoU | 0.336 IoU | 0.619 | 4/4 | printed | 19 |
| M5 assignment accuracy | 80.0 % | 80.0 | 1/4 | annotated | 5 |

## G1 criteria

| criterion | pass rate | median | threshold | judged | unjudged |
|---|---|---|---|---|---|
| self_consistency | 0.667 | 9.023 | 10.0 | 3 | 1 |
| plausibility | 0.25 | 0.5 | 0.8 | 4 | 0 |
| arrangement | 1.0 | 0.8 | 0.7 | 1 | 3 |
| cross_model_scale | 0.5 | 11.6 | 15.0 | 4 | 0 |
| shell_budget | 1.0 | 4884.0 | 1048576 | 4 | 0 |

## QA flags

| flag | listings |
|---|---|
| `low_assignment_margin` | 4 |
| `under_80pct_plausible_ceilings` | 3 |
| `unmatched_plan_polygons` | 3 |
| `not_to_scale_disclaimer` | 3 |
| `room_area_disagrees_with_printed` | 2 |
| `cross_model_scale_disagreement` | 2 |
| `rooms_omitted_from_shell` | 2 |
| `scale_candidates_disagree` | 2 |
| `scale_constraints_disagree` | 2 |
| `unmatched_reconstructed_rooms` | 2 |
| `large_regularisation_snap` | 1 |
| `multiple_plan_outlines` | 1 |
| `no_room_captions` | 1 |
| `no_room_labels` | 1 |
| `no_doors_detected` | 1 |

> No tape-measure ground truth exists for this set. Every number here is plausibility and self-consistency, not accuracy — ROADMAP §0b.
