# Eval — full channel, dev split

8/10 listings scored (2 incomplete) · 2026-08-21T13:21:18+00:00

| metric | median | p90 | coverage | reference | n obs |
|---|---|---|---|---|---|
| M1 total area error | 21.65 % | 57.213 | 7/8 | stated | 7 |
| M2 per-room area error | 32.02 % | 167.447 | 6/8 | printed | 15 |
| M3 adjacency accuracy | 0.0 fraction | 0.0 | 8/8 | annotated | 42 |
| M4 layout IoU | 1.0 IoU | 1.0 | 8/8 | printed | 74 |
| M5 assignment accuracy | 80.0 % | 80.0 | 1/8 | annotated | 5 |

## G1 criteria

| criterion | pass rate | median | threshold | judged | unjudged |
|---|---|---|---|---|---|
| self_consistency | 0.333 | 32.023 | 10.0 | 6 | 2 |
| plausibility | 0.625 | 0.8 | 0.8 | 8 | 0 |
| arrangement | 1.0 | 0.8 | 0.7 | 1 | 7 |
| cross_model_scale | 0.75 | 10.63 | 15.0 | 8 | 0 |
| shell_budget | 1.0 | 28820.0 | 1048576 | 8 | 0 |

## QA flags

| flag | listings |
|---|---|
| `low_assignment_margin` | 8 |
| `unmatched_plan_polygons` | 6 |
| `wall_source_wallnet` | 6 |
| `walls_from_pretrained_net` | 6 |
| `not_to_scale_disclaimer` | 5 |
| `room_area_disagrees_with_printed` | 4 |
| `scale_constraint_rejected` | 3 |
| `wall_source_close_call` | 3 |
| `wall_source_coverage_0.99ink_vs_1.00net` | 3 |
| `scale_constraints_disagree` | 3 |
| `self_consistency_outside_10pct` | 3 |
| `unmatched_reconstructed_rooms` | 3 |
| `under_80pct_plausible_ceilings` | 3 |
| `majority_of_rooms_unphotographed` | 2 |
| `no_room_labels` | 2 |

> No tape-measure ground truth exists for this set. Every number here is plausibility and self-consistency, not accuracy — ROADMAP §0b.
