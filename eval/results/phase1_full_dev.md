# Eval — full channel, dev split

8/10 listings scored (2 incomplete) · 2026-08-21T10:24:26+00:00

| metric | median | p90 | coverage | reference | n obs |
|---|---|---|---|---|---|
| M1 total area error | 21.814 % | 51.541 | 7/8 | stated | 7 |
| M2 per-room area error | 9.05 % | 21.98 | 6/8 | printed | 12 |
| M3 adjacency accuracy | 0.8 fraction | 1.0 | 7/8 | annotated | 26 |
| M4 layout IoU | 0.443 IoU | 0.598 | 8/8 | printed | 37 |
| M5 assignment accuracy | 80.0 % | 80.0 | 1/8 | annotated | 5 |

## G1 criteria

| criterion | pass rate | median | threshold | judged | unjudged |
|---|---|---|---|---|---|
| self_consistency | 0.667 | 9.046 | 10.0 | 6 | 2 |
| plausibility | 0.375 | 0.55 | 0.8 | 8 | 0 |
| arrangement | 1.0 | 0.8 | 0.7 | 1 | 7 |
| cross_model_scale | 0.5 | 14.125 | 15.0 | 8 | 0 |
| shell_budget | 1.0 | 4884.0 | 1048576 | 8 | 0 |

## QA flags

| flag | listings |
|---|---|
| `low_assignment_margin` | 8 |
| `not_to_scale_disclaimer` | 5 |
| `under_80pct_plausible_ceilings` | 5 |
| `unmatched_plan_polygons` | 4 |
| `room_area_disagrees_with_printed` | 4 |
| `cross_model_scale_disagreement` | 4 |
| `rooms_omitted_from_shell` | 4 |
| `scale_candidates_disagree` | 4 |
| `scale_constraints_disagree` | 4 |
| `unmatched_reconstructed_rooms` | 4 |
| `no_doors_detected` | 3 |
| `plan_area_disagrees_with_stated` | 3 |
| `multiple_plan_outlines` | 2 |
| `scale_constraint_rejected` | 2 |
| `self_consistency_outside_10pct` | 2 |

> No tape-measure ground truth exists for this set. Every number here is plausibility and self-consistency, not accuracy — ROADMAP §0b.
