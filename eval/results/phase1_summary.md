30/30 listings have been run (24 carry a floor plan). Split: 10 dev / 20 holdout (16 of the holdout carry a plan).


### Stage completion

| stage | ok | partial/skipped | failed |
|---|---|---|---|
| 5-plan | 25 | 5 | 0 |
| 6-assembly | 25 | 5 | 0 |
| 7-scale | 25 | 5 | 0 |
| 8-shell | 25 | 5 | 0 |
| 9-package | 25 | 5 | 0 |

### Plan channel (stage 5) — 25 listings

| | value |
|---|---|
| scale source | printed_dimensions 11, stated_area 10, none 3, printed_area 1 |
| rooms found per listing | median 8.0 (p10 4.4, p90 15.0) |
| rooms carrying a label | 59% |
| doors found per listing | median 5.0 |
| adjacency edges per listing | median 6.0 |
| plan area / stated area | median 1.0 (n=18) |
| carry a 'not to scale' disclaimer | 48% |
| stage confidence | median 0.55 |

### Geometry and layout (stages 3-4) — 30 listings

| | value |
|---|---|
| rooms reporting a ceiling height | 105 |
| rooms where we refused to report one | 66 (floor_ceiling_gap_too_small 37, ceiling_or_floor_barely_observed 27, no_floor_ceiling_found 2) |
| ceiling height | median 2.574 m (p10 2.07, p90 2.88) |
| within 2.3-3.2 m | 82% |

### Assembly (stage 6) — 25 listings

| | value |
|---|---|
| rooms matched per listing | median 6.0 |
| reconstructed rooms left unmatched | median 0.0 |
| plan polygons left unmatched | median 2.0 |
| cost margin over the runner-up | median 0.017 |
| polygon fit (IoU after SE(2)) | median 0.56 |

### Scale (stage 7) — 25 listings

| | value |
|---|---|
| scale factor applied | median 1.069 (p10 0.81, p90 1.34) |
| solve quality | median 0.8 |
| residual RMS | median 11.413% |
| **self-consistency vs printed dimensions** | median 11.947% (n=11) |
| rooms within ±10% of their printed size | median 0.333 |
| cross-model scale disagreement | median 12.95% |
| constraints rejected as outliers | 10 total |

### Shell (stages 8-9) — 25 listings

| | value |
|---|---|
| glTF size | median 26216.0 bytes (max 65772) |
| triangles | median 484.0 |
| rooms in the shell | median 8.0 |
| over the 1 MB budget | 0 |

### Latency (M12) — 232 runs

| stage | p50 | p95 | max |
|---|---|---|---|
| 0-triage | 4.396 | 10.037 | 117.885 |
| 1-conditioning | 0.0 | 0.0 | 0.0 |
| 2-grouping | 0.0 | 0.0 | 0.0 |
| 3-geometry | 76.996 | 94.794 | 422.039 |
| 4-layout | 2.244 | 4.089 | 5.082 |
| 5-plan | 4.382 | 8.754 | 237.997 |
| 6-assembly | 0.052 | 0.126 | 0.151 |
| 7-scale | 0.001 | 0.002 | 0.004 |
| 8-shell | 0.005 | 0.076 | 0.167 |
| 9-package | 0.001 | 0.005 | 0.009 |
| **end to end (full runs, n=33)** | **87.627** | 149.084 | — |

### QA flags, by how many listings raised them

| flag | listings |
|---|---|
| `low_assignment_margin` | 25 |
| `not_to_scale_disclaimer` | 24 |
| `room_area_disagrees_with_printed` | 22 |
| `unmatched_plan_polygons` | 18 |
| `under_80pct_plausible_ceilings` | 14 |
| `no_doors_detected` | 12 |
| `scale_candidates_disagree` | 12 |
| `cross_model_scale_disagreement` | 9 |
| `large_regularisation_snap` | 8 |
| `dropped_1_implausible_regions` | 8 |
| `unmatched_reconstructed_rooms` | 7 |
| `scale_constraints_disagree` | 7 |
| `majority_of_rooms_unphotographed` | 7 |
| `scale_constraint_rejected` | 7 |
| `no_room_labels` | 6 |
| `self_consistency_outside_10pct` | 6 |
| `no_plan_scale` | 6 |
| `dropped_2_implausible_regions` | 4 |
| `no_room_captions` | 4 |
| `poor_polygon_fit` | 3 |
