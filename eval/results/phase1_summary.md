30/30 listings have been run (24 carry a floor plan). Split: 10 dev / 20 holdout (16 of the holdout carry a plan).


### Stage completion

| stage | ok | partial/skipped | failed |
|---|---|---|---|
| 6-assembly | 22 | 8 | 0 |
| 7-scale | 22 | 8 | 0 |
| 8-shell | 22 | 8 | 0 |
| 9-package | 22 | 8 | 0 |

### Plan channel (stage 5) — 25 listings

| | value |
|---|---|
| scale source | printed_dimensions 12, stated_area 8, none 3, printed_area 2 |
| rooms found per listing | median 7.0 (p10 2.4, p90 16.0) |
| rooms carrying a label | 63% |
| doors found per listing | median 1.0 |
| adjacency edges per listing | median 3.0 |
| plan area / stated area | median 1.0 (n=18) |
| carry a 'not to scale' disclaimer | 48% |
| stage confidence | median 0.497 |

### Geometry and layout (stages 3-4) — 30 listings

| | value |
|---|---|
| rooms reporting a ceiling height | 105 |
| rooms where we refused to report one | 66 (floor_ceiling_gap_too_small 37, ceiling_or_floor_barely_observed 27, no_floor_ceiling_found 2) |
| ceiling height | median 2.574 m (p10 2.07, p90 2.88) |
| within 2.3-3.2 m | 82% |

### Assembly (stage 6) — 22 listings

| | value |
|---|---|
| rooms matched per listing | median 5.0 |
| reconstructed rooms left unmatched | median 0.5 |
| plan polygons left unmatched | median 2.0 |
| cost margin over the runner-up | median 0.037 |
| polygon fit (IoU after SE(2)) | median 0.517 |

### Scale (stage 7) — 22 listings

| | value |
|---|---|
| scale factor applied | median 1.069 (p10 0.77, p90 1.32) |
| solve quality | median 0.803 |
| residual RMS | median 14.563% |
| **self-consistency vs printed dimensions** | median 10.816% (n=13) |
| rooms within ±10% of their printed size | median 0.5 |
| cross-model scale disagreement | median 15.63% |
| constraints rejected as outliers | 8 total |

### Shell (stages 8-9) — 22 listings

| | value |
|---|---|
| glTF size | median 5168.0 bytes (max 6912) |
| triangles | median 64.0 |
| rooms in the shell | median 5.0 |
| over the 1 MB budget | 0 |

### Latency (M12) — 159 runs

| stage | p50 | p95 | max |
|---|---|---|---|
| 0-triage | 4.396 | 10.037 | 117.885 |
| 1-conditioning | 0.0 | 0.0 | 0.0 |
| 2-grouping | 0.0 | 0.0 | 0.0 |
| 3-geometry | 76.996 | 94.794 | 422.039 |
| 4-layout | 2.244 | 4.089 | 5.082 |
| 5-plan | 3.811 | 59.062 | 237.997 |
| 6-assembly | 0.044 | 0.066 | 0.078 |
| 7-scale | 0.001 | 0.002 | 0.002 |
| 8-shell | 0.004 | 0.006 | 0.007 |
| 9-package | 0.001 | 0.002 | 0.002 |
| **end to end (full runs, n=33)** | **87.627** | 149.084 | — |

### QA flags, by how many listings raised them

| flag | listings |
|---|---|
| `not_to_scale_disclaimer` | 24 |
| `room_area_disagrees_with_printed` | 20 |
| `low_assignment_margin` | 20 |
| `scale_candidates_disagree` | 20 |
| `no_doors_detected` | 17 |
| `under_80pct_plausible_ceilings` | 15 |
| `unmatched_plan_polygons` | 13 |
| `plan_area_disagrees_with_stated` | 12 |
| `cross_model_scale_disagreement` | 11 |
| `rooms_omitted_from_shell` | 11 |
| `unmatched_reconstructed_rooms` | 11 |
| `large_regularisation_snap` | 10 |
| `scale_constraints_disagree` | 9 |
| `room_dominates_plan` | 8 |
| `multiple_plan_outlines` | 8 |
| `scale_constraint_rejected` | 7 |
| `self_consistency_outside_10pct` | 6 |
| `poor_polygon_fit` | 4 |
| `no_room_captions` | 4 |
| `no_room_labels` | 4 |
