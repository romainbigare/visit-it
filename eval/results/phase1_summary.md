12/30 listings have been run (24 carry a floor plan). Split: 10 dev / 20 holdout (16 of the holdout carry a plan).


### Stage completion

| stage | ok | partial/skipped | failed |
|---|---|---|---|
| 0-triage | 8 | 0 | 0 |
| 1-conditioning | 8 | 0 | 0 |
| 2-grouping | 8 | 0 | 0 |
| 3-geometry | 8 | 0 | 0 |
| 4-layout | 11 | 0 | 0 |
| 5-plan | 12 | 0 | 0 |
| 6-assembly | 10 | 2 | 0 |
| 7-scale | 10 | 2 | 0 |
| 8-shell | 10 | 2 | 0 |
| 9-package | 10 | 2 | 0 |

### Plan channel (stage 5) — 12 listings

| | value |
|---|---|
| scale source | printed_dimensions 7, stated_area 3, none 2 |
| rooms found per listing | median 6.5 (p10 3.0, p90 15.5) |
| rooms carrying a label | 62% |
| doors found per listing | median 1.5 |
| adjacency edges per listing | median 4.0 |
| plan area / stated area | median 0.865 (n=7) |
| carry a 'not to scale' disclaimer | 58% |
| stage confidence | median 0.562 |

### Geometry and layout (stages 3-4) — 12 listings

| | value |
|---|---|
| rooms reporting a ceiling height | 65 |
| rooms where we refused to report one | 6 (floor_ceiling_gap_too_small 4, ceiling_or_floor_barely_observed 1, no_floor_ceiling_found 1) |
| ceiling height | median 2.51 m (p10 1.61, p90 2.84) |
| within 2.3-3.2 m | 69% |

### Assembly (stage 6) — 10 listings

| | value |
|---|---|
| rooms matched per listing | median 4.5 |
| reconstructed rooms left unmatched | median 1.0 |
| plan polygons left unmatched | median 2.5 |
| cost margin over the runner-up | median 0.026 |
| polygon fit (IoU after SE(2)) | median 0.498 |

### Scale (stage 7) — 10 listings

| | value |
|---|---|
| scale factor applied | median 1.101 (p10 0.94, p90 1.25) |
| solve quality | median 0.82 |
| residual RMS | median 13.021% |
| **self-consistency vs printed dimensions** | median 16.333% (n=7) |
| rooms within ±10% of their printed size | median 0.5 |
| cross-model scale disagreement | median 11.175% |
| constraints rejected as outliers | 4 total |

### Shell (stages 8-9) — 10 listings

| | value |
|---|---|
| glTF size | median 4884.0 bytes (max 6432) |
| triangles | median 61.0 |
| rooms in the shell | median 4.5 |
| over the 1 MB budget | 0 |

### Latency (M12) — 21 runs

| stage | p50 | p95 | max |
|---|---|---|---|
| 0-triage | 4.396 | 12.86 | 117.885 |
| 1-conditioning | 0.0 | 0.0 | 0.0 |
| 2-grouping | 0.0 | 0.0 | 0.0 |
| 3-geometry | 81.247 | 168.996 | 422.039 |
| 4-layout | 2.307 | 5.005 | 5.082 |
| 5-plan | 3.943 | 140.692 | 237.997 |
| 6-assembly | 0.054 | 0.066 | 0.071 |
| 7-scale | 0.001 | 0.001 | 0.002 |
| 8-shell | 0.005 | 0.006 | 0.006 |
| 9-package | 0.001 | 0.002 | 0.002 |
| **end to end** | **90.941** | 318.145 | — |

### QA flags, by how many listings raised them

| flag | listings |
|---|---|
| `not_to_scale_disclaimer` | 14 |
| `room_area_disagrees_with_printed` | 8 |
| `low_assignment_margin` | 8 |
| `unmatched_plan_polygons` | 8 |
| `scale_candidates_disagree` | 8 |
| `no_doors_detected` | 7 |
| `large_regularisation_snap` | 6 |
| `under_80pct_plausible_ceilings` | 6 |
| `rooms_omitted_from_shell` | 5 |
| `unmatched_reconstructed_rooms` | 5 |
| `scale_constraints_disagree` | 4 |
| `room_dominates_plan` | 4 |
| `plan_area_disagrees_with_stated` | 4 |
| `cross_model_scale_disagreement` | 3 |
| `multiple_plan_outlines` | 3 |
| `scale_constraint_rejected` | 2 |
| `poor_polygon_fit` | 2 |
| `no_room_captions` | 2 |
| `no_room_labels` | 2 |
| `self_consistency_outside_10pct` | 2 |
