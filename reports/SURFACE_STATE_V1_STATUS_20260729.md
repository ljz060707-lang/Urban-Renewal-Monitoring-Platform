# SurfaceState v1 extension status

## Actual results

| Item | Status | Evidence |
|---|---|---|
| CWLD label definition | RESOLVED | `docs/third_party/CWLD_LABEL_DEFINITION.md` |
| CWLD codec | PASS | Train-only colour-distance audit; anchors are black, white, blue, red; engineering facility remains distinct from construction site. |
| CWLD smoke | PASS | 20 CUDA optimizer steps, finite mean loss `1.3329783380`; checkpoint is smoke-only. |
| CWLD full training | NOT RUN | Extension currently has no full-epoch evaluator/checkpoint-selection workflow. |
| CWLD full Validation | NOT RUN | No formal model was selected. |
| FUSU package | SAMPLE/PARTIAL | 500 paired samples vs expected 62,752; formal training is disabled. |
| FUSU formal split | BLOCKED | No official AOI/city/Train-Validation metadata; filename grid is not a formal validation split. |
| FUSU sample smoke | PASS | Two real sample images completed TFR/change-head backward and MTI forward. |
| SurfaceState dual-head code | READY | New independent shared-encoder dual-head extension. |
| Joint formal training | BLOCKED | Incomplete FUSU package. |
| TFR implementation / smoke | READY / PASS | T=2 real FUSU and T=4 synthetic missing-time smoke. |
| MTI implementation / smoke | READY / PASS | Explainable two-state temporal integration smoke. |
| MTI formal calibration | BLOCKED | No continuous building-state ground truth. |
| Master overlay | READY | Non-invasive manifest only; not a main-model merge. |
| Main run.py integration | PENDING | Existing-file patch capability remains unavailable. |
| Full20 hash unchanged | PASS | `70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621` |
| RSBuilding hash unchanged | PASS | `f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7` |
| WHU Test reads / SECOND Test reads | `0 / 0` | Overlay does not enumerate either dataset. |
| Inference ready | FALSE | No selected full-validation SurfaceState checkpoint. |

## Provenance and scope

The extension is `non_invasive_overlay`: it does not edit `run.py`, existing registries, frozen Full20, or RSBuilding. CWLD labels are candidates for the stated four native classes only. `engineering_facility_candidate` is never asserted as construction-site ground truth.
