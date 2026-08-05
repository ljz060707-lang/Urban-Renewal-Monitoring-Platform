# SurfaceState v1 training acceptance — 2026-07-29

## Current status

| Item | Status | Evidence |
|---|---|---|
| CWLD full training | RUNNING | `outputs/training/surface_state_v1/cwld_20260729_165008` |
| CWLD full Validation | PARTIAL | Epoch 0 completed on all 731 Validation samples |
| CWLD best checkpoint | AVAILABLE, TRAINING NOT FINAL | epoch 0 best in the running directory |
| FUSU package scope | `current_500_pair_sample` | 500 paired samples |
| FUSU split mode | `grouped_sample_split`, weak isolation | 416 train / 84 validation; hash overlap 0 |
| FUSU sample baseline | MINI EPOCH PASS | `fusu_sample_20260729_165233` |
| FUSU sample TFR | MINI EPOCH PASS | `tfr_sample_20260729_165545` |
| FUSU formal 60-epoch training | NOT STARTED | waits for CWLD GPU job |
| Joint sample training | NOT STARTED | waits for preceding candidates |
| MTI calibration | PASS ON MINI CANDIDATE | 192-grid sample Validation calibration |
| Master overlay loading | PASS | actual FUSU candidate checkpoint loaded |
| Overlay inference smoke | PASS | `outputs/integration/surface_state_v1_overlay/20260729_165834` |
| Alignment guard | PASS | pair aligned; absent CRS recorded as pixel coordinates only |
| Main `run.py` integration | PENDING | workspace helper still cannot update existing files |
| Inference ready | `false` | Tongzhou alignment/numerical/manual review not completed |

## Actual metrics

- CWLD 512 epoch 0: macro mIoU `0.2516223`, macro F1 `0.3756749`, validation loss `1.7369520`.
- CWLD low-resolution full mini epoch: macro mIoU `0.2712356`, macro F1 `0.3873593`.
- FUSU baseline mini selection score: `0.1280565`.
- FUSU TFR mini selection score: `0.1150404`; semantic change F1 was `0`, so no TFR improvement claim is made.
- MTI selected: unary `2.0`, change `0.5`, persistence `1.0`, impossible transition `1.0`; selection score `0.3369905`.

## Scientific boundaries

- `dataset_status = sample_or_partial`
- `training_scope = current_500_pair_sample`
- `validation_scope = sample_spatial_holdout`
- `full_fusu_validated = false`
- `national_generalization_verified = false`
- `joint_training_status = allowed_on_current_sample`
- `joint_candidate_scope = sample_domain_only`
- `inference_ready = false`
- WHU Test reads `0`; SECOND Test reads `0`.

The extension never maps CWLD engineering-facility pixels to construction-site ground truth. It does not load RSBuilding in the project environment and does not alter Full20 or RSBuilding checkpoints.
