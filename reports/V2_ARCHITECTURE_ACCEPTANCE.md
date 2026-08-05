# Urban Renewal V2 acceptance

## Executed evidence

- `python run.py urban-renewal-preflight` succeeded. CUDA device: NVIDIA GeForce RTX 4060 Laptop GPU (8,585,216,000 bytes).
- `python run.py change-preflight --config configs/change_public_v2_multisource_windows.yaml` succeeded with `READY_TO_TRAIN=true`.
- `python run.py change-all --config configs/change_public_v2_multisource_windows.yaml --device cuda:0 --quick` succeeded as run `20260726_171450`.
- Relevant V2 tests: `12 passed`.

## Data and supervision safety

| source | development records discovered | V2 heads | test images read |
|---|---:|---|---:|
| DSIFN | 3,940 train/validation | any-change | 0 |
| S2Looking | 4,000 train/validation | any-change, new, demolition | 0 |
| SECOND | 2,968 train | any-change, semantic T1/T2 | 0 |

SECOND has `mapping_verified=false`, `building_class_id=null`, and no inferred building mapping. Its `0..6` semantic labels are routed only to semantic heads; directional building losses remain disabled. The frozen SECOND holdout protocol remains metadata-only with `image_contents_read=false`.

## Teacher and merge evidence

- DSIFN source SHA256: `d458e9b6bed969fcf79a38603608834b5bcafd8c055ba6b61ffcba8387819210`.
- S2Looking source SHA256: `4fd4c10cc8ceb0ce66c10c7e1f6bde4902404ff465a01423bdf4585653a6978c`.
- Merge report: `outputs/pretraining/change_v2/preflight/CHECKPOINT_MERGE_REPORT.json`.
- Merge copies 290 DSIFN tensors, 4 S2Looking directional-head tensors, and retains 4 semantic-head tensors as random initialization. No checkpoint averaging occurs.
- RSBuilding is `external_ready`: repository and Swin-T SHA256 are present; its Python dependencies are not imported into this environment.

## Quick run evidence

Quick selected 32 DSIFN, 32 S2Looking, and 32 SECOND train samples; validation used 8 DSIFN and 8 S2Looking samples only. One V2 masked-multitask epoch completed with total loss `4.659965`. The log recorded binary loss `1.694539`, new loss `1.280420`, demolition loss `1.235976`, peak CUDA allocation `903,654,912` bytes, and no NaN/Inf failure.

Validation is smoke evidence only, not a replacement for full validation: binary F1 `0.302404`, new F1 `0.0`, demolition F1 `0.0`; semantic validation is unavailable because SECOND has no configured validation partition. No model selection or threshold calibration used test.

## Remaining intentional limits

- `READY_FOR_V2_FULL=false` until an auditable SECOND building-class mapping is supplied.
- `READY_FOR_FULL_FUSION=false` until an RSBuilding scene export is supplied for a concrete inference scene.
- `urban-renewal-infer` is registered for PNG/JPG/GeoTIFF inputs, tiled inference, raster/vector/preview manifests. It was not executed here because no user-selected T1/T2 scene was provided.
