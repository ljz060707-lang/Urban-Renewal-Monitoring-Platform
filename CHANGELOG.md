# Changelog

## V2 multi-teacher architecture

- Added explicit teacher registry and dataset × head supervision routing.
- Added checkpoint bootstrap merge with tensor-level provenance checks.
- Added partial fusion, tiled GeoTIFF inference, alignment, vectorization, preview and MUDI-ready features.
- SECOND mapping pending now preserves semantic supervision while blocking directional labels.
- Added unified development test-read guard and train-only V2 hard-example mining.
- Added `urban-renewal-preflight` and `urban-renewal-infer`.
- Verified one GPU Quick run without reading test data; V2 Full remains intentionally unrun.

## 0.5.1 - Training-readiness hardening

- Added public and personal preflight commands, Windows paths/configuration and helper scripts.
- Isolated validation from test, made quick mode sample-bounded, and skip unsupervised public stages explicitly.
- Added external personal `--data-root` support and deterministic GeoPackage `buildings` layer selection.

## 0.5.0 - 2026-07-19

- Added independent S2Looking, SECOND and DSIFN adapters with a unified missing-supervision-aware index.
- Added MultiTask Siamese U-Net with shared ResNet34 encoder and binary, directional, semantic and uncertainty heads.
- Added masked multi-task losses, dataset-balanced sampling, synchronized pair augmentation and staged binary/semantic/directional training.
- Added strict CUDA gates, AMP, GradScaler, gradient clipping, bounded auto-batch, DDP, resume and deduplicated best/last/epoch checkpoints.
- Added per-dataset metrics, predictions, error sets, plots, static training dashboard and offline review UI.
- Added optional Gradio review, correction precedence, review-aware retraining and local AOI candidate labels/fine-tuning schedule.
- Added double-teacher fusion, joint review and rejected-object exclusion from student fusion labels.
- Preserved all v0.4 building-instance commands and tests.
- Actual local evidence: 113 base tests passed; 3 train smoke tests passed; synthetic CPU three-stage demo completed. GPU smoke and real public-data training were not completed because the active Torch build is CPU-only and dataset paths are absent.

## 0.4.0 - 2026-07-19

- Added `python run.py all --aoi <aoi_id>` and the `doctor`, `prepare`, `train`, `evaluate`, `predict`, `track`, and `demo` commands.
- Added automatic image/label discovery, date parsing, manifest generation and optional per-AOI `project.yaml`.
- Added geometry-only Ground Truth normalization, field aliases, make-valid, clipping, rejection/filter reports and stable object IDs.
- Added automatic local UTM analysis grids, aligned working copies, spatial block holdout and deterministic train/val COCO with RLE holes.
- Added real Mask R-CNN CPU/CUDA training, AMP, gradient clipping, best/last/epoch checkpoints, resume, early stopping and validation inference.
- Added tiled full-image inference with checkpoint hashes, raw/regularized/WGS84 outputs and polygon NMS.
- Fixed reviewed state priority and rejected-object exclusion in student export.
- Replaced unbounded tracking/lineage comparisons with spatial-index candidate generation and bidirectional ambiguity margins.
- Added full observation statuses, configurable structured confidence, preflight, stage hashes, cache reuse and safe `latest.json` updates.
- Added a network-free one-epoch CPU demo; final validation is 90 base tests plus one real Torch CPU training smoke.

## 0.3.0 - 2026-07-18

- Added schema, CRS, coordinate-encoding and provenance contracts.
- Added local UTM selection, alignment diagnostics and quality-mask composition.
- Added shape similarity, ambiguity, deterministic IDs and connected-component lineage.
- Preserved MultiPolygon, courtyard and L-shaped geometry during regularization.
- Added reviewer provenance, MMU exclusions, GeoTIFF tags and honest evaluation states.
- Added CPU-only isolated smoke workflow and expanded tests from 6 to 51.

## 0.2.0

- Robust tracking edition with off-nadir buffers, short-gap repair, lineage candidates, review UI and fractional student labels.
