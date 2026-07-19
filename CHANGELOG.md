# Changelog

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
