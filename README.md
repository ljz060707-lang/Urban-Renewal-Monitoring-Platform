# Urban Renewal Phase 1 Starter — v0.3 China Real-data Hardening Edition

A small-area, high-resolution, multi-temporal urban-renewal monitoring starter.

The first phase is deliberately limited to a **reviewable high-resolution teacher system**:

1. validate several 0.3–2 m images of one small AOI;
2. train/fine-tune a building instance-segmentation teacher;
3. regularize model masks into QGIS-friendly building polygons;
4. match buildings through time with off-nadir-tolerant geometry metrics;
5. detect `stable`, `new_building`, `demolished`, `construction_candidate`, and uncertain states;
6. record many-to-one / one-to-many parent-child lineage;
7. review object state, date-specific image quality, and lineage in a browser;
8. export only MMU-qualified events as fractional 10 m labels for a later student model.

This repository does **not** bundle satellite imagery, commercial data, third-party weights, or copied external repository code. It provides clean-room interfaces and a runnable local pipeline.

## Why this phase comes first

Do not begin with a nationwide end-to-end Sentinel temporal model. First create a reliable high-resolution object-time database in one AOI. The reviewed data then supports:

- a second teacher such as RSBuilding;
- a later ContUrbanCD-style temporal model;
- robust lower-resolution label distillation;
- defensible urban-renewal event statistics.

## v0.3 hardening

v0.3 keeps the v0.2 teacher/tracking/review interfaces and adds the trust boundaries needed for a real Chinese-city AOI:

- preserve source CRS, select local UTM, and reject EPSG:4326/3857 metric work;
- block GCJ-02, BD-09 and unknown vectors from training until verified conversion;
- validate v0.3 schemas, Chinese paths and UTF-8 BOM CSV;
- diagnose ground alignment separately from high-rise roof displacement;
- compose manual > provider > RGB > default quality evidence;
- record shape similarity, ambiguity, deterministic IDs and complex lineage;
- preserve automatic results under review overlays with reviewer provenance;
- emit MMU exclusions, fractional labels, GeoTIFF tags and honest evaluation states;
- isolate every smoke run and record input hashes in `RUN_MANIFEST.json`.

## V2 improvements

The robust-tracking edition adds the safeguards required for real high-resolution imagery:

- **off-nadir robustness:** raw IoU, buffered IoU, bounding-box IoU, centroid distance and area similarity are jointly weighted;
- **dynamic displacement buffer:** optional `off_nadir_deg` metadata expands the matching tolerance when viewing angles differ;
- **canonical tracking geometry:** a `tracking_geometry_wkt` or `canonical_geometry_wkt` field may override the visible roof geometry for matching;
- **polygon regularization:** conservative simplification and orthogonalization before GPKG export;
- **seasonal gap logic:** bounded missing runs are treated as possible vegetation/shadow occlusion rather than immediate demolition;
- **construction transition logic:** a missing interval followed by a materially different geometry becomes `construction_candidate`;
- **parent-child lineage:** explicit `merge_redevelopment` and `split_or_expansion` events; many-to-one children receive a new track ID;
- **quality-mask generator:** conservative RGB heuristics plus human date-level quality override;
- **enhanced review UI:** per-date `usable / unusable / occluded / registration_issue` controls and lineage confirmation;
- **MMU filtering:** only sufficiently large and temporally dated events are exported to 10 m fractional labels.

## Reference architecture

- **ZAsolar pattern:** COCO export, Mask R-CNN fine-tuning, tiled inference, hard-negative review and GeoPackage output.
- **solar_backdating pattern:** explicit usable/unusable observations, monotonicity checks and latest-absent / earliest-present intervals.
- **RSBuilding:** optional external building/change teacher.
- **ContUrbanCD:** phase-2 temporal feature model after local temporal labels are adequate.
- **SpaceNet 7:** mask-to-polygon conversion, cross-date identity and temporal evaluation concepts.

See `SOURCES_AND_ADAPTATION.md`, `GROUND_TRUTH_SPEC.md` and `PHASE1_IMPLEMENTATION.md`.

## Data layout

```text
data/<aoi_id>/
├── manifest.csv
├── images/
│   ├── 2022-04-01.tif
│   └── ...
├── labels/
│   ├── 2022-04-01.gpkg
│   └── ...
└── quality/
    ├── 2022-04-01.tif
    └── ...
```

Recommended `manifest.csv`:

```csv
date,image_path,label_path,quality_mask_path,split,source,off_nadir_deg,sun_elevation_deg,season,quality_notes
2022-04-01,images/2022-04-01.tif,labels/2022-04-01.gpkg,quality/2022-04-01.tif,train,GF-2,6.2,51.0,spring,
2023-08-01,images/2023-08-01.tif,labels/2023-08-01.gpkg,quality/2023-08-01.tif,train,WorldView-3,20.1,66.0,summer,thin haze in northwest
```

All images must use the same projected CRS, extent, pixel size, width and height.

## Geometry policy

Do not silently mix roof and ground-footprint definitions.

Recommended two-geometry contract:

- active `geometry`: the visible per-date roof/object polygon used for segmentation supervision;
- optional `tracking_geometry_wkt`: a manually corrected canonical ground footprint or stable tracking anchor.

When the optional canonical field exists, the tracker uses it for matching while retaining the observed roof geometry for visual review and per-date measurement.

## Environment

Recommended: Linux/WSL2, Python 3.10–3.11, CUDA GPU with at least 12 GB VRAM.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Full smoke test

```bash
python scripts/run_phase1_smoke.py --config configs/example.yaml
```

The unified command is CPU-only and does not import Torch or download weights. It writes `outputs/<aoi_id>/runs/<run_id>/`. The legacy step-by-step commands remain compatible:

```bash
python scripts/make_synthetic_example.py --output data/example/demo_aoi
python scripts/validate_alignment.py --config configs/example.yaml
python scripts/build_coco.py --config configs/example.yaml
python scripts/make_mock_predictions.py --config configs/example.yaml
python scripts/regularize_predictions.py --config configs/example.yaml
python scripts/track_states.py --config configs/example.yaml
python scripts/build_review_queue.py --config configs/example.yaml
python scripts/export_student_labels.py --config configs/example.yaml
```

The synthetic example contains an off-nadir roof shift and a two-buildings-to-one redevelopment event.

Open:

```text
outputs/demo_aoi/review/index.html
```

## Real-data workflow

### 1. Validate alignment

```bash
python scripts/validate_alignment.py --config configs/phase1.yaml
```

Inspect difficult high-rise zones manually even when ground roads align. Ground registration does not guarantee roof alignment across viewing angles.

### 2. Generate or import quality masks

```bash
python scripts/generate_quality_masks.py --config configs/phase1.yaml --update-manifest
```

The generated mask is conservative and should be combined with provider QA and manual review.

### 3. Export COCO and train

```bash
python scripts/build_coco.py --config configs/phase1.yaml
python scripts/train_teacher.py --config configs/phase1.yaml
```

### 4. Infer and regularize polygons

```bash
python scripts/infer_timeseries.py --config configs/phase1.yaml
```

Regularization is integrated into inference. To process pre-existing predictions separately:

```bash
python scripts/regularize_predictions.py --config configs/phase1.yaml
```

### 5. Track, infer states and detect lineage

```bash
python scripts/track_states.py --config configs/phase1.yaml
```

Outputs include:

```text
tracking/tracks_long.gpkg
tracking/tracks_summary.gpkg
tracking/state_intervals.csv
tracking/lineage_events.gpkg
tracking/lineage_events.csv
```

### 6. Human review

```bash
python scripts/build_review_queue.py --config configs/phase1.yaml
```

The page allows:

- final track-state selection;
- date-specific image-quality overrides;
- explicit confirmation/rejection of merge/split lineage;
- notes for geometry edits and roof/footprint inconsistency.

Save the exported CSV at `review.review_csv`, then run:

```bash
python scripts/apply_reviews.py --config configs/phase1.yaml
```

To make reviewed date-quality flags affect state inference, rerun `track_states.py` after saving the review CSV.

### 7. Export later student labels

```bash
python scripts/export_student_labels.py --config configs/phase1.yaml
```

Only events above `student_export.min_mapping_unit_m2` and with a defensible transition date are exported. Output bands are:

1. `new_fraction`
2. `demolition_fraction`
3. `construction_fraction`
4. `any_change_fraction`

## Initial operational states

- `stable`: persists through the usable sequence;
- `new_building`: absent before, then persists for the required number of usable dates;
- `demolished`: present before, then remains absent for the required number of usable dates;
- `construction_candidate`: strong area/geometry transition or a repaired non-monotonic gap with materially different pre/post geometry;
- `uncertain`: unresolved non-monotonicity, poor quality, low confidence or incompatible geometry.

A separate `temporal_flag` records `seasonal_occlusion_or_shadow`, `nonmonotonic_redevelopment_transition`, or other review priorities.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The current tests cover interval inference, off-nadir displacement, seasonal gap repair, construction transitions, merge lineage and polygon regularization.

The v0.3 suite additionally covers schema, CRS, coordinate encodings, Chinese paths, alignment diagnostics, mask priority, geometry preservation, ambiguity, deterministic IDs, complex lineage, review idempotency, MMU boundaries, fractional labels and evaluation behavior.

## First real Chinese-city AOI

1. Create `data/<ascii_aoi_id>/images`, `labels`, and optional `quality` folders.
2. Place 3–8 legally obtained 0.3–2 m GeoTIFF/COG orthophotos in `images`.
3. Inspect the true CRS of every image; never assign EPSG:4326 merely to silence missing metadata.
4. Create a v0.3 `manifest.csv` using `configs/phase1.yaml` and `GROUND_TRUTH_SPEC.md`.
5. Run `validate_schemas.py`, `validate_crs.py`, and `diagnose_alignment.py` with the AOI config.
6. Select a near-nadir, well-covered reference grid and reproject working copies; preserve originals.
7. Generate/combine provider, manual and RGB quality evidence.
8. Draw or correct visible-roof Ground Truth in QGIS and record `geometry_role`.
9. Export COCO; install the optional Torch stack and train only after reviewing data/weight rights.
10. Infer, regularize, track, detect lineage and infer states.
11. Open `review/index.html`, enter reviewer ID, review state/date quality/lineage, and export CSV.
12. Correct geometry in QGIS, apply review overlays, rerun tracking/state, export MMU fractions and evaluate.

GCJ-02/BD-09 web maps may be visual references only. Their coordinates cannot enter formal training or metric analysis without a separately verified, versioned conversion.
