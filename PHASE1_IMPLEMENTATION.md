# Phase-1 implementation specification — robust high-resolution teacher

## 1. Scope and completion criterion

Use one AOI of roughly 0.5–5 km², 3–8 dates and 0.3–2 m imagery. The phase is complete when the system can produce a reviewed building-object time series containing:

- clean per-date building polygons;
- persistent object IDs where identity is meaningful;
- parent-child lineage where identity is not one-to-one;
- defensible first/last-seen intervals;
- explicit unusable-image observations;
- model and reviewer provenance;
- MMU-filtered labels that can later supervise a 10 m model.

The target is not yet a national urban-renewal index and not an end-to-end temporal foundation model.

## 2. Image acquisition and physical-risk control

### 2.1 Minimum sequence

Three dates are an absolute minimum; five or more are preferable. Include observations before, during and after a known change wherever possible.

### 2.2 Orthorectification is necessary but not sufficient

Ground roads can align while high-rise roofs remain displaced because of differing off-nadir angles and relief displacement. Therefore:

1. retain `off_nadir_deg`, source and acquisition metadata in `manifest.csv`;
2. avoid extreme-angle scenes where a suitable alternative exists;
3. test several high-rise buildings separately from ground control points;
4. use buffered and centroid-based matching rather than raw polygon IoU alone;
5. create a canonical ground-footprint/tracking geometry for difficult buildings when needed.

The tracker dynamically increases its buffer when off-nadir metadata differs between dates. The increase is capped to prevent unrelated nearby buildings from being matched.

### 2.3 Seasonal vegetation and shadow

The tracker may reconnect a building after one or two missing observations. A bounded `Present → Absent → Present` sequence is evaluated as follows:

- similar pre/post geometry: `seasonal_occlusion_or_shadow` flag, retaining stable identity;
- materially different pre/post area or buffered overlap: `construction_candidate`;
- long or unresolved gap: `uncertain`.

The threshold must be based on both number of epochs and elapsed days, because one missed monthly observation is different from a two-year gap.

## 3. Ground-truth geometry contract

### 3.1 Do not mix definitions

Every annotation campaign must declare one segmentation target:

- `visible_roof`: the roof/object boundary visible in the source image;
- `ground_footprint`: the building-ground contact boundary;
- `canonical_tracking_anchor`: a stable geometry used only for temporal identity.

The recommended phase-1 contract is:

- active GPKG `geometry` = per-date visible roof polygon;
- `geometry_role` = `visible_roof`;
- optional `tracking_geometry_wkt` = canonical footprint/anchor used for matching;
- `object_id` = known local ID when manually established;
- `review_status` = `draft`, `reviewed`, `rejected`.

Visible roofs are practical model targets. Canonical tracking geometry avoids forcing the model to infer a ground footprint that is not visually observable in oblique imagery.

### 3.2 Annotation rules

- Keep adjacent buildings separate when a visible boundary exists.
- Do not include shadows, trees, scaffolding or cranes in the completed-building polygon.
- Label construction sites separately in reviewer notes until a dedicated construction branch is trained.
- Record difficult roof/footprint disagreements rather than silently editing each date into an artificial common geometry.
- Use the best near-nadir image or trusted cadastral/building data to create canonical tracking geometry.

See `GROUND_TRUTH_SPEC.md` for field-level rules.

## 4. Quality-mask system

### 4.1 Sources

Priority order:

1. provider QA/UDM mask;
2. manual scene/region mask;
3. conservative RGB heuristic mask;
4. human per-track-date override in the review page.

The included generator detects only obvious no-data, bright cloud/glare, optional extreme dark areas, local blur and globally low contrast. It is not a substitute for multispectral cloud detection.

### 4.2 Inference rule

An absence is valid only when the usable fraction within the object/anchor geometry exceeds `tracking.min_quality_fraction`. The review page can override a date as:

- `usable`;
- `unusable`;
- `occluded`;
- `registration_issue`.

All non-usable categories are excluded from absence-based demolition or appearance inference.

## 5. Teacher model and vectorization

### 5.1 Practical first teacher

Use Mask R-CNN ResNet-50 FPN for building instances. Fine-tune on local polygons and hard negatives. A second RSBuilding/Open-CD teacher can be introduced after the local data contract works.

### 5.2 Polygon post-processing

Raw mask polygons are unsuitable for efficient QGIS editing. The pipeline therefore applies:

1. geometry repair;
2. topology-preserving simplification;
3. minimum-rotated-rectangle replacement for highly rectangular objects;
4. conservative dominant-axis edge snapping for more complex footprints;
5. area-change safeguards;
6. polygon NMS across overlapping inference chips.

Do not maximize orthogonality at the cost of destroying L-shaped, courtyard or irregular industrial buildings. The system retains the simplified original when regularization changes area excessively or produces invalid geometry.

## 6. Cross-date tracking

### 6.1 Matching score

For candidate polygons A and B, the assignment score combines:

- raw polygon IoU;
- buffered IoU;
- external bounding-box IoU;
- centroid proximity;
- area similarity.

The default weights intentionally place more emphasis on buffered overlap and centroid proximity than raw IoU. The matching gate accepts a pair when at least one spatial criterion is satisfied, then the Hungarian algorithm performs the one-to-one primary assignment.

### 6.2 Short-memory reconnection

A track remains eligible for reconnection for `max_track_gap` epochs and optionally `max_track_gap_days`. This prevents summer canopy, winter shadows or one failed model inference from creating a false demolition/new-build pair.

### 6.3 Canonical geometry

When `tracking_geometry_wkt` or `canonical_geometry_wkt` exists in the prediction/annotation table, the tracker uses that geometry for pair metrics while retaining the observed geometry in outputs and review chips.

## 7. Parent-child lineage

One-to-one assignment is not sufficient for urban renewal. After primary tracking, the system examines consecutive dates for:

### 7.1 Merge/redevelopment

Several former footprints are substantially absorbed by one current footprint. Output:

```text
event_type = merge_redevelopment
parent_track_ids = old_1;old_2;...
child_track_ids = newly assigned child track
```

Both parent absorption and child counterpart coverage are recorded. This covers demolition of several small buildings followed by a larger replacement.

### 7.2 Split/expansion

One former footprint maps spatially to several current footprints. Output:

```text
event_type = split_or_expansion
parent_track_ids = old_track
child_track_ids = child_1;child_2;...
```

These events are review objects, not automatic proof of physical subdivision. Segmentation errors and connected roofs can create the same pattern.

### 7.3 Identity rule

For confirmed many-to-one redevelopment candidates, the child receives a new track ID and the old parent tracks terminate. Lineage records the inheritance graph without pretending that one parent alone continued unchanged. Split candidates remain reviewable because one retained portion may or may not preserve identity.

## 8. State engine

### 8.1 Operational states

- `stable`;
- `new_building`;
- `demolished`;
- `construction_candidate`;
- `uncertain`.

### 8.2 Construction candidate

The building teacher is not assumed to recognize excavations, cranes or scaffolding. A construction candidate is generated when:

- a repaired non-monotonic gap has strongly different pre/post geometry;
- a new or continuing object displays substantial area change;
- a reviewer assigns construction;
- a lineage event indicates likely redevelopment.

A dedicated `construction / bare_land` branch should be added only after local labels are available.

### 8.3 Interval inference

New construction:

```text
interval_start = latest usable absent date
interval_end   = earliest persistent present date
```

Demolition:

```text
interval_start = latest persistent present date
interval_end   = earliest usable absent date
```

Do not report an exact date when only an interval is observed. Confidence decreases as the interval grows.

## 9. Human review interface

The review page shows all dates for each track. Red outlines indicate actual per-date detections; yellow outlines indicate the representative location when that date has no detection.

Reviewers can:

- choose the final track state;
- flag each date as usable/unusable/occluded/registration issue;
- confirm or reject merge/split lineage;
- note geometry corrections and roof/footprint disagreement.

Geometry edits remain in QGIS. State, temporal quality and lineage decisions remain in the review CSV/GPKG outputs.

Priority order:

1. lineage events;
2. non-monotonic tracks;
3. low-confidence changes;
4. high-rise/off-nadir zones;
5. large industrial roofs and dense old settlements;
6. random high-confidence audit sample.

## 10. Data flywheel

### Round 0

Manually draw complete buildings in representative chips across all selected scene types. Include difficult negatives.

### Round 1

Train, infer, regularize, track and review. Preserve a spatial holdout.

### Round 2

Add corrected polygons, date-quality labels and hard negatives. Retrain. Do not modify the holdout to hide errors.

### Round 3

Add a second teacher or a construction/land-surface branch only after error analysis shows that the data—not basic engineering—is the limiting factor.

## 11. Lower-resolution handoff

### 11.1 Minimum Mapping Unit

A 10 m pixel represents 100 m², so many high-resolution micro-updates are weak sub-pixel signals. The exporter therefore filters events below a configurable MMU, initially recommended around 200 m² for experimentation.

### 11.2 Fractional labels

Do not resample high-resolution binary labels as hard 10 m classes. Supersample and aggregate to:

- new-building fraction;
- demolition fraction;
- construction fraction;
- any-change fraction.

Events without a defensible transition date are not exported automatically. Undated high-resolution redevelopment remains in the review system until lineage or human evidence supplies timing.

### 11.3 Multi-scale task split

- high-resolution system: individual buildings, local demolition, small additions and micro-renewal;
- 10 m student: larger construction sites, building groups, neighborhood-scale demolition and expansion;
- 30 m historical products: long-run urban expansion and coarse screening only.

## 12. Acceptance gates before Phase 2

- all dates pass projected alignment checks;
- high-rise/off-nadir stress test is manually inspected;
- polygon regularization reduces editing burden without material area distortion;
- building segmentation is evaluated on a spatial holdout;
- new/demolished precision is measured only on usable dates;
- lineage event precision is audited separately;
- review CSV can alter date usability and rerun the state engine;
- every event contains geometry, dates, model version, confidence and review status;
- MMU and fractional-label export are validated against the high-resolution source.

## 13. Commands

```bash
python scripts/validate_alignment.py --config configs/phase1.yaml
python scripts/generate_quality_masks.py --config configs/phase1.yaml --update-manifest
python scripts/build_coco.py --config configs/phase1.yaml
python scripts/train_teacher.py --config configs/phase1.yaml
python scripts/infer_timeseries.py --config configs/phase1.yaml
python scripts/regularize_predictions.py --config configs/phase1.yaml  # optional if inference already regularized
python scripts/track_states.py --config configs/phase1.yaml
python scripts/build_review_queue.py --config configs/phase1.yaml
python scripts/apply_reviews.py --config configs/phase1.yaml
python scripts/track_states.py --config configs/phase1.yaml           # apply reviewed date-quality flags
python scripts/export_student_labels.py --config configs/phase1.yaml
```

## 14. v0.3 data contracts

- `schema.py` validates manifest, annotation, prediction, tracking, observation, lineage and review tables with `schema_version=0.3.0`.
- `crs.py` preserves source CRS, selects local WGS84 UTM by AOI centroid and rejects geographic/Web Mercator metric work.
- `coordinate_encoding.py` treats GCJ-02/BD-09 as encodings rather than fake EPSG values and blocks unsafe training input.
- `alignment_diagnostics.py` separates adjacent-date ground shift from roof displacement.
- `quality.py` composes manual, provider and RGB evidence into `usable` plus `reason_code`.
- Tracking stores raw/buffer/bbox IoU, centroid, area, shape, ambiguity margin and deterministic IDs.
- Bipartite connected components support merge, split and complex lineage.
- Review overlays never replace automatic state columns or source annotations.
- Student exports remain fractional, never invent missing dates and emit explicit exclusion inventories.
- Evaluation writes `not computable` for unavailable task-level Ground Truth.

## 15. Reproducibility

`python scripts/run_phase1_smoke.py --config configs/example.yaml` creates an isolated `outputs/<aoi_id>/runs/<run_id>` directory and records config/input hashes, environment, CRS decisions, status and output inventory in `RUN_MANIFEST.json`.
