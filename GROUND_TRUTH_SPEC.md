# Ground-truth and review specification

## 1. Annotation unit

The primary annotation unit is a building object at one acquisition date. The temporal unit is a track or lineage event.

## 2. Required GPKG fields

| Field | Type | Rule |
|---|---|---|
| `object_id` | string | Persistent manual ID when known; blank allowed before temporal review |
| `class_name` | string | Initially `building`; avoid forcing construction into the completed-building class |
| `geometry_role` | string | `visible_roof`, `ground_footprint`, or `canonical_tracking_anchor` |
| `review_status` | string | `draft`, `reviewed`, `rejected` |
| `source_date` | date/string | Acquisition date |
| `notes` | string | Shadow, canopy, off-nadir or ambiguity notes |
| `tracking_geometry_wkt` | string, optional | Canonical geometry used for matching |
| `source_image` | string | Source image or manifest reference |
| `model_version` | string | Model/checkpoint identity |
| `schema_version` | string | `0.3.0` for this release |
| `coordinate_encoding` | string | `wgs84`, `cgcs2000`, `projected`, `gcj02`, `bd09`, or `unknown` |

## 3. Visible roof versus footprint

Use `visible_roof` as the default segmentation target when the image is oblique. Do not invent a ground footprint by shifting the roof without defensible height/view geometry.

Create a canonical tracking anchor when:

- the building is tall;
- roof displacement changes substantially between dates;
- raw roof polygons have little overlap despite obvious identity;
- a trusted cadastral/footprint source exists.

## 4. Completed building

Include the permanent roof/object surface. Exclude:

- cast shadows;
- vegetation;
- temporary coverings;
- cranes and scaffolding;
- adjacent parking or hardscape.

## 5. Construction

During Phase 1, do not force excavation or scaffolding into a completed-building mask. Mark the temporal object as `construction_candidate` and describe the visual evidence. A dedicated construction/bare-ground branch can be trained later.

## 6. Demolition

Confirm demolition only when:

- the image is usable at the object location;
- the prior building position is visible;
- absence persists or is corroborated by a later date;
- the result is not explained by registration shift, tree canopy, shadow or missing coverage.

## 7. Merge/redevelopment

For several old buildings replaced by a larger new object:

- retain each old parent track to its last valid observation;
- create/retain the child track;
- confirm a `merge_redevelopment` lineage event;
- do not arbitrarily assign the child as the continuing identity of only one parent in analytical outputs.

## 8. Split/partial reconstruction

For one old building becoming several new objects or a partially retained footprint:

- retain observed child tracks;
- confirm or reject the proposed lineage;
- note whether the event is physical change or segmentation correction.

## 9. Date-quality review

Per object and date, select:

- `usable`: absence/presence may support temporal inference;
- `unusable`: no valid judgment;
- `occluded`: canopy/shadow obstruction;
- `registration_issue`: object location cannot be compared reliably.

Only `usable` dates count as valid absence evidence.

## 10. Holdout discipline

Keep at least one spatial block untouched by hard-negative mining and iterative geometry correction. Report both building segmentation performance and event-level new/demolition precision on this block.

## 11. China coordinate encodings

- Formal satellite/orthophoto geometry follows its file CRS.
- `gcj02`, `bd09`, and `unknown` annotation geometry is review-only until a verified conversion is recorded.
- Never use `set_crs("EPSG:4326")` to disguise GCJ-02 or BD-09 as WGS84.
- Preserve conversion method, version and timestamp as provenance.

## 12. Review overlay fields

Review CSV/GPKG overlays support `reviewer`, `review_time`, `review_version`, Chinese `notes`, date-level quality JSON and lineage actions. Automatic `state`, confidence and geometry remain unchanged; the effective reviewed decision is written as `state_final`.
