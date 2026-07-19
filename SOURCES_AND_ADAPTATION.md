# Sources and adaptation decisions

## Directly relevant repositories

| Module | Repository | What to reuse conceptually | Phase-1 decision |
|---|---|---|---|
| High-resolution detection workflow | `Robertgao0818/ZAsolar` | region/image registry, COCO export, Mask R-CNN teacher, tiled inference, GPKG output, hard-negative review | Generic clean-room implementation; no PV-specific rules |
| Temporal back-dating | `Robertgao0818/solar_backdating` | anchor/date observations, usable flags, review queue, monotonic transitions, latest-absent / earliest-present intervals | Building state engine and interval contract |
| General building/change model | `Meize0729/RSBuilding` | optional pretrained building/change encoder; SECOND/S2Looking loaders | Optional second teacher after local pipeline validation |
| Continuous temporal model | `SebastianHafner/ContUrbanCD` | TFR, change features, multi-task consistency, SpaceNet 7 sequence loading | Phase 2 after sufficient reviewed temporal masks |
| Polygon tracking | `CosmiQ/CosmiQ_SN7_Baseline` | mask-to-polygon conversion, persistent IDs and temporal object concepts | Lightweight primary assignment plus explicit lineage graph |
| Winning tracking solutions | `SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions` | later comparison of tracking and post-processing strategies | Not imported into the first prototype |

## Important departures from the references

### 1. Roof displacement and off-nadir imagery

SpaceNet-style overlap tracking assumes relatively consistent image geometry. The current scaffold adds buffered IoU, bounding-box overlap, centroid proximity, area similarity and optional off-nadir metadata. A canonical WKT tracking geometry can be supplied separately from the visible roof polygon.

### 2. Urban renewal is not one-to-one tracking

The scaffold retains Hungarian assignment for ordinary persistence but adds a parent-child lineage graph. A many-to-one redevelopment candidate forces a new child track instead of allowing one arbitrary old building ID to continue through the replacement.

### 3. Mask polygons are regularized before review

The output geometry is repaired, simplified and conservatively orthogonalized. Regularization is rejected when it produces invalid geometry or excessive area change.

### 4. Construction remains a candidate state

A building-only teacher cannot reliably classify excavation, scaffolding or bare construction sites. The first phase identifies construction candidates from non-monotonic geometry and reviewer evidence rather than pretending that building masks alone solve the problem.

### 5. Lower-resolution labels are fractional and MMU-filtered

High-resolution events are not naively resampled into hard 10 m labels. Events below a configurable minimum mapping unit or without a defensible transition date are not exported automatically.

## PDF modules that are data rather than standalone models

- `SECOND` and `S2Looking` are pretraining/fine-tuning datasets through RSBuilding/Open-CD-compatible loaders.
- `SpaceNet 7` supplies monthly imagery, footprints and temporal IDs; it is both a dataset and a code baseline.
- `UBGGset` is suitable for a later blue-green-gray branch, not required for the first building-state prototype.
- Construction-waste or bare-land datasets may strengthen a later construction branch, but local manual labels should not be blocked on them.

## Licensing rule

This starter does not vendor third-party code, data or weights. Clone optional external repositories separately and interact through file contracts or adapters. Preserve original licenses and citations when using their code or weights.

## v0.3 implementation record

| Source repository | Reference surface | Idea used | Local implementation | Direct code migration |
|---|---|---|---|---|
| `Robertgao0818/ZAsolar` | registry/COCO/training/review | local manifest, tiled COCO, reviewable GPKG | existing clean-room modules extended with schema/CRS/provenance | No |
| `Robertgao0818/solar_backdating` | temporal observations | usable evidence and date intervals | `summarize_tracks()` plus review quality overlays | No |
| `Meize0729/RSBuilding` | SECOND/S2Looking interfaces | later second teacher boundary | COCO/GPKG contracts retained; no OpenMMLab install | No |
| `SebastianHafner/ContUrbanCD` | temporal organization | later feature handoff | fractional grids and event inventories only | No |
| `CosmiQ/CosmiQ_SN7_Baseline` | temporal IDs/evaluation | object tracking concepts | deterministic multi-metric assignment/evaluation | No |
| `SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions` | post-processing | parent-child identity | bipartite connected-component lineage | No |

No source file from these repositories was copied in v0.3. Original licenses and citations remain the responsibility of any separately installed optional adapter or weight.
