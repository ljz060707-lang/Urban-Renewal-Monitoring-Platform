# v0.3 China real-data hardening summary

This revision converts the robust-tracking scaffold into an auditable Chinese-city small-AOI teacher-data system.

## Implemented

1. Off-nadir-aware matching with dynamic buffers and multi-metric costs.
2. Optional canonical tracking geometry through WKT fields.
3. Conservative polygon regularization and a standalone reprocessing command.
4. Multi-epoch soft-gap repair with occlusion versus construction discrimination.
5. Explicit merge/redevelopment and split/expansion lineage outputs.
6. Conservative RGB quality-mask generation with reason codes.
7. Date-specific quality controls in the browser review UI.
8. Lineage confirmation/rejection in the review UI.
9. MMU-filtered fractional 10 m label export.
10. Synthetic stress test containing roof displacement and many-to-one redevelopment.
11. CRS preservation, local UTM selection and EPSG:4326/3857 metric-operation rejection.
12. GCJ-02/BD-09/unknown training-input blocking.
13. v0.3 schema validation, UTF-8 BOM CSV and Chinese path tests.
14. Ground/roof alignment diagnostics and reference-grid reprojection.
15. Manual/provider/RGB quality-mask composition with stable reason codes.
16. Shape-aware ambiguity, deterministic IDs and complex lineage.
17. Idempotent review overlays with reviewer provenance.
18. MMU exclusions, GeoTIFF tags and explicit `not computable` evaluation.
19. CPU-only isolated smoke runs with input hashes.

## Remaining deliberate limitations

- RGB heuristics cannot reliably identify all thin cloud, haze or shadow.
- Building-only Mask R-CNN is not a construction-site detector.
- Canonical ground footprints still require trusted vectors or manual work.
- Lineage is a candidate graph and needs review.
- The 10 m exporter does not make micro-renewal visible; it intentionally filters it.
- GCJ-02/BD-09 conversion algorithms are deliberately not bundled.
- Real Mask R-CNN training/inference still requires optional Torch/CUDA and user-approved local weights.
- Synthetic segmentation metrics validate plumbing, not real-China generalization.
