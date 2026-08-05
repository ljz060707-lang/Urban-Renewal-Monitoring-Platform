"""Canonical run layout and manifest; absent modules remain pending/unsupported."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


RASTERS = (
    "any_change_probability.tif", "new_building_probability.tif", "demolition_probability.tif",
    "vacant_land_probability.tif", "construction_probability.tif", "building_probability.tif",
    "building_mask.tif", "building_boundary.tif", "uncertainty.tif",
)
VECTORS = ("new_building.gpkg", "demolition.gpkg", "construction.gpkg", "buildings.gpkg", "urban_renewal_events.gpkg")
VISUALS = ("overview.png", "change_map.png", "semantic_map.png", "object_map.png", "temporal_map.png")


def initialize_inference_run(scene: str, run_id: str, *, output_root: str | Path = "outputs/inference", capabilities: set[str] | None = None) -> Path:
    root = Path(output_root) / scene / run_id
    for name in ("rasters", "vectors", "temporal", "visualization"):
        (root / name).mkdir(parents=True, exist_ok=True)
    capabilities = capabilities or set()
    products = {}
    for filename in RASTERS:
        key = filename.removesuffix(".tif")
        supported = key in capabilities
        products[key] = {
            "status": "pending" if supported else "unsupported",
            "path": None,
            "expected_path": str(root / "rasters" / filename) if supported else None,
        }
    for filename in VECTORS:
        key = filename.removesuffix(".gpkg")
        products[key] = {"status": "pending", "path": None}
    for filename in VISUALS:
        products[filename.removesuffix(".png")] = {"status": "pending", "path": None}
    manifest = {
        "schema_version": "1.0", "scene": scene, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "products": products,
        "terminology": "All detected objects are candidates unless independently reviewed.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root
