from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box


def write_reviewed_region_index(records: list[dict], decisions: dict, output_path: Path) -> dict:
    rows = []
    crs = None
    for record in records:
        decision = decisions.get(record.get("sample_id"))
        if record.get("review_group") != "geometry" or not decision:
            continue
        source = Path(record.get("source_paths", {}).get("probability", ""))
        bounds = record.get("window_bounds") or []
        if not source.is_file() or len(bounds) != 4:
            continue
        try:
            with rasterio.open(source) as src:
                window = Window(float(bounds[0]), float(bounds[1]), float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1]))
                geometry = box(*window_bounds(window, src.transform))
                crs = src.crs
            rows.append({
                "sample_id": record["sample_id"],
                "review_group": "geometry",
                "verdict": decision.get("verdict", ""),
                "review_purpose": "quality_assurance",
                "label_source": "human_reviewed_quality_assessment",
                "ground_truth_status": "not_pixel_ground_truth",
                "geometry": geometry,
            })
        except Exception:
            continue
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows and crs:
        gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_file(output_path, layer="reviewed_regions", driver="GPKG")
        status = "written"
    else:
        status = "no_geometrically_resolvable_reviewed_regions"
    return {"path": str(output_path), "status": status, "count": len(rows)}
