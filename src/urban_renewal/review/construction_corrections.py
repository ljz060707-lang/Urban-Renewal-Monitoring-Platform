from __future__ import annotations

from pathlib import Path


def export_qgis_corrections(review_dir, inventory):
    """Export existing RSBuilding candidates for QGIS; never invent empty geometry."""
    import geopandas as gpd
    import pandas as pd

    source_paths = []
    for record in inventory.get("records", []):
        for date, sources in record.get("source_paths", {}).items():
            path = Path(sources.get("gpkg", ""))
            if path.is_file() and path not in source_paths:
                source_paths.append(path)
    if not source_paths:
        raise FileNotFoundError("no audited building candidate GeoPackage is available for QGIS export")
    frames = []
    for path in source_paths:
        frame = gpd.read_file(path, layer="buildings")
        frame["geometry_source"] = "rsbuilding_candidate"
        frame["boundary_review"] = "pending"
        frame["construction_track_id"] = None
        frame["label_source"] = "model_prediction"
        frame["scientific_status"] = "candidate_not_ground_truth"
        frames.append(frame)
    crs = frames[0].crs
    if any(frame.crs != crs for frame in frames):
        raise ValueError("candidate GeoPackages use different CRS; export stopped without resampling")
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    target = Path(review_dir).resolve() / "construction_activity" / "REVIEW_CORRECTIONS.gpkg"
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(target, layer="building_candidates_for_review", driver="GPKG")
    return {"status": "success", "path": str(target), "feature_count": len(merged), "layer": "building_candidates_for_review", "scientific_status": "candidate_not_ground_truth"}
