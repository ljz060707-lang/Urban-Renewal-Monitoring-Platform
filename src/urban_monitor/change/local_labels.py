from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from urban_monitor.project_discovery import parse_date_from_filename


CAUTION_FLAGS = ["high_rise_roof_displacement", "shadow", "tree_occlusion", "registration_issue", "merge", "split", "construction_transition", "outline_difference", "segmentation_error"]


def _latest_run(repo: Path, aoi_id: str) -> Path:
    pointer = repo / "outputs" / aoi_id / "latest.json"
    if not pointer.is_file(): raise FileNotFoundError(f"Successful v0.4 run is required: {pointer}")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    run = Path(payload["run_directory"])
    if payload.get("status") != "success" or not run.is_dir(): raise ValueError("latest.json does not reference a successful run")
    return run


def _unmatched(source: gpd.GeoDataFrame, target: gpd.GeoDataFrame, iou_threshold: float, min_area: float):
    rows = []
    target_index = target.sindex
    for _, row in source.iterrows():
        geometry = row.geometry
        candidates = list(target_index.query(geometry, predicate="intersects")) if len(target) else []
        best_iou = 0.0
        for index in candidates:
            other = target.iloc[index].geometry
            union = geometry.union(other).area
            if union: best_iou = max(best_iou, geometry.intersection(other).area / union)
        uncovered = geometry.difference(target.iloc[candidates].geometry.union_all()) if candidates else geometry
        if best_iou < iou_threshold or uncovered.area >= min_area:
            item = row.to_dict(); item["match_iou"] = best_iou; item["difference_area_m2"] = float(uncovered.area)
            item["geometry"] = geometry if best_iou < iou_threshold else uncovered
            if not item["geometry"].is_empty and item["geometry"].area >= min_area: rows.append(item)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=source.crs) if rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=source.crs)


def _write_mask(path: Path, geometries, reference: Path):
    with rasterio.open(reference) as source:
        profile = source.profile.copy(); profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        mask = rasterize(((geometry, 1) for geometry in geometries if geometry is not None and not geometry.is_empty), out_shape=(source.height, source.width), transform=source.transform, fill=0, dtype="uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as target: target.write(mask, 1)


def generate_local_change_candidates(aoi_id: str, *, repo_root: str | Path, iou_threshold: float = 0.2,
                                     min_area_m2: float = 10.0, run: str | Path | None = None) -> Path:
    repo = Path(repo_root).resolve(); project = repo / "data" / aoi_id
    labels = {parse_date_from_filename(path.name): path for path in (project / "labels").iterdir() if path.suffix.casefold() in {".gpkg", ".geojson", ".shp"}}
    images = {parse_date_from_filename(path.name): path for path in (project / "images").iterdir() if path.suffix.casefold() in {".tif", ".tiff"}}
    dates = sorted(set(labels) & set(images))
    if len(dates) < 2: raise ValueError("change-labels requires at least two matched image/label dates")
    run = Path(run) if run else _latest_run(repo, aoi_id)
    root = run / "change_labels"; root.mkdir(parents=True, exist_ok=True)
    summary = []
    for first_date, second_date in zip(dates, dates[1:]):
        with rasterio.open(images[first_date]) as raster: analysis_crs = raster.crs
        if analysis_crs is None: raise ValueError(f"Image CRS is missing: {images[first_date]}")
        first = gpd.read_file(labels[first_date]); second = gpd.read_file(labels[second_date])
        if first.crs is None or second.crs is None: raise ValueError("Local building labels require CRS")
        first, second = first.to_crs(analysis_crs), second.to_crs(analysis_crs)
        new = _unmatched(second, first, iou_threshold, min_area_m2)
        demolition = _unmatched(first, second, iou_threshold, min_area_m2)
        pair = root / f"{first_date}_to_{second_date}"; pair.mkdir(parents=True, exist_ok=True)
        for frame, name in ((first, "T1_buildings"), (second, "T2_buildings"), (new, "GT_NEW_candidate"), (demolition, "GT_DEM_candidate")):
            output = frame.copy(); output["candidate_status"] = "requires_human_review" if "candidate" in name else "source_buildings"
            output.to_file(pair / f"{name}.gpkg", driver="GPKG")
        change = pd.concat([new.assign(change_type="new"), demolition.assign(change_type="demolition")], ignore_index=True)
        change = gpd.GeoDataFrame(change, geometry="geometry", crs=analysis_crs)
        change["candidate_status"] = "requires_human_review"; change.to_file(pair / "GT_CD_candidate.gpkg", driver="GPKG")
        for frame, name in ((first, "T1_building_mask"), (second, "T2_building_mask"), (new, "GT_NEW_candidate"), (demolition, "GT_DEM_candidate"), (change, "GT_CD_candidate")):
            _write_mask(pair / f"{name}.tif", frame.geometry, images[first_date])
        summary.append({"pair": f"{first_date}_to_{second_date}", "t1_date": first_date, "t2_date": second_date,
                        "new_candidates": len(new), "demolition_candidates": len(demolition), "status": "requires_human_review"})
    pd.DataFrame(summary).to_csv(root / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    review = run / "review" / "local_change_labels"; review.mkdir(parents=True, exist_ok=True)
    flags = "".join(f"<label><input type='checkbox' value='{flag}'>{flag}</label>" for flag in CAUTION_FLAGS)
    rows = "".join(f"<article><h2>{html.escape(row['pair'])}</h2><p>新建候选 {row['new_candidates']}，拆除候选 {row['demolition_candidates']}</p><p>{flags}</p><select><option>pending</option><option>confirmed</option><option>rejected</option><option>needs_edit</option></select><textarea placeholder='审核备注'></textarea></article>" for row in summary)
    page = f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>本地变化候选审核</title><style>body{{font:15px system-ui;max-width:1000px;margin:30px auto}}article{{border:1px solid #ccc;padding:18px;margin:12px}}label{{display:inline-block;margin:6px}}textarea{{display:block;width:100%;margin-top:10px}}</style><h1>本地变化候选审核</h1><p>候选不是 Ground Truth；仅 confirmed 项可进入微调。</p>{rows}</html>"
    (review / "index.html").write_text(page, encoding="utf-8")
    return root
