from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import mapping
import pandas as pd


def _mean(source, geometry) -> float | None:
    try:
        window = geometry_window(source, [mapping(geometry)], pad_x=0, pad_y=0)
    except Exception:
        return None
    data = source.read(1, window=window, masked=True)
    transform = source.window_transform(window)
    inside = ~geometry_mask([mapping(geometry)], out_shape=data.shape, transform=transform, invert=False)
    valid = inside & ~np.ma.getmaskarray(data)
    return float(np.asarray(data)[valid].mean()) if valid.any() else None


def _state(row) -> str:
    for field in ("state_final", "reviewed_state", "state"):
        value = row.get(field)
        if value is not None and str(value) not in {"", "nan", "None"}: return str(value)
    return "unknown"


def fuse_teachers(tracks_path: str | Path, any_change_path: str | Path, new_path: str | Path,
                  demolition_path: str | Path, output_dir: str | Path, *, uncertainty_path: str | Path | None = None,
                  high_threshold: float = 0.65, low_threshold: float = 0.35, uncertainty_threshold: float = 0.60) -> Path:
    tracks = gpd.read_file(tracks_path)
    if tracks.crs is None: raise ValueError("Building teacher tracks require CRS")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if "fusion_id" not in tracks:
        tracks["fusion_id"] = [f"fusion_{index:08d}" for index in range(len(tracks))]
    with rasterio.open(any_change_path) as any_source, rasterio.open(new_path) as new_source, rasterio.open(demolition_path) as dem_source:
        sources = [any_source, new_source, dem_source]
        if any(source.crs is None for source in sources): raise ValueError("Change probability rasters require CRS")
        uncertainty_source = rasterio.open(uncertainty_path) if uncertainty_path else None
        try:
            analysis = tracks.to_crs(any_source.crs)
            for index, row in analysis.iterrows():
                geometry = row.geometry
                any_probability, new_probability, demolition_probability = (_mean(source, geometry) for source in sources)
                uncertainty = _mean(uncertainty_source, geometry) if uncertainty_source else 0.0
                state = _state(tracks.iloc[index])
                high_new, high_dem = (new_probability or 0) >= high_threshold, (demolition_probability or 0) >= high_threshold
                high_any = (any_probability or 0) >= high_threshold
                if high_new and not high_dem: change_state = "new_building"
                elif high_dem and not high_new: change_state = "demolished"
                elif high_new and high_dem: change_state = "new_dem_conflict"
                elif high_any: change_state = "other_change"
                else: change_state = "stable"
                normalized = state.casefold()
                building_new = normalized in {"new", "new_building"}
                building_dem = normalized in {"demolished", "demolition", "removed"}
                building_stable = normalized in {"stable", "unchanged"}
                agreement = ((building_new and high_new) or (building_dem and high_dem) or (building_stable and not high_any))
                issues = []
                if building_stable and high_any: issues.append("stable_vs_high_change")
                if building_new and (new_probability or 0) < low_threshold: issues.append("new_vs_low_probability")
                if building_dem and (demolition_probability or 0) < low_threshold: issues.append("demolition_vs_low_probability")
                if high_new and high_dem: issues.append("new_dem_conflict")
                if (uncertainty or 0) >= uncertainty_threshold: issues.append("high_uncertainty")
                for field, label in (("ambiguous_match", "ambiguity"), ("lineage_event", "lineage_event"), ("construction_candidate", "construction_candidate")):
                    if str(row.get(field, "")).casefold() in {"true", "1", "yes"}: issues.append(label)
                if str(row.get("registration_status", "")).casefold() in {"failed", "issue", "registration_issue"}: issues.append("registration_issue")
                confidence = float(np.clip((1.0 - (uncertainty or 0)) * (max(new_probability or 0, demolition_probability or 0, any_probability or 0) if not building_stable else 1.0 - (any_probability or 0)), 0, 1))
                tracks.loc[index, "new_probability_mean"] = new_probability
                tracks.loc[index, "demolition_probability_mean"] = demolition_probability
                tracks.loc[index, "any_change_probability_mean"] = any_probability
                tracks.loc[index, "uncertainty_mean"] = uncertainty
                tracks.loc[index, "building_teacher_state"] = state
                tracks.loc[index, "change_teacher_state"] = change_state
                tracks.loc[index, "teacher_agreement"] = bool(agreement and not issues)
                tracks.loc[index, "fusion_confidence"] = confidence
                tracks.loc[index, "review_priority"] = "high" if issues else ("low" if agreement else "medium")
                tracks.loc[index, "fusion_issues"] = ";".join(issues)
        finally:
            if uncertainty_source: uncertainty_source.close()
    output = output_dir / "fused_candidates.gpkg"; tracks.to_file(output, driver="GPKG")
    disagreements = tracks[(tracks["teacher_agreement"] == False) | (tracks["fusion_issues"].astype(str) != "")].copy()  # noqa: E712
    priority = tracks[tracks["review_priority"].isin(["high", "medium"])].copy()
    if len(disagreements): disagreements.to_file(output_dir / "model_disagreements.gpkg", driver="GPKG")
    if len(priority): priority.to_file(output_dir / "review_priority.gpkg", driver="GPKG")
    inventory_columns = [column for column in tracks.columns if column != tracks.geometry.name]
    pd.DataFrame(tracks[inventory_columns]).to_csv(output_dir / "fusion_inventory.csv", index=False, encoding="utf-8-sig")
    report = ["# Double-teacher fusion report", "", f"- candidates: {len(tracks)}", f"- disagreements: {len(disagreements)}",
              f"- high priority: {int((tracks['review_priority']=='high').sum())}", f"- mean fusion confidence: {tracks['fusion_confidence'].mean():.4f}",
              "", "Fusion confidence is a rule-based score, not a calibrated probability."]
    (output_dir / "fusion_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output
