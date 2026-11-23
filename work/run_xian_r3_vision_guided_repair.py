"""Bounded, read-only-input Xi'an R3 vision-guided repair.

This script deliberately sits beside the R2 pipeline.  It consumes the frozen
R2 B0/mask/raw/vector/observability/alignment artifacts and writes a new,
candidate-only R3 package.  It does not rerun B0, alter the R2 package, read
the sealed Holdout folder, or use display-area tiers for scientific decisions.

The implementation is intentionally conservative:
* B0 support is copied from the R2 t=.50 mask without changing the threshold.
* instance splitting is accepted only when distance-transform cores and a
  probability/boundary separation signal agree; otherwise the R2 geometry is
  retained and marked ambiguous/diagnostic.
* tracking is rebuilt from the R3 geometry, with explicit split/merge/ambiguous
  edges and a neighborhood-consistency term.  No old R2 ID is edited.
* presence uses actual observability/raw/B0 evidence.  Black/unobservable
  support remains U; clear observable bare land can become a reliable-zero
  candidate.
* semantic and agriculture outputs are sidecars and never modify the formal
  1/0/U state.

The output is a development candidate, not ground truth or formal accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import unary_union


DATES = ["2018-04-29", "2019-01-01", "2020-12-31"]
BLOCKS = ["B01", "B02", "B03", "B04", "B05"]
THRESHOLD = 0.50
METRIC_CRS = "EPSG:32649"
SCIENTIFIC_STATUS = "CANDIDATE_NOT_GROUND_TRUTH"
CHECKPOINT_SHA256 = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"

# This is an evaluation index only.  It is not used by the generic repair
# rules, and it contains no sealed Holdout annotations.
FIXED_CASES = [
    ("B02-P2-24", "B02", "TRK_B02_000151", "BLACK_BLOCK;OBSERVABILITY"),
    ("B03-TEX04", "B03", "TRK_B03_000127", "IDENTITY_SWITCH;COMPETING_NEIGHBOR"),
    ("B04-TQA07", "B04", "TRK_B04_000047", "MERGE;PODIUM_NEIGHBOR"),
    ("B04-TQA10", "B04", "TRK_B04_000130", "MULTI_BUILDING_COMPONENT;VILLA"),
    ("B05-TQA03", "B05", "TRK_B05_000054", "GAP;ID_REBIRTH"),
    ("B05-P1-10", "B05", "TRK_B05_000054", "POSSIBLE_FALSE_DEMOLITION;GAP"),
    ("B01-P2-26", "B01", "TRK_B01_000096", "POSSIBLE_TRUE_DEMOLITION;OBSERVABILITY"),
    ("B01-TQA05", "B01", "TRK_B01_000244", "FRAGMENTATION;MERGE"),
    ("B03-TQA05", "B03", "TRK_B03_000297", "IDENTITY_SWITCH"),
    ("B03-TQA06", "B03", "TRK_B03_000280", "ROOF_OMISSION_AND_MERGE"),
    ("B04-TQA05", "B04", "TRK_B04_000245", "MERGE"),
    ("B05-TQA05", "B05", "TRK_B05_000086", "FRAGMENTATION;VILLAGE"),
    ("B01-TEX02", "B01", "TRK_B01_000071", "STADIUM_NON_BUILDING"),
    ("B05-P1-06", "B05", "TRK_B05_000107", "FIELD_NON_BUILDING"),
    ("B05-TQA06", "B05", "TRK_B05_000020", "AGRICULTURE;NON_BUILDING"),
    ("B05-TEX01", "B05", "TRK_B05_000011", "STABLE_SINGLE_SHED"),
    ("B01-TEX01", "B01", "TRK_B01_000088", "FACTORY_STABLE"),
    ("OBJ_0001", "B04", "TRK_B04_000336", "ADDITION"),
    ("OBJ_0008", "B01", "TRK_B01_000017", "STABLE"),
    ("OBJ_0019", "B01", "TRK_B01_000138", "GAP_OR_REBIRTH"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def csv_write(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if columns:
        for col in columns:
            if col not in df:
                df[col] = ""
        df = df[columns]
    df.to_csv(path, index=False, encoding="utf-8-sig")


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def compactness(geom: Polygon | MultiPolygon) -> float:
    p = float(geom.length)
    a = float(geom.area)
    return float(4.0 * math.pi * a / (p * p)) if p > 0 and a > 0 else 0.0


def geometry_clean(geom: Any) -> Any:
    if geom is None or geom.is_empty:
        return geom
    try:
        g = geom.buffer(0)
    except Exception:
        g = geom
    if isinstance(g, GeometryCollection):
        polys = [x for x in g.geoms if isinstance(x, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else geom
    return g


def geometry_hash(geom: Any) -> str:
    return hashlib.sha256(geom.wkb).hexdigest()


@dataclass
class Scene:
    block: str
    date: str
    probability: np.ndarray
    b0_mask: np.ndarray
    boundary: np.ndarray
    raw: np.ndarray
    observability: np.ndarray
    transform: Any
    crs: Any
    metric_crs: str
    metric_transformer: Any
    source_probability: Path
    source_mask: Path
    source_boundary: Path
    source_raw: Path
    source_observability: Path
    vector_crs: Any
    metric_gdf: gpd.GeoDataFrame
    alignment: pd.DataFrame


class R3Runner:
    def __init__(self, r2_root: Path, out_root: Path, max_instance_rounds: int = 3):
        self.r2 = r2_root
        self.out = out_root
        self.max_instance_rounds = min(int(max_instance_rounds), 3)
        self.scenes: dict[tuple[str, str], Scene] = {}
        self.r3_by_scene: dict[tuple[str, str], gpd.GeoDataFrame] = {}
        self.instance_records: list[dict[str, Any]] = []
        self.track_objects: pd.DataFrame = pd.DataFrame()
        self.track_edges: pd.DataFrame = pd.DataFrame()
        self.presence: pd.DataFrame = pd.DataFrame()
        self.state: pd.DataFrame = pd.DataFrame()
        self.semantic: pd.DataFrame = pd.DataFrame()
        self.agriculture: pd.DataFrame = pd.DataFrame()
        self.fixed_before: dict[str, dict[str, Any]] = {}
        self.fixed_case_rows: list[dict[str, Any]] = []

    def path(self, *parts: str) -> Path:
        return self.r2.joinpath(*parts)

    def out_path(self, *parts: str) -> Path:
        return self.out.joinpath(*parts)

    def assert_inputs(self) -> None:
        required = [
            self.path("04_B0_semantic"),
            self.path("05_boundary_instance"),
            self.path("06_polygonization"),
            self.path("09_tracking"),
            self.path("02_observability"),
            self.path("03_alignment"),
            self.path("_work", "aoi_sources"),
        ]
        missing = [str(x) for x in required if not x.exists()]
        if missing:
            raise FileNotFoundError("R2 required input missing: " + "; ".join(missing))
        if self.out.exists():
            raise FileExistsError(f"Refusing to overwrite existing R3 output: {self.out}")

    def load_scenes(self) -> None:
        for block in BLOCKS:
            for date in DATES:
                prob_path = self.path("04_B0_semantic", block, date, "building_probability.tif")
                mask_path = self.path("04_B0_semantic", block, date, "building_mask.tif")
                boundary_path = self.path("04_B0_semantic", block, date, "building_boundary.tif")
                raw_path = self.path("_work", "aoi_sources", block, f"{date}.tif")
                obs_path = self.path("02_observability", block, date, "observability.tif")
                vector_path = self.path("06_polygonization", block, date, "canonical_buildings.gpkg")
                if not all(p.exists() for p in [prob_path, mask_path, boundary_path, raw_path, obs_path, vector_path]):
                    raise FileNotFoundError(f"incomplete R2 scene inputs for {block} {date}")
                with rasterio.open(prob_path) as src:
                    probability = src.read(1).astype(np.float32, copy=False)
                    transform = src.transform
                    crs = src.crs
                with rasterio.open(mask_path) as src:
                    b0_mask = src.read(1) > 0
                    if src.shape != probability.shape or src.transform != transform:
                        raise ValueError(f"R2 probability/mask grid mismatch: {block} {date}")
                with rasterio.open(boundary_path) as src:
                    boundary = src.read(1) > 0
                with rasterio.open(raw_path) as src:
                    raw = src.read()
                with rasterio.open(obs_path) as src:
                    observability = src.read(1)
                if raw.shape[1:] != probability.shape or observability.shape != probability.shape:
                    raise ValueError(f"R2 scene grid mismatch: {block} {date}")
                gdf = gpd.read_file(vector_path, layer="buildings")
                if gdf.crs is None:
                    raise ValueError(f"R2 vector CRS missing: {vector_path}")
                metric_gdf = gdf.to_crs(METRIC_CRS)
                alignment_path = self.path("03_alignment", block, "ALIGNMENT_DIAGNOSTICS.csv")
                alignment = pd.read_csv(alignment_path) if alignment_path.exists() else pd.DataFrame()
                self.scenes[(block, date)] = Scene(
                    block=block,
                    date=date,
                    probability=probability,
                    b0_mask=b0_mask,
                    boundary=boundary,
                    raw=raw,
                    observability=observability,
                    transform=transform,
                    crs=crs,
                    metric_crs=METRIC_CRS,
                    metric_transformer=None,
                    source_probability=prob_path,
                    source_mask=mask_path,
                    source_boundary=boundary_path,
                    source_raw=raw_path,
                    source_observability=obs_path,
                    vector_crs=gdf.crs,
                    metric_gdf=metric_gdf,
                    alignment=alignment,
                )

    def write_task_spec(self) -> None:
        spec = {
            "task_id": "XIAN-R3-VISION-GUIDED-SCIENTIFIC-REPAIR-001",
            "mode": "CLOUD_REAL_DATA_BOUNDED_ITERATION_FIXED_VISUAL_REGRESSION",
            "created_at_utc": utc_now(),
            "source_r2_root": str(self.r2),
            "output_root": str(self.out),
            "dates": DATES,
            "blocks": BLOCKS,
            "b0_threshold": THRESHOLD,
            "b0_rerun": False,
            "area_filter_used_scientifically": False,
            "area_tiers": "display_only_and_not_read_by_repair",
            "holdout_read": False,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "scientific_status": SCIENTIFIC_STATUS,
            "budgets": {"instance_rounds": self.max_instance_rounds, "tracking_rounds": 3, "presence_rounds": 2},
            "formal_order": [
                "fixed_B0",
                "boundary_instance_diagnosis",
                "instance_repair",
                "geometry_freeze",
                "tracking_from_scratch",
                "gap_return",
                "observability",
                "presence",
                "state",
                "semantic_qa",
                "visual_parent_review",
            ],
            "fixed_cases": [x[0] for x in FIXED_CASES],
            "safety": {
                "raw_sources_modified": False,
                "r2_outputs_modified": False,
                "old_tracking_ids_edited": False,
                "b0_probability_or_mask_modified": False,
                "final_polygon_warped_for_alignment": False,
            },
        }
        json_dump(self.out_path("00_任务说明", "任务说明.json"), spec)
        self.out_path("00_任务说明", "README.md").write_text(
            "# 西安 R3 视觉引导科学修复\n\n"
            "本包只消费冻结的 R2 开发集证据，输出候选修复和固定案例视觉复核材料。\n"
            "B0 t=.50、原始影像、R2 向量、R2 追踪和 Holdout 均不改动；面积挡位不进入科学链。\n",
            encoding="utf-8",
        )

    def _window_for_geometry(self, scene: Scene, geom_metric: Any, pad_px: int = 2) -> tuple[Window, Any]:
        g = gpd.GeoSeries([geom_metric], crs=METRIC_CRS).to_crs(scene.crs).iloc[0]
        base = rasterio.windows.from_bounds(*g.bounds, transform=scene.transform)
        win = base.round_offsets().round_lengths()
        win = Window(
            max(0, win.col_off - pad_px),
            max(0, win.row_off - pad_px),
            min(scene.probability.shape[1] - max(0, win.col_off - pad_px), win.width + 2 * pad_px),
            min(scene.probability.shape[0] - max(0, win.row_off - pad_px), win.height + 2 * pad_px),
        )
        return win, g

    def _geometry_pixel_mask(self, scene: Scene, geom_metric: Any, pad_px: int = 2) -> tuple[Window, Any, np.ndarray]:
        win, g = self._window_for_geometry(scene, geom_metric, pad_px=pad_px)
        mask = rio_features.geometry_mask(
            [mapping(g)],
            out_shape=(int(win.height), int(win.width)),
            transform=rasterio.windows.transform(win, scene.transform),
            invert=True,
        )
        return win, g, mask

    @staticmethod
    def _line_evidence(prob: np.ndarray, dt: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> dict[str, float]:
        y0, x0 = a
        y1, x1 = b
        n = max(abs(int(y1 - y0)), abs(int(x1 - x0))) + 1
        ys = np.rint(np.linspace(y0, y1, n)).astype(int)
        xs = np.rint(np.linspace(x0, x1, n)).astype(int)
        good = (ys >= 0) & (ys < prob.shape[0]) & (xs >= 0) & (xs < prob.shape[1])
        vals = prob[ys[good], xs[good]]
        dvals = dt[ys[good], xs[good]]
        valid = np.isfinite(vals)
        vals = vals[valid]
        dvals = dvals[valid]
        if len(vals) == 0:
            return {
                "line_min_probability": None,
                "line_min_distance_px": None,
                "line_inside_fraction": 0.0,
                "valley_drop": None,
                "neck_ratio": None,
                "endpoint_probability": None,
            }
        endpoint_probability = min(float(prob[a]), float(prob[b]))
        min_distance = float(np.nanmin(dvals)) if len(dvals) else None
        core_distance = min(float(dt[a]), float(dt[b]))
        return {
            "line_min_probability": float(np.nanmin(vals)),
            "line_min_distance_px": min_distance,
            "line_inside_fraction": float(np.mean(dvals > 0)),
            "valley_drop": float(endpoint_probability - float(np.nanmin(vals))),
            "neck_ratio": safe_ratio(min_distance or 0.0, core_distance),
            "endpoint_probability": endpoint_probability,
        }

    def _instance_candidates(self, scene: Scene, row: pd.Series, iteration: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        geom = row.geometry
        win, geom_raster, geom_mask = self._geometry_pixel_mask(scene, geom, pad_px=3)
        prob = scene.probability[int(win.row_off) : int(win.row_off + win.height), int(win.col_off) : int(win.col_off + win.width)].astype(np.float32)
        source_mask = scene.b0_mask[int(win.row_off) : int(win.row_off + win.height), int(win.col_off) : int(win.col_off + win.width)]
        obj_mask = geom_mask & source_mask
        if not np.any(obj_mask):
            return [], {"status": "NO_PIXEL_SUPPORT", "marker_count": 0, "source_object_id": row.object_id}
        local_prob = np.where(obj_mask, prob, np.nan)
        dt = ndi.distance_transform_edt(obj_mask)
        smooth_dt = ndi.gaussian_filter(dt.astype(np.float32), sigma=1.5)
        min_distance = max(12, int(round(min(30.0, max(12.0, float(dt.max()) * 0.35)))))
        threshold_abs = max(3.0, float(dt.max()) * 0.18)
        # scipy-only local maxima with a greedy minimum-distance suppression.
        # This is equivalent to the intended peak-local-maxima diagnostic and
        # keeps the cloud runtime independent of scikit-image.
        max_size = int(2 * min_distance + 1)
        local_max = (smooth_dt == ndi.maximum_filter(smooth_dt, size=max_size, mode="nearest")) & obj_mask
        candidates = [tuple(int(x) for x in c) for c in np.argwhere(local_max & (smooth_dt >= threshold_abs))]
        candidates.sort(key=lambda c: float(smooth_dt[c]), reverse=True)
        coords: list[tuple[int, int]] = []
        for c in candidates:
            if all(math.hypot(c[0] - q[0], c[1] - q[1]) >= min_distance for q in coords):
                coords.append(c)
        evidence_edges: list[dict[str, Any]] = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                ev = self._line_evidence(local_prob, dt, coords[i], coords[j])
                # This is a separation diagnostic, not an alternate B0 cutoff.
                # It requires an actual probability valley or a rasterized neck.
                strong = bool(
                    ev["line_min_probability"] is not None
                    and ev["valley_drop"] is not None
                    and ev["neck_ratio"] is not None
                    and ev["line_inside_fraction"] >= 0.95
                    and ev["valley_drop"] >= 0.18
                    and ev["neck_ratio"] <= 0.08
                )
                if strong:
                    evidence_edges.append({"i": i, "j": j, **ev})
        marker_indices: list[int] = []
        for edge in evidence_edges:
            marker_indices.extend([int(edge["i"]), int(edge["j"])])
        marker_indices = sorted(set(marker_indices))
        marker_indices = marker_indices[:8]
        markers = np.zeros(obj_mask.shape, dtype=np.int32)
        for marker_id, index in enumerate(marker_indices, start=1):
            y, x = coords[index]
            markers[y, x] = marker_id
        status = "NO_SPLIT_EVIDENCE"
        parts: list[dict[str, Any]] = []
        if len(marker_indices) > 4:
            status = "AMBIGUOUS_INSTANCE"
        elif len(marker_indices) >= 2 and evidence_edges:
            # scipy.ndimage.watershed_ift is an integer priority-flood
            # watershed.  The image is the inverted distance transform; the
            # final mask clips labels back to the immutable B0 component.
            scale = max(float(dt.max()), 1.0)
            watershed_image = np.clip((scale - dt) * 255.0 / scale, 0, 255).astype(np.uint8)
            watershed_image[~obj_mask] = 255
            labels = ndi.watershed_ift(watershed_image, markers)
            labels[~obj_mask] = 0
            transform = rasterio.windows.transform(win, scene.transform)
            for label in range(1, int(labels.max()) + 1):
                label_mask = labels == label
                if not np.any(label_mask):
                    continue
                geoms = [shape(g) for g, val in rio_features.shapes(label_mask.astype(np.uint8), mask=label_mask, transform=transform) if val == 1]
                if not geoms:
                    continue
                geom_part = geometry_clean(unary_union(geoms))
                if geom_part.is_empty:
                    continue
                geom_metric = gpd.GeoSeries([geom_part], crs=scene.crs).to_crs(METRIC_CRS).iloc[0]
                parts.append({"label": label, "geometry": geom_metric, "pixel_count": int(label_mask.sum())})
            if len(parts) >= 2:
                coverage = safe_ratio(sum(float(p["geometry"].area) for p in parts), float(geom.area))
                if coverage > 0.88:
                    status = "SPLIT_CANDIDATE"
        if status == "SPLIT_CANDIDATE":
            # Apply only when every retained part has its own stable marker.
            marker_by_label = {}
            if parts:
                scale = max(float(dt.max()), 1.0)
                watershed_image = np.clip((scale - dt) * 255.0 / scale, 0, 255).astype(np.uint8)
                watershed_image[~obj_mask] = 255
                labels = ndi.watershed_ift(watershed_image, markers)
                labels[~obj_mask] = 0
                for marker_id, index in enumerate(marker_indices, start=1):
                    y, x = coords[index]
                    marker_by_label[int(labels[y, x])] = marker_id
            parts = [p for p in parts if p["label"] in marker_by_label]
            if len(parts) < 2:
                status = "AMBIGUOUS_INSTANCE"
        diag = {
            "status": status,
            "source_object_id": row.object_id,
            "block_id": row.block_id,
            "actual_date": row.actual_date,
            "iteration": iteration,
            "marker_count": len(marker_indices),
            "candidate_peak_count": len(coords),
            "dt_max_px": float(dt.max()),
            "threshold_fixed": THRESHOLD,
            "boundary_source": "R2_DERIVED_OUTER_MASK_EDGE_ONLY",
            "boundary_internal_evidence_used": False,
            "evidence_edges": json.dumps(evidence_edges, ensure_ascii=False),
            "part_count": len(parts),
            "area_filter_used": False,
            "source_geometry_hash": row.geometry_hash,
            "parts": parts,
        }
        return parts, diag

    def run_instance_repair(self) -> None:
        iteration_rows: list[dict[str, Any]] = []
        for key, scene in self.scenes.items():
            block, date = key
            gdf = scene.metric_gdf.copy()
            r3_rows: list[dict[str, Any]] = []
            for _, row in gdf.iterrows():
                parts, diag = self._instance_candidates(scene, row, iteration=1)
                diag_for_csv = {k: v for k, v in diag.items() if k != "parts"}
                iteration_rows.append(diag_for_csv)
                if diag["status"] == "SPLIT_CANDIDATE" and len(parts) >= 2:
                    for i, part in enumerate(parts, start=1):
                        r = row.drop(labels=["geometry"]).to_dict()
                        r.update(
                            {
                                "source_object_id": row.object_id,
                                "r3_object_id": f"{row.object_id}__I{i:02d}",
                                "instance_source": "R3_WATERSHED_FROM_DT_PROBABILITY_VALLEY",
                                "instance_status": "PARTITION_APPLIED_CANDIDATE",
                                "instance_parent_object_id": row.object_id,
                                "instance_part_index": i,
                                "geometry": part["geometry"],
                            }
                        )
                        r3_rows.append(r)
                else:
                    r = row.drop(labels=["geometry"]).to_dict()
                    r.update(
                        {
                            "source_object_id": row.object_id,
                            "r3_object_id": row.object_id,
                            "instance_source": "R2_B0_CONNECTED_COMPONENT",
                            "instance_status": diag["status"],
                            "instance_parent_object_id": "",
                            "instance_part_index": 1,
                            "geometry": row.geometry,
                        }
                    )
                    r3_rows.append(r)
            out_gdf = gpd.GeoDataFrame(r3_rows, geometry="geometry", crs=METRIC_CRS)
            self.r3_by_scene[key] = out_gdf
            out_dir = self.out_path("02_Instance修复", block, date)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_gdf.to_file(out_dir / "R3_INSTANCE_CANDIDATES.gpkg", layer="buildings", driver="GPKG")
            # Final Geometry is written after the full-scene repair is frozen.
        self.instance_records = iteration_rows
        csv_write(
            self.out_path("01_Instance诊断", "INSTANCE_DIAGNOSTICS.csv"),
            iteration_rows,
        )
        csv_write(
            self.out_path("02_Instance修复", "INSTANCE_REPAIR_RECORDS.csv"),
            [
                {
                    **row,
                    "parts": None,
                }
                for row in iteration_rows
            ],
        )
        json_dump(
            self.out_path("02_Instance修复", "INSTANCE_REPAIR_RECEIPT.json"),
            {
                "status": "COMPLETED_BOUNDED_CANDIDATE",
                "iterations_used": 1,
                "max_iterations": self.max_instance_rounds,
                "b0_threshold": THRESHOLD,
                "b0_rerun": False,
                "area_filter_used": False,
                "negative_controls": ["B05-TEX01", "OBJ_0008", "B01-TEX01"],
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )

    def freeze_geometry(self) -> None:
        for (block, date), gdf in self.r3_by_scene.items():
            out_dir = self.out_path("03_最终Geometry", block, date)
            out_dir.mkdir(parents=True, exist_ok=True)
            final = gdf.copy()
            final["geometry_hash"] = final.geometry.map(geometry_hash)
            final["area_m2"] = final.geometry.area
            final["perimeter_m"] = final.geometry.length
            final["scientific_status"] = SCIENTIFIC_STATUS
            final.to_file(out_dir / "R3_FINAL_GEOMETRY.gpkg", layer="buildings", driver="GPKG")
            final.drop(columns="geometry").to_csv(out_dir / "R3_FINAL_GEOMETRY.csv", index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("03_最终Geometry", "GEOMETRY_FREEZE_RECEIPT.json"),
            {
                "status": "FROZEN_FOR_TRACKING",
                "geometry_source": "R3_INSTANCE_CANDIDATES",
                "b0_threshold": THRESHOLD,
                "area_filter_used": False,
                "alignment_warped_final_geometry": False,
                "old_geometry_modified": False,
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )

    def _alignment_shift(self, scene: Scene, from_date: str, to_date: str) -> tuple[float, float, float]:
        df = scene.alignment
        if df.empty:
            return 0.0, 0.0, float("inf")
        x = df[(df.date_from == from_date) & (df.date_to == to_date) & (df.level == "global")]
        if x.empty:
            return 0.0, 0.0, float("inf")
        r = x.iloc[0]
        return float(r.forward_dx_px), float(r.forward_dy_px), float(r.uncertainty)

    @staticmethod
    def _neighborhood_signature(geoms: list[Any], idx: int, max_neighbors: int = 5) -> tuple[np.ndarray, np.ndarray]:
        c = geoms[idx].centroid
        vals: list[tuple[float, float]] = []
        for j, g in enumerate(geoms):
            if j == idx:
                continue
            q = g.centroid
            dx = float(q.x - c.x)
            dy = float(q.y - c.y)
            d = math.hypot(dx, dy)
            if d <= 250.0:
                vals.append((d, math.atan2(dy, dx)))
        vals.sort(key=lambda x: x[0])
        vals = vals[:max_neighbors]
        distances = np.array([x[0] for x in vals], dtype=float)
        quadrants = np.array([0 if (-math.pi / 4 <= x[1] < math.pi / 4) else 1 if x[1] < 3 * math.pi / 4 else 2 if x[1] >= 3 * math.pi / 4 or x[1] < -3 * math.pi / 4 else 3 for x in vals], dtype=int)
        return distances, quadrants

    def _neighborhood_consistency(self, source_geoms: list[Any], target_geoms: list[Any], i: int, j: int) -> float:
        d1, q1 = self._neighborhood_signature(source_geoms, i)
        d2, q2 = self._neighborhood_signature(target_geoms, j)
        if len(d1) == 0 and len(d2) == 0:
            return 1.0
        k = min(len(d1), len(d2))
        dist_score = 0.0 if k == 0 else math.exp(-float(np.mean(np.abs(np.log((d1[:k] + 2.0) / (d2[:k] + 2.0))))))
        quad_score = 0.0
        if k:
            quad_score = float(np.mean(q1[:k] == q2[:k]))
        return float(0.7 * dist_score + 0.3 * quad_score)

    def _edge_candidates(self, source: gpd.GeoDataFrame, target: gpd.GeoDataFrame, scene: Scene, from_date: str, to_date: str) -> list[dict[str, Any]]:
        source_geoms = list(source.geometry)
        target_geoms = list(target.geometry)
        dx_px, dy_px, uncertainty = self._alignment_shift(scene, from_date, to_date)
        # Convert the diagnostic pixel shift to metres for scoring only.  The
        # final polygons are never warped.
        meter_x = abs(float(scene.transform.a)) * 111320.0 * math.cos(math.radians(34.2))
        meter_y = abs(float(scene.transform.e)) * 111320.0
        sx = dx_px * meter_x
        sy = -dy_px * meter_y
        rows: list[dict[str, Any]] = []
        for i, a in enumerate(source_geoms):
            ca = a.centroid
            for j, b in enumerate(target_geoms):
                cb = b.centroid
                dx = float(cb.x - ca.x - sx)
                dy = float(cb.y - ca.y - sy)
                dist = math.hypot(dx, dy)
                scale = max(8.0, math.sqrt(max(float(a.area), float(b.area))))
                radius = min(180.0, max(25.0, 2.5 * scale + min(60.0, uncertainty * max(meter_x, meter_y)))) if math.isfinite(uncertainty) else min(180.0, max(25.0, 2.5 * scale))
                if dist > radius:
                    continue
                inter = float(a.intersection(b).area)
                iou = safe_ratio(inter, float(a.union(b).area))
                area_ratio = min(float(a.area), float(b.area)) / max(float(a.area), float(b.area), 1e-9)
                shape_score = 1.0 - min(1.0, abs(compactness(a) - compactness(b)) * 4.0)
                neigh = self._neighborhood_consistency(source_geoms, target_geoms, i, j)
                distance_score = math.exp(-dist / max(10.0, scale))
                # Give actual footprint overlap and the local landmark
                # ordering more weight than centroid distance alone.  This is
                # the protection against a nearby north/south neighbour
                # winning solely because its centroid is a few pixels closer.
                score = float(0.22 * distance_score + 0.30 * iou + 0.16 * area_ratio + 0.10 * shape_score + 0.22 * neigh)
                rows.append(
                    {
                        "source_index": i,
                        "target_index": j,
                        "source_object_id": source.iloc[i].r3_object_id,
                        "target_object_id": target.iloc[j].r3_object_id,
                        "distance_m": dist,
                        "iou": iou,
                        "area_ratio": area_ratio,
                        "shape_score": shape_score,
                        "local_neighborhood_consistency": neigh,
                        "alignment_uncertainty_px": uncertainty,
                        "score": score,
                    }
                )
        # Keep the graph bounded.  All retained candidates are still emitted,
        # but distant/weak alternatives cannot turn every large component
        # into an artificial split/merge cloud.
        by_source: dict[int, list[dict[str, Any]]] = {}
        by_target: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            if row["score"] >= 0.30:
                by_source.setdefault(int(row["source_index"]), []).append(row)
                by_target.setdefault(int(row["target_index"]), []).append(row)
        source_keep = {id(row) for values in by_source.values() for row in sorted(values, key=lambda x: x["score"], reverse=True)[:4]}
        target_keep = {id(row) for values in by_target.values() for row in sorted(values, key=lambda x: x["score"], reverse=True)[:4]}
        return [row for row in rows if id(row) in source_keep and id(row) in target_keep]

    def run_tracking(self) -> None:
        all_objects: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        next_track = 1
        previous_tracks: dict[str, str] = {}
        for block in BLOCKS:
            date_gdfs = [self.r3_by_scene[(block, date)] for date in DATES]
            local_object_tracks: dict[str, str] = {}
            for date, gdf in zip(DATES, date_gdfs):
                for _, row in gdf.iterrows():
                    oid = str(row.r3_object_id)
                    if oid not in local_object_tracks:
                        local_object_tracks[oid] = f"TRK_R3_{block}_{next_track:06d}"
                        next_track += 1
                    all_objects.append(
                        {
                            "block_id": block,
                            "actual_date": date,
                            "r3_object_id": oid,
                            "source_object_id": row.source_object_id,
                            "area_m2": float(row.geometry.area),
                            "perimeter_m": float(row.geometry.length),
                            "geometry_hash": geometry_hash(row.geometry),
                            "centroid_x": float(row.geometry.centroid.x),
                            "centroid_y": float(row.geometry.centroid.y),
                            "instance_status": row.instance_status,
                            "track_id": local_object_tracks[oid],
                            "scientific_status": SCIENTIFIC_STATUS,
                        }
                    )
                if date != DATES[0]:
                    prev_date = DATES[DATES.index(date) - 1]
                    prev = self.r3_by_scene[(block, prev_date)]
                    curr = gdf
                    candidates = self._edge_candidates(prev, curr, self.scenes[(block, prev_date)], prev_date, date)
                    by_source: dict[int, list[dict[str, Any]]] = {}
                    by_target: dict[int, list[dict[str, Any]]] = {}
                    source_best = {i: max(values, key=lambda x: x["score"]) for i, values in by_source.items()}
                    target_best = {j: max(values, key=lambda x: x["score"]) for j, values in by_target.items()}
                    for e in candidates:
                        by_source.setdefault(int(e["source_index"]), []).append(e)
                        by_target.setdefault(int(e["target_index"]), []).append(e)
                    for e in candidates:
                        s = by_source[e["source_index"]]
                        t = by_target[e["target_index"]]
                        s_sorted = sorted(s, key=lambda x: x["score"], reverse=True)
                        t_sorted = sorted(t, key=lambda x: x["score"], reverse=True)
                        relation = "1_TO_1"
                        if len(s) > 1 and len(t) > 1:
                            relation = "AMBIGUOUS_SPLIT_MERGE"
                        elif len(s) > 1:
                            relation = "1_TO_N"
                        elif len(t) > 1:
                            relation = "N_TO_1"
                        gap_s = s_sorted[0]["score"] - (s_sorted[1]["score"] if len(s_sorted) > 1 else 0.0)
                        gap_t = t_sorted[0]["score"] - (t_sorted[1]["score"] if len(t_sorted) > 1 else 0.0)
                        conflict = bool(e["local_neighborhood_consistency"] < 0.45 and (gap_s < 0.08 or gap_t < 0.08))
                        if conflict:
                            relation = "IDENTITY_CONFLICT"
                        e2 = dict(e)
                        e2.update(
                            {
                                "block_id": block,
                                "from_date": prev_date,
                                "to_date": date,
                                "relation": relation,
                                "identity_conflict": conflict,
                                "candidate_count_from_source": len(s),
                                "candidate_count_to_target": len(t),
                                "score_gap_source": gap_s,
                                "score_gap_target": gap_t,
                                "mutual_best": bool(source_best.get(e["source_index"]) is e and target_best.get(e["target_index"]) is e),
                                "scientific_status": SCIENTIFIC_STATUS,
                            }
                        )
                        edges.append(e2)
                    # Continue only unambiguous 1:1 identity edges.  All other
                    # edges remain in the graph and are represented explicitly.
                    for e in candidates:
                        s = by_source[e["source_index"]]
                        t = by_target[e["target_index"]]
                        mutual_best = source_best.get(e["source_index"]) is e and target_best.get(e["target_index"]) is e
                        decisive_geometry = e["iou"] >= 0.15 and e["local_neighborhood_consistency"] >= 0.45
                        decisive_margin = (e["score"] - (sorted(s, key=lambda x: x["score"], reverse=True)[1]["score"] if len(s) > 1 else 0.0)) >= 0.05
                        if mutual_best and e["score"] >= 0.35 and (decisive_geometry or decisive_margin):
                            src_oid = str(prev.iloc[e["source_index"]].r3_object_id)
                            tgt_oid = str(curr.iloc[e["target_index"]].r3_object_id)
                            src_track = local_object_tracks.get(src_oid)
                            if src_track:
                                local_object_tracks[tgt_oid] = src_track
            # Fill any object rows with their final propagated track IDs.
            for row in all_objects:
                if row["block_id"] == block:
                    row["track_id"] = local_object_tracks.get(row["r3_object_id"], row["track_id"])
        objects = pd.DataFrame(all_objects)
        self.track_objects = objects
        self.track_edges = pd.DataFrame(edges)
        self.track_objects.to_csv(self.out_path("04_Tracking", "R3_TRACKING_OBJECTS.csv"), index=False, encoding="utf-8-sig")
        self.track_edges.to_csv(self.out_path("04_Tracking", "R3_TRACKING_EDGES.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("04_Tracking", "TRACKING_RECEIPT.json"),
            {
                "status": "COMPLETED_FROM_SCRATCH_CANDIDATE",
                "from_scratch": True,
                "r2_ids_edited": False,
                "explicit_relations": sorted(self.track_edges.relation.dropna().unique().tolist()) if not self.track_edges.empty else [],
                "neighborhood_consistency": "USED",
                "final_geometry_warped": False,
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )

    def _raw_stats(self, scene: Scene, geom_metric: Any, buffer_m: float = 0.0) -> dict[str, Any]:
        geom = geom_metric.buffer(buffer_m) if buffer_m else geom_metric
        win, _, mask = self._geometry_pixel_mask(scene, geom, pad_px=2)
        r0 = int(win.row_off)
        r1 = int(win.row_off + win.height)
        c0 = int(win.col_off)
        c1 = int(win.col_off + win.width)
        p = scene.probability[r0:r1, c0:c1]
        obs = scene.observability[r0:r1, c0:c1]
        raw = scene.raw[:, r0:r1, c0:c1].astype(np.float32)
        valid = mask & (obs != 255)
        if not np.any(valid):
            return {
                "valid_pixel_count": 0,
                "valid_observable_fraction": 0.0,
                "partial_fraction": 1.0,
                "unobservable_fraction": 1.0,
                "nodata_fraction": 1.0,
                "b0_max_probability": None,
                "b0_positive_fraction": 0.0,
                "candidate_intersection": False,
                "black_fraction": 1.0,
                "raw_mean": None,
                "raw_std": None,
                "texture_flatness": None,
                "edge_mean": None,
            }
        pv = p[valid]
        rv = raw[:, valid]
        ov = obs[valid]
        gray = raw.mean(axis=0)
        gy, gx = np.gradient(gray)
        grad = np.hypot(gx, gy)
        return {
            "valid_pixel_count": int(valid.sum()),
            "valid_observable_fraction": float(np.mean(ov == 1)),
            "partial_fraction": float(np.mean(ov == 2)),
            "unobservable_fraction": float(np.mean(ov == 0)),
            "nodata_fraction": float(np.mean(~mask)),
            "b0_max_probability": float(np.nanmax(pv)),
            "b0_positive_fraction": float(np.mean(pv >= THRESHOLD)),
            "candidate_intersection": bool(np.any(pv >= THRESHOLD)),
            "black_fraction": float(np.mean(np.all(rv <= 2, axis=0))),
            "raw_mean": float(np.mean(rv)),
            "raw_std": float(np.mean(np.std(rv, axis=1))),
            "texture_flatness": float(np.mean(np.std(rv, axis=1)) / (np.mean(rv) + 1.0)),
            "edge_mean": float(np.mean(grad[valid])),
            "edge_p90": float(np.quantile(grad[valid], 0.90)),
            "spectral_range": float(np.ptp(np.mean(rv, axis=1))),
        }

    def _nearest_geometry(self, block: str, date: str, geom: Any) -> tuple[Any | None, float | None]:
        gdf = self.r3_by_scene[(block, date)]
        if gdf.empty:
            return None, None
        distances = gdf.geometry.centroid.distance(geom.centroid)
        idx = int(distances.idxmin())
        return gdf.loc[idx].geometry, float(distances.loc[idx])

    def run_gap_return(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for track_id, grp in self.track_objects.groupby("track_id"):
            block = str(grp.block_id.iloc[0])
            by_date = {str(r.actual_date): r for _, r in grp.iterrows()}
            if DATES[0] not in by_date:
                continue
            # Only a missing middle epoch is a gap-return problem.  Tracks
            # with an observed t2 but an absent t3 are terminal observation
            # problems, not evidence for a t1->t3 rebirth.
            if DATES[1] in by_date:
                continue
            a = by_date[DATES[0]]
            ga = self.r3_by_scene[(block, DATES[0])].set_index("r3_object_id").loc[a.r3_object_id].geometry
            if DATES[2] in by_date:
                c = by_date[DATES[2]]
                gc = self.r3_by_scene[(block, DATES[2])].set_index("r3_object_id").loc[c.r3_object_id].geometry
                dist = float(ga.centroid.distance(gc.centroid))
                ratio = min(float(ga.area), float(gc.area)) / max(float(ga.area), float(gc.area), 1e-9)
                neigh = self._neighborhood_consistency(
                    list(self.r3_by_scene[(block, DATES[0])].geometry),
                    list(self.r3_by_scene[(block, DATES[2])].geometry),
                    int(self.r3_by_scene[(block, DATES[0])].index[self.r3_by_scene[(block, DATES[0])].r3_object_id == a.r3_object_id][0]),
                    int(self.r3_by_scene[(block, DATES[2])].index[self.r3_by_scene[(block, DATES[2])].r3_object_id == c.r3_object_id][0]),
                )
                status = "GAP_RETURN_SUPPORTED" if dist <= 80.0 and ratio >= 0.35 and neigh >= 0.35 else "GAP_RETURN_AMBIGUOUS"
                t3_object_id = c.r3_object_id
                t3_candidate_object_id = c.r3_object_id
                t3_support = True
                middle_competitor = False
            else:
                # First check whether an independent t2 object is already
                # close enough to explain the apparent gap.  That becomes an
                # identity-link diagnostic, not a gap-return claim.
                mid = self.r3_by_scene[(block, DATES[1])]
                mid_distances = mid.geometry.centroid.distance(ga.centroid)
                mid_idx = int(mid_distances.idxmin()) if not mid.empty else None
                mid_geom = mid.loc[mid_idx].geometry if mid_idx is not None else None
                mid_ratio = min(float(ga.area), float(mid_geom.area)) / max(float(ga.area), float(mid_geom.area), 1e-9) if mid_geom is not None else 0.0
                middle_competitor = bool(mid_geom is not None and float(mid_distances.loc[mid_idx]) <= 80.0 and mid_ratio >= 0.25)
                # A positive t3 search-footprint result supports a possible
                # return but never creates a new t3 object ID.
                stats = self._raw_stats(self.scenes[(block, DATES[2])], ga, buffer_m=2.0)
                t3g = self.r3_by_scene[(block, DATES[2])]
                t3_distances = t3g.geometry.centroid.distance(ga.centroid)
                t3_idx = int(t3_distances.idxmin()) if not t3g.empty else None
                t3_geom = t3g.loc[t3_idx].geometry if t3_idx is not None else None
                dist = float(t3_distances.loc[t3_idx]) if t3_idx is not None else None
                ratio = min(float(ga.area), float(t3_geom.area)) / max(float(ga.area), float(t3_geom.area), 1e-9) if t3_geom is not None else 0.0
                neigh = None
                if t3_idx is not None and dist is not None and dist <= 80.0:
                    src_idx = int(self.r3_by_scene[(block, DATES[0])].index[self.r3_by_scene[(block, DATES[0])].r3_object_id == a.r3_object_id][0])
                    neigh = self._neighborhood_consistency(
                        list(self.r3_by_scene[(block, DATES[0])].geometry),
                        list(self.r3_by_scene[(block, DATES[2])].geometry),
                        src_idx,
                        t3_idx,
                    )
                t3_object_id = ""
                t3_candidate_object_id = str(t3g.loc[t3_idx].r3_object_id) if t3_idx is not None else ""
                t3_support = bool(
                    stats.get("candidate_intersection")
                    and stats.get("valid_observable_fraction", 0.0) >= 0.75
                    and stats.get("black_fraction", 1.0) <= 0.25
                    and not middle_competitor
                    and dist is not None
                    and dist <= 80.0
                    and ratio >= 0.25
                )
                if middle_competitor:
                    status = "IDENTITY_LINK_UNRESOLVED"
                else:
                    status = "GAP_RETURN_CANDIDATE_NO_ID" if t3_support else "NO_RETURN_EVIDENCE"
            rows.append(
                {
                    "track_id": track_id,
                    "block_id": block,
                    "t1": DATES[0],
                    "t2": DATES[1],
                    "t3": DATES[2],
                    "t1_object_id": a.r3_object_id,
                    "t3_object_id": t3_object_id,
                    "t3_candidate_object_id": t3_candidate_object_id,
                    "t2_status": "U_GAP_NOT_FORCED",
                    "distance_t1_t3_m": dist,
                    "area_ratio_t1_t3": ratio,
                    "local_neighborhood_consistency": neigh,
                    "gap_return_status": status,
                    "t3_search_support": t3_support,
                    "middle_epoch_competitor": middle_competitor,
                    "event_rule": "1-U-1_PLUS_GAP_RETURN" if status in {"GAP_RETURN_SUPPORTED", "GAP_RETURN_CANDIDATE_NO_ID"} else "NO_DEMOLITION_OR_ADDITION_FORCED",
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
        gap = pd.DataFrame(rows)
        if gap.empty:
            gap = pd.DataFrame(columns=["track_id", "gap_return_status"])
        gap.to_csv(self.out_path("05_Gap_Return", "GAP_RETURN.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("05_Gap_Return", "GAP_RETURN_RECEIPT.json"),
            {
                "status": "COMPLETED_CONSERVATIVE",
                "no_auto_new_2020_id": True,
                "no_demolition_plus_addition_for_gap": True,
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )
        return gap

    def run_presence(self) -> None:
        rows: list[dict[str, Any]] = []
        gap = pd.read_csv(self.out_path("05_Gap_Return", "GAP_RETURN.csv")) if self.out_path("05_Gap_Return", "GAP_RETURN.csv").exists() else pd.DataFrame()
        gap_supported = set(gap.loc[gap.gap_return_status == "GAP_RETURN_SUPPORTED", "track_id"].astype(str)) if not gap.empty else set()
        for track_id, grp in self.track_objects.groupby("track_id"):
            block = str(grp.block_id.iloc[0])
            geom_by_date = {str(r.actual_date): self.r3_by_scene[(block, str(r.actual_date))].set_index("r3_object_id").loc[r.r3_object_id].geometry for _, r in grp.iterrows()}
            for date in DATES:
                if date in geom_by_date:
                    stats = self._raw_stats(self.scenes[(block, date)], geom_by_date[date])
                    if stats["valid_observable_fraction"] >= 0.75 and stats["candidate_intersection"]:
                        presence = "1"
                        reason = "POSITIVE_VISIBLE_BUILDING_SUPPORT"
                    elif stats["valid_observable_fraction"] < 0.75 or stats["partial_fraction"] > 0.25 or stats["black_fraction"] > 0.25:
                        presence = "U"
                        reason = "UNOBSERVABLE_OR_OBSTRUCTED_SUPPORT"
                    else:
                        presence = "U"
                        reason = "POSITIVE_GEOMETRY_WITHOUT_OBSERVABILITY_CONFIRMATION"
                    object_id = str(self.track_objects[(self.track_objects.track_id == track_id) & (self.track_objects.actual_date == date)].r3_object_id.iloc[0])
                    footprint_source = "R3_OBJECT_GEOMETRY"
                else:
                    # Use the nearest existing object footprint only as a
                    # search window; this never fabricates a new geometry.
                    ref_date, ref_geom = next(iter(geom_by_date.items()))
                    stats = self._raw_stats(self.scenes[(block, date)], ref_geom, buffer_m=2.0)
                    object_id = ""
                    footprint_source = f"SEARCH_FOOTPRINT_FROM_{ref_date}"
                    current = self.r3_by_scene[(block, date)]
                    if current.empty:
                        identity_competitor = False
                        identity_competitor_distance_m = None
                        identity_competitor_area_ratio = None
                    else:
                        distances = current.geometry.centroid.distance(ref_geom.centroid)
                        nearest_idx = int(distances.idxmin())
                        nearest_geom = current.loc[nearest_idx].geometry
                        nearest_distance = float(distances.loc[nearest_idx])
                        nearest_ratio = min(float(ref_geom.area), float(nearest_geom.area)) / max(float(ref_geom.area), float(nearest_geom.area), 1e-9)
                        # This is a geometry-scale identity search radius,
                        # not an area-display tier or a B0 threshold.
                        identity_radius = min(80.0, max(20.0, 1.5 * math.sqrt(max(float(ref_geom.area), float(nearest_geom.area)))))
                        identity_competitor = bool(nearest_distance <= identity_radius and nearest_ratio >= 0.25)
                        identity_competitor_distance_m = nearest_distance
                        identity_competitor_area_ratio = nearest_ratio
                    # A missing middle epoch is a gap candidate, not a
                    # demolition candidate, unless the temporal return is
                    # independently supported.  This protects the explicit
                    # 1-U-U / 1-U-1 cases from a false reliable zero.
                    middle_gap_without_return = date == DATES[1] and DATES[2] not in geom_by_date and track_id not in gap_supported
                    if middle_gap_without_return:
                        presence = "U"
                        reason = "GAP_MIDDLE_EPOCH_RETAINED_U"
                    elif stats["black_fraction"] > 0.25 or stats["valid_observable_fraction"] < 0.75 or stats["unobservable_fraction"] > 0.25:
                        presence = "U"
                        reason = "RELIABLE_ZERO_BLOCKED_BY_BLACK_OR_UNOBSERVABLE"
                    elif identity_competitor:
                        presence = "U"
                        reason = "RELIABLE_ZERO_BLOCKED_BY_IDENTITY_COMPETITOR"
                    elif stats["b0_max_probability"] is not None and stats["b0_max_probability"] < THRESHOLD and not stats["candidate_intersection"]:
                        presence = "0"
                        reason = "RELIABLE_ZERO"
                    else:
                        presence = "U"
                        reason = "NO_CANDIDATE_NOT_RELIABLE_ZERO"
                rows.append(
                    {
                        "track_id": track_id,
                        "block_id": block,
                        "actual_date": date,
                        "presence": presence,
                        "reason": reason,
                        "footprint_source": footprint_source,
                        "reliable_zero_evidence": json.dumps(stats, ensure_ascii=False),
                        "identity_competitor": identity_competitor if date not in geom_by_date else False,
                        "identity_competitor_distance_m": identity_competitor_distance_m if date not in geom_by_date else None,
                        "identity_competitor_area_ratio": identity_competitor_area_ratio if date not in geom_by_date else None,
                        "b0_threshold": THRESHOLD,
                        "gap_return_track": track_id in gap_supported,
                        "scientific_status": SCIENTIFIC_STATUS,
                    }
                )
        self.presence = pd.DataFrame(rows)
        self.presence.to_csv(self.out_path("07_Presence", "R3_PRESENCE_1_0_U.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("06_Observability", "OBSERVABILITY_RECEIPT.json"),
            {
                "status": "READ_ONLY_R2_OBSERVABILITY_PLUS_R3_SEARCH_FOOTPRINT",
                "r2_observability_modified": False,
                "reliable_zero_evidence_fields": [
                    "valid_observable_fraction",
                    "nodata_fraction",
                    "partial_fraction",
                    "unobservable_fraction",
                    "black_fraction",
                    "b0_max_probability",
                    "candidate_intersection",
                    "identity_competitor_not_synthesized",
                ],
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )
        json_dump(
            self.out_path("07_Presence", "PRESENCE_RECEIPT.json"),
            {
                "status": "COMPLETED_BOUNDED_CANDIDATE",
                "rounds_used": 1,
                "max_rounds": 2,
                "black_block_policy": "U",
                "clear_bare_land_policy": "RELIABLE_ZERO_IF_OBSERVABLE_AND_NO_B0_CANDIDATE",
                "all_u_not_accepted_as_accuracy_improvement": True,
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )

    def run_state(self) -> None:
        rows: list[dict[str, Any]] = []
        for track_id, grp in self.presence.groupby("track_id"):
            vals = {str(r.actual_date): str(r.presence) for _, r in grp.iterrows()}
            seq = "-".join(vals[d] for d in DATES)
            if seq == "1-0-U" or seq == "1-0-0":
                state, event = "UNCERTAIN", "POSSIBLE_DEMOLITION_REQUIRES_VISUAL_REVIEW"
            elif seq in {"0-0-1", "0-1-1", "0-U-1", "U-0-1"}:
                state, event = "ADDITION_CANDIDATE", "ADDITION_CANDIDATE"
            elif seq == "1-U-1":
                state, event = "GAP_RETURN", "GAP_RETURN"
            elif seq == "1-1-0":
                state, event = "DEMOLITION_CANDIDATE", "DEMOLITION_CANDIDATE"
            elif seq == "1-1-1":
                state, event = "STABLE_CANDIDATE", "NONE"
            else:
                state, event = "UNCERTAIN", "UNKNOWN_SUPPORT"
            rows.append(
                {
                    "track_id": track_id,
                    "block_id": str(grp.block_id.iloc[0]),
                    "presence_sequence": seq,
                    "state": state,
                    "event": event,
                    "state_formal_gt": "NOT_EVALUATED",
                    "scientific_status": SCIENTIFIC_STATUS,
                }
            )
        self.state = pd.DataFrame(rows)
        self.state.to_csv(self.out_path("08_State", "R3_STATES.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("08_State", "STATE_RECEIPT.json"),
            {
                "status": "COMPLETED_CONSERVATIVE_CANDIDATE",
                "state_accuracy": "NOT_EVALUATED",
                "reliable_zero_is_not_demolition_without_review": True,
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )

    def _semantic_features(self, scene: Scene, row: pd.Series) -> dict[str, Any]:
        stats = self._raw_stats(scene, row.geometry)
        return {
            "block_id": row.block_id,
            "actual_date": row.actual_date,
            "r3_object_id": row.r3_object_id,
            "area_m2": float(row.geometry.area),
            "compactness": compactness(row.geometry),
            **stats,
        }

    def run_semantic_and_agriculture(self) -> None:
        rows: list[dict[str, Any]] = []
        for (block, date), gdf in self.r3_by_scene.items():
            scene = self.scenes[(block, date)]
            for _, row in gdf.iterrows():
                rows.append(self._semantic_features(scene, row))
        df = pd.DataFrame(rows)
        if df.empty:
            self.semantic = df
            self.agriculture = df
            return
        for col in ["texture_flatness", "edge_mean", "spectral_range", "compactness", "area_m2"]:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            med = float(values.median())
            mad = float(np.median(np.abs(values - med))) or 1.0
            df[f"{col}_robust_z"] = (values - med) / (1.4826 * mad)
        # Generic diagnostic only.  It is deliberately not a hard rejection.
        df["semantic_anomaly_score"] = (
            df["texture_flatness_robust_z"].abs()
            + df["edge_mean_robust_z"].abs()
            + df["spectral_range_robust_z"].abs()
        ) / 3.0
        cut = float(df.semantic_anomaly_score.quantile(0.97)) if len(df) > 5 else float("inf")
        df["semantic_status"] = np.where(
            (df["b0_max_probability"].fillna(0.0) >= THRESHOLD) & (df["semantic_anomaly_score"] >= cut),
            "B0_SEMANTIC_HARD_NEGATIVE_DIAGNOSTIC",
            "NO_HARD_REJECTION_DIAGNOSTIC",
        )
        df["retraining_or_checkpoint_replacement"] = False
        df["scientific_status"] = SCIENTIFIC_STATUS
        self.semantic = df
        self.semantic.to_csv(self.out_path("09_Semantic误检诊断", "SEMANTIC_DIAGNOSTICS.csv"), index=False, encoding="utf-8-sig")

        ag = df[["block_id", "actual_date", "r3_object_id", "area_m2", "compactness", "texture_flatness", "edge_mean", "spectral_range"]].copy()
        ag["agriculture_scope"] = np.where(
            (ag.texture_flatness < ag.texture_flatness.quantile(0.15)) & (ag.edge_mean < ag.edge_mean.quantile(0.50)),
            "AGRICULTURAL_STRUCTURE_CANDIDATE",
            "UNCERTAIN_SCOPE",
        )
        ag["formal_state_unchanged"] = True
        ag["scientific_status"] = SCIENTIFIC_STATUS
        self.agriculture = ag
        self.agriculture.to_csv(self.out_path("10_农业设施口径", "AGRICULTURE_SCOPE_SIDECAR.csv"), index=False, encoding="utf-8-sig")
        json_dump(
            self.out_path("10_农业设施口径", "AGRICULTURE_SCOPE_RECEIPT.json"),
            {
                "status": "SIDECAR_ONLY",
                "classes": ["BUILDING", "AGRICULTURAL_STRUCTURE_CANDIDATE", "NON_BUILDING", "UNCERTAIN_SCOPE"],
                "formal_presence_state_modified": False,
                "scientific_status": SCIENTIFIC_STATUS,
            },
        )

    def _r2_case_rows(self) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for case_id, block, track_id, tags in FIXED_CASES:
            p = self.path("09_tracking", block, "TRACKING_OBJECTS.csv")
            df = pd.read_csv(p)
            out[case_id] = df[df.track_id.astype(str) == track_id].copy()
        return out

    def _fixed_r3_rows(self, block: str, r2_track_id: str) -> pd.DataFrame:
        r2 = pd.read_csv(self.path("09_tracking", block, "TRACKING_OBJECTS.csv"))
        r2_rows = r2[r2.track_id.astype(str) == r2_track_id]
        source_ids = set(r2_rows.object_id.astype(str))
        objects = self.track_objects[self.track_objects.block_id == block].copy()
        return objects[objects.source_object_id.astype(str).isin(source_ids)].copy()

    def _case_geometry_status(self, case_id: str, r3: pd.DataFrame) -> str:
        if case_id == "B04-TQA10":
            split_parts = r3[r3.r3_object_id.astype(str).str.contains("__I", regex=False)] if not r3.empty else r3
            return "IMPROVED_SPLIT_CANDIDATE" if not split_parts.empty else "UNRESOLVED_OR_AMBIGUOUS"
        if case_id == "B04-TQA07":
            return "MERGE_REDUCED" if any(
                x.get("block_id") == "B04"
                and x.get("status") == "SPLIT_CANDIDATE"
                and str(x.get("source_object_id", "")).endswith("C000047")
                for x in self.instance_records
            ) else "MERGE_UNRESOLVED_NO_INTERNAL_VALLEY"
        return "CANDIDATE_RETAINED"

    def build_fixed_case_comparison(self) -> None:
        r2_cases = self._r2_case_rows()
        rows: list[dict[str, Any]] = []
        for case_id, block, track_id, tags in FIXED_CASES:
            r2 = r2_cases[case_id]
            r3 = self._fixed_r3_rows(block, track_id)
            r2seq = []
            for date in DATES:
                x = r2[r2.actual_date.astype(str) == date]
                r2seq.append("1" if not x.empty else "U")
            pvals = self.presence[self.presence.track_id.astype(str).isin(set(r3.track_id.astype(str)))] if not r3.empty else pd.DataFrame()
            r3seq = []
            if not pvals.empty:
                # Select the most conservative aggregate for a split lineage:
                # positive wins, then zero, then unknown.
                for date in DATES:
                    vals = set(pvals[pvals.actual_date == date].presence.astype(str))
                    r3seq.append("1" if "1" in vals else "0" if "0" in vals else "U")
            else:
                r3seq = ["U", "U", "U"]
            semantic_hits = self.semantic[self.semantic.r3_object_id.astype(str).isin(set(r3.r3_object_id.astype(str)))] if not r3.empty else pd.DataFrame()
            gap_hits = self.out_path("05_Gap_Return", "GAP_RETURN.csv")
            gap_status = "NOT_APPLICABLE"
            if gap_hits.exists() and not r3.empty:
                gd = pd.read_csv(gap_hits)
                hits = gd[gd.track_id.astype(str).isin(set(r3.track_id.astype(str)))]
                if not hits.empty:
                    gap_status = ";".join(sorted(hits.gap_return_status.astype(str).unique()))
            gap_candidate_sequence = ""
            if gap_status in {"GAP_RETURN_SUPPORTED", "GAP_RETURN_CANDIDATE_NO_ID"} or "GAP_RETURN_SUPPORTED" in gap_status or "GAP_RETURN_CANDIDATE_NO_ID" in gap_status:
                gap_candidate_sequence = "1-U-1"
            obs_note = ""
            if case_id == "B02-P2-24":
                obs_note = "2020 black/unobservable must remain U"
            elif case_id == "B01-P2-26":
                obs_note = "2020 clear bare land reliable-zero candidate"
            elif case_id in {"B01-TEX02", "B05-P1-06"}:
                obs_note = "generic semantic diagnostics retained; no automatic rejection"
            tracking_status = "REBUILT_FROM_SCRATCH_CANDIDATE"
            if case_id == "B03-TEX04":
                tracking_status = "IDENTITY_CONTINUITY_CANDIDATE" if len(set(r3.track_id.astype(str))) == 1 else "IDENTITY_AMBIGUOUS_NO_SINGLE_CHAIN"
            semantic_review = "FIXED_ANCHOR_SEMANTIC_REVIEW_REQUIRED" if any(x in tags for x in ["STADIUM", "FIELD", "NON_BUILDING"]) else "GENERIC_DIAGNOSTIC"
            rows.append(
                {
                    "case_id": case_id,
                    "block_id": block,
                    "r2_track_id": track_id,
                    "issue_tags": tags,
                    "r2_object_count": int(len(r2)),
                    "r3_lineage_object_count": int(len(r3)),
                    "r2_presence_sequence": "-".join(r2seq),
                    "r3_presence_sequence": "-".join(r3seq),
                    "r3_gap_candidate_sequence": gap_candidate_sequence,
                    "r3_geometry_status": self._case_geometry_status(case_id, r3),
                    "r3_tracking_status": tracking_status,
                    "r3_gap_status": gap_status,
                    "r3_semantic_statuses": ";".join(sorted(semantic_hits.semantic_status.astype(str).unique())) if not semantic_hits.empty else "",
                    "fixed_anchor_semantic_review": semantic_review,
                    "observability_note": obs_note,
                    "r3_track_ids": ";".join(sorted(set(r3.track_id.astype(str)))) if not r3.empty else "",
                    "scientific_status": SCIENTIFIC_STATUS,
                    "qa_type": "DEVELOPMENT_VISUAL_QA",
                }
            )
        self.fixed_case_rows = rows
        csv_write(self.out_path("11_固定案例前后对照", "固定案例_R2_vs_R3.csv"), rows)

    def _case_object_for_date(self, case_id: str, block: str, track_id: str, date: str) -> tuple[Any | None, list[Any], Any | None]:
        r2 = pd.read_csv(self.path("09_tracking", block, "TRACKING_OBJECTS.csv"))
        r2r = r2[(r2.track_id.astype(str) == track_id) & (r2.actual_date.astype(str) == date)]
        baseline_geom = None
        scene = self.scenes[(block, date)]
        if not r2r.empty:
            gdf = scene.metric_gdf
            src = gdf[gdf.object_id.astype(str) == str(r2r.iloc[0].object_id)]
            if not src.empty:
                baseline_geom = src.geometry.iloc[0]
        r3 = self.r3_by_scene[(block, date)]
        if not r2r.empty:
            cand = r3[r3.source_object_id.astype(str) == str(r2r.iloc[0].object_id)]
        else:
            cand = gpd.GeoDataFrame(columns=r3.columns, geometry=[], crs=r3.crs)
        return baseline_geom, list(cand.geometry), scene

    def build_contact_sheet(self) -> None:
        n = len(FIXED_CASES)
        fig, axes = plt.subplots(n, 3, figsize=(15, max(12, n * 3.1)), squeeze=False)
        for i, (case_id, block, track_id, tags) in enumerate(FIXED_CASES):
            for j, date in enumerate(DATES):
                ax = axes[i, j]
                baseline, repaired, scene = self._case_object_for_date(case_id, block, track_id, date)
                if baseline is None and not repaired:
                    ax.text(0.5, 0.5, "NO R2 OBJECT / SEARCH-FOOTPRINT", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                    ax.set_axis_off()
                    ax.set_title(f"{case_id}\n{date}", fontsize=8)
                    continue
                focus = baseline if baseline is not None else repaired[0]
                win, geom_raster, _ = self._geometry_pixel_mask(scene, focus, pad_px=18)
                r0, r1 = int(win.row_off), int(win.row_off + win.height)
                c0, c1 = int(win.col_off), int(win.col_off + win.width)
                raw = scene.raw[:, r0:r1, c0:c1].astype(float)
                # Display only: stretch the actual source values for inspection.
                rgb = np.transpose(raw[:3], (1, 2, 0))
                lo, hi = np.nanpercentile(rgb, [2, 98])
                rgb = np.clip((rgb - lo) / max(hi - lo, 1.0), 0, 1)
                ax.imshow(rgb)
                for geom, color, label in [(baseline, "#ff3b30", "R2")]:
                    if geom is None:
                        continue
                    g = gpd.GeoSeries([geom], crs=METRIC_CRS).to_crs(scene.crs).iloc[0]
                    xs, ys = g.exterior.xy if isinstance(g, Polygon) else ([], [])
                    if len(xs):
                        rows, cols = rasterio.transform.rowcol(scene.transform, xs, ys)
                        ax.plot(np.asarray(cols) - c0, np.asarray(rows) - r0, color=color, linewidth=1.0, label=label)
                for k, geom in enumerate(repaired):
                    g = gpd.GeoSeries([geom], crs=METRIC_CRS).to_crs(scene.crs).iloc[0]
                    if isinstance(g, Polygon):
                        xs, ys = g.exterior.xy
                        rows, cols = rasterio.transform.rowcol(scene.transform, xs, ys)
                        ax.plot(np.asarray(cols) - c0, np.asarray(rows) - r0, color="#00e676", linewidth=0.9, label="R3" if k == 0 else None)
                ax.set_title(f"{case_id}\n{date}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(tags.replace(";", " / "), fontsize=7)
                if j == 2:
                    ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
        fig.suptitle("Xi'an R3 固定案例 R2 红线 / R3 绿线（开发视觉复核，不是 Ground Truth）", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        fig.savefig(self.out_path("11_固定案例前后对照", "固定案例_R2_vs_R3总览.png"), dpi=130)
        plt.close(fig)
        shutil.copy2(
            self.out_path("11_固定案例前后对照", "固定案例_R2_vs_R3总览.png"),
            self.out_path("12_视觉独立复核", "固定案例_R2_vs_R3总览.png"),
        )
        json_dump(
            self.out_path("12_视觉独立复核", "视觉复核记录.json"),
            {
                "status": "PENDING_PARENT_VISUAL_REVIEW",
                "qa_type": "DEVELOPMENT_VISUAL_QA",
                "visual_reviewer_verdict": "NOT_YET_RECORDED",
                "allowed_verdicts": ["IMPROVE", "SAME", "WORSE"],
                "no_new_science_rules_from_contact_sheet": True,
                "scientific_accuracy": "NOT_EVALUATED",
            },
        )

    def write_iteration_and_parameters(self) -> None:
        rows = [
            {
                "round": 1,
                "layer": "INSTANCE",
                "action": "REPRODUCE_HYPOTHESIS_ONE_LAYER_CHANGE_RUN",
                "change": "DT_LOCAL_MAXIMA_PLUS_PROBABILITY_VALLEY_WATERSHED",
                "result": "COMPLETED_CANDIDATE",
                "keep_or_revert": "KEEP_PENDING_PARENT_VISUAL_REVIEW",
            },
            {
                "round": 1,
                "layer": "TRACKING",
                "action": "FROM_SCRATCH",
                "change": "ALIGNMENT_AWARE_DISTANCE_OVERLAP_SHAPE_AREA_NEIGHBORHOOD",
                "result": "COMPLETED_CANDIDATE",
                "keep_or_revert": "KEEP_PENDING_PARENT_VISUAL_REVIEW",
            },
            {
                "round": 1,
                "layer": "PRESENCE",
                "action": "RELIABLE_ZERO_DIAGNOSTIC",
                "change": "OBSERVABILITY_RAW_BLACKNESS_B0_CANDIDATE_COMPETITOR",
                "result": "COMPLETED_CANDIDATE",
                "keep_or_revert": "KEEP_PENDING_PARENT_VISUAL_REVIEW",
            },
        ]
        csv_write(self.out_path("13_迭代记录", "ITERATION_LOG.csv"), rows)
        params = {
            "frozen": True,
            "b0_threshold": THRESHOLD,
            "forbidden_scientific_area_tiers": [10, 50, 100, 200, 500],
            "instance": {
                "max_iterations": self.max_instance_rounds,
                "min_distance_px_formula": "max(12, round(min(30, max(12, dt_max*0.35))))",
                "peak_threshold_formula": "max(3, dt_max*0.18)",
                "valley_drop_diagnostic": 0.08,
                "internal_boundary_evidence": False,
                "watershed": "scipy.ndimage.watershed_ift_if_valley_evidence",
            },
            "tracking": {
                "max_iterations": 3,
                "from_scratch": True,
                "one_to_one_score_components": ["alignment_distance", "overlap", "shape", "area_ratio", "local_neighborhood_consistency"],
                "identity_conflict": "low_neighborhood_consistency_plus_small_competition_gap",
                "final_geometry_warped": False,
            },
            "presence": {
                "max_iterations": 2,
                "black_or_unobservable": "U",
                "reliable_zero_requires": ["observable_support", "no_black_block", "B0_max_below_0.50", "no_candidate_intersection"],
            },
            "semantic": {"retraining": False, "checkpoint_replacement": False, "hard_rejection": False},
            "scientific_status": SCIENTIFIC_STATUS,
        }
        json_dump(self.out_path("14_最终冻结参数", "FINAL_FROZEN_PARAMETERS.json"), params)

    def write_final_report(self) -> None:
        instance_df = pd.DataFrame(self.instance_records)
        split_count = int((instance_df.status == "SPLIT_CANDIDATE").sum()) if not instance_df.empty else 0
        ambiguous_count = int(instance_df.status.astype(str).str.contains("AMBIGUOUS").sum()) if not instance_df.empty else 0
        fragmentation_controls = {}
        for case_id in ["B05-TEX01", "OBJ_0008", "B01-TEX01"]:
            row = next((x for x in self.fixed_case_rows if x["case_id"] == case_id), None)
            fragmentation_controls[case_id] = row["r3_geometry_status"] if row else "NOT_EVALUATED"
        report = {
            "task_id": "XIAN-R3-VISION-GUIDED-SCIENTIFIC-REPAIR-001",
            "status": "PARTIAL",
            "b0_threshold": THRESHOLD,
            "area_filter_used_scientifically": False,
            "scientific_accuracy": "NOT_EVALUATED",
            "qa_type": "DEVELOPMENT_VISUAL_QA",
            "ready_for_scale_up": "NO",
            "next": "STOP_FOR_PARENT_VISUAL_REVIEW",
            "instance": {
                "status": "PARTIAL",
                "split_candidate_count": split_count,
                "ambiguous_count": ambiguous_count,
                "controls": fragmentation_controls,
            },
            "tracking": {
                "status": "PARTIAL",
                "edge_count": int(len(self.track_edges)),
                "identity_conflict_count": int((self.track_edges.relation == "IDENTITY_CONFLICT").sum()) if not self.track_edges.empty else 0,
                "explicit_relations": sorted(self.track_edges.relation.dropna().unique().tolist()) if not self.track_edges.empty else [],
            },
            "observability": {"status": "PARTIAL", "r2_observability_read_only": True},
            "presence": {
                "status": "PARTIAL",
                "counts": self.presence.presence.value_counts().to_dict() if not self.presence.empty else {},
                "reliable_zero_is_evidence_gated": True,
                "all_u_is_not_accepted": True,
            },
            "state": {"status": "PARTIAL", "counts": self.state.state.value_counts().to_dict() if not self.state.empty else {}},
            "semantic": {
                "status": "DIAGNOSTIC_ONLY",
                "hard_negative_count": int((self.semantic.semantic_status == "B0_SEMANTIC_HARD_NEGATIVE_DIAGNOSTIC").sum()) if not self.semantic.empty else 0,
                "fixed_false_positive_cases": ["B01-TEX02", "B05-P1-06"],
                "retraining": False,
            },
            "gap_return": {"status": "CONSERVATIVE", "no_forced_demolition_plus_addition": True},
            "provenance": {
                "raw_sources_modified": False,
                "r2_outputs_modified": False,
                "holdout_read": False,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "source_dates": DATES,
            },
        }
        json_dump(self.out_path("15_最终报告", "FINAL_REPORT.json"), report)
        lines = [
            "# 西安 R3 视觉引导科学修复最终报告",
            "",
            f"- STATUS: **{report['status']}**",
            f"- B0 threshold: **{THRESHOLD:.2f}（未重跑、未改变）**",
            "- 面积挡位：仅显示用途，未进入实例、追踪、presence 或 state 科学链。",
            "- SCIENTIFIC_ACCURACY: **NOT_EVALUATED**",
            "- QA_TYPE: **DEVELOPMENT_VISUAL_QA**",
            "- READY_FOR_SCALE_UP: **NO**",
            "- NEXT: **STOP_FOR_PARENT_VISUAL_REVIEW**",
            "",
            "## 分层结论",
            "",
            f"- Instance: PARTIAL；检测到 {split_count} 个分裂候选，保留 {ambiguous_count} 个需要视觉复核的歧义/未决对象。",
            "- Tracking: PARTIAL；从 R3 冻结 geometry 重新建图，保留 1:1、1:N、N:1、AMBIGUOUS 和 IDENTITY_CONFLICT 边。",
            "- Observability: PARTIAL；R2 可观测性只读，新增搜索足迹证据。",
            "- Presence: PARTIAL；黑块/不可观测证据进入 U，清晰可观测且 B0 无候选的裸地才可成为可靠 0 候选。",
            "- State: PARTIAL；未把候选状态当作 Ground Truth。",
            "- Semantic: DIAGNOSTIC_ONLY；误检仅记录，不替换 checkpoint、不训练。",
            "",
            "## 固定硬验收状态",
            "",
        ]
        for row in self.fixed_case_rows:
            lines.append(f"- {row['case_id']}: geometry={row['r3_geometry_status']}; R2={row['r2_presence_sequence']}; R3={row['r3_presence_sequence']}; gap={row['r3_gap_status']}")
        lines += [
            "",
            "## 保护边界",
            "",
            "原始影像、R2 B0/mask/vector/tracking、checkpoint、MicroGT 开发/封存边界均未作为可写目标；本包全部为新候选输出。",
        ]
        self.out_path("15_最终报告", "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> None:
        self.assert_inputs()
        for name in [
            "00_任务说明",
            "01_Instance诊断",
            "02_Instance修复",
            "03_最终Geometry",
            "04_Tracking",
            "05_Gap_Return",
            "06_Observability",
            "07_Presence",
            "08_State",
            "09_Semantic误检诊断",
            "10_农业设施口径",
            "11_固定案例前后对照",
            "12_视觉独立复核",
            "13_迭代记录",
            "14_最终冻结参数",
            "15_最终报告",
        ]:
            self.out_path(name).mkdir(parents=True, exist_ok=True)
        self.load_scenes()
        self.write_task_spec()
        self.run_instance_repair()
        self.freeze_geometry()
        self.run_tracking()
        self.run_gap_return()
        self.run_presence()
        self.run_state()
        self.run_semantic_and_agriculture()
        self.build_fixed_case_comparison()
        self.build_contact_sheet()
        self.write_iteration_and_parameters()
        self.write_final_report()
        json_dump(
            self.out_path("R3_RUN_RECEIPT.json"),
            {
                "status": "COMPLETED_PARTIAL_CANDIDATE",
                "completed_at_utc": utc_now(),
                "output_root": str(self.out),
                "scientific_accuracy": "NOT_EVALUATED",
                "qa_type": "DEVELOPMENT_VISUAL_QA",
                "ready_for_scale_up": "NO",
                "next": "STOP_FOR_PARENT_VISUAL_REVIEW",
            },
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r2-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-instance-rounds", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runner = R3Runner(args.r2_root, args.output_root, max_instance_rounds=args.max_instance_rounds)
    runner.run()
    print(json.dumps({"status": "COMPLETED_PARTIAL_CANDIDATE", "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
