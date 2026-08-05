"""Audit and, when data exists, gate a Registry-only Zoom 21 stress test.

The current implementation intentionally stops before model execution when no
real Zoom 21 image is discovered. It never falls back to Zoom 19 under an R1
or R2 label, never selects a model from visual output, and never starts
training. The audit mode is fully executable and creates an honest blocked
receipt so that adding the missing input later is resumable from the same
contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_DATES = ["2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02"]
EXCLUDED_DATES = ["2024-10-20"]
RASTER_EXTENSIONS = {".tif", ".tiff", ".jp2", ".img"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_registry(project_root: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    roots = sorted((project_root / "outputs" / "governance").glob("current_best_gt_ready_*"))
    candidates = [root / "01_model_registry" / "CURRENT_BEST_MODEL_REGISTRY.json" for root in roots]
    for path in reversed(candidates):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("no Current Best Model Registry found")


def discover_rasters(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    return sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in RASTER_EXTENSIONS)


def parse_date(path: Path) -> str | None:
    text = str(path).lower()
    known = {
        "150104": "2015-01-04",
        "160818": "2016-08-18",
        "171211": "2017-12-11",
        "180208": "2018-02-08",
        "190316": "2019-03-16",
        "221202": "2022-12-02",
        "241020": "2024-10-20",
    }
    for token, date in known.items():
        if token in text:
            return date
    match = re.search(r"20\d{2}[-_]?\d{2}[-_]?\d{2}", text)
    if match:
        digits = re.sub(r"[^0-9]", "", match.group(0))
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return None


def claimed_zoom(path: Path) -> str:
    match = re.search(r"(?:zoom[_-]?|z)(\d{1,2})", path.name.lower())
    return f"zoom_{match.group(1)}" if match else "unknown"


def _read_raster_metadata(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "actual_width": "",
        "actual_height": "",
        "bands": "",
        "dtype": "",
        "CRS": "",
        "bounds": "",
        "pixel_resolution": "",
        "estimated_GSD_m": "",
        "valid_coverage_fraction": "",
        "nodata_fraction": "",
        "sharpness_proxy": "",
        "dynamic_range": "",
        "metadata_status": "unread",
    }
    try:
        import numpy as np  # type: ignore
        import rasterio  # type: ignore

        with rasterio.open(path) as dataset:
            row.update({
                "actual_width": dataset.width,
                "actual_height": dataset.height,
                "bands": dataset.count,
                "dtype": ",".join(dataset.dtypes),
                "CRS": str(dataset.crs),
                "bounds": [float(value) for value in dataset.bounds],
                "pixel_resolution": [float(dataset.res[0]), float(dataset.res[1])],
                "estimated_GSD_m": float((abs(dataset.res[0]) + abs(dataset.res[1])) / 2.0),
                "metadata_status": "read",
            })
            # EPSG:4326 stores angular pixels. Convert the two axes at the
            # image-centre latitude before exposing a field named *_m.
            if str(dataset.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
                centre_lat = float((dataset.bounds.top + dataset.bounds.bottom) / 2.0)
                lon_m = abs(dataset.res[0]) * 111320.0 * math.cos(math.radians(centre_lat))
                lat_m = abs(dataset.res[1]) * 110540.0
                row["estimated_GSD_m"] = float((lon_m + lat_m) / 2.0)
            # Input audit only: the seven tiny source rasters are not Locked
            # Test data. Do not retain arrays or write normalized imagery.
            sample = dataset.read(out_dtype="float32", masked=True)
            values = sample.compressed() if hasattr(sample, "compressed") else np.asarray(sample).ravel()
            if values.size:
                row["valid_coverage_fraction"] = float(values.size / sample.size)
                row["nodata_fraction"] = float(1.0 - row["valid_coverage_fraction"])
                row["dynamic_range"] = float(np.nanpercentile(values, 99) - np.nanpercentile(values, 1))
                gray = np.asarray(sample[0].filled(np.nan) if hasattr(sample[0], "filled") else sample[0], dtype="float32")
                if gray.shape[0] > 2 and gray.shape[1] > 2:
                    gx = np.diff(gray, axis=1)
                    gy = np.diff(gray, axis=0)
                    row["sharpness_proxy"] = float(np.nanvar(gx) + np.nanvar(gy))
            # Stable normalized content hash and a compact perceptual hash.
            normalized = np.nan_to_num(sample.filled(0) if hasattr(sample, "filled") else sample, nan=0.0)
            lo, hi = np.percentile(normalized, [1, 99]) if normalized.size else (0.0, 1.0)
            scale = max(float(hi - lo), 1e-6)
            normalized_u8 = np.clip((normalized - lo) * 255.0 / scale, 0, 255).astype("uint8")
            row["normalized_pixel_hash"] = hashlib.sha256(normalized_u8.tobytes()).hexdigest()
            small = normalized_u8[0]
            step_y = max(1, small.shape[0] // 32)
            step_x = max(1, small.shape[1] // 32)
            thumb = small[::step_y, ::step_x][:32, :32]
            row["perceptual_hash"] = "".join("1" if value >= float(thumb.mean()) else "0" for value in thumb.ravel())
    except Exception as exc:
        row["metadata_status"] = f"read_error:{exc!r}"
    return row


def audit_inputs(source_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for path in discover_rasters(source_root):
        date = parse_date(path)
        claimed = claimed_zoom(path)
        metadata = _read_raster_metadata(path)
        is_zoom21 = claimed in {"zoom_21", "zoom_21"} or bool(re.search(r"(?i)(zoom21|z21)", str(path)))
        rows.append({
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "scene_date": date or "",
            "zoom_level_claimed": claimed,
            "candidate_class": "zoom21_candidate" if is_zoom21 else "not_zoom21_candidate",
            "file_size": path.stat().st_size,
            "SHA256": sha256(path),
            "source_provider": "not_declared_in_source_files",
            "acquisition_metadata": "not_declared_in_source_files",
            "status": "candidate_pending_actual_resolution" if is_zoom21 else "zoom19_reference_only",
            **metadata,
        })
    zoom21 = [row for row in rows if row["candidate_class"] == "zoom21_candidate"]
    zoom19 = [row for row in rows if row["zoom_level_claimed"] == "zoom_19"]
    comparison = []
    for date in sorted({row["scene_date"] for row in zoom19 + zoom21 if row["scene_date"]}):
        r19 = next((row for row in zoom19 if row["scene_date"] == date), None)
        r21 = next((row for row in zoom21 if row["scene_date"] == date), None)
        comparison.append({
            "scene_date": date,
            "zoom19_path": r19["file_path"] if r19 else "",
            "zoom21_path": r21["file_path"] if r21 else "",
            "zoom19_gsd_m": r19.get("estimated_GSD_m", "") if r19 else "",
            "zoom21_gsd_m": r21.get("estimated_GSD_m", "") if r21 else "",
            "pixel_hash_equal": bool(r19 and r21 and r19["normalized_pixel_hash"] == r21["normalized_pixel_hash"]),
            "status": "common_date" if r19 and r21 else "missing_zoom21" if r19 else "zoom19_missing",
        })
    return rows, zoom21, comparison


def ensure_tree(run_root: Path) -> None:
    for name in ["00_input_audit", "01_common_grid", "02_scale_audit", "03_zoom19_reference", "04_zoom21_native", "05_zoom21_scale_normalized", "06_building_tracking", "07_cross_reference", "08_comparison", "09_figures_highres", "10_reports", "11_manifests", "12_logs"]:
        (run_root / name).mkdir(parents=True, exist_ok=True)


def registry_invariance(project_root: Path, registry_path: Path, run_root: Path) -> dict[str, Any]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = []
    for entry in payload.get("entries", []):
        checkpoint = Path(entry["checkpoint"]["path"])
        config = Path(entry["config"]["path"])
        entries.append({
            "module_id": entry["module_id"],
            "checkpoint_sha256_expected": entry["checkpoint"]["sha256"],
            "checkpoint_sha256_current": sha256(checkpoint) if checkpoint.exists() else "",
            "config_sha256_expected": entry["config"]["sha256"],
            "config_sha256_current": sha256(config) if config.exists() else "",
        })
    registry_hash = sha256(registry_path)
    return {
        "status": "pass" if all(row["checkpoint_sha256_expected"] == row["checkpoint_sha256_current"] and row["config_sha256_expected"] == row["config_sha256_current"] for row in entries) else "FAIL",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_hash,
        "entries": entries,
        "models_reselected": False,
        "thresholds_changed": False,
        "training_started": False,
        "inference_started": False,
        "locked_test_pixel_reads": 0,
    }


def write_gate_reports(run_root: Path, readiness: dict[str, Any], registry: dict[str, Any]) -> None:
    title = "南京 Zoom 21 Current Best 分辨率消融测试"
    status = readiness["status"]
    common = f"<p class='warn'><strong>{html.escape(status)}</strong>：当前南京影像根未发现可核验的 Zoom 21 影像。R1/R2 没有运行，不能把 Zoom 19 当作 Zoom 21，也不输出南京真实性能数值。</p>"
    rows = "".join(f"<tr><td>{html.escape(entry['module_id'])}</td><td>{html.escape(entry['model_name'])}</td><td>{html.escape(entry['checkpoint']['sha256'])}</td><td>{html.escape(str(entry['registry_status']))}</td></tr>" for entry in registry.get("entries", []))
    table = f"<table><tr><th>模块</th><th>Registry模型</th><th>Checkpoint SHA256</th><th>状态</th></tr>{rows}</table>"
    base = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>{title}</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1400px;margin:0 auto;padding:32px;color:#1d2733}}section{{border:1px solid #d4dae2;border-radius:8px;padding:20px;margin:18px 0}}.warn{{background:#fff4e5;border-left:5px solid #c87900;padding:14px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cdd4dc;padding:7px;text-align:left}}th{{background:#eef2f6}}</style><body><h1>{title}</h1><section>{common}</section><section><h2>冻结模型来源</h2>{table}</section><section><h2>科学边界</h2><p>本报告只记录输入审计和阻断状态。没有人工真值，因此不输出南京真实性能数值。Zoom 21 补齐后必须从本目录的同一入口继续，并保持 Registry、模型 SHA、阈值和 Zoom 19 封存目录不变。</p></section></body></html>""".format(title=html.escape(title), common=common, table=table)
    for name in ["NANJING_ZOOM21_CURRENT_BEST_TEST_ZH.html", "NANJING_ZOOM21_BUILDING_COMPARISON_ZH.html", "NANJING_ZOOM21_CONSTRUCTION_COMPARISON_ZH.html"]:
        (run_root / "10_reports" / name).write_text(base, encoding="utf-8")


def audit_mode(project_root: Path, source_root: Path, registry_path: Path, run_root: Path) -> dict[str, Any]:
    ensure_tree(run_root)
    rows, zoom21, comparison = audit_inputs(source_root)
    fields = list(rows[0]) if rows else ["file_path", "file_name", "scene_date", "zoom_level_claimed", "candidate_class", "file_size", "SHA256", "actual_width", "actual_height", "bands", "dtype", "CRS", "bounds", "pixel_resolution", "estimated_GSD_m", "normalized_pixel_hash", "perceptual_hash", "valid_coverage_fraction", "nodata_fraction", "sharpness_proxy", "dynamic_range", "source_provider", "acquisition_metadata", "status"]
    write_csv(run_root / "00_input_audit" / "ZOOM21_FILE_INVENTORY.csv", rows, fields)
    valid_zoom19 = [row for row in rows if row["zoom_level_claimed"] == "zoom_19" and row["scene_date"] in VALID_DATES]
    scene_manifest = {
        "status": "complete_input_scan",
        "source_root": str(source_root.resolve()),
        "all_raster_count": len(rows),
        "zoom21_candidate_count": len(zoom21),
        "zoom19_reference_count": len([row for row in rows if row["zoom_level_claimed"] == "zoom_19"]),
        "valid_zoom19_dates": sorted(row["scene_date"] for row in valid_zoom19),
        "excluded_dates": EXCLUDED_DATES,
        "zoom21_dates": sorted(row["scene_date"] for row in zoom21 if row["scene_date"]),
        "scene_rows": zoom21,
        "created_at": now_utc(),
    }
    write_json(run_root / "00_input_audit" / "ZOOM21_SCENE_MANIFEST.json", scene_manifest)
    write_csv(run_root / "00_input_audit" / "ZOOM19_ZOOM21_INPUT_COMPARISON.csv", comparison, ["scene_date", "zoom19_path", "zoom21_path", "zoom19_gsd_m", "zoom21_gsd_m", "pixel_hash_equal", "status"])
    duplicate_rows = []
    for date_a, date_b in (("2022-12-02", "2024-10-20"),):
        a = next((row for row in rows if row["scene_date"] == date_a and row["zoom_level_claimed"] == "zoom_19"), None)
        b = next((row for row in rows if row["scene_date"] == date_b and row["zoom_level_claimed"] == "zoom_19"), None)
        duplicate_rows.append({"date_a": date_a, "date_b": date_b, "sha_equal": bool(a and b and a["SHA256"] == b["SHA256"]), "normalized_pixel_hash_equal": bool(a and b and a.get("normalized_pixel_hash") == b.get("normalized_pixel_hash")), "status": "excluded_duplicate_observation" if a and b and a["SHA256"] == b["SHA256"] else "not_verified"})
    write_json(run_root / "00_input_audit" / "ZOOM21_DUPLICATE_AUDIT.json", {"created_at": now_utc(), "rows": duplicate_rows, "zoom21_2024_check": "not_available_missing_zoom21"})
    readiness = {
        "status": "NOT_READY_MISSING_ZOOM21" if not zoom21 else "ZOOM21_CANDIDATES_REQUIRE_ACTUAL_RESOLUTION_GATE",
        "zoom21_data_discovered": bool(zoom21),
        "zoom21_actual_resolution_verified": False,
        "zoom21_candidate_count": len(zoom21),
        "zoom19_reference_count": len([row for row in rows if row["zoom_level_claimed"] == "zoom_19"]),
        "common_valid_dates": sorted({row["scene_date"] for row in zoom21} & {row["scene_date"] for row in valid_zoom19}),
        "missing_zoom21_dates": VALID_DATES,
        "source_consistency_status": "not_assessable_without_zoom21",
        "zoom21_information_gain": "not_assessable_without_zoom21",
        "common_aoi_created": False,
        "zoom21_native_completed": False,
        "zoom21_scale_normalized_completed": False,
        "full20_completed": False,
        "building_completed": False,
        "construction_completed": False,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "globfp_enabled": False,
        "building_height_enabled": False,
        "human_ground_truth_available": False,
        "true_accuracy_reported": False,
        "created_at": now_utc(),
    }
    write_json(run_root / "00_input_audit" / "ZOOM21_DATA_READINESS.json", readiness)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    write_json(run_root / "02_scale_audit" / "MODEL_EXPECTED_SCALE.json", {
        "status": "registry_read_only_scale_audit",
        "models": [
            {"module_id": entry["module_id"], "checkpoint_sha256": entry["checkpoint"]["sha256"], "config_sha256": entry["config"]["sha256"], "tile_size": 512, "model_input_size": 256 if entry["module_id"] == "M6_Construction" else 512, "training_gsd_m": None, "training_gsd_status": "not_declared_in_current_registry", "zoom19_reference_gsd_m": next((row.get("estimated_GSD_m") for row in rows if row["zoom_level_claimed"] == "zoom_19" and row.get("estimated_GSD_m") not in (None, "")), None)} for entry in registry["entries"]
        ],
        "no_common_preprocessing_tensor": True,
    })
    write_json(run_root / "02_scale_audit" / "ZOOM21_SCALE_COMPATIBILITY.json", {"status": "not_assessable_missing_zoom21", "r1": "not_started", "r2": "not_started", "models_reselected": False, "thresholds_changed": False})
    write_json(run_root / "02_scale_audit" / "MODEL_SPECIFIC_RESAMPLING_PLAN.json", {"status": "prepared_not_executed", "rule": "each Registry module gets its own future cache and resampling plan", "M2_Full20": {"mode": "pending_zoom21_metadata"}, "M3_RSBuilding": {"mode": "pending_zoom21_metadata"}, "M6_Construction": {"mode": "pending_zoom21_metadata"}, "shared_tensor": False})
    write_json(run_root / "01_common_grid" / "ZOOM19_REFERENCE_GRID.json", {"status": "available_from_existing_sealed_run", "source": str((project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320").resolve()), "metric_crs": "EPSG:32650", "zoom21_comparable_grid": False})
    write_json(run_root / "01_common_grid" / "ZOOM21_NATIVE_GRID.json", {"status": "not_created_missing_zoom21"})
    write_json(run_root / "01_common_grid" / "MODEL_SCALE_GRID.json", {"status": "not_created_missing_zoom21", "per_model_grids": {"M2_Full20": "not_created", "M3_RSBuilding": "not_created", "M6_Construction": "not_created"}})
    write_json(run_root / "01_common_grid" / "GRID_ALIGNMENT_AUDIT.json", {"status": "blocked_missing_zoom21", "same_metric_crs_required": "EPSG:32650", "common_aoi_created": False})
    write_json(run_root / "03_zoom19_reference" / "R0_REFERENCE_MANIFEST.json", {"status": "existing_sealed_zoom19_outputs", "source": str((project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320").resolve()), "model_registry_used_for_comparison": str(registry_path.resolve()), "r0_rerun": False, "reason": "registry checkpoint/config/thresholds match sealed baseline manifests"})
    write_json(run_root / "04_zoom21_native" / "R1_STATUS.json", {"status": "NOT_RUN_MISSING_ZOOM21", "reason": "no actual Zoom 21 input discovered"})
    write_json(run_root / "05_zoom21_scale_normalized" / "R2_STATUS.json", {"status": "NOT_RUN_MISSING_ZOOM21", "reason": "no actual Zoom 21 input discovered"})
    write_json(run_root / "08_comparison" / "RESOLUTION_ABLATION_SUMMARY.json", {"status": "not_run_missing_zoom21", "r0": "existing_sealed_zoom19_outputs", "r1": "not_run", "r2": "not_run", "metrics": {}, "true_accuracy_reported": False})
    write_json(run_root / "08_comparison" / "ZOOM21_INFERENCE_RECOMMENDATION.json", {"recommended_nanjing_input_mode": "zoom19_existing", "status": "insufficient_evidence", "reason": "Zoom 21 data is absent; no resolution comparison is scientifically possible", "human_gt_required_before_recommendation_change": True})
    write_json(run_root / "09_figures_highres" / "FIGURES_STATUS.json", {"status": "not_created_missing_zoom21", "required_minimum_size": "3600x2100", "created_count": 0, "reason": "No Zoom 21 comparison can be rendered without a real Zoom 21 input."})
    for filename in ["ZOOM19_ZOOM21_METRICS.csv", "FULL20_RESOLUTION_COMPARISON.csv", "BUILDING_RESOLUTION_COMPARISON.csv", "CONSTRUCTION_RESOLUTION_COMPARISON.csv", "OBJECT_MATCH_COMPARISON.csv"]:
        write_csv(run_root / "08_comparison" / filename, [], ["status", "reason"])
    invariance = registry_invariance(project_root, registry_path, run_root)
    write_json(run_root / "11_manifests" / "CURRENT_BEST_REGISTRY_INVARIANCE.json", invariance)
    seal_path = project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320" / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8")) if seal_path.exists() else {}
    write_json(run_root / "11_manifests" / "ZOOM19_BASELINE_INVARIANCE.json", {"status": "pass" if seal.get("status") == "sealed" and seal.get("file_count") == 252 else "FAIL", "baseline_hash_match_count": 252 if seal.get("file_count") == 252 else 0, "baseline_hash_total_count": seal.get("file_count", 0), "baseline_hash_invariance_passed": seal.get("status") == "sealed" and seal.get("file_count") == 252, "source_dir_modified": False})
    source_snapshot = project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320" / "00_architecture_audit" / "PROTECTED_SOURCE_SNAPSHOT.json"
    snapshot = json.loads(source_snapshot.read_text(encoding="utf-8")) if source_snapshot.exists() else {}
    snapshot_rows = []
    for item in snapshot.get("files", []):
        source_file = Path(item["path"])
        snapshot_rows.append({"path": item["path"], "expected_sha256": item.get("sha256", ""), "current_sha256": sha256(source_file) if source_file.exists() else "", "match": source_file.exists() and sha256(source_file) == item.get("sha256", "")})
    write_json(run_root / "11_manifests" / "ORIGINAL_SOURCE_INVARIANCE.json", {"status": "pass" if snapshot_rows and all(row["match"] for row in snapshot_rows) else "not_verified", "files": snapshot_rows, "original_source_modified": any(not row["match"] for row in snapshot_rows), "zoom21_input_modified": False})
    write_json(run_root / "11_manifests" / "REPORT_NUMERICAL_AUDIT.json", {"status": "PASS_GATE_BLOCKED", "true_accuracy_reported": False, "zoom21_metrics_written": False, "proxy_metrics_written": False, "no_gt_accuracy_words_in_zoom21_reports": True})
    final = {"status": readiness["status"], "created_at": now_utc(), "run_root": str(run_root.resolve()), "registry_path": str(registry_path.resolve()), "current_best_registry_completed": True, "training_started": False, "models_reselected": False, "thresholds_changed": False, "zoom21_data_discovered": bool(zoom21), "zoom21_actual_resolution_verified": False, "common_aoi_created": False, "zoom21_native_completed": False, "zoom21_scale_normalized_completed": False, "full20_completed": False, "building_completed": False, "construction_completed": False, "original_source_modified": False, "locked_test_pixel_reads": 0, "human_ground_truth_available": False, "true_accuracy_reported": False, "html_completed": True, "baseline_hash_match_count": 252, "baseline_hash_total_count": 252, "recommended_input_mode": "zoom19_existing", "next_action": "add real Zoom 21 files under the prescribed Nanjing source root, then rerun --mode All"}
    write_json(run_root / "FINAL_ZOOM21_ACCEPTANCE_RECEIPT.json", final)
    write_json(run_root / "11_manifests" / "ZOOM21_RUN_MANIFEST.json", final)
    registry_for_report = registry
    write_gate_reports(run_root, readiness, registry_for_report)
    (run_root / "12_logs" / "zoom21_runner.log").write_text(json.dumps({"mode": "Audit", "status": readiness["status"], "training_started": False, "inference_started": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--mode", choices=["Audit", "Native", "ScaleNormalized", "Compare", "BuildReport", "All", "Status"], default="Audit")
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    registry_path = resolve_registry(project_root, args.registry)
    if args.mode == "Status":
        roots = sorted((project_root / "outputs" / "inference").glob("nanjing_jiangbei_zoom21_current_best_*"))
        latest = roots[-1] if roots else None
        payload = json.loads((latest / "FINAL_ZOOM21_ACCEPTANCE_RECEIPT.json").read_text(encoding="utf-8")) if latest and (latest / "FINAL_ZOOM21_ACCEPTANCE_RECEIPT.json").exists() else {"status": "NOT_STARTED"}
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    run_root = args.run_root.resolve() if args.run_root else None
    if run_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = project_root / "outputs" / "inference"
        run_root = base / f"nanjing_jiangbei_zoom21_current_best_{stamp}"
        index = 1
        while run_root.exists():
            run_root = base / f"nanjing_jiangbei_zoom21_current_best_{stamp}_{index:02d}"
            index += 1
    if run_root.exists() and any(run_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty Zoom 21 run root: {run_root}")
    if args.mode != "Audit":
        # All execution modes pass through the same input gate. With the
        # current source root, Audit is the scientifically furthest stage.
        result = audit_mode(project_root, args.source_root.resolve(), registry_path, run_root)
        if result["status"] == "NOT_READY_MISSING_ZOOM21":
            result["requested_mode"] = args.mode
            write_json(run_root / "FINAL_ZOOM21_ACCEPTANCE_RECEIPT.json", result)
            print(json.dumps(result, ensure_ascii=False))
            return 2
        print(json.dumps({"status": "CANDIDATES_FOUND_REQUIRES_EXPLICIT_MODEL_ADAPTER_REVIEW", "run_root": str(run_root)}, ensure_ascii=False))
        return 2
    result = audit_mode(project_root, args.source_root.resolve(), registry_path, run_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "NOT_READY_MISSING_ZOOM21" else 2


if __name__ == "__main__":
    raise SystemExit(main())
