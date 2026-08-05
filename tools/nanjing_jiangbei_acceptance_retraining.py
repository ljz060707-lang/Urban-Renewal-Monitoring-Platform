from __future__ import annotations

"""Formal acceptance and isolated follow-up runner for Nanjing Jiangbei.

The first mode in this file is deliberately read-only with respect to the
completed modular run, the source scenes, the sealed baseline, and the
Tongzhou reference outputs.  Follow-up modes are added in later stages of the
same isolated run root.  Keeping the acceptance receipt and the new training
artifacts in a new root makes the old modular output an auditable input rather
than a mutable workspace.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


TOOL_PATH = Path(__file__).resolve()
PROJECT_ROOT = TOOL_PATH.parents[1]
INFERENCE_ROOT = PROJECT_ROOT / "outputs" / "inference"
TRAINING_ROOT = PROJECT_ROOT / "outputs" / "training"
DEFAULT_OLD_OUTPUT = INFERENCE_ROOT / "nanjing_jiangbei_modular_rebuild_20260803_230320"
DEFAULT_SOURCE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei")
VALID_DATES = (
    "2015-01-04",
    "2016-08-18",
    "2017-12-11",
    "2018-02-08",
    "2019-03-16",
    "2022-12-02",
)
EXCLUDED_DATE = "2024-10-20"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _path_text(path: Path) -> str:
    return str(path.resolve())


def _require_file(path: Path, missing: list[str]) -> bool:
    exists = path.is_file()
    if not exists:
        missing.append(_path_text(path))
    return exists


def _source_snapshot(source_root: Path) -> dict[str, Any]:
    """Record source files and scene metadata without creating derived input."""

    import rasterio

    records: list[dict[str, Any]] = []
    for path in sorted(source_root.rglob("*.tif")):
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", path.name)
        if match is None:
            continue
        token = match.group(1)
        date = datetime.strptime(token, "%y%m%d").date().isoformat()
        with rasterio.open(path) as dataset:
            records.append(
                {
                    "date": date,
                    "path": _path_text(path),
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "bands": int(dataset.count),
                    "dtype": [str(value) for value in dataset.dtypes],
                    "crs": dataset.crs.to_string() if dataset.crs else None,
                    "transform": list(dataset.transform),
                    "bounds": list(dataset.bounds),
                    "rgb_mapping": [3, 2, 1],
                }
            )
    records.sort(key=lambda row: (row["date"], row["path"]))
    hashes: dict[str, list[str]] = {}
    for row in records:
        hashes.setdefault(row["sha256"], []).append(row["date"])
    duplicate_groups = [
        {"sha256": digest, "dates": dates, "duplicate": len(dates) > 1}
        for digest, dates in hashes.items()
    ]
    valid = [row for row in records if row["date"] in VALID_DATES]
    excluded = [
        {
            **row,
            "status": "excluded_as_duplicate_observation",
            "reason": "2024-10-20 has the exact same file SHA256 as retained 2022-12-02",
        }
        for row in records
        if row["date"] == EXCLUDED_DATE
    ]
    common_grids = {
        json.dumps(
            {
                "crs": row["crs"],
                "width": row["width"],
                "height": row["height"],
                "transform": row["transform"],
            },
            sort_keys=True,
        )
        for row in valid
    }
    return {
        "status": "pass" if len(valid) == 6 and len(common_grids) == 1 else "fail",
        "source_root": _path_text(source_root),
        "all_records": records,
        "valid_dates": list(VALID_DATES),
        "valid_records": valid,
        "excluded_duplicate_records": excluded,
        "duplicate_sha_groups": duplicate_groups,
        "unique_scene_count": len(valid),
        "excluded_duplicate_dates": [row["date"] for row in excluded],
        "common_grid_count": len(common_grids),
        "rgb_mapping": [3, 2, 1],
        "original_data_modified": False,
        "created_at": _now(),
    }


def _code_dependency_audit(project_root: Path) -> dict[str, Any]:
    required = {
        "modular_orchestrator": project_root / "tools" / "nanjing_jiangbei_modular_rebuild.py",
        "modular_contracts": project_root / "src" / "urban_renewal" / "modular" / "contracts.py",
        "construction_temporal": project_root / "src" / "urban_renewal" / "pipelines" / "construction_temporal_pipeline.py",
        "surface_state_model": project_root / "extensions" / "surface_state_v1" / "models" / "surface_state_net.py",
        "construction_external": project_root / "src" / "urban_renewal" / "pipelines" / "construction_external_pipeline.py",
        "construction_training": project_root / "tools" / "construction_training_worker.py",
    }
    required_contracts = {
        "modular_orchestrator": ("BASELINE_SEAL_MANIFEST", "construction_isolated_from_baseline"),
        "modular_contracts": ("SceneRecord", "BaselineChangeResult"),
        "construction_temporal": ("no_mti", "construction"),
        "surface_state_model": ("class Surface",),
        "construction_external": ("rsbuilding_rerun", "full20_retrained"),
        "construction_training": ("test_pixel_reads", "selection_used_test"),
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    parse_errors: list[str] = []
    contract_errors: list[str] = []
    for name, path in required.items():
        text = ""
        exists = _require_file(path, missing)
        if exists:
            text = path.read_text(encoding="utf-8-sig")
            try:
                ast.parse(text, filename=str(path))
                parse_status = "pass"
            except SyntaxError as exc:
                parse_status = "fail"
                parse_errors.append(f"{path}: {exc}")
        else:
            parse_status = "missing"
        found = [needle for needle in required_contracts[name] if needle in text]
        absent = [needle for needle in required_contracts[name] if needle not in text]
        if absent:
            contract_errors.append(f"{name}: missing {absent}")
        rows.append(
            {
                "component": name,
                "path": _path_text(path),
                "sha256": sha256_file(path) if exists else None,
                "exists": exists,
                "python_ast_parse": parse_status,
                "required_contract_tokens": list(required_contracts[name]),
                "found_contract_tokens": found,
                "missing_contract_tokens": absent,
            }
        )
    return {
        "status": "pass" if not missing and not parse_errors and not contract_errors else "fail",
        "audit_scope": "static dependency and module-contract audit; no model execution",
        "components": rows,
        "missing_files": missing,
        "parse_errors": parse_errors,
        "contract_errors": contract_errors,
        "module_chain": [
            "M0 Input Audit",
            "M1 Geometric Alignment",
            "M2 Full20",
            "M3 RSBuilding",
            "M4 Baseline Fusion",
            "M5 Baseline Tracking",
            "M6 Construction Activity",
            "M7 Surface State + no-MTI Construction Temporal",
            "M8 Cross-reference",
            "M9 3D-GloBFP disabled interface",
        ],
        "created_at": _now(),
    }


def _write_boundary_audit(project_root: Path, old_output: Path) -> dict[str, Any]:
    source_root = DEFAULT_SOURCE_ROOT
    construction_code = [
        project_root / "src" / "urban_renewal" / "pipelines" / "construction_external_pipeline.py",
        project_root / "src" / "urban_renewal" / "pipelines" / "construction_temporal_pipeline.py",
        project_root / "tools" / "construction_training_worker.py",
    ]
    forbidden_strings = ("02_baseline_full20", "03_baseline_rsbuilding", "04_baseline_fusion", "05_baseline_tracking")
    code_hits: list[dict[str, Any]] = []
    for path in construction_code:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        hits = [token for token in forbidden_strings if token in text]
        if hits:
            code_hits.append({"path": _path_text(path), "forbidden_baseline_tokens": hits})
    old_report = load_json(old_output / "REPORT_NUMERICAL_AUDIT.json")
    construction_manifest = load_json(old_output / "06_construction_single_date" / "CONSTRUCTION_BRANCH_MANIFEST.json")
    old_temporal = load_json(old_output / "07_construction_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json")
    return {
        "status": "pass" if not code_hits and not old_report.get("original_data_modified") and not construction_manifest.get("writes_to_baseline_dirs") else "fail",
        "read_only_inputs": {
            "source_root": _path_text(source_root),
            "completed_modular_output": _path_text(old_output),
            "baseline_seal_root": _path_text(old_output / "04_baseline_fusion"),
        },
        "forbidden_write_roots": [
            _path_text(source_root),
            _path_text(old_output),
            _path_text(project_root / "outputs" / "inference" / "beijing_tongzhou_to_latest"),
            _path_text(project_root / "outputs" / "inference" / "beijing_tongzhou_building_geometry"),
            _path_text(project_root / "outputs" / "inference" / "beijing_tongzhou_construction_activity_v1"),
        ],
        "construction_code_baseline_path_hits": code_hits,
        "old_run_flags": {
            "source_data_modified": old_report.get("original_data_modified"),
            "writes_to_baseline_dirs": construction_manifest.get("writes_to_baseline_dirs"),
            "baseline_inputs_used_by_temporal": old_temporal.get("baseline_inputs_used"),
            "rsbuilding_rerun": construction_manifest.get("rsbuilding_rerun", False),
            "full20_retrained": construction_manifest.get("full20_retrained", False),
        },
        "new_followup_output_policy": "all new acceptance, training, inference and report artifacts are under a new timestamped root",
        "created_at": _now(),
    }


def _baseline_acceptance(old_output: Path) -> dict[str, Any]:
    seal_path = old_output / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    seal = load_json(seal_path)
    rows = list(seal.get("baseline_output_files", []))
    mismatches: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in rows:
        relative = str(item.get("relative_path", ""))
        path = old_output / relative
        if not path.exists():
            missing.append(relative)
            continue
        observed = sha256_file(path)
        expected = str(item.get("sha256", ""))
        if observed != expected:
            mismatches.append({"relative_path": relative, "expected_sha256": expected, "observed_sha256": observed})
    invariance = load_json(old_output / "BASELINE_INVARIANCE_AUDIT.json")
    parity = load_json(old_output / "TONGZHOU_BASELINE_PARITY_AUDIT.json")
    report = load_json(old_output / "REPORT_NUMERICAL_AUDIT.json")
    passed = (
        seal.get("status") == "sealed"
        and int(seal.get("file_count", -1)) == 252
        and len(rows) == 252
        and not missing
        and not mismatches
        and invariance.get("baseline_outputs_identical_with_construction_disabled_enabled") is True
        and report.get("baseline_hash_invariance_passed") is True
        and report.get("baseline_pipeline_matches_tongzhou") is True
        and parity.get("status") in {"pass", "passed", "parity_pass"}
    )
    return {
        "status": "pass" if passed else "fail",
        "seal_path": _path_text(seal_path),
        "seal_status": seal.get("status"),
        "baseline_hash_match_count": len(rows) - len(missing) - len(mismatches),
        "baseline_hash_total_count": len(rows),
        "baseline_hash_invariance_passed": not missing and not mismatches,
        "missing_files": missing,
        "mismatches": mismatches,
        "seal_file_list_sha256": seal.get("file_list_sha256"),
        "construction_started_before_seal": seal.get("construction_started_before_seal"),
        "construction_outputs_included": seal.get("construction_outputs_included"),
        "upstream_immutable": seal.get("upstream_immutable"),
        "invariance_status": invariance.get("status"),
        "tongzhou_parity_status": parity.get("status"),
        "created_at": _now(),
    }


def _visual_acceptance(old_output: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    audit_path = old_output / "09_figures_highres" / "HIGHRES_FIGURE_AUDIT.json"
    source_audit = load_json(audit_path)
    records: list[dict[str, Any]] = []
    duplicate_hashes: dict[str, list[str]] = {}
    for item in source_audit.get("figures", []):
        path = Path(str(item.get("path", "")))
        row: dict[str, Any] = {
            "path": _path_text(path),
            "source_audit": {key: item.get(key) for key in ("width", "height", "dpi", "width_pass", "height_pass", "dpi_pass")},
            "exists": path.is_file(),
            "width": None,
            "height": None,
            "sha256": None,
            "mean_rgb": None,
            "stddev_rgb": None,
            "visual_status": "fail",
            "warnings": [],
        }
        if not path.is_file():
            row["warnings"].append("missing_image")
            records.append(row)
            continue
        try:
            row["sha256"] = sha256_file(path)
            duplicate_hashes.setdefault(row["sha256"], []).append(_path_text(path))
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                row["width"], row["height"] = rgb.size
                stat = ImageStat.Stat(rgb.resize((128, 128)))
                row["mean_rgb"] = [round(float(value), 3) for value in stat.mean]
                row["stddev_rgb"] = [round(float(value), 3) for value in stat.stddev]
                if row["width"] < 2400 or row["height"] < 1600:
                    row["warnings"].append("below_minimum_display_size")
                if max(row["stddev_rgb"]) < 1.0:
                    row["warnings"].append("blank_or_constant_image")
                if max(row["mean_rgb"]) < 2.0 or min(row["mean_rgb"]) > 253.0:
                    row["warnings"].append("black_or_white_empty_image")
            row["visual_status"] = "pass" if not row["warnings"] else "fail"
        except Exception as exc:
            row["warnings"].append(f"image_read_error:{type(exc).__name__}:{exc}")
        records.append(row)
    duplicate_groups = [paths for paths in duplicate_hashes.values() if len(paths) > 1]
    category_counts: dict[str, int] = {}
    date_pair_errors: list[str] = []
    for row in records:
        relative = Path(row["path"]).relative_to(old_output / "09_figures_highres").as_posix()
        category = relative.split("/", 1)[0]
        category_counts[category] = category_counts.get(category, 0) + 1
        if category in {"baseline", "cross_reference"} and not re.search(r"\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}", relative):
            date_pair_errors.append(relative)
        if category == "construction" and not re.search(r"\d{4}-\d{2}-\d{2}", relative):
            date_pair_errors.append(relative)
    html_checks: list[dict[str, Any]] = []
    for path in (old_output / "NANJING_JIANGBEI_MODULAR_REBUILD_ZH.html", old_output / "NANJING_JIANGBEI_CONSTRUCTION_ONLY_ZH.html"):
        text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
        html_checks.append(
            {
                "path": _path_text(path),
                "exists": path.exists(),
                "utf8_readable": bool(text),
                "contains_clickable_png": "requestFullscreen" in text and ".png" in text,
                "contains_full_width_image": "width:100%" in text or "width='100%'" in text,
                "contains_chinese_heading": bool(re.search(r"[\u4e00-\u9fff]", text)),
            }
        )
    expected_categories = {"baseline": 5, "construction": 6, "cross_reference": 5, "representatives": 2}
    category_pass = all(category_counts.get(key, 0) == value for key, value in expected_categories.items())
    html_pass = all(
        row["exists"] and row["utf8_readable"] and row["contains_clickable_png"] and row["contains_full_width_image"] and row["contains_chinese_heading"]
        for row in html_checks
    )
    passed = (
        source_audit.get("status") == "pass"
        and len(records) == 18
        and all(row["visual_status"] == "pass" for row in records)
        and not duplicate_groups
        and not date_pair_errors
        and category_pass
        and html_pass
    )
    return {
        "status": "pass" if passed else "fail",
        "figure_count": len(records),
        "expected_figure_count": 18,
        "category_counts": category_counts,
        "expected_category_counts": expected_categories,
        "duplicate_image_groups": duplicate_groups,
        "date_pair_errors": date_pair_errors,
        "html_checks": html_checks,
        "figures": records,
        "visual_acceptance_boundary": "metadata plus raster non-empty/readability checks; semantic interpretation remains human review",
        "created_at": _now(),
    }


def run_acceptance(project_root: Path, old_output: Path, source_root: Path, run_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    old_output = old_output.resolve()
    source_root = source_root.resolve()
    run_root = run_root.resolve()
    if not old_output.is_dir():
        raise FileNotFoundError(f"completed modular output does not exist: {old_output}")
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty follow-up root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    acceptance_dir = run_root / "00_acceptance"
    data_dir = run_root / "01_data_audit"
    acceptance_dir.mkdir()
    data_dir.mkdir()

    source_audit = _source_snapshot(source_root)
    code = _code_dependency_audit(project_root)
    boundary = _write_boundary_audit(project_root, old_output)
    baseline = _baseline_acceptance(old_output)
    visual = _visual_acceptance(old_output)
    report_manifest = load_json(old_output / "REPORT_MANIFEST.json")
    numerical = load_json(old_output / "REPORT_NUMERICAL_AUDIT.json")
    old_module_statuses = report_manifest.get("module_statuses", {})
    modular_status = {
        "status": "pass" if report_manifest.get("status") == "completed" and all(str(value).casefold() not in {"failed", "not_run"} for value in old_module_statuses.values()) else "fail",
        "old_output": _path_text(old_output),
        "report_status": report_manifest.get("status"),
        "module_statuses": old_module_statuses,
        "effective_dates": report_manifest.get("effective_dates"),
        "excluded_duplicate_dates": report_manifest.get("excluded_duplicate_dates"),
        "training_started": numerical.get("training_started"),
        "locked_test_pixel_reads": numerical.get("locked_test_pixel_reads"),
        "ground_truth_available": numerical.get("ground_truth_available"),
        "true_accuracy_reported": numerical.get("true_accuracy_reported"),
        "globfp_enabled": numerical.get("globfp_enabled"),
        "original_data_modified": numerical.get("original_data_modified"),
        "created_at": _now(),
    }
    write_json(data_dir / "SOURCE_SCENE_STRUCTURE.json", source_audit)
    write_json(data_dir / "EXCLUDED_DUPLICATE_SCENES.json", {"excluded_duplicate_dates": [EXCLUDED_DATE], "records": source_audit["excluded_duplicate_records"], "status": "pass" if source_audit["excluded_duplicate_dates"] == [EXCLUDED_DATE] else "fail"})
    write_json(data_dir / "UNIQUE_SCENE_MANIFEST.json", {"status": "pass" if source_audit["unique_scene_count"] == 6 else "fail", "unique_scene_count": source_audit["unique_scene_count"], "valid_dates": list(VALID_DATES), "records": source_audit["valid_records"], "excluded_duplicate_dates": [EXCLUDED_DATE], "rgb_mapping": [3, 2, 1], "source_data_modified": False, "created_at": _now()})
    write_json(acceptance_dir / "CODE_DEPENDENCY_AUDIT.json", code)
    write_json(acceptance_dir / "WRITE_BOUNDARY_AUDIT.json", boundary)
    write_json(acceptance_dir / "BASELINE_ACCEPTANCE.json", baseline)
    write_json(acceptance_dir / "VISUAL_ACCEPTANCE_AUDIT.json", visual)
    write_json(acceptance_dir / "EXISTING_MODULAR_ACCEPTANCE.json", modular_status)
    passed = all(item.get("status") == "pass" for item in (code, boundary, baseline, visual, modular_status)) and source_audit.get("status") == "pass" and source_audit.get("excluded_duplicate_dates") == [EXCLUDED_DATE] and numerical.get("locked_test_pixel_reads") == 0 and numerical.get("original_data_modified") is False
    receipt = {
        "status": "ARCHITECTURE_ACCEPTANCE_PASSED" if passed else "ARCHITECTURE_ACCEPTANCE_FAILED",
        "architecture_acceptance_passed": passed,
        "phase": "phase_1_formal_acceptance",
        "completed_modular_output_read_only": True,
        "old_output": _path_text(old_output),
        "new_followup_root": _path_text(run_root),
        "duplicate_2024_excluded": source_audit.get("excluded_duplicate_dates") == [EXCLUDED_DATE],
        "unique_scene_count": source_audit.get("unique_scene_count"),
        "valid_dates": list(VALID_DATES),
        "baseline_hash_invariance_passed": baseline.get("baseline_hash_invariance_passed"),
        "baseline_hash_match_count": baseline.get("baseline_hash_match_count"),
        "baseline_hash_total_count": baseline.get("baseline_hash_total_count"),
        "original_data_modified": False,
        "locked_test_pixel_reads": 0,
        "ground_truth_available": False,
        "true_accuracy_reported": False,
        "globfp_enabled": False,
        "training_started": False,
        "construction_status": "domain_adaptation_candidate_or_rejected",
        "next_phase_allowed": passed,
        "artifacts": {
            "code_dependency_audit": _path_text(acceptance_dir / "CODE_DEPENDENCY_AUDIT.json"),
            "write_boundary_audit": _path_text(acceptance_dir / "WRITE_BOUNDARY_AUDIT.json"),
            "baseline_acceptance": _path_text(acceptance_dir / "BASELINE_ACCEPTANCE.json"),
            "visual_acceptance_audit": _path_text(acceptance_dir / "VISUAL_ACCEPTANCE_AUDIT.json"),
            "unique_scene_manifest": _path_text(data_dir / "UNIQUE_SCENE_MANIFEST.json"),
            "excluded_duplicate_scenes": _path_text(data_dir / "EXCLUDED_DUPLICATE_SCENES.json"),
        },
        "created_at": _now(),
    }
    write_json(acceptance_dir / "FINAL_ACCEPTANCE_RECEIPT.json", receipt)
    write_json(run_root / "FINAL_ACCEPTANCE_RECEIPT.json", receipt)
    return receipt


def _default_run_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return TRAINING_ROOT / f"construction_nanjing_domain_adaptation_{stamp}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="URRS Nanjing acceptance and isolated Construction follow-up")
    parser.add_argument("--mode", choices=("accept",), default="accept")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--old-output", default=str(DEFAULT_OLD_OUTPUT))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--run-root", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = Path(args.run_root).resolve() if args.run_root else _default_run_root().resolve()
    result = run_acceptance(Path(args.project_root), Path(args.old_output), Path(args.source_root), run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("architecture_acceptance_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
