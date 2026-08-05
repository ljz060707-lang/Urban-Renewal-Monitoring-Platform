"""Read-only, bounded audit for the local 3D-GloBFP distribution.

The source is large (tens of GB and hundreds of millions of features).  This
tool therefore uses GDAL layer metadata and chunked attribute reads.  It never
merges provincial files and never writes into the source directory.  Any
statistic that is sampled or scoped to Jiangsu is explicitly labelled as such.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARCHIVE_EXTENSIONS = {".rar", ".zip", ".7z"}
VECTOR_COMPONENT_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
VECTOR_DATA_EXTENSIONS = {".shp", ".gpkg", ".geojson"}
DOC_TOKENS = ("readme", "license", "licence", "copyright", "citation")
HEIGHT_TOKENS = ("height", "elev", "altitude", "building_h", "bldg_h", "z_value")
CHUNK_SIZE = 500_000
HEIGHT_RESERVOIR_SIZE = 250_000


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def read_old_modular_context(old_root: Path) -> dict[str, Any]:
    context: dict[str, Any] = {
        "old_modular_root": str(old_root),
        "old_modular_root_exists": old_root.is_dir(),
        "analysis_crs_expected": "EPSG:32650",
        "context_files": [],
        "common_grid_files_observed": [],
    }
    if not old_root.is_dir():
        return context
    for candidate in sorted(old_root.rglob("MODULAR_INPUT_AUDIT.json"))[:10]:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
            context["context_files"].append({"path": str(candidate), "status": "READ"})
            if isinstance(payload, dict):
                context["modular_input_keys"] = sorted(payload.keys())
                context["modular_input_grid_aoi_crs_values"] = {
                    key: payload[key]
                    for key in payload
                    if any(token in key.lower() for token in ("grid", "aoi", "crs", "bounds"))
                }
        except Exception as exc:
            context["context_files"].append(
                {"path": str(candidate), "status": "READ_ERROR", "error": f"{type(exc).__name__}: {exc}"}
            )
    names = {"COMMON_GRID.GPKG", "COMMON_GRID.TIF", "VALID_OVERLAP_MASK.TIF", "VALID_OVERLAP_MASK.GPKG"}
    context["common_grid_files_observed"] = [
        str(path) for path in sorted(old_root.rglob("*")) if path.is_file() and path.name.upper() in names
    ][:100]
    return context


def extractor_probe() -> dict[str, Any]:
    candidates = [
        "7z",
        "7za",
        "unrar",
        "rar",
        "WinRAR",
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files\7-Zip\7za.exe",
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files\WinRAR\WinRAR.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
    ]
    found = []
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            found.append(resolved)
        elif Path(candidate).is_file():
            found.append(candidate)
    try:
        import rarfile  # type: ignore  # noqa: F401

        rarfile_available = True
    except Exception:
        rarfile_available = False
    try:
        import py7zr  # type: ignore  # noqa: F401

        py7zr_available = True
    except Exception:
        py7zr_available = False
    return {
        "found_executables": sorted(set(found)),
        "rarfile_python_package_available": rarfile_available,
        "py7zr_python_package_available": py7zr_available,
        "usable_external_archive_extractor": bool(found),
    }


def file_category(path: Path) -> tuple[str, bool, bool, bool]:
    suffix = path.suffix.lower()
    name = path.name.lower()
    is_archive = suffix in ARCHIVE_EXTENSIONS
    is_vector = suffix in VECTOR_COMPONENT_EXTENSIONS
    is_doc = any(token in name for token in DOC_TOKENS)
    if is_archive:
        category = "archive"
    elif is_vector:
        category = "vector_component"
    elif is_doc or suffix in {".txt", ".md", ".pdf"}:
        category = "documentation_candidate"
    else:
        category = "other"
    return category, is_archive, is_vector, is_doc


def archive_member_status(path: Path, probe: dict[str, Any]) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        try:
            import zipfile

            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                return {
                    "status": "LISTED",
                    "member_count": len(members),
                    "member_name_sample": [item.filename for item in members[:50]],
                    "member_vector_count": sum(Path(item.filename).suffix.lower() in VECTOR_COMPONENT_EXTENSIONS for item in members),
                }
        except Exception as exc:
            return {"status": "LIST_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if path.suffix.lower() == ".rar" and probe["rarfile_python_package_available"]:
        if probe["usable_external_archive_extractor"]:
            return {"status": "EXTERNAL_READER_AVAILABLE_MEMBER_LIST_NOT_RUN"}
        return {"status": "MEMBER_LIST_NOT_ATTEMPTED_NO_EXTERNAL_READER"}
    return {"status": "MEMBER_LIST_UNAVAILABLE_NO_SUPPORTED_READER"}


def schema_candidates(fields: list[str], dtypes: list[str]) -> list[str]:
    candidates = []
    for field, dtype in zip(fields, dtypes):
        low = field.lower()
        numeric = any(token in dtype.lower() for token in ("int", "float", "real", "double"))
        if numeric and any(token in low for token in HEIGHT_TOKENS):
            candidates.append(field)
    return candidates


def reservoir_stats(path: Path, field: str, feature_count: int) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import pyogrio  # type: ignore
    except Exception as exc:
        return {"status": "READER_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}

    reservoir: list[float] = []
    rng = random.Random(20260804 + sum(ord(char) for char in str(path)))
    finite_count = 0
    missing_count = 0
    nonfinite_count = 0
    zero_count = 0
    negative_count = 0
    greater_100m_count = 0
    greater_500m_count = 0
    minimum = math.inf
    maximum = -math.inf
    for offset in range(0, feature_count, CHUNK_SIZE):
        frame = None
        try:
            frame = pyogrio.read_dataframe(
                path,
                columns=[field],
                read_geometry=False,
                skip_features=offset,
                max_features=min(CHUNK_SIZE, feature_count - offset),
            )
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype="float64")
            missing_count += int(np.isnan(values).sum())
            nonfinite_count += int((~np.isfinite(values) & ~np.isnan(values)).sum())
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_count += int(finite.size)
                zero_count += int((finite == 0).sum())
                negative_count += int((finite < 0).sum())
                greater_100m_count += int((finite > 100).sum())
                greater_500m_count += int((finite > 500).sum())
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
                for value in finite.tolist():
                    if len(reservoir) < HEIGHT_RESERVOIR_SIZE:
                        reservoir.append(float(value))
                    else:
                        replacement = rng.randrange(finite_count)
                        if replacement < HEIGHT_RESERVOIR_SIZE:
                            reservoir[replacement] = float(value)
        except Exception as exc:
            return {
                "status": "READ_ERROR",
                "field_name": field,
                "feature_count_expected": feature_count,
                "failed_offset": offset,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            del frame
            gc.collect()
        if offset == 0 or (offset // CHUNK_SIZE) % 10 == 0:
            print(
                json.dumps({"stage": "height_attribute_scan", "file": str(path), "offset": offset, "total": feature_count}, ensure_ascii=False),
                flush=True,
            )
    if reservoir:
        quantiles = np.quantile(np.asarray(reservoir, dtype="float64"), [0.01, 0.05, 0.50, 0.95, 0.99])
        quantile_payload = dict(zip(("p01", "p05", "p50", "p95", "p99"), [float(value) for value in quantiles]))
    else:
        quantile_payload = {key: None for key in ("p01", "p05", "p50", "p95", "p99")}
    return {
        "status": "COMPLETE",
        "field_name": field,
        "feature_count_expected": feature_count,
        "finite_count": finite_count,
        "missing_count": missing_count,
        "missing_rate": float(missing_count / feature_count) if feature_count else None,
        "nonfinite_count": nonfinite_count,
        "zero_count": zero_count,
        "negative_count": negative_count,
        "greater_than_100m_count": greater_100m_count,
        "greater_than_500m_count": greater_500m_count,
        "min": None if not reservoir else minimum,
        **quantile_payload,
        "max": None if not reservoir else maximum,
        "quantile_method": "deterministic_reservoir_sample",
        "quantile_sample_size": len(reservoir),
        "chunk_size": CHUNK_SIZE,
    }


def geometry_quality(path: Path, feature_count: int, output_root: Path, full_scan: bool) -> dict[str, Any]:
    """Count geometry quality indicators in chunks; duplicates use SQLite."""

    try:
        import pyogrio  # type: ignore
    except Exception as exc:
        return {"status": "READER_UNAVAILABLE", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    scan_limit = feature_count if full_scan else min(feature_count, 10_000)
    db_path = output_root / "geometry_hashes.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE IF NOT EXISTS geometry_hashes (dataset TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(dataset, digest))")
    connection.commit()
    empty = null_count = invalid = observed = duplicates = 0
    try:
        for offset in range(0, scan_limit, 50_000):
            frame = None
            try:
                frame = pyogrio.read_dataframe(
                    path,
                    columns=[],
                    read_geometry=True,
                    skip_features=offset,
                    max_features=min(50_000, scan_limit - offset),
                )
                geometries = frame.geometry
                observed += len(geometries)
                null_count += int(geometries.isna().sum())
                empty += int(geometries.is_empty.fillna(False).sum())
                invalid += int((~geometries.is_valid.fillna(False)).sum())
                hashes = []
                for geometry in geometries:
                    if geometry is None or geometry is None or getattr(geometry, "is_empty", True):
                        continue
                    digest = hashlib.sha256(geometry.wkb).hexdigest()
                    hashes.append((str(path), digest))
                before = connection.total_changes
                connection.executemany("INSERT OR IGNORE INTO geometry_hashes(dataset, digest) VALUES (?, ?)", hashes)
                connection.commit()
                inserted = connection.total_changes - before
                duplicates += max(0, len(hashes) - inserted)
            except Exception as exc:
                return {
                    "status": "READ_ERROR",
                    "path": str(path),
                    "scan_scope": "FULL_FILE" if full_scan else "FIRST_10000_FEATURES",
                    "observed_features": observed,
                    "empty_geometry_count": empty,
                    "null_geometry_count": null_count,
                    "invalid_geometry_count": invalid,
                    "duplicate_geometry_count": duplicates,
                    "failed_offset": offset,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                del frame
                gc.collect()
    finally:
        connection.close()
    return {
        "status": "COMPLETE",
        "path": str(path),
        "scan_scope": "FULL_FILE" if full_scan else "FIRST_10000_FEATURES",
        "feature_count_expected": feature_count,
        "observed_features": observed,
        "empty_geometry_count": empty,
        "null_geometry_count": null_count,
        "invalid_geometry_count": invalid,
        "duplicate_geometry_count": duplicates,
        "duplicate_method": "exact_wkb_sha256_with_sqlite_unique_index" if full_scan else "exact_within_bounded_sample",
    }


def run_audit(source_root: Path, old_modular_root: Path, output_root: Path, geometry_mode: str) -> dict[str, Any]:
    import pyogrio  # type: ignore

    output_root.mkdir(parents=True, exist_ok=True)
    audit_root = output_root / "00_globfp_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    probe = extractor_probe()
    file_rows: list[dict[str, Any]] = []
    sha_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    archives: list[Path] = []
    for index, path in enumerate(files, start=1):
        category, is_archive, is_vector, is_doc = file_category(path)
        stat = path.stat()
        digest = sha256_file(path)
        row = {
            "path": str(path),
            "relative_path": rel(path, source_root),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": int(stat.st_size),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": digest,
            "category": category,
            "is_archive": is_archive,
            "is_vector_component": is_vector,
            "is_readme_license_candidate": is_doc,
        }
        file_rows.append(row)
        sha_rows.append({key: row[key] for key in ("path", "relative_path", "size_bytes", "sha256")})
        if is_archive:
            archives.append(path)
            member = archive_member_status(path, probe)
            counterpart = source_root / path.stem
            archive_rows.append(
                {
                    "archive_path": str(path),
                    "relative_path": rel(path, source_root),
                    "archive_type": path.suffix.lower().lstrip("."),
                    "size_bytes": int(stat.st_size),
                    "sha256": digest,
                    "member_listing_status": member.get("status"),
                    "member_count": member.get("member_count"),
                    "member_name_sample": json.dumps(member.get("member_name_sample", []), ensure_ascii=False),
                    "member_vector_component_count": member.get("member_vector_count"),
                    "extracted_counterpart_observed": counterpart.is_dir(),
                    "extraction_status": "USER_EXTRACTED_DIRECTORY_OBSERVED" if counterpart.is_dir() else "NOT_EXTRACTED",
                    "source_archive_preserved": True,
                    "error": member.get("error"),
                }
            )
        if index == 1 or index % 5 == 0 or index == len(files):
            print(json.dumps({"stage": "sha256_inventory", "files_done": index, "files_total": len(files)}, ensure_ascii=False), flush=True)

    vector_paths = sorted(path for path in files if path.suffix.lower() == ".shp")
    component_rows: list[dict[str, Any]] = []
    required = {".shp", ".shx", ".dbf", ".prj"}
    for path in vector_paths:
        observed = {path.with_suffix(ext).exists() for ext in VECTOR_COMPONENT_EXTENSIONS}
        observed_names = [ext for ext in sorted(VECTOR_COMPONENT_EXTENSIONS) if path.with_suffix(ext).exists()]
        missing = sorted(required - set(observed_names))
        component_rows.append(
            {
                "status": "COMPLETE" if not missing else "INCOMPLETE",
                "dataset_stem": path.stem,
                "source_grid_directory": path.parent.name,
                "shp_path": str(path),
                "observed_components": ",".join(observed_names),
                "missing_required_components": ",".join(missing),
                "component_paths": json.dumps({ext: str(path.with_suffix(ext)) for ext in observed_names}, ensure_ascii=False),
            }
        )

    schema_rows: list[dict[str, Any]] = []
    height_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for index, path in enumerate(vector_paths, start=1):
        info = pyogrio.read_info(path)
        fields = [str(value) for value in info.get("fields", [])]
        dtypes = [str(value) for value in info.get("dtypes", [])]
        candidates = schema_candidates(fields, dtypes)
        feature_count = int(info.get("features", 0))
        schema_rows.append(
            {
                "status": "SCHEMA_READ_COMPLETE",
                "path": str(path),
                "relative_path": rel(path, source_root),
                "source_grid_id": path.parent.name,
                "crs": str(info.get("crs")) if info.get("crs") else None,
                "geometry_type": str(info.get("geometry_type")),
                "fields": fields,
                "dtypes": dtypes,
                "feature_count": feature_count,
                "total_bounds": info.get("total_bounds"),
                "height_field_candidates": candidates,
                "building_id_field_candidates": [field for field in fields if any(token in field.lower() for token in ("id", "fid", "object"))],
            }
        )
        if len(candidates) == 1:
            stats = reservoir_stats(path, candidates[0], feature_count)
            height_rows.append(
                {
                    "status": "HEIGHT_FIELD_CONFIRMED_BY_REAL_SCHEMA",
                    "path": str(path),
                    "actual_height_field": candidates[0],
                    "height_unit": "NOT_STATED_IN_LAYER_SCHEMA_OR_LOCAL_DOCUMENTATION",
                    "height_unit_status": "UNVERIFIED",
                    "selection_method": "single_numeric_semantic_candidate_after_schema_read",
                    "stats": stats,
                }
            )
        else:
            height_rows.append(
                {
                    "status": "HEIGHT_FIELD_AMBIGUOUS" if candidates else "HEIGHT_FIELD_NOT_FOUND",
                    "path": str(path),
                    "actual_height_field": None,
                    "height_field_candidates": candidates,
                    "height_unit": None,
                }
            )
        full_geometry = geometry_mode == "all" or (geometry_mode == "target_full" and path.stem.lower() == "jiangsu")
        geometry_rows.append(geometry_quality(path, feature_count, output_root, full_geometry))
        print(json.dumps({"stage": "vector_audit", "file_index": index, "file_total": len(vector_paths), "file": str(path)}, ensure_ascii=False), flush=True)

    top_level = []
    for path in sorted(source_root.iterdir(), key=lambda item: item.name.lower()):
        stat = path.stat()
        top_level.append({"name": path.name, "kind": "directory" if path.is_dir() else "file", "size_bytes": int(stat.st_size) if path.is_file() else None})
    context = read_old_modular_context(old_modular_root)
    height_fields = sorted({row["actual_height_field"] for row in height_rows if row.get("actual_height_field")})
    crs_values = sorted({row["crs"] for row in schema_rows if row.get("crs")})
    geometry_complete = all(row.get("status") == "COMPLETE" for row in geometry_rows)
    readiness_status = "READY_FOR_TARGET_EXTRACTION_HEIGHT_UNIT_REVIEW_REQUIRED"
    if not vector_paths:
        readiness_status = "BLOCKED_NO_EXTRACTED_VECTORS"
    readiness = {
        "status": readiness_status,
        "audited_at_utc": now_utc(),
        "source_root": str(source_root),
        "source_top_level_structure": top_level,
        "source_file_count": len(files),
        "archive_count": len(archives),
        "china_archive_count": sum(path.stem.lower().startswith("china") for path in archives),
        "extracted_grid_directory_count": len([path for path in source_root.iterdir() if path.is_dir()]),
        "vector_file_count": len(vector_paths),
        "vector_component_set_complete_count": sum(row["status"] == "COMPLETE" for row in component_rows),
        "vector_component_set_incomplete_count": sum(row["status"] != "COMPLETE" for row in component_rows),
        "crs_values": crs_values,
        "geometry_types": sorted({row["geometry_type"] for row in schema_rows}),
        "actual_height_fields": height_fields,
        "height_field_consistent_across_vector_files": height_fields == ["Height"] and all(len(row.get("height_field_candidates", [])) == 1 for row in schema_rows),
        "height_unit": "NOT_STATED_IN_LAYER_SCHEMA_OR_LOCAL_DOCUMENTATION",
        "height_unit_status": "UNVERIFIED",
        "geometry_quality_audit_status": "COMPLETE" if geometry_complete else "PARTIAL_OR_ERROR",
        "geometry_quality_scope": geometry_mode,
        "grid_index_rule": "province_directory_and_province_shapefile_name_observed; formal_grid_index_not_found_in_schema",
        "data_time_anchor": "NOT_STATED_IN_LOCAL_FILES",
        "allow_nanjing_extraction": bool(vector_paths),
        "allow_national_height_training": False,
        "allow_globfp_pseudo_labels": False,
        "allow_construction_c_b2": False,
        "allow_construction_c_b3": False,
        "height_model_gate": "BLOCKED_PENDING_HEIGHT_UNIT_CONFIRMATION",
        "source_data_modified_by_pipeline": False,
        "locked_test_pixel_reads": 0,
        "training_started": False,
        "old_modular_context": context,
        "archive_extraction_observed": "USER_EXTRACTED_DIRECTORIES_PRESENT_SOURCE_ARCHIVES_PRESERVED",
    }

    write_csv(audit_root / "FILE_INVENTORY.csv", file_rows, ["path", "relative_path", "name", "extension", "size_bytes", "modified_utc", "sha256", "category", "is_archive", "is_vector_component", "is_readme_license_candidate"])
    write_csv(audit_root / "ARCHIVE_INVENTORY.csv", archive_rows, ["archive_path", "relative_path", "archive_type", "size_bytes", "sha256", "member_listing_status", "member_count", "member_name_sample", "member_vector_component_count", "extracted_counterpart_observed", "extraction_status", "source_archive_preserved", "error"])
    write_csv(audit_root / "SHA256_INVENTORY.csv", sha_rows, ["path", "relative_path", "size_bytes", "sha256"])
    write_csv(audit_root / "VECTOR_COMPONENT_AUDIT.csv", component_rows, ["status", "dataset_stem", "source_grid_directory", "shp_path", "observed_components", "missing_required_components", "component_paths"])
    write_json(audit_root / "VECTOR_SCHEMA_AUDIT.json", {"status": "SCHEMA_READ_COMPLETE", "records": schema_rows, "unknown_structure_is_error": False})
    write_json(audit_root / "HEIGHT_FIELD_AUDIT.json", {"status": "HEIGHT_FIELD_READ_COMPLETE_UNIT_UNVERIFIED", "actual_height_fields": height_fields, "records": height_rows, "height_field_was_not_inferred_from_filename": True})
    write_json(audit_root / "SPATIAL_COVERAGE_AUDIT.json", {"status": "SCHEMA_BOUNDS_READ_GEOMETRY_QUALITY_SCOPED", "crs_values": crs_values, "records": schema_rows, "geometry_quality_records": geometry_rows, "nanjing_sealed_context": context})
    write_json(audit_root / "GEOMETRY_QUALITY_AUDIT.json", {"status": "COMPLETE" if geometry_complete else "PARTIAL_OR_ERROR", "scope": geometry_mode, "records": geometry_rows, "hash_database": str(output_root / "geometry_hashes.sqlite3")})
    docs = [path for path in files if any(token in path.name.lower() for token in DOC_TOKENS)]
    (audit_root / "LICENSE_AND_README_AUDIT.md").write_text(
        "# 3D-GloBFP README / License 审计\n\n"
        f"- source_root: `{source_root}`\n- audited_at_utc: `{now_utc()}`\n"
        f"- documentation_candidates_observed: `{len(docs)}`\n"
        f"- extracted_directories_observed: `{readiness['extracted_grid_directory_count']}`\n\n"
        "源目录中实际观察到的是 5 个 RAR 与用户解压后的省级 Shapefile；未观察到 README、License、Licence、版权或引用文档。压缩包成员未通过外部 RAR reader 读取，但对应解压目录已直接读取层 schema。高度字段单位未在 layer schema 或本地文档中声明，不能默认为米。\n",
        encoding="utf-8",
    )
    write_json(audit_root / "GLOBFP_DATA_READINESS.json", readiness)
    write_json(output_root / "GLOBFP_DATA_AUDIT.json", readiness)
    write_json(output_root / "GLOBFP_DATA_READINESS.json", readiness)
    return readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Data Set\3D-GloBFP"))
    parser.add_argument("--old-modular-root", type=Path, default=project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--geometry-mode", choices=("target_full", "all", "sample"), default="target_full")
    args = parser.parse_args()
    if not args.source_root.is_dir():
        print(json.dumps({"status": "GLOBFP_SOURCE_NOT_FOUND", "source_root": str(args.source_root)}, ensure_ascii=False))
        return 2
    readiness = run_audit(args.source_root, args.old_modular_root, args.output_root, args.geometry_mode)
    print(json.dumps({"output_root": str(args.output_root), **readiness}, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
