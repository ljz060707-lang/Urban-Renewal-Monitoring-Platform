"""Auditable preparation entry point for the local 3D-GloBFP branch.

This first stage is intentionally conservative.  It inventories the source as
found, hashes source files, and reads vector schemas only when an extracted
vector is actually available.  A compressed-only source is never represented
as an empty vector dataset and no height field is inferred from a filename.

The later extraction, matching, height-model, and Construction stages should
consume the receipts produced here rather than bypassing this gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARCHIVE_EXTENSIONS = {".rar", ".zip", ".7z"}
VECTOR_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".gpkg", ".geojson"}
VECTOR_DATA_EXTENSIONS = {".shp", ".gpkg", ".geojson"}
README_LICENSE_TOKENS = ("readme", "license", "licence", "copyright", "citation")


def utc_now() -> str:
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


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def archive_tool_probe() -> dict[str, Any]:
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
    found: list[str] = []
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            found.append(resolved)
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            found.append(str(candidate_path))
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
        "known_extractor_candidates": candidates,
        "found_executables": sorted(set(found)),
        "rarfile_python_package_available": rarfile_available,
        "py7zr_python_package_available": py7zr_available,
        "usable_rar_extractor": bool(found),
    }


def read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except Exception as exc:  # pragma: no cover - defensive for legacy receipts
        return None, f"{type(exc).__name__}: {exc}"


def find_modular_context(old_root: Path) -> dict[str, Any]:
    context: dict[str, Any] = {
        "old_modular_root": str(old_root),
        "old_modular_root_exists": old_root.is_dir(),
        "context_files": [],
        "common_grid_context": None,
        "valid_overlap_context": None,
        "analysis_crs_expected": "EPSG:32650",
    }
    if not old_root.is_dir():
        return context
    candidates = sorted(old_root.rglob("MODULAR_INPUT_AUDIT.json"))
    for candidate in candidates[:10]:
        payload, error = read_json(candidate)
        context["context_files"].append({"path": str(candidate), "read_error": error})
        if error is None and isinstance(payload, dict):
            context["common_grid_context"] = {
                key: payload.get(key)
                for key in payload
                if "grid" in key.lower() or "aoi" in key.lower() or "crs" in key.lower()
            }
            if context["common_grid_context"]:
                break
    relevant = []
    for candidate in sorted(old_root.rglob("*")):
        if candidate.is_file() and candidate.name.upper() in {
            "COMMON_GRID.GPKG",
            "COMMON_GRID.TIF",
            "VALID_OVERLAP_MASK.TIF",
            "VALID_OVERLAP_MASK.GPKG",
        }:
            relevant.append(str(candidate))
    context["common_grid_files_observed"] = relevant[:100]
    return context


def classify_file(path: Path) -> tuple[str, bool, bool, bool]:
    suffix = path.suffix.lower()
    name = path.name.lower()
    is_archive = suffix in ARCHIVE_EXTENSIONS
    is_vector = suffix in VECTOR_EXTENSIONS
    is_readme_license = any(token in name for token in README_LICENSE_TOKENS)
    if is_archive:
        category = "archive"
    elif is_vector:
        category = "vector_component"
    elif is_readme_license or suffix in {".txt", ".md", ".pdf"}:
        category = "documentation_candidate"
    else:
        category = "other"
    return category, is_archive, is_vector, is_readme_license


def list_archive_members(path: Path, tool_probe: dict[str, Any]) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        try:
            import zipfile

            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                return {
                    "status": "LISTED",
                    "member_count": len(members),
                    "member_name_sample": [item.filename for item in members[:50]],
                    "member_vector_component_count": sum(
                        Path(item.filename).suffix.lower() in VECTOR_EXTENSIONS for item in members
                    ),
                }
        except Exception as exc:
            return {"status": "LIST_ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if suffix == ".rar" and tool_probe["rarfile_python_package_available"]:
        # rarfile still requires an external unrar/bsdtar tool for most RARs.
        # Do not call it when no extractor was found; the explicit state is more
        # informative than a silent empty member list.
        return {
            "status": "MEMBER_LIST_NOT_ATTEMPTED_NO_EXTERNAL_EXTRACTOR",
            "member_count": None,
        }
    return {
        "status": "MEMBER_LIST_UNAVAILABLE_NO_SUPPORTED_READER",
        "member_count": None,
    }


def inspect_extracted_vectors(vector_paths: list[Path], source_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schema_rows: list[dict[str, Any]] = []
    height_rows: list[dict[str, Any]] = []
    try:
        import geopandas as gpd  # type: ignore
    except Exception as exc:
        return [
            {
                "status": "SCHEMA_READER_UNAVAILABLE",
                "path": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        ], [
            {
                "status": "HEIGHT_SCHEMA_UNAVAILABLE",
                "actual_height_fields": [],
                "reason": "geopandas_unavailable",
            }
        ]

    for path in vector_paths:
        row: dict[str, Any] = {
            "status": "READ_ATTEMPTED",
            "path": str(path),
            "relative_path": relative_path(path, source_root),
            "crs": None,
            "geometry_types": [],
            "columns": [],
            "feature_count": None,
            "error": None,
        }
        try:
            frame = gpd.read_file(path)
            row["crs"] = frame.crs.to_string() if frame.crs is not None else None
            row["geometry_types"] = sorted(
                {str(value) for value in frame.geometry.geom_type.dropna().unique()}
            )
            row["columns"] = [str(column) for column in frame.columns]
            row["feature_count"] = int(len(frame))
            schema_rows.append(row)

            numeric_fields = [
                str(column)
                for column in frame.columns
                if column != frame.geometry.name
                and str(frame[column].dtype) not in {"object", "string"}
                and not str(frame[column].dtype).startswith("datetime")
            ]
            field_stats = []
            for field in numeric_fields:
                values = frame[field]
                finite = values[values.notna()]
                if len(finite) == 0:
                    continue
                finite_numeric = finite.astype(float)
                field_stats.append(
                    {
                        "field_name": field,
                        "dtype": str(frame[field].dtype),
                        "non_null_count": int(len(finite_numeric)),
                        "missing_count": int(len(frame) - len(finite_numeric)),
                        "missing_rate": float((len(frame) - len(finite_numeric)) / len(frame))
                        if len(frame)
                        else None,
                        "zero_count": int((finite_numeric == 0).sum()),
                        "negative_count": int((finite_numeric < 0).sum()),
                        "min": float(finite_numeric.min()),
                        "p01": float(finite_numeric.quantile(0.01)),
                        "p05": float(finite_numeric.quantile(0.05)),
                        "p50": float(finite_numeric.quantile(0.50)),
                        "p95": float(finite_numeric.quantile(0.95)),
                        "p99": float(finite_numeric.quantile(0.99)),
                        "max": float(finite_numeric.max()),
                    }
                )
            height_rows.append(
                {
                    "status": "HEIGHT_FIELD_CANDIDATES_FROM_REAL_SCHEMA",
                    "path": str(path),
                    "numeric_field_stats": field_stats,
                    "actual_height_field": None,
                    "selection_status": "REQUIRES_SEMANTIC_CONFIRMATION_FROM_SCHEMA_AND_METADATA",
                }
            )
        except Exception as exc:
            row["status"] = "SCHEMA_READ_ERROR"
            row["error"] = f"{type(exc).__name__}: {exc}"
            schema_rows.append(row)
            height_rows.append(
                {
                    "status": "HEIGHT_FIELD_AUDIT_READ_ERROR",
                    "path": str(path),
                    "numeric_field_stats": [],
                    "actual_height_field": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return schema_rows, height_rows


def component_audit(files: list[Path], source_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[str]] = {}
    paths: dict[tuple[str, str], dict[str, Path]] = {}
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
            continue
        key = (str(path.parent), path.stem.lower())
        grouped.setdefault(key, set()).add(suffix)
        paths.setdefault(key, {})[suffix] = path
    rows: list[dict[str, Any]] = []
    required = {".shp", ".shx", ".dbf", ".prj"}
    for (parent, stem), suffixes in sorted(grouped.items()):
        missing = sorted(required - suffixes)
        rows.append(
            {
                "status": "COMPLETE" if not missing else "INCOMPLETE",
                "dataset_stem": stem,
                "directory": parent,
                "relative_directory": relative_path(Path(parent), source_root),
                "observed_components": ",".join(sorted(suffixes)),
                "missing_required_components": ",".join(missing),
                "has_cpg": ".cpg" in suffixes,
                "component_paths": json.dumps(
                    {suffix: str(path) for suffix, path in paths[(parent, stem)].items()},
                    ensure_ascii=False,
                ),
            }
        )
    if not rows:
        rows.append(
            {
                "status": "NO_EXTRACTED_VECTOR_COMPONENTS",
                "dataset_stem": None,
                "directory": None,
                "relative_directory": None,
                "observed_components": "",
                "missing_required_components": ".shp,.shx,.dbf,.prj",
                "has_cpg": False,
                "component_paths": "{}",
            }
        )
    return rows


def build_readme_license_audit(source_root: Path, files: list[Path], tool_probe: dict[str, Any]) -> str:
    docs = [path for path in files if any(token in path.name.lower() for token in README_LICENSE_TOKENS)]
    lines = [
        "# 3D-GloBFP 本地 README / License 审计",
        "",
        f"- source_root: `{source_root}`",
        f"- audited_at_utc: `{utc_now()}`",
        f"- source_file_count: `{len(files)}`",
        f"- documentation_candidates: `{len(docs)}`",
        "",
        "## 实际观察",
        "",
        "本轮递归扫描只观察到源目录中的 RAR 压缩包；没有在源目录中发现可直接读取的 README、License、Licence、版权或引用文件。压缩包成员未能列出，因此不能把压缩包内部是否存在文档写成已确认。",
        "",
        "## 归档工具边界",
        "",
        f"- known extractor executables found: `{json.dumps(tool_probe['found_executables'], ensure_ascii=False)}`",
        f"- rarfile Python package available: `{tool_probe['rarfile_python_package_available']}`",
        "- source archives were not modified or extracted in place.",
        "",
        "该文件是数据可用性审计，不是对未读取的压缩包内容作出的许可推断。继续使用数据前必须补充可读取的 README/License/引用信息。",
    ]
    if docs:
        lines.extend(["", "## 发现的文档候选", ""])
        lines.extend(f"- `{path}`" for path in docs)
    return "\n".join(lines) + "\n"


def audit_source(source_root: Path, old_modular_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    audit_root = output_root / "00_globfp_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    if not source_root.is_dir():
        readiness = {
            "status": "GLOBFP_SOURCE_NOT_FOUND",
            "source_root": str(source_root),
            "source_data_modified": False,
            "locked_test_pixel_reads": 0,
            "training_started": False,
        }
        write_json(audit_root / "GLOBFP_DATA_READINESS.json", readiness)
        write_json(output_root / "GLOBFP_DATA_AUDIT.json", readiness)
        return readiness

    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    tool_probe = archive_tool_probe()
    file_rows: list[dict[str, Any]] = []
    sha_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    hash_errors: list[dict[str, str]] = []
    archive_paths: list[Path] = []
    for index, path in enumerate(files, start=1):
        category, is_archive, is_vector, is_doc = classify_file(path)
        size = None
        modified = None
        try:
            stat = path.stat()
            size = int(stat.st_size)
            modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        except Exception as exc:
            hash_errors.append({"path": str(path), "stage": "stat", "error": f"{type(exc).__name__}: {exc}"})
        digest = None
        if size is not None:
            try:
                digest = sha256_file(path)
            except Exception as exc:
                hash_errors.append({"path": str(path), "stage": "sha256", "error": f"{type(exc).__name__}: {exc}"})
        row = {
            "path": str(path),
            "relative_path": relative_path(path, source_root),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": size,
            "modified_utc": modified,
            "sha256": digest,
            "category": category,
            "is_archive": is_archive,
            "is_vector_component": is_vector,
            "is_readme_license_candidate": is_doc,
        }
        file_rows.append(row)
        sha_rows.append(
            {
                "path": str(path),
                "relative_path": relative_path(path, source_root),
                "size_bytes": size,
                "sha256": digest,
                "hash_status": "OK" if digest else "ERROR",
            }
        )
        if is_archive:
            archive_paths.append(path)
            member_info = list_archive_members(path, tool_probe)
            archive_rows.append(
                {
                    "archive_path": str(path),
                    "relative_path": relative_path(path, source_root),
                    "archive_type": path.suffix.lower().lstrip("."),
                    "size_bytes": size,
                    "sha256": digest,
                    "member_listing_status": member_info.get("status"),
                    "member_count": member_info.get("member_count"),
                    "member_name_sample": json.dumps(member_info.get("member_name_sample", []), ensure_ascii=False),
                    "member_vector_component_count": member_info.get("member_vector_component_count"),
                    "extracted_counterpart_observed": False,
                    "extraction_status": "NOT_EXTRACTED_SOURCE_PRESERVED",
                    "error": member_info.get("error"),
                }
            )
        if index % 10 == 0 or index == len(files):
            print(json.dumps({"stage": "hashing_source", "files_done": index, "files_total": len(files)}), flush=True)

    vector_paths = [path for path in files if path.suffix.lower() in VECTOR_DATA_EXTENSIONS]
    schema_rows, height_rows = inspect_extracted_vectors(vector_paths, source_root) if vector_paths else (
        [
            {
                "status": "NOT_AVAILABLE_COMPRESSED_ONLY",
                "path": None,
                "relative_path": None,
                "crs": None,
                "geometry_types": [],
                "columns": [],
                "feature_count": None,
                "error": "No extracted vector data was observed under the source root.",
            }
        ],
        [
            {
                "status": "NOT_AVAILABLE_COMPRESSED_ONLY",
                "actual_height_fields": [],
                "height_field_known": False,
                "reason": "The source contains compressed archives only; no schema was read.",
            }
        ],
    )

    component_rows = component_audit(files, source_root)
    context = find_modular_context(old_modular_root)
    top_level_entries = []
    for path in sorted(source_root.iterdir(), key=lambda item: item.name.lower()):
        try:
            stat = path.stat()
            top_level_entries.append(
                {
                    "name": path.name,
                    "kind": "directory" if path.is_dir() else "file",
                    "size_bytes": int(stat.st_size) if path.is_file() else None,
                    "extension": path.suffix.lower() if path.is_file() else None,
                }
            )
        except Exception as exc:
            top_level_entries.append({"name": path.name, "kind": "unreadable", "error": f"{type(exc).__name__}: {exc}"})

    source_vector_count = sum(path.suffix.lower() in VECTOR_EXTENSIONS for path in files)
    archive_count = len(archive_paths)
    extracted_file_count = len(vector_paths)
    extraction_blocked = archive_count > 0 and extracted_file_count == 0
    readiness_status = "BLOCKED_ARCHIVE_EXTRACTION_TOOL_UNAVAILABLE" if extraction_blocked else "AUDIT_COMPLETE_EXTRACTION_REQUIRED"
    readiness = {
        "status": readiness_status,
        "audited_at_utc": utc_now(),
        "source_root": str(source_root),
        "source_top_level_structure": top_level_entries,
        "source_file_count": len(files),
        "source_vector_component_count": source_vector_count,
        "extracted_vector_file_count": extracted_file_count,
        "archive_count": archive_count,
        "china_archive_count": sum(path.stem.lower().startswith("china") for path in archive_paths),
        "china_extracted_package_count": 0 if extraction_blocked else None,
        "archives_are_compressed_only": extraction_blocked,
        "archive_extraction_tool": tool_probe,
        "height_field_known": bool(vector_paths),
        "actual_height_fields": [row.get("actual_height_field") for row in height_rows if row.get("actual_height_field")],
        "schema_read_status": "NOT_AVAILABLE_COMPRESSED_ONLY" if extraction_blocked else "READ_ATTEMPTED",
        "extraction_required_before_spatial_or_height_analysis": extraction_blocked,
        "allow_national_height_training": False if extraction_blocked else None,
        "allow_nanjing_extraction": False if extraction_blocked else None,
        "allow_globfp_pseudo_labels": False,
        "allow_construction_c_b2": False,
        "allow_construction_c_b3": False,
        "source_data_modified": False,
        "locked_test_pixel_reads": 0,
        "training_started": False,
        "reason": (
            "Five China RAR archives are present but no usable RAR extractor is installed or discoverable; "
            "no archive was modified or extracted in place."
            if extraction_blocked
            else "See the schema and archive audits for the next permitted stage."
        ),
        "old_modular_context": context,
        "hash_errors": hash_errors,
    }

    write_csv(
        audit_root / "FILE_INVENTORY.csv",
        file_rows,
        [
            "path",
            "relative_path",
            "name",
            "extension",
            "size_bytes",
            "modified_utc",
            "sha256",
            "category",
            "is_archive",
            "is_vector_component",
            "is_readme_license_candidate",
        ],
    )
    write_csv(
        audit_root / "ARCHIVE_INVENTORY.csv",
        archive_rows,
        [
            "archive_path",
            "relative_path",
            "archive_type",
            "size_bytes",
            "sha256",
            "member_listing_status",
            "member_count",
            "member_name_sample",
            "member_vector_component_count",
            "extracted_counterpart_observed",
            "extraction_status",
            "error",
        ],
    )
    write_csv(
        audit_root / "SHA256_INVENTORY.csv",
        sha_rows,
        ["path", "relative_path", "size_bytes", "sha256", "hash_status"],
    )
    write_csv(
        audit_root / "VECTOR_COMPONENT_AUDIT.csv",
        component_rows,
        [
            "status",
            "dataset_stem",
            "directory",
            "relative_directory",
            "observed_components",
            "missing_required_components",
            "has_cpg",
            "component_paths",
        ],
    )
    write_json(
        audit_root / "VECTOR_SCHEMA_AUDIT.json",
        {
            "status": "NOT_AVAILABLE_COMPRESSED_ONLY" if extraction_blocked else "READ_ATTEMPTED",
            "source_root": str(source_root),
            "vector_paths_observed": [str(path) for path in vector_paths],
            "records": schema_rows,
            "unknown_structure_is_error": extraction_blocked,
        },
    )
    write_json(
        audit_root / "HEIGHT_FIELD_AUDIT.json",
        {
            "status": "NOT_AVAILABLE_COMPRESSED_ONLY" if extraction_blocked else "READ_ATTEMPTED",
            "source_root": str(source_root),
            "height_field_known": bool(vector_paths),
            "actual_height_field": None,
            "records": height_rows,
            "height_field_was_not_inferred_from_filename": True,
        },
    )
    write_json(
        audit_root / "SPATIAL_COVERAGE_AUDIT.json",
        {
            "status": "NOT_AVAILABLE_COMPRESSED_ONLY" if extraction_blocked else "READ_ATTEMPTED",
            "source_root": str(source_root),
            "crs": None,
            "geometry_bounds": None,
            "grid_index_rule": None,
            "data_time_anchor": None,
            "reason": "No extracted vector schema or features were available for spatial reading.",
            "nanjing_context_from_sealed_modular_run": context,
        },
    )
    (audit_root / "LICENSE_AND_README_AUDIT.md").write_text(
        build_readme_license_audit(source_root, files, tool_probe), encoding="utf-8"
    )
    write_json(audit_root / "GLOBFP_DATA_READINESS.json", readiness)
    write_json(output_root / "GLOBFP_DATA_AUDIT.json", readiness)
    write_json(output_root / "GLOBFP_DATA_READINESS.json", readiness)

    return readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Data Set\3D-GloBFP"))
    parser.add_argument(
        "--old-modular-root",
        type=Path,
        default=script_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        args.project_root / "outputs" / "data_preparation" / f"globfp_china_nanjing_{timestamp}"
    )
    readiness = audit_source(args.source_root, args.old_modular_root, output_root)
    print(json.dumps({"output_root": str(output_root), **readiness}, ensure_ascii=False, indent=2, default=json_default))
    return 0 if readiness.get("status") != "GLOBFP_SOURCE_NOT_FOUND" else 2


if __name__ == "__main__":
    sys.exit(main())
