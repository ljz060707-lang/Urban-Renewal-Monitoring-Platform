"""Audit a local 3D-GloBFP distribution without modifying source data.

The audit deliberately separates vector footprints from image/label pairs.  A
large provincial Shapefile is not silently treated as an image segmentation
training set.  Metadata is read through pyogrio and only a bounded geometry
sample is loaded for quality checks; full feature materialisation is avoided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_info(path: Path) -> dict[str, Any]:
    import pyogrio  # type: ignore

    try:
        info = pyogrio.read_info(path)
        return {
            "status": "ok",
            "path": str(path),
            "feature_count": int(info.get("features", 0) or 0),
            "crs": str(info.get("crs")) if info.get("crs") else None,
            "geometry_type": str(info.get("geometry_type")),
            "fields": [str(v) for v in info.get("fields", [])],
            "dtypes": [str(v) for v in info.get("dtypes", [])],
            "bounds": [float(v) for v in (info.get("total_bounds") or [])],
        }
    except Exception as exc:
        return {"status": "error", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _sample_geometry_quality(path: Path, sample_size: int) -> dict[str, Any]:
    import numpy as np  # type: ignore
    import pyogrio  # type: ignore

    try:
        frame = pyogrio.read_dataframe(path, max_features=sample_size, use_arrow=False)
        geometries = frame.geometry
        empty = int(geometries.is_empty.fillna(True).sum())
        null = int(geometries.isna().sum())
        invalid = int((~geometries.is_valid.fillna(False)).sum())
        area_values = np.asarray([float(g.area) for g in geometries if g is not None and not g.is_empty], dtype="float64")
        return {
            "status": "sampled",
            "sample_size": int(len(frame)),
            "empty_geometry_count": empty,
            "null_geometry_count": null,
            "invalid_geometry_count": invalid,
            "area_min": float(np.min(area_values)) if area_values.size else None,
            "area_median": float(np.median(area_values)) if area_values.size else None,
            "area_max": float(np.max(area_values)) if area_values.size else None,
            "finite_area_fraction": float(np.isfinite(area_values).mean()) if area_values.size else 0.0,
        }
    except Exception as exc:
        return {"status": "sample_error", "error": f"{type(exc).__name__}: {exc}"}


def classify_asset(info: dict[str, Any], source_root: Path) -> str:
    # GloBFP in this distribution is a vector-only footprint/height archive.
    # Keep the classification explicit instead of inferring an image pairing.
    if info.get("status") != "ok":
        return "E_INVALID_OR_EXCLUDED"
    height = any(str(f).lower() in {"height", "elevation", "building_height"} for f in info.get("fields", []))
    return "C_FOOTPRINT_HEIGHT_ONLY" if height else "D_FOOTPRINT_ONLY"


def audit(source_root: Path, output_root: Path, *, hash_files: bool = True, sample_size: int = 1000) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    audit_root = output_root / "01_globfp_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(source_root.rglob("*")) if p.is_file()]
    file_rows: list[dict[str, Any]] = []
    for p in files:
        stat = p.stat()
        file_rows.append({
            "relative_path": p.relative_to(source_root).as_posix(),
            "path": str(p),
            "extension": p.suffix.lower(),
            "size_bytes": int(stat.st_size),
            "sha256": sha256_file(p) if hash_files else None,
        })
    vectors = [p for p in files if p.suffix.lower() in {".shp", ".gpkg", ".geojson"}]
    records: list[dict[str, Any]] = []
    for p in vectors:
        info = _safe_info(p)
        if info.get("status") == "ok":
            info["height_field_candidates"] = [f for f in info.get("fields", []) if str(f).lower() in {"height", "elevation", "building_height"}]
            info["geometry_quality"] = _sample_geometry_quality(p, sample_size)
        info["relative_path"] = p.relative_to(source_root).as_posix()
        info["asset_class"] = classify_asset(info, source_root)
        info["height_unit_status"] = "unverified" if info.get("height_field_candidates") else "not_applicable"
        records.append(info)

    image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2"}
    raster_files = [r for r in file_rows if r["extension"] in image_exts]
    counts = {name: sum(r.get("asset_class") == name for r in records) for name in {
        "A_IMAGE_FOOTPRINT_HEIGHT", "B_IMAGE_FOOTPRINT", "C_FOOTPRINT_HEIGHT_ONLY", "D_FOOTPRINT_ONLY", "E_INVALID_OR_EXCLUDED"
    }}
    readiness = {
        "status": "vector_only_no_image_pairs" if not raster_files else "image_assets_observed_pending_pairing",
        "audited_at_utc": utc_now(),
        "source_root": str(source_root.resolve()),
        "file_count": len(files),
        "vector_file_count": len(vectors),
        "raster_image_count": len(raster_files),
        "asset_class_counts": counts,
        "height_fields_observed": sorted({f for r in records for f in r.get("height_field_candidates", [])}),
        "height_unit_status": "unverified",
        "trainable_image_pairs": 0,
        "allow_image_segmentation_training": bool(
            raster_files and any(r.get("asset_class") in {"A_IMAGE_FOOTPRINT_HEIGHT", "B_IMAGE_FOOTPRINT"} for r in records)
        ),
        "source_data_modified": False,
        "locked_test_pixel_reads": 0,
        "records": records,
    }
    write_csv(audit_root / "FILE_INVENTORY.csv", file_rows)
    write_csv(audit_root / "VECTOR_RASTER_INVENTORY.csv", records)
    write_json(audit_root / "GLOBFP_SCHEMA_AUDIT.json", {"status": "complete", "records": records})
    write_json(audit_root / "GLOBFP_HEIGHT_AUDIT.json", {
        "status": "unverified_units", "height_fields": readiness["height_fields_observed"],
        "records": [{"path": r.get("path"), "field": (r.get("height_field_candidates") or [None])[0], "unit_status": r.get("height_unit_status")} for r in records],
    })
    write_json(audit_root / "GLOBFP_READINESS.json", readiness)
    write_json(output_root / "GLOBFP_DATA_READINESS.json", readiness)
    write_json(output_root / "DATASET_AUDIT_RECEIPT.json", {
        "dataset": "3D-GloBFP", "status": readiness["status"], "source_data_modified": False,
        "locked_test_pixel_reads": 0, "created_at": utc_now(), "audit_root": str(audit_root.resolve()),
    })
    print(json.dumps(readiness, ensure_ascii=False, indent=2, default=str))
    return readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"D:\Data Set\3D-GloBFP"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args()
    if not args.root.is_dir():
        print(json.dumps({"status": "blocked_source_missing", "root": str(args.root)}, ensure_ascii=False))
        return 2
    audit(args.root, args.output, hash_files=not args.no_hash, sample_size=max(1, args.sample_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
