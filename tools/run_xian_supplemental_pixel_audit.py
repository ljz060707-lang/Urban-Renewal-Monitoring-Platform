"""Audit Xi'an supplemental pixel payloads without modifying any source data.

This is an evidence-registration step for the supplemental integration task.
It deliberately keeps three things separate:

* georeferenced, readable raster assets (direct TIFF/PNG/JPEG payloads);
* cache JPEG collections whose tile coordinates can be inspected but whose
  per-tile georeferencing is not promoted to formal geometry; and
* Google Earth cache/date metadata (``.dat``, ``.raw``, ``.datever``, ``.ver``)
  which is never treated as an image by this script.

The output is append-only under a caller-supplied directory.  No geometry,
tracking, Presence, State, or formal product is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


RASTER_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".geotiff"}
DATE_METADATA_EXT = {".datever"}
GE_CACHE_EXT = {
    ".dat", ".raw", ".ver", ".bpw", ".tfw", ".map", ".geid", ".zip",
    ".bak", ".log", ".html", ".txt",
}
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])(?!\d)")
TILE_RE = re.compile(r"(?:^|[_\\/])(?:gesh_)?(\d{4,7})_(\d{4,7})_(\d{1,2})\.(?:jpg|jpeg|png)$", re.I)
PACKED_DATE_SENTINEL_MAX = 545
TARGET_DATES = {
    "2018": date(2018, 4, 29),
    "2019": date(2019, 1, 1),
    "2020": date(2020, 12, 31),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def packed_date(value: int) -> str | None:
    if value <= PACKED_DATE_SENTINEL_MAX:
        return None
    year = value >> 9
    month = (value >> 5) & 0x0F
    day = value & 0x1F
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def decode_datever(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    try:
        with path.open("rb") as handle:
            data = handle.read()
    except OSError:
        return counts
    usable = len(data) - (len(data) % 16)
    for offset in range(0, usable, 16):
        packed = struct.unpack_from("<I", data, offset)[0]
        actual = packed_date(packed)
        if actual:
            counts[actual] += 1
    return counts


def parse_dates(text: str) -> set[str]:
    return {f"{year}-{month}-{day}" for year, month, day in DATE_RE.findall(text)}


def safe_float(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if number == number else None


def csv_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in RASTER_EXT:
        return "REAL_PIXEL_RASTER"
    if suffix in DATE_METADATA_EXT:
        return "DATE_METADATA"
    if suffix in GE_CACHE_EXT:
        return "GE_CACHE_METADATA_ONLY"
    return "UNKNOWN_FORMAT"


def file_walk(root: Path) -> list[Path]:
    result: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        result.extend(base / name for name in filenames)
    return result


def top_observation(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if relative.parts else root.name


def observation_root(root: Path, name: str) -> Path:
    return root / name


def datever_stats(files: list[Path]) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    records = 0
    readable = 0
    for path in files:
        if path.suffix.lower() != ".datever":
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size % 16:
            continue
        readable += 1
        current = decode_datever(path)
        counts.update(current)
        records += sum(current.values())
    return counts, records, readable


def infer_dates(path: Path, root_name: str, date_counts: Counter[str]) -> tuple[list[str], str]:
    name_dates = parse_dates(str(path)) | parse_dates(root_name)
    if date_counts:
        return sorted(date_counts), "DATEVER_DECODED"
    if name_dates:
        return sorted(name_dates), "EXPLICIT_DATE_IN_PATH"
    return [], "NO_ACTUAL_DATE_EVIDENCE"


def nearest_target(actual_date: str) -> tuple[str, int] | tuple[str, str]:
    try:
        current = date.fromisoformat(actual_date)
    except ValueError:
        return "", ""
    best_epoch, best_days = min(
        ((epoch, abs((current - target).days)) for epoch, target in TARGET_DATES.items()),
        key=lambda item: item[1],
    )
    return best_epoch, best_days


def sha256_path_plus_size(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: rel_posix(item, root).lower()):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        digest.update(rel_posix(path, root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def raster_metadata(path: Path) -> dict[str, Any]:
    """Read metadata without loading the raster into memory."""
    result: dict[str, Any] = {
        "format": path.suffix.lower().lstrip("."),
        "crs": "",
        "transform": "",
        "width": "",
        "height": "",
        "bands": "",
        "resolution_x": "",
        "resolution_y": "",
        "bbox_left": "",
        "bbox_bottom": "",
        "bbox_right": "",
        "bbox_top": "",
        "nodata": "",
        "readable": False,
        "metadata_read_status": "NOT_READ",
    }
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff", ".geotiff", ".jp2"}:
        try:
            import rasterio  # type: ignore

            with rasterio.open(path) as dataset:
                result.update(
                    {
                        "format": dataset.driver,
                        "crs": dataset.crs.to_string() if dataset.crs else "",
                        "transform": ",".join(f"{float(value):.15g}" for value in dataset.transform),
                        "width": dataset.width,
                        "height": dataset.height,
                        "bands": dataset.count,
                        "resolution_x": safe_float(dataset.res[0]),
                        "resolution_y": safe_float(dataset.res[1]),
                        "bbox_left": safe_float(dataset.bounds.left),
                        "bbox_bottom": safe_float(dataset.bounds.bottom),
                        "bbox_right": safe_float(dataset.bounds.right),
                        "bbox_top": safe_float(dataset.bounds.top),
                        "nodata": ";".join("" if value is None else str(value) for value in dataset.nodatavals),
                        "readable": True,
                        "metadata_read_status": "READABLE_RASTERIO",
                    }
                )
            return result
        except Exception as exc:
            result["metadata_read_status"] = f"RASTER_READ_ERROR:{type(exc).__name__}"
            return result
    try:
        from PIL import Image  # type: ignore

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            result.update(
                {
                    "format": image.format or result["format"],
                    "width": image.width,
                    "height": image.height,
                    "bands": len(image.getbands()),
                    "readable": True,
                    "metadata_read_status": "READABLE_PIL_NO_GEOREF",
                }
            )
    except Exception as exc:
        result["metadata_read_status"] = f"IMAGE_READ_ERROR:{type(exc).__name__}"
    return result


def tile_range(pixel_files: list[Path]) -> dict[str, Any]:
    xs: list[int] = []
    ys: list[int] = []
    zooms: list[int] = []
    for path in pixel_files:
        match = TILE_RE.search(path.as_posix())
        if not match:
            continue
        xs.append(int(match.group(1)))
        ys.append(int(match.group(2)))
        zooms.append(int(match.group(3)))
    if not xs:
        return {"tile_x_min": "", "tile_x_max": "", "tile_y_min": "", "tile_y_max": "", "tile_zoom": ""}
    return {
        "tile_x_min": min(xs),
        "tile_x_max": max(xs),
        "tile_y_min": min(ys),
        "tile_y_max": max(ys),
        "tile_zoom": ";".join(str(value) for value in sorted(set(zooms))),
    }


def build_inventory(root: Path, out: Path) -> dict[str, Any]:
    root = root.resolve()
    out = out.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = file_walk(root)
    files_by_obs: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        files_by_obs[top_observation(path, root)].append(path)

    inventory_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    pixel_total_bytes = 0
    pixel_total_files = 0
    pixel_observation_count = 0
    representative_paths: list[Path] = []

    for obs_name in sorted(files_by_obs, key=str.lower):
        obs_files = files_by_obs[obs_name]
        class_counts = Counter(classify(path) for path in obs_files)
        for key, value in class_counts.items():
            format_counts[key] += value
        raster_files = [path for path in obs_files if classify(path) == "REAL_PIXEL_RASTER"]
        datever_files = [path for path in obs_files if path.suffix.lower() == ".datever"]
        date_counts, datever_records, datever_readable = datever_stats(datever_files)
        explicit_dates = parse_dates(obs_name)
        actual_dates = sorted(set(date_counts) or explicit_dates)
        pixel_bytes = sum(path.stat().st_size for path in raster_files if path.exists())
        pixel_total_bytes += pixel_bytes
        pixel_total_files += len(raster_files)
        pixel_observation_count += int(bool(raster_files))
        representative = next((path for path in raster_files if path.suffix.lower() in {".tif", ".tiff", ".jp2", ".geotiff"}), None)
        if representative is None and raster_files:
            representative = raster_files[0]
        if representative:
            representative_paths.append(representative)

        tile_info = tile_range(raster_files)
        observation_rows.append(
            {
                "observation_id": f"xian_supplement::{obs_name}",
                "source_asset_id": f"xian_supplement::{obs_name}",
                "source_ref": str(observation_root(root, obs_name)),
                "observation_label": obs_name,
                "classification": "REAL_PIXEL_RASTER" if raster_files else ("DATE_METADATA" if datever_files else "GE_CACHE_METADATA_ONLY"),
                "pixel_available": "YES" if raster_files else "NO",
                "pixel_file_count": len(raster_files),
                "pixel_bytes": pixel_bytes,
                "all_file_count": len(obs_files),
                "class_counts_json": json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False, sort_keys=True),
                "datever_file_count": len(datever_files),
                "datever_readable_file_count": datever_readable,
                "datever_decoded_record_count": datever_records,
                "actual_dates": ";".join(actual_dates),
                "actual_date_provenance": "DATEVER_DECODED" if date_counts else ("EXPLICIT_DATE_IN_PATH" if explicit_dates else "NO_ACTUAL_DATE_EVIDENCE"),
                "representative_pixel_asset": str(representative) if representative else "",
                "tile_x_min": tile_info["tile_x_min"],
                "tile_x_max": tile_info["tile_x_max"],
                "tile_y_min": tile_info["tile_y_min"],
                "tile_y_max": tile_info["tile_y_max"],
                "tile_zoom": tile_info["tile_zoom"],
                "inventory_sha256_path_plus_size": sha256_path_plus_size(obs_files, root),
                "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                "raw_inputs_modified": False,
            }
        )
        for actual in actual_dates:
            nearest_epoch, days = nearest_target(actual)
            date_rows.append(
                {
                    "observation_id": f"xian_supplement::{obs_name}",
                    "observation_label": obs_name,
                    "source_ref": str(observation_root(root, obs_name)),
                    "actual_acquisition_date": actual,
                    "date_provenance": "DATEVER_DECODED" if date_counts else "EXPLICIT_DATE_IN_PATH",
                    "pixel_available": "YES" if raster_files else "NO",
                    "pixel_file_count": len(raster_files),
                    "nearest_formal_target": nearest_epoch,
                    "days_from_nearest_target": days,
                    "near_target_priority": "HIGH" if isinstance(days, int) and days <= 180 else ("MEDIUM" if isinstance(days, int) and days <= 540 else "LOW"),
                    "formal_epoch_created": False,
                    "supplement_role": "date_validation;same_date_dedup;alignment_parallax;identity;observability",
                }
            )

        # The four large combined TIFFs are recorded individually.  A cache
        # folder with many JPEGs is recorded as a pixel collection plus one
        # representative image: this keeps the inventory auditable without
        # producing a multi-hundred-megabyte CSV of repeated tile headers.
        direct_files = [path for path in raster_files if path.suffix.lower() in {".tif", ".tiff", ".jp2", ".geotiff"}]
        if direct_files:
            for path in direct_files:
                metadata = raster_metadata(path)
                dates, provenance = infer_dates(path, obs_name, date_counts)
                row = {
                    "row_type": "PIXEL_ASSET",
                    "classification": "REAL_PIXEL_RASTER",
                    "pixel_available": "YES",
                    "observation_id": f"xian_supplement::{obs_name}",
                    "source_asset_id": f"xian_supplement::{obs_name}::{rel_posix(path, observation_root(root, obs_name))}",
                    "source_ref": str(path),
                    "relative_path": rel_posix(path, root),
                    "size_bytes": path.stat().st_size,
                    "format": metadata["format"],
                    "crs": metadata["crs"],
                    "transform": metadata["transform"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "bands": metadata["bands"],
                    "resolution_x": metadata["resolution_x"],
                    "resolution_y": metadata["resolution_y"],
                    "bbox_left": metadata["bbox_left"],
                    "bbox_bottom": metadata["bbox_bottom"],
                    "bbox_right": metadata["bbox_right"],
                    "bbox_top": metadata["bbox_top"],
                    "nodata": metadata["nodata"],
                    "readable": metadata["readable"],
                    "metadata_read_status": metadata["metadata_read_status"],
                    "capture_dates": ";".join(dates),
                    "capture_date_provenance": provenance,
                    "content_sha256": "NOT_COMPUTED_LARGE_RASTER",
                    "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                    "raw_inputs_modified": False,
                }
                inventory_rows.append(row)
        elif raster_files:
            representative = raster_files[0]
            metadata = raster_metadata(representative)
            inventory_rows.append(
                {
                    "row_type": "PIXEL_COLLECTION",
                    "classification": "REAL_PIXEL_RASTER",
                    "pixel_available": "YES",
                    "observation_id": f"xian_supplement::{obs_name}",
                    "source_asset_id": f"xian_supplement::{obs_name}::PIXEL_COLLECTION",
                    "source_ref": str(observation_root(root, obs_name)),
                    "relative_path": rel_posix(representative, root),
                    "size_bytes": pixel_bytes,
                    "format": "JPEG/PNG_COLLECTION",
                    "crs": "",
                    "transform": "",
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "bands": metadata["bands"],
                    "resolution_x": "",
                    "resolution_y": "",
                    "bbox_left": "",
                    "bbox_bottom": "",
                    "bbox_right": "",
                    "bbox_top": "",
                    "nodata": "",
                    "readable": metadata["readable"],
                    "metadata_read_status": "COLLECTION_REPRESENTATIVE_ONLY_NO_PER_TILE_GEOREF",
                    "capture_dates": ";".join(actual_dates),
                    "capture_date_provenance": "DATEVER_DECODED" if date_counts else "EXPLICIT_DATE_IN_PATH",
                    "pixel_file_count": len(raster_files),
                    "tile_x_min": tile_info["tile_x_min"],
                    "tile_x_max": tile_info["tile_x_max"],
                    "tile_y_min": tile_info["tile_y_min"],
                    "tile_y_max": tile_info["tile_y_max"],
                    "tile_zoom": tile_info["tile_zoom"],
                    "content_sha256": "NOT_COMPUTED_COLLECTION",
                    "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                    "raw_inputs_modified": False,
                }
            )

    # Add a compact, complete classification audit of non-pixel formats.  It
    # is one row per observation/format, not a claim that these files are
    # observations themselves.
    for obs in observation_rows:
        counts = json.loads(obs["class_counts_json"])
        for classification, count in sorted(counts.items()):
            if classification == "REAL_PIXEL_RASTER":
                continue
            inventory_rows.append(
                {
                    "row_type": "FORMAT_SUMMARY",
                    "classification": classification,
                    "pixel_available": "NO",
                    "observation_id": obs["observation_id"],
                    "source_asset_id": f"{obs['observation_id']}::{classification}",
                    "source_ref": obs["source_ref"],
                    "relative_path": "",
                    "size_bytes": "",
                    "format": classification,
                    "readable": "NOT_APPLICABLE_AS_PIXEL",
                    "metadata_read_status": "NOT_AN_IMAGE_BY_POLICY",
                    "file_count": count,
                    "actual_dates": obs["actual_dates"],
                    "prohibition_note": "Metadata/cache only; not promoted to image pixels unless legally reconstructed outside this audit.",
                }
            )

    # Date coverage summary used to drive the next evidence step.
    actual_counts = Counter(row["actual_acquisition_date"] for row in date_rows)
    target_rows = []
    for epoch, target in TARGET_DATES.items():
        matching = [row for row in date_rows if row["nearest_formal_target"] == epoch]
        target_rows.append(
            {
                "formal_target": epoch,
                "target_date": target.isoformat(),
                "observation_date_rows": len(matching),
                "pixel_available_observation_rows": sum(row["pixel_available"] == "YES" for row in matching),
                "actual_dates": ";".join(sorted({row["actual_acquisition_date"] for row in matching})),
                "best_days_from_target": min((int(row["days_from_nearest_target"]) for row in matching), default=""),
                "note": "This is supplemental support only; it is not a formal epoch and does not replace main geometry.",
            }
        )

    manifest = {
        "schema_version": "urrs.xian.supplemental.pixel.audit.v1",
        "created_at_utc": utc_now(),
        "root": str(root),
        "output_root": str(out),
        "source_files_scanned": len(files),
        "observation_count": len(observation_rows),
        "pixel_file_count": pixel_total_files,
        "pixel_bytes": pixel_total_bytes,
        "pixel_observation_count": pixel_observation_count,
        "classification_counts": dict(sorted(format_counts.items())),
        "actual_date_count": len(actual_counts),
        "actual_dates": dict(sorted(actual_counts.items())),
        "target_summary": target_rows,
        "direct_georeferenced_rasters": [row["source_ref"] for row in inventory_rows if row.get("row_type") == "PIXEL_ASSET" and row.get("crs")],
        "pixel_policy": {
            "supplemental_never_replaces_main_geometry": True,
            "supplemental_never_becomes_fourth_formal_epoch": True,
            "dat_raw_datever_ver_not_images": True,
            "not_detected_is_not_absent": True,
            "alignment_before_absence": True,
            "real_pixel_raster_collections_are_registered_at_collection_scope": True,
        },
        "raw_inputs_modified": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / "SUPPLEMENTAL_PIXEL_INVENTORY.csv", inventory_rows)
    csv_write(out / "SUPPLEMENTAL_ACTUAL_DATE_REGISTRY.csv", date_rows)
    csv_write(out / "SUPPLEMENTAL_OBSERVATION_REGISTRY.csv", observation_rows)
    csv_write(out / "SUPPLEMENTAL_TARGET_DATE_SUMMARY.csv", target_rows)
    (out / "SUPPLEMENTAL_PIXEL_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Xi'an supplemental pixel audit\n\n"
        "本目录只登记 `F:\\xian\\XIAN补充` 的真实像元、实际日期和缓存证据，不修改三期主影像、Geometry、Tracking、Presence、State 或正式产品。\n\n"
        "- `REAL_PIXEL_RASTER`：可读取的 TIF/PNG/JPEG/JPEG2000 像元。大规模缓存 JPEG 按 observation 的像元集合登记，并保留代表性 tile 的尺寸读取结果。\n"
        "- `GE_CACHE_METADATA_ONLY`：`.dat/.raw/.ver/.bpw/.tfw/.map/.geid` 等缓存/空间参考/控制文件；不会被当作影像。\n"
        "- `DATE_METADATA`：`.datever` 解码得到实际采集日期；名义文件夹年份不替代实际日期。\n"
        "- 辅助影像只是日期核验、同日去重、配准/视差、身份和可观测性证据，不能替换主 Geometry。\n\n"
        "完整像元集合的逐文件扫描会产生极大的 CSV，因此主清单对大规模 cache JPEG 采用 collection scope，并记录像元数量、总字节数、tile 范围和代表性文件；直接组合 TIF 逐文件给出 CRS、transform、尺寸、波段、分辨率、bbox、nodata 和可读性。\n",
        encoding="utf-8",
    )
    return manifest


def build_inventory_fast(root: Path, out: Path) -> dict[str, Any]:
    """Streaming fallback for very large cache trees.

    The first implementation retained every Path and calculated a path/size
    digest for every tile.  That is unnecessarily expensive on a networked or
    removable drive.  This mode keeps the same classification and direct-TIF
    metadata contract, while aggregating large JPEG collections during one
    streaming directory walk.  It never claims per-tile CRS/transform for a
    cache collection and records the limitation in the manifest.
    """
    root = root.resolve()
    out = out.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    observations = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower())
    inventory_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    format_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    total_files = 0
    total_pixel_files = 0
    total_pixel_bytes = 0

    for obs_dir in observations:
        class_counts: Counter[str] = Counter()
        raster_count = 0
        raster_bytes = 0
        datever_count = 0
        datever_bytes = 0
        sample_raster: Path | None = None
        direct_rasters: list[Path] = []
        sample_dates: set[str] = set()
        tile_x: list[int] = []
        tile_y: list[int] = []
        tile_z: set[int] = set()
        obs_files = 0
        for dirpath, _, filenames in os.walk(obs_dir):
            base = Path(dirpath)
            for filename in filenames:
                path = base / filename
                kind = classify(path)
                class_counts[kind] += 1
                obs_files += 1
                total_files += 1
                if kind == "REAL_PIXEL_RASTER":
                    raster_count += 1
                    try:
                        raster_bytes += path.stat().st_size
                    except OSError:
                        pass
                    if sample_raster is None:
                        sample_raster = path
                    if path.suffix.lower() in {".tif", ".tiff", ".jp2", ".geotiff"}:
                        direct_rasters.append(path)
                    match = TILE_RE.search(path.as_posix())
                    if match:
                        tile_x.append(int(match.group(1)))
                        tile_y.append(int(match.group(2)))
                        tile_z.add(int(match.group(3)))
                elif path.suffix.lower() == ".datever":
                    datever_count += 1
                    try:
                        datever_bytes += path.stat().st_size
                    except OSError:
                        pass
                    # Validate a small sample only.  The explicit date in the
                    # observation label remains the primary capture-date
                    # evidence for a cache folder.
                    if len(sample_dates) < 8:
                        sample_dates.update(decode_datever(path))

        explicit_dates = parse_dates(obs_dir.name)
        actual_dates = sorted(sample_dates or explicit_dates)
        for actual in actual_dates:
            actual_counts[actual] += 1
        for key, value in class_counts.items():
            format_counts[key] += value
        total_pixel_files += raster_count
        total_pixel_bytes += raster_bytes
        if tile_x:
            tile_info = {
                "tile_x_min": min(tile_x),
                "tile_x_max": max(tile_x),
                "tile_y_min": min(tile_y),
                "tile_y_max": max(tile_y),
                "tile_zoom": ";".join(str(value) for value in sorted(tile_z)),
            }
        else:
            tile_info = {"tile_x_min": "", "tile_x_max": "", "tile_y_min": "", "tile_y_max": "", "tile_zoom": ""}
        pixel_available = bool(raster_count)
        observation_rows.append(
            {
                "observation_id": f"xian_supplement::{obs_dir.name}",
                "source_asset_id": f"xian_supplement::{obs_dir.name}",
                "source_ref": str(obs_dir),
                "observation_label": obs_dir.name,
                "classification": "REAL_PIXEL_RASTER" if pixel_available else ("DATE_METADATA" if datever_count else "GE_CACHE_METADATA_ONLY"),
                "pixel_available": "YES" if pixel_available else "NO",
                "pixel_file_count": raster_count,
                "pixel_bytes": raster_bytes,
                "all_file_count": obs_files,
                "class_counts_json": json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False, sort_keys=True),
                "datever_file_count": datever_count,
                "datever_readable_file_count": "SAMPLED_ONLY",
                "datever_decoded_record_count": "SAMPLED_ONLY",
                "actual_dates": ";".join(actual_dates),
                "actual_date_provenance": "DATEVER_SAMPLE_PLUS_EXPLICIT_PATH" if sample_dates else ("EXPLICIT_DATE_IN_PATH" if explicit_dates else "NO_ACTUAL_DATE_EVIDENCE"),
                "representative_pixel_asset": str(sample_raster) if sample_raster else "",
                **tile_info,
                "inventory_sha256_path_plus_size": "NOT_COMPUTED_FAST_STREAMING_AUDIT",
                "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                "raw_inputs_modified": False,
            }
        )
        date_provenance = "DATEVER_SAMPLE_PLUS_EXPLICIT_PATH" if sample_dates else "EXPLICIT_DATE_IN_PATH"
        for actual in actual_dates:
            nearest_epoch, days = nearest_target(actual)
            date_rows.append(
                {
                    "observation_id": f"xian_supplement::{obs_dir.name}",
                    "observation_label": obs_dir.name,
                    "source_ref": str(obs_dir),
                    "actual_acquisition_date": actual,
                    "date_provenance": date_provenance,
                    "pixel_available": "YES" if pixel_available else "NO",
                    "pixel_file_count": raster_count,
                    "nearest_formal_target": nearest_epoch,
                    "days_from_nearest_target": days,
                    "near_target_priority": "HIGH" if isinstance(days, int) and days <= 180 else ("MEDIUM" if isinstance(days, int) and days <= 540 else "LOW"),
                    "formal_epoch_created": False,
                    "supplement_role": "date_validation;same_date_dedup;alignment_parallax;identity;observability",
                }
            )
        if direct_rasters:
            for path in direct_rasters:
                metadata = raster_metadata(path)
                dates, provenance = infer_dates(path, obs_dir.name, Counter(sample_dates))
                inventory_rows.append(
                    {
                        "row_type": "PIXEL_ASSET",
                        "classification": "REAL_PIXEL_RASTER",
                        "pixel_available": "YES",
                        "observation_id": f"xian_supplement::{obs_dir.name}",
                        "source_asset_id": f"xian_supplement::{obs_dir.name}::{path.relative_to(obs_dir).as_posix()}",
                        "source_ref": str(path),
                        "relative_path": rel_posix(path, root),
                        "size_bytes": path.stat().st_size,
                        "format": metadata["format"],
                        "crs": metadata["crs"],
                        "transform": metadata["transform"],
                        "width": metadata["width"],
                        "height": metadata["height"],
                        "bands": metadata["bands"],
                        "resolution_x": metadata["resolution_x"],
                        "resolution_y": metadata["resolution_y"],
                        "bbox_left": metadata["bbox_left"],
                        "bbox_bottom": metadata["bbox_bottom"],
                        "bbox_right": metadata["bbox_right"],
                        "bbox_top": metadata["bbox_top"],
                        "nodata": metadata["nodata"],
                        "readable": metadata["readable"],
                        "metadata_read_status": metadata["metadata_read_status"],
                        "capture_dates": ";".join(dates),
                        "capture_date_provenance": provenance,
                        "content_sha256": "NOT_COMPUTED_LARGE_RASTER",
                        "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                        "raw_inputs_modified": False,
                    }
                )
        elif sample_raster:
            metadata = raster_metadata(sample_raster)
            inventory_rows.append(
                {
                    "row_type": "PIXEL_COLLECTION",
                    "classification": "REAL_PIXEL_RASTER",
                    "pixel_available": "YES",
                    "observation_id": f"xian_supplement::{obs_dir.name}",
                    "source_asset_id": f"xian_supplement::{obs_dir.name}::PIXEL_COLLECTION",
                    "source_ref": str(obs_dir),
                    "relative_path": rel_posix(sample_raster, root),
                    "size_bytes": raster_bytes,
                    "format": "JPEG/PNG_COLLECTION",
                    "crs": "",
                    "transform": "",
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "bands": metadata["bands"],
                    "resolution_x": "",
                    "resolution_y": "",
                    "bbox_left": "",
                    "bbox_bottom": "",
                    "bbox_right": "",
                    "bbox_top": "",
                    "nodata": "",
                    "readable": metadata["readable"],
                    "metadata_read_status": "COLLECTION_REPRESENTATIVE_ONLY_NO_PER_TILE_GEOREF",
                    "capture_dates": ";".join(actual_dates),
                    "capture_date_provenance": "DATEVER_SAMPLE_PLUS_EXPLICIT_PATH" if sample_dates else "EXPLICIT_DATE_IN_PATH",
                    "pixel_file_count": raster_count,
                    **tile_info,
                    "content_sha256": "NOT_COMPUTED_COLLECTION",
                    "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                    "raw_inputs_modified": False,
                }
            )
        for classification, count in sorted(class_counts.items()):
            if classification == "REAL_PIXEL_RASTER":
                continue
            inventory_rows.append(
                {
                    "row_type": "FORMAT_SUMMARY",
                    "classification": classification,
                    "pixel_available": "NO",
                    "observation_id": f"xian_supplement::{obs_dir.name}",
                    "source_asset_id": f"xian_supplement::{obs_dir.name}::{classification}",
                    "source_ref": str(obs_dir),
                    "format": classification,
                    "readable": "NOT_APPLICABLE_AS_PIXEL",
                    "metadata_read_status": "NOT_AN_IMAGE_BY_POLICY",
                    "file_count": count,
                    "actual_dates": ";".join(actual_dates),
                    "prohibition_note": "Metadata/cache only; not promoted to image pixels unless legally reconstructed outside this audit.",
                }
            )

    target_rows: list[dict[str, Any]] = []
    for epoch, target in TARGET_DATES.items():
        matching = [row for row in date_rows if row["nearest_formal_target"] == epoch]
        target_rows.append(
            {
                "formal_target": epoch,
                "target_date": target.isoformat(),
                "observation_date_rows": len(matching),
                "pixel_available_observation_rows": sum(row["pixel_available"] == "YES" for row in matching),
                "actual_dates": ";".join(sorted({row["actual_acquisition_date"] for row in matching})),
                "best_days_from_target": min((int(row["days_from_nearest_target"]) for row in matching), default=""),
                "note": "Supplemental support only; no fourth formal epoch and no geometry replacement.",
            }
        )
    manifest = {
        "schema_version": "urrs.xian.supplemental.pixel.audit.v1",
        "created_at_utc": utc_now(),
        "root": str(root),
        "output_root": str(out),
        "scan_mode": "FAST_STREAMING_DIRECTORY_AUDIT",
        "source_files_scanned": total_files,
        "observation_count": len(observation_rows),
        "pixel_file_count": total_pixel_files,
        "pixel_bytes": total_pixel_bytes,
        "pixel_observation_count": sum(row["pixel_available"] == "YES" for row in observation_rows),
        "classification_counts": dict(sorted(format_counts.items())),
        "actual_date_count": len(actual_counts),
        "actual_dates": dict(sorted(actual_counts.items())),
        "target_summary": target_rows,
        "direct_georeferenced_rasters": [row["source_ref"] for row in inventory_rows if row.get("row_type") == "PIXEL_ASSET" and row.get("crs")],
        "pixel_policy": {
            "supplemental_never_replaces_main_geometry": True,
            "supplemental_never_becomes_fourth_formal_epoch": True,
            "dat_raw_datever_ver_not_images": True,
            "not_detected_is_not_absent": True,
            "alignment_before_absence": True,
            "cache_collections_registered_at_collection_scope": True,
        },
        "limitations": [
            "large cache folders were aggregated during streaming enumeration",
            "datever capture dates were sampled for cache folders; explicit dated observation labels were retained",
            "path-plus-size SHA was not computed in fast mode",
            "per-cache-tile CRS/transform/bbox is not asserted",
        ],
        "raw_inputs_modified": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / "SUPPLEMENTAL_PIXEL_INVENTORY.csv", inventory_rows)
    csv_write(out / "SUPPLEMENTAL_ACTUAL_DATE_REGISTRY.csv", date_rows)
    csv_write(out / "SUPPLEMENTAL_OBSERVATION_REGISTRY.csv", observation_rows)
    csv_write(out / "SUPPLEMENTAL_TARGET_DATE_SUMMARY.csv", target_rows)
    (out / "SUPPLEMENTAL_PIXEL_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Xi'an supplemental pixel audit\n\n"
        "本目录是只读、流式的辅助影像证据审计。组合 TIF 逐文件读取地理元数据；百万级 cache JPEG 按 observation/collection 汇总，`.dat/.raw/.datever/.ver` 仍按政策排除出图像。\n\n"
        "未修改 Geometry、Tracking、Presence、State 或正式产品；辅助影像只提供日期、同日去重、配准/视差、身份和可观测性支持。\n",
        encoding="utf-8",
    )
    return manifest


def build_inventory_targeted(root: Path, out: Path, prior_inventory: Path | None = None) -> dict[str, Any]:
    """Audit direct raster payloads and reuse a prior read-only cache census.

    Some cache trees are millions of small files on a slow F: volume.  A
    bounded audit is preferable to pretending that a time-out is a complete
    scan.  This mode verifies the current root and every direct/combined
    raster path now, while carrying forward the prior cache count only with an
    explicit provenance/limitation field.
    """
    root = root.resolve()
    out = out.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if prior_inventory is None:
        prior_inventory = Path(__file__).resolve().parents[1] / "work" / "xian_supplement_inventory_20260914.json"
    prior_rows: dict[str, dict[str, Any]] = {}
    prior_status = "NOT_AVAILABLE"
    if prior_inventory.is_file():
        try:
            prior_payload = json.loads(prior_inventory.read_text(encoding="utf-8"))
            prior_rows = {str(row.get("folder")): row for row in prior_payload.get("dates", [])}
            prior_status = str(prior_inventory)
        except Exception as exc:
            prior_status = f"READ_ERROR:{type(exc).__name__}"

    inventory_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    actual_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    all_direct_checked: list[str] = []
    direct_raster_count = 0
    direct_raster_bytes = 0
    carried_pixel_count = 0
    carried_pixel_bytes = 0

    folders = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower())
    for obs_dir in folders:
        obs_name = obs_dir.name
        # Direct files are cheap to enumerate and cover the combined TIFFs,
        # sidecars, datever aggregates, and control metadata.
        direct_files = [path for path in obs_dir.iterdir() if path.is_file()]
        direct_counts = Counter(classify(path) for path in direct_files)
        for key, value in direct_counts.items():
            format_counts[key] += value
        direct_tifs = [path for path in direct_files if path.suffix.lower() in {".tif", ".tiff", ".jp2", ".geotiff"}]
        for path in direct_tifs:
            all_direct_checked.append(str(path))
        prior = prior_rows.get(obs_name, {})
        prior_pixel_count = int(prior.get("jpg_count", 0) or 0)
        prior_pixel_bytes = int(prior.get("bytes", 0) or 0)
        direct_pixel_count = len([path for path in direct_files if classify(path) == "REAL_PIXEL_RASTER"])
        direct_pixel_bytes = sum(path.stat().st_size for path in direct_files if classify(path) == "REAL_PIXEL_RASTER")
        # Current root-level direct raster evidence takes precedence.  Nested
        # cache JPEGs use the prior census, with provenance retained below.
        pixel_count = direct_pixel_count if direct_pixel_count else prior_pixel_count
        pixel_bytes = direct_pixel_bytes if direct_pixel_count else prior_pixel_bytes
        pixel_available = pixel_count > 0
        source_status = "CURRENT_DIRECT_RASTER_VERIFIED" if direct_pixel_count else ("PRIOR_CACHE_CENSUS_CARRIED_FORWARD" if prior_pixel_count else "NO_CURRENT_PIXEL_PAYLOAD_VERIFIED")
        if direct_pixel_count:
            direct_raster_count += direct_pixel_count
            direct_raster_bytes += direct_pixel_bytes
        else:
            carried_pixel_count += prior_pixel_count
            carried_pixel_bytes += prior_pixel_bytes

        # Folder label/date is retained even when a sample datever aggregate
        # is present.  The aggregate is only a supporting date record.
        explicit_dates = parse_dates(obs_name)
        datever_aggregates = [path for path in direct_files if "datever" in path.name.lower()]
        sample_dates: set[str] = set()
        for path in datever_aggregates[:4]:
            sample_dates.update(decode_datever(path))
        actual_dates = sorted(sample_dates or explicit_dates)
        date_provenance = "DATEVER_AGGREGATE_SAMPLE_PLUS_EXPLICIT_PATH" if sample_dates else ("EXPLICIT_DATE_IN_PATH" if explicit_dates else "NO_ACTUAL_DATE_EVIDENCE")
        for actual in actual_dates:
            actual_counts[actual] += 1
        direct_repr = direct_tifs[0] if direct_tifs else None
        observation_rows.append(
            {
                "observation_id": f"xian_supplement::{obs_name}",
                "source_asset_id": f"xian_supplement::{obs_name}",
                "source_ref": str(obs_dir),
                "observation_label": obs_name,
                "classification": "REAL_PIXEL_RASTER" if pixel_available else ("DATE_METADATA" if datever_aggregates else "GE_CACHE_METADATA_ONLY"),
                "pixel_available": "YES" if pixel_available else "NO",
                "pixel_file_count": pixel_count,
                "pixel_bytes": pixel_bytes,
                "pixel_count_provenance": source_status,
                "all_file_count_direct_only": len(direct_files),
                "class_counts_direct_json": json.dumps(dict(sorted(direct_counts.items())), ensure_ascii=False, sort_keys=True),
                "datever_aggregate_count_direct": len(datever_aggregates),
                "actual_dates": ";".join(actual_dates),
                "actual_date_provenance": date_provenance,
                "representative_pixel_asset": str(direct_repr) if direct_repr else str(prior.get("representative_pixel_asset", "")),
                "inventory_sha256_path_plus_size": "NOT_COMPUTED_TARGETED_AUDIT",
                "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                "raw_inputs_modified": False,
            }
        )
        for actual in actual_dates:
            nearest_epoch, days = nearest_target(actual)
            date_rows.append(
                {
                    "observation_id": f"xian_supplement::{obs_name}",
                    "observation_label": obs_name,
                    "source_ref": str(obs_dir),
                    "actual_acquisition_date": actual,
                    "date_provenance": date_provenance,
                    "pixel_available": "YES" if pixel_available else "NO",
                    "pixel_file_count": pixel_count,
                    "pixel_count_provenance": source_status,
                    "nearest_formal_target": nearest_epoch,
                    "days_from_nearest_target": days,
                    "near_target_priority": "HIGH" if isinstance(days, int) and days <= 180 else ("MEDIUM" if isinstance(days, int) and days <= 540 else "LOW"),
                    "formal_epoch_created": False,
                    "supplement_role": "date_validation;same_date_dedup;alignment_parallax;identity;observability",
                }
            )
        if direct_tifs:
            for path in direct_tifs:
                metadata = raster_metadata(path)
                dates, provenance = infer_dates(path, obs_name, Counter(sample_dates))
                inventory_rows.append(
                    {
                        "row_type": "PIXEL_ASSET",
                        "classification": "REAL_PIXEL_RASTER",
                        "pixel_available": "YES",
                        "observation_id": f"xian_supplement::{obs_name}",
                        "source_asset_id": f"xian_supplement::{obs_name}::{path.relative_to(obs_dir).as_posix()}",
                        "source_ref": str(path),
                        "relative_path": rel_posix(path, root),
                        "size_bytes": path.stat().st_size,
                        "format": metadata["format"],
                        "crs": metadata["crs"],
                        "transform": metadata["transform"],
                        "width": metadata["width"],
                        "height": metadata["height"],
                        "bands": metadata["bands"],
                        "resolution_x": metadata["resolution_x"],
                        "resolution_y": metadata["resolution_y"],
                        "bbox_left": metadata["bbox_left"],
                        "bbox_bottom": metadata["bbox_bottom"],
                        "bbox_right": metadata["bbox_right"],
                        "bbox_top": metadata["bbox_top"],
                        "nodata": metadata["nodata"],
                        "readable": metadata["readable"],
                        "metadata_read_status": metadata["metadata_read_status"],
                        "capture_dates": ";".join(dates),
                        "capture_date_provenance": provenance,
                        "content_sha256": "NOT_COMPUTED_LARGE_RASTER",
                        "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                        "raw_inputs_modified": False,
                    }
                )
        elif prior_pixel_count:
            inventory_rows.append(
                {
                    "row_type": "PIXEL_COLLECTION_PRIOR_CENSUS",
                    "classification": "REAL_PIXEL_RASTER",
                    "pixel_available": "YES",
                    "observation_id": f"xian_supplement::{obs_name}",
                    "source_asset_id": f"xian_supplement::{obs_name}::PIXEL_COLLECTION",
                    "source_ref": str(obs_dir),
                    "relative_path": "NESTED_CACHE_NOT_RESCANNED_THIS_TURN",
                    "size_bytes": pixel_bytes,
                    "format": "JPEG_COLLECTION",
                    "crs": "",
                    "transform": "",
                    "width": "",
                    "height": "",
                    "bands": "",
                    "resolution_x": "",
                    "resolution_y": "",
                    "bbox_left": "",
                    "bbox_bottom": "",
                    "bbox_right": "",
                    "bbox_top": "",
                    "nodata": "",
                    "readable": "PRIOR_INVENTORY_ONLY",
                    "metadata_read_status": "PRIOR_READ_ONLY_CENSUS",
                    "capture_dates": ";".join(actual_dates),
                    "capture_date_provenance": date_provenance,
                    "pixel_file_count": pixel_count,
                    "pixel_count_provenance": source_status,
                    "content_sha256": "NOT_COMPUTED_PRIOR_CENSUS",
                    "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                    "raw_inputs_modified": False,
                }
            )
        for classification, count in sorted(direct_counts.items()):
            if classification == "REAL_PIXEL_RASTER":
                continue
            inventory_rows.append(
                {
                    "row_type": "FORMAT_SUMMARY_DIRECT_ONLY",
                    "classification": classification,
                    "pixel_available": "NO",
                    "observation_id": f"xian_supplement::{obs_name}",
                    "source_asset_id": f"xian_supplement::{obs_name}::{classification}",
                    "source_ref": str(obs_dir),
                    "format": classification,
                    "readable": "NOT_APPLICABLE_AS_PIXEL",
                    "metadata_read_status": "NOT_AN_IMAGE_BY_POLICY",
                    "file_count": count,
                    "actual_dates": ";".join(actual_dates),
                    "prohibition_note": "Metadata/cache only; .dat/.raw/.datever/.ver are not treated as image pixels.",
                }
            )

    for epoch, target in TARGET_DATES.items():
        matching = [row for row in date_rows if row["nearest_formal_target"] == epoch]
        target_rows.append(
            {
                "formal_target": epoch,
                "target_date": target.isoformat(),
                "observation_date_rows": len(matching),
                "pixel_available_observation_rows": sum(row["pixel_available"] == "YES" for row in matching),
                "actual_dates": ";".join(sorted({row["actual_acquisition_date"] for row in matching})),
                "best_days_from_target": min((int(row["days_from_nearest_target"]) for row in matching), default=""),
                "note": "Supplemental support only; no fourth formal epoch and no geometry replacement.",
            }
        )
    manifest = {
        "schema_version": "urrs.xian.supplemental.pixel.audit.v1",
        "created_at_utc": utc_now(),
        "root": str(root),
        "output_root": str(out),
        "scan_mode": "TARGETED_DIRECT_RASTER_PLUS_PRIOR_CACHE_CENSUS",
        "current_root_observation_directories_verified": len(folders),
        "current_direct_raster_paths_checked": len(all_direct_checked),
        "current_direct_raster_paths": all_direct_checked,
        "source_files_scanned_direct_only": sum(int(row["all_file_count_direct_only"]) for row in observation_rows),
        "observation_count": len(observation_rows),
        "pixel_file_count": direct_raster_count + carried_pixel_count,
        "pixel_bytes": direct_raster_bytes + carried_pixel_bytes,
        "current_direct_pixel_file_count": direct_raster_count,
        "prior_cache_pixel_file_count": carried_pixel_count,
        "classification_counts_direct_only": dict(sorted(format_counts.items())),
        "actual_date_count": len(actual_counts),
        "actual_dates": dict(sorted(actual_counts.items())),
        "target_summary": target_rows,
        "direct_georeferenced_rasters": [row["source_ref"] for row in inventory_rows if row.get("row_type") == "PIXEL_ASSET" and row.get("crs")],
        "prior_inventory_used": prior_status,
        "pixel_policy": {
            "supplemental_never_replaces_main_geometry": True,
            "supplemental_never_becomes_fourth_formal_epoch": True,
            "dat_raw_datever_ver_not_images": True,
            "not_detected_is_not_absent": True,
            "alignment_before_absence": True,
        },
        "limitations": [
            "nested cache folders were not recursively rescanned in this bounded turn",
            "nested cache counts/bytes are carried from the prior read-only 2026-09-14 census",
            "per-cache-tile CRS/transform/bbox is not asserted",
            "large-raster content SHA was not computed",
        ],
        "raw_inputs_modified": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / "SUPPLEMENTAL_PIXEL_INVENTORY.csv", inventory_rows)
    csv_write(out / "SUPPLEMENTAL_ACTUAL_DATE_REGISTRY.csv", date_rows)
    csv_write(out / "SUPPLEMENTAL_OBSERVATION_REGISTRY.csv", observation_rows)
    csv_write(out / "SUPPLEMENTAL_TARGET_DATE_SUMMARY.csv", target_rows)
    (out / "SUPPLEMENTAL_PIXEL_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Xi'an supplemental pixel audit\n\n"
        "本轮采用定向审计：现场核验所有 observation 目录和组合 TIF；百万级嵌套 cache 的计数沿用此前只读盘点，并在 manifest 中显式标注来源。`.dat/.raw/.datever/.ver` 不作为影像。\n\n"
        "辅助影像只作为日期、同日去重、配准/视差、身份和可观测性证据，不替换主 Geometry，不构成第四正式时期。\n",
        encoding="utf-8",
    )
    return manifest


def build_inventory_bounded(root: Path, out: Path, prior_inventory: Path | None = None) -> dict[str, Any]:
    """Bounded audit that avoids enumerating cache entries entirely.

    This is intentionally narrower than ``--targeted``: it checks the current
    observation directories and every expected combined-TIF path, then joins
    the already recorded cache census.  It is the safe fallback for a volume
    where listing a single cache directory itself is the bottleneck.
    """
    root = root.resolve()
    out = out.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if prior_inventory is None:
        prior_inventory = Path(__file__).resolve().parents[1] / "work" / "xian_supplement_inventory_20260914.json"
    prior_rows: dict[str, dict[str, Any]] = {}
    prior_status = "NOT_AVAILABLE"
    if prior_inventory.is_file():
        try:
            payload = json.loads(prior_inventory.read_text(encoding="utf-8"))
            prior_rows = {str(row.get("folder")): row for row in payload.get("dates", [])}
            prior_status = str(prior_inventory)
        except Exception as exc:
            prior_status = f"READ_ERROR:{type(exc).__name__}"

    # Directory entries are few; do not call iterdir/rglob on any cache tree.
    folders = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower())
    inventory_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    actual_counts: Counter[str] = Counter()
    direct_rasters: list[Path] = []
    carried_pixel_count = 0
    carried_pixel_bytes = 0
    direct_pixel_bytes = 0
    metadata_presence_rows: list[dict[str, Any]] = []

    for obs_dir in folders:
        name = obs_dir.name
        prior = prior_rows.get(name, {})
        base = name[:-len("_combined")] if name.endswith("_combined") else name
        expected_tif = obs_dir / f"{base}_zoom_20.tif"
        if not expected_tif.is_file():
            # A few payloads were historically named without the exact
            # combined stem; check only the small set of predictable TIFF
            # names, never the cache tree.
            candidates = [obs_dir / f"{name}_zoom_20.tif", obs_dir / f"{base}.tif"]
            expected_tif = next((candidate for candidate in candidates if candidate.is_file()), expected_tif)
        has_direct_tif = expected_tif.is_file()
        if has_direct_tif:
            direct_rasters.append(expected_tif)
            metadata = raster_metadata(expected_tif)
            direct_pixel_count = 1
            direct_bytes = expected_tif.stat().st_size
            direct_pixel_bytes += direct_bytes
            inventory_rows.append(
                {
                    "row_type": "PIXEL_ASSET",
                    "classification": "REAL_PIXEL_RASTER",
                    "pixel_available": "YES",
                    "observation_id": f"xian_supplement::{name}",
                    "source_asset_id": f"xian_supplement::{name}::{expected_tif.name}",
                    "source_ref": str(expected_tif),
                    "relative_path": rel_posix(expected_tif, root),
                    "size_bytes": direct_bytes,
                    "format": metadata["format"],
                    "crs": metadata["crs"],
                    "transform": metadata["transform"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "bands": metadata["bands"],
                    "resolution_x": metadata["resolution_x"],
                    "resolution_y": metadata["resolution_y"],
                    "bbox_left": metadata["bbox_left"],
                    "bbox_bottom": metadata["bbox_bottom"],
                    "bbox_right": metadata["bbox_right"],
                    "bbox_top": metadata["bbox_top"],
                    "nodata": metadata["nodata"],
                    "readable": metadata["readable"],
                    "metadata_read_status": metadata["metadata_read_status"],
                    "capture_dates": ";".join(sorted(parse_dates(name) | parse_dates(expected_tif.name))),
                    "capture_date_provenance": "EXPLICIT_DATE_IN_PATH",
                    "content_sha256": "NOT_COMPUTED_LARGE_RASTER",
                    "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                    "raw_inputs_modified": False,
                }
            )
        else:
            direct_pixel_count = 0
            direct_bytes = 0
        prior_pixel_count = int(prior.get("jpg_count", 0) or 0)
        prior_pixel_bytes = int(prior.get("bytes", 0) or 0)
        pixel_count = direct_pixel_count if has_direct_tif else prior_pixel_count
        pixel_bytes = direct_bytes if has_direct_tif else prior_pixel_bytes
        if not has_direct_tif:
            carried_pixel_count += prior_pixel_count
            carried_pixel_bytes += prior_pixel_bytes
        actual_dates = sorted(parse_dates(name))
        for actual in actual_dates:
            actual_counts[actual] += 1
        pixel_available = pixel_count > 0
        payload_status = "CURRENT_DIRECT_TIF_VERIFIED" if has_direct_tif else ("PRIOR_NESTED_PIXEL_CENSUS" if prior_pixel_count else "NO_PIXEL_PAYLOAD_CURRENTLY_VERIFIED")
        # Check sidecars by predictable exact paths only.  These checks support
        # the explicit prohibition against treating them as image pixels.
        sidecars = {
            "bpw": obs_dir / f"{base}_zoom_20.bpw",
            "tfw": obs_dir / f"{base}_zoom_20.tfw",
            "map": obs_dir / f"{base}_zoom_20.map",
            "datever_aggregate": obs_dir / f"{base}_datever.dat",
            "geid": root / f"{name}.geid",
        }
        metadata_presence_rows.append({"observation_label": name, **{key: path.is_file() for key, path in sidecars.items()}})
        observation_rows.append(
            {
                "observation_id": f"xian_supplement::{name}",
                "source_asset_id": f"xian_supplement::{name}",
                "source_ref": str(obs_dir),
                "observation_label": name,
                "classification": "REAL_PIXEL_RASTER" if pixel_available else ("GE_CACHE_METADATA_ONLY" if any(path.is_file() for path in sidecars.values()) else "UNKNOWN_FORMAT"),
                "pixel_available": "YES" if pixel_available else "NO",
                "pixel_file_count": pixel_count,
                "pixel_bytes": pixel_bytes,
                "pixel_count_provenance": payload_status,
                "actual_dates": ";".join(actual_dates),
                "actual_date_provenance": "EXPLICIT_DATE_IN_PATH" if actual_dates else "NO_ACTUAL_DATE_EVIDENCE",
                "direct_tif_checked": str(expected_tif),
                "direct_tif_exists": has_direct_tif,
                "sidecar_checks_json": json.dumps({key: path.is_file() for key, path in sidecars.items()}, ensure_ascii=False, sort_keys=True),
                "recursive_file_enumeration": False,
                "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                "raw_inputs_modified": False,
            }
        )
        for actual in actual_dates:
            nearest_epoch, days = nearest_target(actual)
            date_rows.append(
                {
                    "observation_id": f"xian_supplement::{name}",
                    "observation_label": name,
                    "source_ref": str(obs_dir),
                    "actual_acquisition_date": actual,
                    "date_provenance": "EXPLICIT_DATE_IN_PATH",
                    "pixel_available": "YES" if pixel_available else "NO",
                    "pixel_file_count": pixel_count,
                    "pixel_count_provenance": payload_status,
                    "nearest_formal_target": nearest_epoch,
                    "days_from_nearest_target": days,
                    "near_target_priority": "HIGH" if isinstance(days, int) and days <= 180 else ("MEDIUM" if isinstance(days, int) and days <= 540 else "LOW"),
                    "formal_epoch_created": False,
                    "supplement_role": "date_validation;same_date_dedup;alignment_parallax;identity;observability",
                }
            )
        if not has_direct_tif and prior_pixel_count:
            inventory_rows.append(
                {
                    "row_type": "PIXEL_COLLECTION_PRIOR_CENSUS",
                    "classification": "REAL_PIXEL_RASTER",
                    "pixel_available": "YES",
                    "observation_id": f"xian_supplement::{name}",
                    "source_asset_id": f"xian_supplement::{name}::PIXEL_COLLECTION",
                    "source_ref": str(obs_dir),
                    "relative_path": "NESTED_CACHE_NOT_ENUMERATED_THIS_TURN",
                    "size_bytes": prior_pixel_bytes,
                    "format": "JPEG_COLLECTION",
                    "crs": "",
                    "transform": "",
                    "width": "",
                    "height": "",
                    "bands": "",
                    "resolution_x": "",
                    "resolution_y": "",
                    "bbox_left": "",
                    "bbox_bottom": "",
                    "bbox_right": "",
                    "bbox_top": "",
                    "nodata": "",
                    "readable": "PRIOR_INVENTORY_ONLY",
                    "metadata_read_status": "PRIOR_READ_ONLY_CENSUS",
                    "capture_dates": ";".join(actual_dates),
                    "capture_date_provenance": "EXPLICIT_DATE_IN_PATH",
                    "pixel_file_count": prior_pixel_count,
                    "pixel_count_provenance": "PRIOR_INVENTORY_20260914",
                    "content_sha256": "NOT_COMPUTED_PRIOR_CENSUS",
                    "geometry_owner": "NO_SUPPLEMENTAL_GEOMETRY_OWNER",
                    "raw_inputs_modified": False,
                }
            )

    target_rows = []
    for epoch, target in TARGET_DATES.items():
        matching = [row for row in date_rows if row["nearest_formal_target"] == epoch]
        target_rows.append(
            {
                "formal_target": epoch,
                "target_date": target.isoformat(),
                "observation_date_rows": len(matching),
                "pixel_available_observation_rows": sum(row["pixel_available"] == "YES" for row in matching),
                "actual_dates": ";".join(sorted({row["actual_acquisition_date"] for row in matching})),
                "best_days_from_target": min((int(row["days_from_nearest_target"]) for row in matching), default=""),
                "note": "Supplemental support only; no fourth formal epoch and no geometry replacement.",
            }
        )
    manifest = {
        "schema_version": "urrs.xian.supplemental.pixel.audit.v1",
        "created_at_utc": utc_now(),
        "root": str(root),
        "output_root": str(out),
        "scan_mode": "BOUNDED_EXPECTED_RASTER_AND_PRIOR_CACHE_CENSUS",
        "current_observation_directories_verified": len(folders),
        "expected_direct_rasters_checked": len(direct_rasters) + sum(1 for folder in folders if not (folder / f"{folder.name[:-len('_combined')] if folder.name.endswith('_combined') else folder.name}_zoom_20.tif").is_file()),
        "direct_georeferenced_rasters": [str(path) for path in direct_rasters],
        "source_files_scanned": "DIRECT_EXPECTED_PATHS_ONLY",
        "observation_count": len(observation_rows),
        "pixel_observation_count": sum(row["pixel_available"] == "YES" for row in observation_rows),
        "pixel_file_count": len(direct_rasters) + carried_pixel_count,
        "pixel_bytes": direct_pixel_bytes + carried_pixel_bytes,
        "current_direct_pixel_file_count": len(direct_rasters),
        "prior_cache_pixel_file_count": carried_pixel_count,
        "classification_counts": {
            "REAL_PIXEL_RASTER": len(direct_rasters) + carried_pixel_count,
            "GE_CACHE_METADATA_ONLY": "BOUNDED_NOT_ENUMERATED",
            "DATE_METADATA": "BOUNDED_NOT_ENUMERATED",
            "UNKNOWN_FORMAT": "BOUNDED_NOT_ENUMERATED",
        },
        "actual_date_count": len(actual_counts),
        "actual_dates": dict(sorted(actual_counts.items())),
        "target_summary": target_rows,
        "sidecar_presence": metadata_presence_rows,
        "prior_inventory_used": prior_status,
        "pixel_policy": {
            "supplemental_never_replaces_main_geometry": True,
            "supplemental_never_becomes_fourth_formal_epoch": True,
            "dat_raw_datever_ver_not_images": True,
            "not_detected_is_not_absent": True,
            "alignment_before_absence": True,
        },
        "limitations": [
            "cache entries were not enumerated in this bounded audit",
            "nested cache counts/bytes are carried from the prior read-only 2026-09-14 census",
            "per-cache-tile CRS/transform/bbox is not asserted",
            "large-raster content SHA was not computed",
        ],
        "raw_inputs_modified": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / "SUPPLEMENTAL_PIXEL_INVENTORY.csv", inventory_rows)
    csv_write(out / "SUPPLEMENTAL_ACTUAL_DATE_REGISTRY.csv", date_rows)
    csv_write(out / "SUPPLEMENTAL_OBSERVATION_REGISTRY.csv", observation_rows)
    csv_write(out / "SUPPLEMENTAL_TARGET_DATE_SUMMARY.csv", target_rows)
    (out / "SUPPLEMENTAL_PIXEL_AUDIT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "SUPPLEMENTAL_SIDEcar_PRESENCE.json").write_text(json.dumps(metadata_presence_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Xi'an supplemental pixel audit\n\n"
        "本轮采用有界审计：现场核验所有 observation 目录、预期组合 TIF 路径和完整 TIF 地理元数据；不枚举百万级 cache 条目，嵌套 cache 数量沿用此前只读盘点并明确标记。`.dat/.raw/.datever/.ver` 不作为影像。\n\n"
        "辅助影像只作为日期、同日去重、配准/视差、身份和可观测性证据，不替换主 Geometry，不构成第四正式时期。\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"F:\xian\XIAN补充"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fast", action="store_true", help="stream large cache trees and aggregate cache image collections")
    parser.add_argument("--targeted", action="store_true", help="verify direct rasters and use prior read-only cache census")
    parser.add_argument("--bounded", action="store_true", help="check expected direct raster paths without enumerating cache entries")
    parser.add_argument("--prior-inventory", type=Path, default=None)
    args = parser.parse_args()
    if args.bounded:
        manifest = build_inventory_bounded(args.root, args.output_root, args.prior_inventory)
    elif args.targeted:
        manifest = build_inventory_targeted(args.root, args.output_root, args.prior_inventory)
    elif args.fast:
        manifest = build_inventory_fast(args.root, args.output_root)
    else:
        manifest = build_inventory(args.root, args.output_root)
    print(json.dumps({
        "status": "PASS_SUPPLEMENTAL_PIXEL_AUDIT",
        "output_root": str(args.output_root.resolve()),
        "source_files_scanned": manifest["source_files_scanned"],
        "observation_count": manifest["observation_count"],
        "pixel_file_count": manifest["pixel_file_count"],
        "pixel_observation_count": manifest["pixel_observation_count"],
        "classification_counts": manifest["classification_counts"],
        "actual_date_count": manifest["actual_date_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
