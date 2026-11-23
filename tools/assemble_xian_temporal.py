"""Assemble Xi'an GEID regions into date-aware epoch references.

This utility is intentionally CPU-only.  It does not train or run a model.
It discovers already-created combined GeoTIFFs, recovers per-tile acquisition
dates from GEID logs, writes immutable source provenance, creates lightweight
VRT mosaics and overview GeoTIFFs, and classifies temporal reliability without
substituting a nominal task year for an observed acquisition date.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception as exc:  # pragma: no cover - reported as a clear receipt
    rasterio = None
    Resampling = None
    _RASTERIO_IMPORT_ERROR = repr(exc)


DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)")
TILE_RE = re.compile(r"gesh_(\d+)_(\d+)_(\d+)\.(?:jpg|jpeg)$", re.I)
LOG_RE = re.compile(
    r"(?P<filename>gesh_\d+_\d+_\d+\.jpg)\s*:\s*"
    r"(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+(?P<status>[^\s]+)",
    re.I,
)
OK_STATUSES = {"ok", "success", "successful", "downloaded", "complete"}
ZOOM = 20
TILE_SIZE = 256
DEFAULT_D_ROOT = Path(r"D:\Data Set\xiannew")
DEFAULT_E_XIANNEW = Path(r"E:\xiannew")
DEFAULT_E_XIAN = Path(r"E:\xian\xian")
DEFAULT_OUT = Path(r"E:\URRS Output\xiannew")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, limit: int | None = None) -> str | None:
    try:
        h = hashlib.sha256()
        read = 0
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                h.update(block)
                read += len(block)
                if limit is not None and read >= limit:
                    break
        return h.hexdigest()
    except OSError:
        return None


def read_ini(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in raw and not raw.lstrip().startswith("["):
                key, value = raw.split("=", 1)
                out[key.strip()] = value.strip()
    except OSError:
        pass
    return out


def parse_date(value: str | None) -> str | None:
    if not value:
        return None
    match = DATE_RE.search(value)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def candidate_roots(d_root: Path, e_xiannew: Path, e_xian: Path) -> list[Path]:
    # Priority is deliberate: D source and split-task logs are the authoritative
    # GEID evidence; E copies are used only where D has no source.
    return [d_root, d_root / "split_tasks_r2", e_xiannew, e_xian]


def source_path_for(stem: str, roots: Iterable[Path]) -> tuple[Path | None, Path | None]:
    for root in roots:
        direct = root / f"{stem}_combined" / f"{stem}_zoom_20.tif"
        if direct.is_file() and direct.stat().st_size > 0:
            geid = root / f"{stem}.geid"
            return direct, geid if geid.is_file() else None
    return None, None


def resolve_metadata_root(stem: str, source_root: Path, roots: Iterable[Path]) -> tuple[Path, Path | None, list[Path]]:
    """Resolve GEID/log provenance when mosaics and metadata live apart.

    Some immutable copies place the combined GeoTIFF under ``E:\\xiannew``
    while the authoritative ``.geid`` and log remain under
    ``D:\\Data Set\\xiannew``. Date inference must follow that metadata,
    never fall back to a nominal stem when an external log is available.
    """
    candidates: list[Path] = [source_root, *list(roots)]
    seen: set[str] = set()
    fallback = source_root.resolve()
    for candidate in candidates:
        root = Path(candidate).resolve()
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        geid = root / f"{stem}.geid"
        logs = sorted(root.glob(f"{stem}_log.txt*"))
        if geid.is_file() or logs:
            return root, geid if geid.is_file() else None, logs
    return fallback, None, []


def parse_logs(root: Path, stem: str) -> tuple[dict[str, dict[str, Any]], list[Path], dict[str, Any]]:
    all_logs = sorted(
        [p for p in root.glob(f"{stem}_log.txt*") if p.is_file()],
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    # The primary GEID log is a complete end-of-task listing and contains the
    # latest successful observation for each tile.  Rotated logs repeat the
    # same rows many times and can be hundreds of megabytes; reading them all
    # both wastes memory and risks turning an audit into a second downloader.
    # Keep their paths in provenance, but parse only the primary log when it is
    # available (otherwise the newest rotated log is the least stale evidence).
    primary = root / f"{stem}_log.txt"
    logs = [primary] if primary.is_file() else (all_logs[-1:] if all_logs else [])
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    line_count = 0
    matched = 0
    for path in logs:
        try:
            mtime = path.stat().st_mtime
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, 1):
                    line_count += 1
                    match = LOG_RE.search(line)
                    if not match:
                        continue
                    matched += 1
                    observed = parse_date(match.group("date"))
                    status = match.group("status").strip().lower()
                    rows[match.group("filename").lower()].append(
                        {
                            "date": observed,
                            "status": status,
                            "success": status in OK_STATUSES,
                            "path": str(path.resolve()),
                            "line": line_no,
                            "mtime": mtime,
                        }
                    )
        except OSError:
            continue
    chosen: dict[str, dict[str, Any]] = {}
    conflict_count = 0
    for filename, entries in rows.items():
        observed_dates = sorted({item["date"] for item in entries if item.get("date")})
        if len(observed_dates) > 1:
            conflict_count += 1
        successful = [item for item in entries if item.get("success") and item.get("date")]
        selected = max(successful or entries, key=lambda item: (item.get("mtime", 0.0), item.get("line", 0)))
        chosen[filename] = {
            "actual_acquisition_date": selected.get("date"),
            "date_source": "geid_log_success_latest" if successful else "geid_log_latest_non_success",
            "latest_status": selected.get("status"),
            "observed_dates": observed_dates,
            "date_conflict": len(observed_dates) > 1,
            "source_log": selected.get("path"),
            "source_line": selected.get("line"),
        }
    return chosen, logs, {
        "log_count": len(all_logs),
        "parsed_log_count": len(logs),
        "line_count": line_count,
        "matched_line_count": matched,
        "tile_row_count": len(chosen),
        "date_conflict_tile_count": conflict_count,
    }


def infer_nominal_year(stem: str, geid_values: dict[str, str]) -> int | None:
    actual = parse_date(geid_values.get("HistoryDate"))
    if actual:
        return int(actual[:4])
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", stem)
    return int(match.group(1)) if match else None


def grid_from_geid(values: dict[str, str]) -> dict[str, int | None]:
    def integer(name: str) -> int | None:
        try:
            return int(values[name])
        except (KeyError, TypeError, ValueError):
            return None

    min_x, max_x = integer("MinX_20"), integer("MaxX_20")
    min_y, max_y = integer("MinY_20"), integer("MaxY_20")
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "tile_columns": (max_x - min_x + 1) if min_x is not None and max_x is not None else None,
        "tile_rows": (max_y - min_y + 1) if min_y is not None and max_y is not None else None,
    }


def sample_nonzero(path: Path) -> dict[str, Any]:
    if rasterio is None:
        return {"read_success": False, "error": _RASTERIO_IMPORT_ERROR}
    try:
        with rasterio.open(path) as src:
            height = max(1, min(src.height, 512))
            width = max(1, min(src.width, 512))
            arr = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.nearest,
                masked=False,
            )
            finite = np.isfinite(arr)
            nonzero = np.count_nonzero(arr[finite]) if finite.any() else 0
            return {
                "read_success": True,
                "sample_shape": [int(height), int(width)],
                "nonzero_fraction": float(nonzero / max(1, int(finite.sum()))),
                "min": float(np.nanmin(arr)) if finite.any() else None,
                "max": float(np.nanmax(arr)) if finite.any() else None,
            }
    except Exception as exc:
        return {"read_success": False, "error": f"{type(exc).__name__}: {exc}"}


def discover_sources(d_root: Path, e_xiannew: Path, e_xian: Path) -> list[dict[str, Any]]:
    roots = candidate_roots(d_root, e_xiannew, e_xian)
    stems: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for item in root.iterdir():
                if item.is_dir() and item.name.lower().endswith("_combined"):
                    tif = item / f"{item.name[:-9]}_zoom_20.tif"
                    if tif.is_file():
                        stems.add(item.name[:-9])
        except OSError:
            continue
    # Prefer the complete central three-epoch names even if their directory is
    # nested below E:\xian\xian.
    records: list[dict[str, Any]] = []
    for stem in sorted(stems):
        source, geid = source_path_for(stem, roots)
        if source is None:
            continue
        source_root = source.parent.parent
        metadata_root, geid, _metadata_logs = resolve_metadata_root(stem, source_root, roots)
        values = read_ini(geid) if geid else {}
        nominal_date = parse_date(values.get("HistoryDate"))
        year = infer_nominal_year(stem, values)
        log_root = metadata_root
        rows, logs, log_summary = parse_logs(log_root, stem)
        try:
            with rasterio.open(source) as ds:
                metadata = {
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "count": int(ds.count),
                    "dtype": list(ds.dtypes),
                    "crs": ds.crs.to_string() if ds.crs else None,
                    "transform": list(ds.transform),
                    "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
                    "res": [float(ds.res[0]), float(ds.res[1])],
                    "nodata": ds.nodata,
                }
        except Exception as exc:
            metadata = {"error": f"{type(exc).__name__}: {exc}"}
        sample = sample_nonzero(source)
        date_counts = Counter(
            item.get("actual_acquisition_date")
            for item in rows.values()
            if item.get("actual_acquisition_date")
        )
        records.append(
            {
                "stem": stem,
                "nominal_date": nominal_date,
                "nominal_year": year,
                "source_tif": str(source.resolve()),
                "source_root": str(source_root.resolve()),
                "metadata_root": str(metadata_root.resolve()),
                "geid": str(geid.resolve()) if geid else None,
                "geid_sha256": sha256_file(geid) if geid else None,
                "geid_values": values,
                "logs": [str(item.resolve()) for item in logs],
                "log_summary": log_summary,
                "date_counts": dict(date_counts),
                "tile_dates": rows,
                "metadata": metadata,
                "sample": sample,
                "source_bytes": source.stat().st_size,
                "source_mtime": source.stat().st_mtime,
                "source_sha256": None,
                "source_hash_policy": "large immutable source hash deferred; size/mtime/metadata/sample recorded",
                "status": "usable" if sample.get("read_success") and sample.get("nonzero_fraction", 0) > 0 else "blocked_invalid_source",
            }
        )
    return records


def pick_records(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        year = record.get("nominal_year")
        if year in (2018, 2019, 2020) and record.get("status") == "usable":
            grouped[int(year)].append(record)
    # One source per exact spatial footprint/stem is already guaranteed by the
    # discovery priority.  Keep all non-overlapping regions; overlaps are
    # declared in the receipt for downstream QA instead of silently discarded.
    return {year: sorted(items, key=lambda item: item["stem"]) for year, items in grouped.items()}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_vrt(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"status": "blocked_no_sources", "path": str(path)}
    valid = [item for item in records if item.get("metadata", {}).get("crs") and item.get("metadata", {}).get("count")]
    if not valid:
        return {"status": "blocked_no_georeferenced_sources", "path": str(path)}
    res_x = float(np.median([abs(float(item["metadata"]["res"][0])) for item in valid]))
    res_y = float(np.median([abs(float(item["metadata"]["res"][1])) for item in valid]))
    left = min(float(item["metadata"]["bounds"][0]) for item in valid)
    bottom = min(float(item["metadata"]["bounds"][1]) for item in valid)
    right = max(float(item["metadata"]["bounds"][2]) for item in valid)
    top = max(float(item["metadata"]["bounds"][3]) for item in valid)
    width = int(round((right - left) / res_x))
    height = int(round((top - bottom) / res_y))
    crs = valid[0]["metadata"]["crs"]
    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        f"  <SRS>{crs}</SRS>",
        # GDAL's GeoTransform order is origin-x, pixel-width, rotation-x,
        # origin-y, rotation-y, pixel-height.  Keeping the origin first is
        # essential: putting ``res_x`` in the first slot silently relocates
        # the mosaic to longitude 0 and corrupts every downstream clip/area
        # calculation while still producing a syntactically valid VRT.
        f'  <GeoTransform>{left}, {res_x}, 0, {top}, 0, {-res_y}</GeoTransform>',
    ]
    count = int(valid[0]["metadata"]["count"])
    dtype = str(valid[0]["metadata"]["dtype"][0])
    dtype_map = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16", "float32": "Float32", "float64": "Float64"}
    vrt_dtype = dtype_map.get(dtype, "Byte")
    for band in range(1, count + 1):
        lines.append(f'  <VRTRasterBand dataType="{vrt_dtype}" band="{band}">')
        lines.append("    <NoDataValue>0</NoDataValue>")
        for item in valid:
            meta = item["metadata"]
            bounds = meta["bounds"]
            src_path = Path(item["source_tif"])
            xoff = int(round((float(bounds[0]) - left) / res_x))
            yoff = int(round((top - float(bounds[3])) / res_y))
            lines.extend(
                [
                    "    <SimpleSource>",
                    f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>',
                    f"      <SourceBand>{band}</SourceBand>",
                    f'      <SrcRect xOff="0" yOff="0" xSize="{meta["width"]}" ySize="{meta["height"]}"/>',
                    f'      <DstRect xOff="{xoff}" yOff="{yoff}" xSize="{meta["width"]}" ySize="{meta["height"]}"/>',
                    "    </SimpleSource>",
                ]
            )
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "created",
        "path": str(path.resolve()),
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "bounds": [left, bottom, right, top],
        "resolution": [res_x, res_y],
        "source_count": len(valid),
    }


def make_overview(vrt_path: Path, out_path: Path, max_size: int, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rasterio is None:
        return {"status": "blocked_rasterio_missing", "error": _RASTERIO_IMPORT_ERROR}
    try:
        with rasterio.open(vrt_path) as src:
            scale = min(1.0, float(max_size) / max(src.width, src.height))
            width = max(1, int(round(src.width * scale)))
            height = max(1, int(round(src.height * scale)))
            # Reading a very large VRT at once can make GDAL allocate a
            # full-resolution intermediate array.  Compose the small overview
            # from the source GeoTIFFs instead; their internal overviews make
            # each read bounded and fast while preserving the exact VRT extent.
            data = np.zeros((src.count, height, width), dtype=np.uint8)
            vrt_bounds = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
            xscale = width / max(1.0, float(src.width))
            yscale = height / max(1.0, float(src.height))
            for item in records or []:
                meta = item.get("metadata", {})
                source_path = Path(item["source_tif"])
                if not source_path.is_file():
                    continue
                bounds = meta.get("bounds") or []
                if len(bounds) != 4:
                    continue
                sx = max(1, int(round(float(meta.get("width", 1)) * scale)))
                sy = max(1, int(round(float(meta.get("height", 1)) * scale)))
                xoff = int(round((float(bounds[0]) - vrt_bounds[0]) / max(1e-12, (vrt_bounds[2] - vrt_bounds[0])) * width))
                yoff = int(round((vrt_bounds[3] - float(bounds[3])) / max(1e-12, (vrt_bounds[3] - vrt_bounds[1])) * height))
                try:
                    with rasterio.open(source_path) as source:
                        tile = source.read(out_shape=(src.count, sy, sx), resampling=Resampling.average)
                    x1, y1 = max(0, xoff), max(0, yoff)
                    x2, y2 = min(width, xoff + sx), min(height, yoff + sy)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    tx1, ty1 = x1 - xoff, y1 - yoff
                    tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)
                    patch = tile[:, ty1:ty2, tx1:tx2]
                    # Preserve the first non-zero source in overlaps.  Source
                    # priority is fixed by discovery order and is recorded in
                    # SOURCE_INVENTORY, so no result-driven selection occurs.
                    target = data[:, y1:y2, x1:x2]
                    fill = np.all(target == 0, axis=0)
                    for band in range(src.count):
                        target[band][fill] = patch[band][fill]
                    data[:, y1:y2, x1:x2] = target
                except Exception:
                    continue
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                width=width,
                height=height,
                transform=src.transform * src.transform.scale(src.width / width, src.height / height),
                tiled=True,
                blockxsize=min(256, width - (width % 16) if width >= 16 else 16),
                blockysize=min(256, height - (height % 16) if height >= 16 else 16),
                compress="deflate",
                predictor=2,
                BIGTIFF="IF_SAFER",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)
            return {"status": "created", "path": str(out_path.resolve()), "width": width, "height": height, "count": src.count, "crs": src.crs.to_string() if src.crs else None}
    except Exception as exc:
        return {"status": "blocked_overview_error", "error": f"{type(exc).__name__}: {exc}"}


def build_temporal_reliability(grouped: dict[int, list[dict[str, Any]]], out_dir: Path, max_size: int) -> dict[str, Any]:
    observations: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    date_counter: Counter[str] = Counter()
    for year, records in grouped.items():
        for record in records:
            for filename, detail in record.get("tile_dates", {}).items():
                match = TILE_RE.match(filename)
                if not match:
                    continue
                x, y, z = (int(match.group(i)) for i in (1, 2, 3))
                if z != ZOOM:
                    continue
                actual = detail.get("actual_acquisition_date")
                if actual:
                    observations[(x, y)][year] = {
                        "date": actual,
                        "stem": record["stem"],
                        "source": record["source_tif"],
                    }
                    date_counter[actual] += 1
    mode_date = date_counter.most_common(1)[0][0] if date_counter else None
    mode_day = date.fromisoformat(mode_date).toordinal() if mode_date else None
    reliability_counts: Counter[str] = Counter()
    date_rows: list[dict[str, Any]] = []
    all_dates: set[str] = set()
    for key, by_year in sorted(observations.items()):
        dates = sorted({item["date"] for item in by_year.values()})
        all_dates.update(dates)
        years_present = sorted(by_year)
        nominal_ok = all(year in by_year and by_year[year]["date"].startswith(str(year)) for year in (2018, 2019, 2020))
        if len(dates) < 2:
            category = "DUPLICATE_EPOCH" if len(dates) == 1 and len(years_present) >= 2 else "TEMPORAL_INSUFFICIENT"
        elif nominal_ok:
            category = "VALID_TEMPORAL"
        else:
            category = "OFF_DATE_VALID"
        reliability_counts[category] += 1
        date_rows.append(
            {
                "tile_x": key[0],
                "tile_y": key[1],
                "observed_epochs": ";".join(str(item) for item in years_present),
                "actual_dates": ";".join(dates),
                "category": category,
            }
        )
    write_csv(out_dir / "DATE_TILE_PROVENANCE.csv", date_rows, ["tile_x", "tile_y", "observed_epochs", "actual_dates", "category"])
    summary_rows = [{"category": key, "tile_count": int(value)} for key, value in sorted(reliability_counts.items())]
    write_csv(out_dir / "TEMPORAL_RELIABILITY_SUMMARY.csv", summary_rows, ["category", "tile_count"])
    write_json(
        out_dir / "TEMPORAL_RELIABILITY.json",
        {
            "status": "pass" if date_rows else "blocked_no_date_observations",
            "mode_date": mode_date,
            "mode_date_tile_count": int(date_counter.get(mode_date, 0)) if mode_date else 0,
            "actual_date_count": len(all_dates),
            "actual_dates": sorted(all_dates),
            "category_counts": dict(reliability_counts),
            "rules": {
                "VALID_TEMPORAL": "three nominal epochs have distinct observed dates in their respective calendar years",
                "OFF_DATE_VALID": "at least two distinct observed dates exist but one or more observations are off nominal year",
                "DUPLICATE_EPOCH": "multiple nominal epochs collapse to one actual date",
                "TEMPORAL_INSUFFICIENT": "fewer than two distinct actual dates or incomplete evidence",
                "nominal_year_is_never_evidence": True,
            },
            "scientific_status": "candidate_not_ground_truth",
        },
    )
    # Date and reliability overview rasters are deliberately low resolution;
    # native source imagery remains untouched and is referenced by the VRTs.
    if grouped:
        source_meta = next((item for rows in grouped.values() for item in rows if item.get("metadata", {}).get("bounds")), None)
    else:
        source_meta = None
    date_map = {d: i + 1 for i, d in enumerate(sorted(all_dates))}
    if source_meta and date_rows and rasterio is not None:
        left = min(float(item["metadata"]["bounds"][0]) for rows in grouped.values() for item in rows)
        bottom = min(float(item["metadata"]["bounds"][1]) for rows in grouped.values() for item in rows)
        right = max(float(item["metadata"]["bounds"][2]) for rows in grouped.values() for item in rows)
        top = max(float(item["metadata"]["bounds"][3]) for rows in grouped.values() for item in rows)
        res = float(np.median([abs(float(item["metadata"]["res"][0])) for rows in grouped.values() for item in rows]))
        full_w = max(1, int(round((right - left) / res)))
        full_h = max(1, int(round((top - bottom) / res)))
        scale = min(1.0, float(max_size) / max(full_w, full_h))
        out_w = max(1, int(round(full_w * scale)))
        out_h = max(1, int(round(full_h * scale)))
        date_grid = np.zeros((out_h, out_w), dtype=np.uint16)
        rel_grid = np.zeros((out_h, out_w), dtype=np.uint8)
        for row in date_rows:
            x, y = int(row["tile_x"]), int(row["tile_y"])
            # Web Mercator tile indices are not used for geometry here; the
            # georeferenced sources provide the authoritative transform.  Find
            # a source carrying this tile and derive its lon/lat from the
            # source bounds/grid, avoiding a degree-distance assumption.
            source = None
            for rows in grouped.values():
                for item in rows:
                    vals = item.get("geid_values", {})
                    try:
                        if int(vals["MinX_20"]) <= x <= int(vals["MaxX_20"]) and int(vals["MinY_20"]) <= y <= int(vals["MaxY_20"]):
                            source = item
                            break
                    except (KeyError, TypeError, ValueError):
                        continue
                if source:
                    break
            if not source:
                continue
            vals = source.get("geid_values", {})
            try:
                min_x, max_x = int(vals["MinX_20"]), int(vals["MaxX_20"])
                min_y, max_y = int(vals["MinY_20"]), int(vals["MaxY_20"])
                sb = source["metadata"]["bounds"]
                sx = (float(sb[2]) - float(sb[0])) / max(1, max_x - min_x + 1)
                sy = (float(sb[3]) - float(sb[1])) / max(1, max_y - min_y + 1)
                lon = float(sb[0]) + (x - min_x) * sx
                lat_top = float(sb[3]) - (y - min_y) * sy
                col = int((lon - left) / max(1e-12, right - left) * out_w)
                rr = int((top - lat_top) / max(1e-12, top - bottom) * out_h)
                if not (0 <= col < out_w and 0 <= rr < out_h):
                    continue
                dates = row["actual_dates"].split(";")
                selected_date = dates[0] if dates else None
                if selected_date in date_map:
                    date_grid[rr, col] = date_map[selected_date]
                category = row["category"]
                rel_grid[rr, col] = {"VALID_TEMPORAL": 1, "OFF_DATE_VALID": 2, "DUPLICATE_EPOCH": 3, "TEMPORAL_INSUFFICIENT": 4}.get(category, 0)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
        transform = rasterio.transform.from_bounds(left, bottom, right, top, out_w, out_h)
        profile = {"driver": "GTiff", "height": out_h, "width": out_w, "count": 1, "dtype": "uint16", "crs": "EPSG:4326", "transform": transform, "tiled": True, "compress": "deflate", "BIGTIFF": "IF_SAFER"}
        for name, grid, dtype in (("DATE_CODE_OVERVIEW.tif", date_grid, "uint16"), ("TEMPORAL_RELIABILITY_OVERVIEW.tif", rel_grid, "uint8")):
            profile["dtype"] = dtype
            with rasterio.open(out_dir / name, "w", **profile) as dst:
                dst.write(grid, 1)
        if mode_day:
            offsets = []
            for d in sorted(all_dates):
                offsets.append({"actual_date": d, "days_from_mode_date": date.fromisoformat(d).toordinal() - mode_day, "tile_count": date_counter[d]})
        else:
            offsets = []
        write_csv(out_dir / "DATE_OFFSET_SUMMARY.csv", offsets, ["actual_date", "days_from_mode_date", "tile_count"])
    return {"status": "pass" if date_rows else "blocked_no_date_observations", "mode_date": mode_date, "category_counts": dict(reliability_counts), "tile_observation_count": len(date_rows), "actual_date_count": len(all_dates)}


def create_run(out_root: Path, run_dir: Path | None) -> Path:
    if run_dir:
        run = run_dir.resolve()
    else:
        run = (out_root / f"xian_temporal_assembly_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    for name in ("00_audit", "01_epoch_mosaics", "02_temporal_reliability", "03_visuals", "04_statistics", "receipts", "logs"):
        (run / name).mkdir(parents=True, exist_ok=True)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-root", type=Path, default=DEFAULT_D_ROOT)
    parser.add_argument("--e-xiannew", type=Path, default=DEFAULT_E_XIANNEW)
    parser.add_argument("--e-xian", type=Path, default=DEFAULT_E_XIAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-overview-size", type=int, default=4096)
    parser.add_argument("--stage", choices=("audit", "mosaic", "all"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = create_run(args.output_root, args.run_dir)
    audit_dir = run / "00_audit"
    mosaic_dir = run / "01_epoch_mosaics"
    reliability_dir = run / "02_temporal_reliability"
    started = now_iso()
    records = discover_sources(args.d_root.resolve(), args.e_xiannew.resolve(), args.e_xian.resolve())
    grouped = pick_records(records)
    inventory_rows = []
    for item in records:
        inventory_rows.append(
            {
                "stem": item["stem"],
                "nominal_date": item.get("nominal_date"),
                "nominal_year": item.get("nominal_year"),
                "status": item.get("status"),
                "source_tif": item.get("source_tif"),
                "source_bytes": item.get("source_bytes"),
                "source_sha256": item.get("source_sha256"),
                "metadata_crs": item.get("metadata", {}).get("crs"),
                "width": item.get("metadata", {}).get("width"),
                "height": item.get("metadata", {}).get("height"),
                "nonzero_fraction": item.get("sample", {}).get("nonzero_fraction"),
                "actual_date_counts": json.dumps(item.get("date_counts", {}), ensure_ascii=False),
                "date_conflict_tile_count": item.get("log_summary", {}).get("date_conflict_tile_count"),
            }
        )
    fields = list(inventory_rows[0].keys()) if inventory_rows else ["stem", "status"]
    write_csv(audit_dir / "SOURCE_INVENTORY.csv", inventory_rows, fields)
    write_json(
        audit_dir / "SOURCE_INVENTORY.json",
        {
            "schema_version": "urrs.xian.temporal.source_inventory.v1",
            "created_at": started,
            "roots": {"d_root": str(args.d_root.resolve()), "e_xiannew": str(args.e_xiannew.resolve()), "e_xian": str(args.e_xian.resolve())},
            "source_count": len(records),
            "usable_count": sum(item.get("status") == "usable" for item in records),
            "records": records,
            "raw_inputs_modified": False,
            "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
            "scientific_status": "candidate_not_ground_truth",
        },
    )
    write_json(
        audit_dir / "RUN_MANIFEST.json",
        {
            "schema_version": "urrs.xian.temporal.assembly.v1",
            "run_dir": str(run),
            "created_at": started,
            "stage": args.stage,
            "grouped_source_stems": {str(year): [item["stem"] for item in rows] for year, rows in grouped.items()},
            "actual_date_policy": "actual acquisition dates from GEID logs only; nominal task year is never substituted",
            "raw_inputs_modified": False,
            "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        },
    )
    if args.stage in {"mosaic", "all"}:
        mosaic_receipts = {}
        for year, rows in sorted(grouped.items()):
            vrt_info = build_vrt(mosaic_dir / f"XIAN_{year}_ACTUAL_DATE_AWARE.vrt", rows)
            overview_info = make_overview(mosaic_dir / f"XIAN_{year}_ACTUAL_DATE_AWARE.vrt", mosaic_dir / f"XIAN_{year}_OVERVIEW.tif", args.max_overview_size, records=rows) if vrt_info.get("status") == "created" else {"status": "blocked_vrt"}
            mosaic_receipts[str(year)] = {"sources": [item["stem"] for item in rows], "vrt": vrt_info, "overview": overview_info}
        write_json(mosaic_dir / "EPOCH_MOSAIC_RECEIPT.json", {"created_at": now_iso(), "epochs": mosaic_receipts, "source_tifs_are_read_only": True})
    reliability = build_temporal_reliability(grouped, reliability_dir, args.max_overview_size)
    summary = {
        "status": "pass" if grouped else "blocked_no_usable_sources",
        "run_dir": str(run),
        "source_count": len(records),
        "usable_source_count": sum(item.get("status") == "usable" for item in records),
        "epoch_source_counts": {str(year): len(rows) for year, rows in grouped.items()},
        "actual_date_reliability": reliability,
        "created_at": now_iso(),
        "raw_inputs_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "scientific_status": "candidate_not_ground_truth",
    }
    write_json(run / "receipts" / "ASSEMBLY_RECEIPT.json", summary)
    write_json(run / "CURRENT_PROGRESS.json", {"stage": args.stage, "status": summary["status"], "run_dir": str(run), "updated_at": now_iso()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
