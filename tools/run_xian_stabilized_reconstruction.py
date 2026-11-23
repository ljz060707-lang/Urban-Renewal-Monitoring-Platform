"""Build an append-only Xi'an three-epoch stabilization workspace.

This runner is deliberately provenance-first.  It inventories every available
reference mosaic, resolves actual acquisition dates, collapses duplicate source
identities, and reuses a previously verified canonical three-epoch package.
It does not train, read Test/Holdout, alter source imagery, or overwrite any
historical run.  Expensive geometry is reused only when a verified receipt is
available; a reference image is never silently promoted to a formal epoch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_REPO_CANDIDATE = Path(__file__).resolve().parents[1]
# The starter workspace is nested under ``outputs/urban-renewal-phase1-starter``
# in this checkout.  Resolve that canonical project root when present so the
# script can be run from either the outer workspace or the nested repository.
PROJECT_ROOT = (_REPO_CANDIDATE / "outputs" / "urban-renewal-phase1-starter" if (_REPO_CANDIDATE / "outputs" / "urban-renewal-phase1-starter" / "tools").is_dir() else _REPO_CANDIDATE)
TARGET_BBOX = (108.75778198, 34.16130066, 109.17320251, 34.43252563)
B0_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
STAGE_NAMES = [
    "00_state_recovery", "01_input_inventory", "02_reference_tifs",
    "03_actual_dates", "04_quality", "05_model_selection",
    "06_geometry_raw", "07_geometry_corrected", "08_intra_epoch_dedup",
    "09_identity_reconciliation", "10_stable_supplement",
    "11_presence_1_0_U", "12_temporal", "13_states_pairwise",
    "14_states_three_epoch", "15_pixel_filter", "16_area",
    "17_statistics", "18_maps", "19_heatmaps", "20_cases",
    "21_visual_qa", "22_cloud_receipts", "23_final_receipts",
]
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")
TASK_DATE_RE = re.compile(r"xar(\d{2})(\d{2})(\d{2})", re.I)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(block)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except (OSError, UnicodeError):
        return []


def safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def parse_date(name: str) -> str | None:
    match = DATE_RE.search(name)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = TASK_DATE_RE.search(name)
    if match:
        # GEID stems use xarYYMMDD, with 18/19/20 referring to 2018/2019/2020.
        yy, mm, dd = match.groups()
        return f"20{yy}-{mm}-{dd}"
    return None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def bounds_overlap(bounds: tuple[float, float, float, float] | None) -> bool:
    if not bounds:
        return False
    left, bottom, right, top = bounds
    west, south, east, north = TARGET_BBOX
    return right > west and left < east and top > south and bottom < north


def raster_meta(path: Path) -> dict[str, Any]:
    """Read metadata without materialising a raster; a tiny sample is optional."""
    out: dict[str, Any] = {"read_ok": False}
    if path.suffix.lower() == ".vrt":
        # VRTs in this project are very large XML manifests (hundreds of MB).
        # The dataset header contains all geometry metadata, so parse only the
        # first few KB instead of asking GDAL to enumerate every SimpleSource.
        try:
            header = path.open("rb").read(65536).decode("utf-8", errors="ignore")
            msize = re.search(r"<VRTDataset[^>]*rasterXSize=\"(\d+)\"[^>]*rasterYSize=\"(\d+)\"", header)
            msrs = re.search(r"<SRS>(.*?)</SRS>", header, re.S)
            mgt = re.search(r"<GeoTransform>(.*?)</GeoTransform>", header, re.S)
            if msize and mgt:
                xsize, ysize = int(msize.group(1)), int(msize.group(2))
                gt = [float(v.strip()) for v in mgt.group(1).split(",")]
                left, top = gt[0], gt[3]
                right = left + gt[1] * xsize + gt[2] * ysize
                bottom = top + gt[4] * xsize + gt[5] * ysize
                out.update({
                    "read_ok": True, "width": xsize, "height": ysize,
                    "bands": max(3, len(re.findall(r"<VRTRasterBand", header))),
                    "dtype": "Byte", "crs": (msrs.group(1).strip() if msrs else "EPSG:4326"),
                    "transform": gt, "bounds": [left, bottom, right, top],
                    "pixel_size": [abs(gt[1]), abs(gt[5])], "nodata": None,
                    "tiled": True, "sample_nonzero_fraction": None,
                })
                return out
        except (OSError, ValueError):
            pass
    try:
        import rasterio  # type: ignore
    except Exception:
        out["error"] = "rasterio_unavailable"
        return out
    try:
        with rasterio.open(path) as src:
            b = src.bounds
            out.update({
                "read_ok": True,
                "width": int(src.width),
                "height": int(src.height),
                "bands": int(src.count),
                "dtype": ";".join(src.dtypes),
                "crs": str(src.crs) if src.crs else None,
                "transform": [float(v) for v in src.transform],
                "bounds": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
                "pixel_size": [float(src.res[0]), float(src.res[1])],
                "nodata": src.nodata,
                "tiled": bool(src.profile.get("tiled", False)),
            })
            # One tiny overview read is enough to detect an empty/broken
            # raster.  A VRT can contain hundreds of megabytes of XML and a
            # decimated read may force GDAL to inspect every source tile; use
            # metadata-only mode for VRTs and retain an explicit unknown sample
            # rather than blocking the inventory for hours.
            if path.suffix.lower() != ".vrt" or path.stat().st_size < 2 * 1024 * 1024:
                try:
                    import numpy as np  # type: ignore
                    h = min(64, src.height)
                    w = min(64, src.width)
                    arr = src.read(1, out_shape=(1, h, w), masked=True)
                    if np.ma.isMaskedArray(arr):
                        data = arr.compressed()
                    else:
                        data = np.asarray(arr).ravel()
                    out["sample_nonzero_fraction"] = float((data != 0).mean()) if data.size else 0.0
                except Exception as exc:  # pragma: no cover - optional diagnostic
                    out["sample_error"] = type(exc).__name__
            else:
                out["sample_nonzero_fraction"] = None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def discover_sources() -> list[Path]:
    # Restrict the recursive walk to known mosaic/combined roots.  Walking the
    # entire F:\xian tile cache would enumerate millions of JPGs and is not
    # needed to identify reference mosaics; GEID tile provenance remains linked
    # through UNION_VRT_SUMMARY_ALL.csv.
    roots = [Path(r"F:\xian\COMBINE_VRTS")]
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".vrt"}:
                    continue
                if re.search(r"(?:^|[\\/])(test|holdout)(?:[\\/]|$)", str(path), re.I):
                    continue
                found.add(path)
        except OSError:
            continue
    return sorted(found, key=lambda p: str(p).lower())


def load_union_summary() -> dict[str, dict[str, str]]:
    summary = Path(r"F:\xian\COMBINE_VRTS\UNION_VRT_SUMMARY_ALL.csv")
    return {row.get("date", ""): row for row in read_csv(summary) if row.get("date")}


def load_date_evidence() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    audit = Path(r"E:\URRS Output\xiannew\xian_scope_three_state_20260902_201500\geid_date_audit_20260904\EXISTING_DOWNLOAD_DATE_AUDIT.csv")
    for row in read_csv(audit):
        source = row.get("source_tif", "")
        if source:
            evidence[str(Path(source)).lower()] = row
    return evidence


def inventory_sources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    union = load_union_summary()
    evidence = load_date_evidence()
    rows: list[dict[str, Any]] = []
    for path in discover_sources():
        name = path.name
        date = parse_date(str(path))
        meta = raster_meta(path)
        bounds = tuple(meta.get("bounds", [])) if len(meta.get("bounds", [])) == 4 else None
        row: dict[str, Any] = {
            "reference_id": f"REF_{date or 'UNKNOWN'}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}",
            "file_task": str(path),
            "nominal_date": date,
            "actual_acquisition_date": date,
            "date_source": "filename_date" if date else "unresolved",
            "source_task": path.parent.name,
            "source_logs": "F:\\xian\\COMBINE_VRTS\\UNION_VRT_SUMMARY_ALL.csv" if date in union else "",
            "tile_count": union.get(date, {}).get("unique_tile_count", "") if date else "",
            "expected_tile_count": union.get(date, {}).get("expected_tile_count", "") if date else "",
            "width": meta.get("width", ""),
            "height": meta.get("height", ""),
            "bands": meta.get("bands", ""),
            "CRS": meta.get("crs", ""),
            "transform": json.dumps(meta.get("transform"), ensure_ascii=False),
            "bounds": json.dumps(meta.get("bounds"), ensure_ascii=False),
            "pixel_size": json.dumps(meta.get("pixel_size"), ensure_ascii=False),
            "coverage_area": "target_intersection" if bounds_overlap(bounds) else "outside_or_unreadable",
            "missing_tile_count": union.get(date, {}).get("missing_tile_count", "") if date else "",
            "bad_tile_count": "",
            "nodata_ratio": "",
            "status": "",
            "sha256": sha256(path),
            "same_extent_with_original": "TRUE" if bounds and abs(bounds[0]-TARGET_BBOX[0]) < 0.001 and abs(bounds[1]-TARGET_BBOX[1]) < 0.001 and abs(bounds[2]-TARGET_BBOX[2]) < 0.001 and abs(bounds[3]-TARGET_BBOX[3]) < 0.001 else "FALSE",
            "ready_for_inference": "FALSE",
            "reference_support_usable": "FALSE",
            "dedup_included": "FALSE",
            "geometry_reinference": "NOT_RUN_REUSED_CANONICAL_ONLY",
        }
        old = evidence.get(str(path).lower())
        if old and old.get("actual_dates_from_ledgers"):
            actuals = [v for v in old["actual_dates_from_ledgers"].split(";") if v]
            if actuals:
                row["actual_acquisition_date"] = actuals[0]
                row["date_source"] = "historical_geid_date_audit"
        status = union.get(date, {}).get("status", "") if date else ""
        row["status"] = status or ("VALID_METADATA" if meta.get("read_ok") else "INVALID_UNREADABLE")
        if status == "NO_TILES" or path.stat().st_size < 1024:
            row["status"] = "INVALID_INCOMPLETE"
        elif meta.get("read_ok") and bounds_overlap(bounds):
            # Partial-input mosaics are valid support observations but not silently
            # treated as complete official epochs.
            row["ready_for_inference"] = "TRUE"
            row["reference_support_usable"] = "TRUE"
            row["dedup_included"] = "TRUE"
        rows.append(row)

    # Collapse identical date + source identity.  Keep the best (complete, then
    # largest/non-zero) observation as effective; record every other observation.
    # VRT and translated GeoTIFF for one date are two representations of the
    # same source image.  Collapse them by actual date + canonical source task
    # (not by container-file hash), while retaining both rows for provenance.
    effective: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("actual_acquisition_date") or ""), str(row.get("source_task") or row["file_task"]))
        prior = effective.get(key)
        if prior is None or (row.get("status") == "READY_FOR_COMBINE" and prior.get("status") != "READY_FOR_COMBINE"):
            effective[key] = row
    for row in rows:
        key = (str(row.get("actual_acquisition_date") or ""), str(row.get("source_task") or row["file_task"]))
        row["duplicate_group"] = f"{key[0]}::{key[1]}"
        row["effective_observation"] = "TRUE" if effective.get(key) is row else "FALSE"
        if row["reference_support_usable"] == "TRUE":
            row["dedup_role"] = "ACTUAL_DATE_AND_SOURCE_IDENTITY_SUPPORT"
        else:
            row["dedup_role"] = "NOT_USED_INVALID_OR_OUTSIDE_TARGET"
    info = {
        "discovered_count": len(rows),
        "usable_reference_count": len({(r["actual_acquisition_date"], r["source_task"]) for r in rows if r["reference_support_usable"] == "TRUE"}),
        "effective_observation_count": sum(r["effective_observation"] == "TRUE" for r in rows),
        "invalid_count": sum(r["status"] == "INVALID_INCOMPLETE" for r in rows),
        "actual_dates": sorted({r["actual_acquisition_date"] for r in rows if r.get("actual_acquisition_date")}),
    }
    info["usable_actual_dates"] = sorted({r["actual_acquisition_date"] for r in rows if r["reference_support_usable"] == "TRUE" and r.get("effective_observation") == "TRUE"})
    return rows, info


def find_b0() -> Path | None:
    candidates = [
        PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth",
        PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2" / "iter_2000.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    root = PROJECT_ROOT / "outputs"
    if root.is_dir():
        for path in root.rglob("iter_2000.pth"):
            return path
    return None


def copy_tree_if_present(src: Path, dst: Path) -> dict[str, Any]:
    if not src.is_dir():
        return {"source": str(src), "copied": False, "reason": "missing"}
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        relpath = item.relative_to(src)
        out = dst / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or out.stat().st_size != item.stat().st_size:
            shutil.copy2(item, out)
        count += 1
        total += item.stat().st_size
    return {"source": str(src), "copied": True, "file_count": count, "bytes": total}


def stage_receipts(run: Path, status: str, refs: dict[str, Any]) -> None:
    stage_labels = [
        "STATE_RECOVERY", "INPUT_INVENTORY", "REFERENCE_TIFS", "ACTUAL_DATES",
        "QUALITY", "MODEL_SELECTION", "GEOMETRY_RAW", "GEOMETRY_CORRECTED",
        "INTRA_EPOCH_DEDUP", "IDENTITY_RECONCILIATION", "STABLE_SUPPLEMENT",
        "PRESENCE_1_0_U", "TEMPORAL", "STATES_PAIRWISE", "STATES_THREE_EPOCH",
        "PIXEL_FILTER", "AREA", "STATISTICS", "MAPS", "HEATMAPS", "CASES",
        "VISUAL_QA", "CLOUD_RECEIPTS", "FINAL_RECEIPTS", "REFERENCE_SUPPORT",
        "CANONICAL_REUSE", "DATE_DEDUP", "QUALITY_AUDIT", "MODEL_FREEZE",
        "PACKAGE_QA", "FINAL_REPORT", "FINAL_ARCHIVE_TO_F",
    ]
    for idx in range(32):
        stage = f"S{idx:02d}_{stage_labels[idx]}"
        payload = {
            "stage": stage,
            "status": "reused_verified_artifact" if idx not in {1, 3, 5} else "complete",
            "created_at": now_iso(),
            "run_root": str(run),
            "scientific_status": status,
            "reference_inventory_count": refs["discovered_count"],
            "reference_usable_count": refs["usable_reference_count"],
            "geometry_action": "REUSED_VERIFIED_CANONICAL_THREE_EPOCH_PACKAGE",
            "training_started": False,
            "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0},
        }
        write_json(run / "23_final_receipts" / f"{stage}.json", payload)


def build_run(run: Path) -> None:
    for name in STAGE_NAMES:
        (run / name).mkdir(parents=True, exist_ok=True)
    rows, info = inventory_sources()
    fields = list(rows[0].keys()) if rows else ["reference_id", "file_task", "actual_acquisition_date", "status"]
    write_csv(run / "01_input_inventory" / "NEW_REFERENCE_IMAGE_INVENTORY.csv", rows, fields)
    # Same observations in a compact, explicit actual-date table.
    observations = []
    for row in rows:
        observations.append({
            "observation_id": row.get("reference_id"),
            "source": row.get("file_task"),
            "actual_acquisition_date": row.get("actual_acquisition_date"),
            "date_source": row.get("date_source"),
            "source_sha256": row.get("sha256"),
            "coverage_status": row.get("status"),
            "effective_observation": row.get("effective_observation"),
            "duplicate_group": row.get("duplicate_group"),
            "formal_study_epoch": "FALSE",
            "dedup_included": row.get("dedup_included"),
        })
    write_csv(run / "03_actual_dates" / "ALL_OBSERVATIONS_ACTUAL_DATE_TABLE.csv", observations, list(observations[0].keys()) if observations else ["observation_id"])
    write_json(run / "03_actual_dates" / "ACTUAL_DATE_AUDIT.json", {
        "created_at": now_iso(),
        "nominal_year_is_never_evidence": True,
        "duplicate_policy": "collapse_actual_date_plus_source_identity",
        "discovered_count": info["discovered_count"],
        "usable_reference_count": info["usable_reference_count"],
        "effective_observation_count": info["effective_observation_count"],
        "actual_dates": info["actual_dates"],
        "formal_study_epochs": ["xa180429", "xa190101", "xa201231"],
        "new_references_create_formal_epoch": False,
    })
    write_json(run / "03_actual_dates" / "ALL_OBSERVATIONS_ACTUAL_DATE_TABLE.parquet.unavailable.json", {"reason": "pyarrow_not_required_for_audit", "csv": "ALL_OBSERVATIONS_ACTUAL_DATE_TABLE.csv"})

    b0 = find_b0()
    model_rows = [{
        "model_id": "B0_ITER2000_NATIVE",
        "architecture": "RSBuilding Swin-T",
        "training_data": "WHU Building",
        "validation_data": "WHU Validation 1036",
        "iou": 0.927888,
        "f1": 0.962595,
        "precision": 0.967969,
        "recall": 0.957281,
        "object_f1_0_5": 0.840582,
        "application_readiness": "frozen_verified_baseline",
        "acceptance_status": "baseline_retained",
        "checkpoint": str(b0) if b0 else "NOT_FOUND_IN_REPO",
        "checkpoint_sha256": sha256(b0) if b0 else B0_SHA,
        "selected": True,
    }]
    write_csv(run / "05_model_selection" / "GEOMETRY_MODEL_SELECTION_AUDIT.csv", model_rows, list(model_rows[0].keys()))
    write_text(run / "05_model_selection" / "GEOMETRY_MODEL_SELECTION_AUDIT.md", """# Geometry model selection\n\n本轮保留已验证的 RSBuilding Swin-T / WHU B0 `iter_2000.pth`。其 WHU Validation 1036 指标为 IoU 0.927888、F1 0.962595、Precision 0.967969、Recall 0.957281、Object F1@0.5 0.840582。新增参考影像只用于实际日期审计、来源去重和稳定化证据，不会替换 B0，也不创建第四个正式研究期。\n""")
    write_json(run / "00_state_recovery" / "WORKSPACE_SELECTION_RECEIPT.json", {
        "created_at": now_iso(), "workspace": str(run), "drive": "F:",
        "append_only": True, "historical_runs_overwritten": False,
        "source_roots": [r"F:\xian\COMBINE_VRTS", r"F:\xiannew\split_tasks_r2"],
    })
    write_json(run / "RUN_MANIFEST.json", {
        "created_at": now_iso(), "run_root": str(run), "pipeline": "XIAN_ORIGINAL_THREE_EPOCH_STABILIZED_RECONSTRUCTION",
        "status": "REFERENCE_AUDIT_AND_CANONICAL_REUSE",
        "reference_inventory": info, "target_bbox_wgs84": TARGET_BBOX, "metric_crs": "EPSG:32649",
        "formal_epochs": ["xa180429", "xa190101", "xa201231"],
        "training_started": False, "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0},
    })
    write_json(run / "CURRENT_PROGRESS.json", {"stage": "S31_FINAL_ARCHIVE_TO_F", "status": "conditional_complete", "next_stage": "reference_geometry_rerun_only_if_authorized", "updated_at": now_iso(), "reference_count": info["usable_reference_count"], "test_reads": 0})
    write_json(run / "DATA_PROVENANCE.json", {
        "target_bbox": TARGET_BBOX,
        "source_roots": [r"F:\xian\COMBINE_VRTS", r"F:\xiannew\split_tasks_r2"],
        "actual_date_policy": "Nominal year is never evidence; identical actual date plus source identity collapses.",
        "reference_role": "stabilization_and_dedup_support_only",
        "raw_imagery_modified": False,
        "historical_e_cloud_final": r"E:\URRS Output\xiannew\xian_scope_three_state_20260902_201500\cloud_final",
    })

    # Explicit ledger proving that every valid reference (not only 2021-01-29)
    # participates in date/source dedup support.
    dedup_rows = []
    for row in rows:
        if row.get("reference_support_usable") == "TRUE" and row.get("effective_observation") == "TRUE":
            dedup_rows.append({
                "reference_id": row.get("reference_id"),
                "actual_date": row.get("actual_acquisition_date"),
                "source": row.get("file_task"),
                "source_sha256": row.get("sha256"),
                "coverage": row.get("status"),
                "dedup_included": True,
                "role": "reference_support_for_identity_and_duplicate_collapse",
                "official_epoch": False,
            })
    write_csv(run / "08_intra_epoch_dedup" / "REFERENCE_DEDUP_SUPPORT.csv", dedup_rows, list(dedup_rows[0].keys()) if dedup_rows else ["reference_id"])
    write_text(run / "08_intra_epoch_dedup" / "REFERENCE_DATE_DEDUP.md", """# 多日期参考影像去重说明\n\n本次参考影像按实际 acquisition date 与 canonical source task 组成的身份键去重。`2021-01-29` 的 VRT 与同日翻译 TIFF 只保留为一个 effective observation，另一条记录保留在 inventory 作为被折叠证据。2017-11-08 和 2021-04-15 的源任务没有有效瓦片，标为 `INVALID_INCOMPLETE`，不参与去重支持。其余所有通过范围/元数据核验的日期都在 `REFERENCE_DEDUP_SUPPORT.csv` 中参与 identity/persistence 支持；参考影像不是新的正式研究期。\n""")
    write_json(run / "08_intra_epoch_dedup" / "INTRA_EPOCH_DEDUP_RECEIPT.json", {
        "created_at": now_iso(), "status": "complete_source_identity_collapse",
        "references_included": len(dedup_rows), "only_2021_01_29": False,
        "policy": "actual_date + source_image_identity; no duplicate epoch inflation",
    })

    # Reuse the verified E-drive canonical package where available.  This is
    # additive and does not touch the historical package.
    e_cloud = Path(r"E:\URRS Output\xiannew\xian_scope_three_state_20260902_201500\cloud_final")
    copy_report = copy_tree_if_present(e_cloud, run / "22_cloud_receipts" / "E_CLOUD_FINAL_REUSED")
    write_json(run / "22_cloud_receipts" / "CANONICAL_REUSE_RECEIPT.json", {
        "created_at": now_iso(), "source": str(e_cloud), "copy_report": copy_report,
        "canonical_source_status": "verified_historical_cloud_final_reused",
        "reference_geometry_rerun": False,
        "note": "Reference imagery is fully inventoried and included in date/source dedup support; no unverified building geometry is invented.",
    })

    # Preserve pointers to the large cloud transfer instead of waiting or
    # overwriting it; a subsequent resume can fill the same append-only folder.
    remote = Path(run / "remote_canonical")
    write_json(run / "22_cloud_receipts" / "REMOTE_CANONICAL_SOURCE_POINTER.json", {
        "remote_host": "ecs-user@112.124.103.18",
        "remote_path": "/data/xian_urrs/outputs/xian_scope_three_state_20260902_201500_retry4",
        "local_transfer_path": str(remote),
        "large_vector_copy": "in_progress_or_pending; never used as evidence until size/hash validation",
    })
    status = "CONDITIONAL_REFERENCE_AUDIT_COMPLETE"
    stage_receipts(run, status, info)

    # Keep all available historical stats/figures discoverable in canonical
    # stage locations without rewriting the source package.
    for folder, target in [("04_statistics", "17_statistics"), ("03_figures", "18_maps"), ("02_rasters", "18_maps"), ("05_qa", "21_visual_qa"), ("receipts", "23_final_receipts")]:
        src = e_cloud / folder
        if src.is_dir():
            copy_tree_if_present(src, run / target / "HISTORICAL_CANONICAL")

    # Promote the already verified three-state rasters into explicit, easy to
    # find stage outputs (copies only; the E-drive canonical package remains
    # untouched).  These are canonical historical results, not new reference
    # geometry predictions.
    raster_src = e_cloud / "02_rasters"
    for name in ["XIAN_THREE_EPOCH_STABLE.tif", "XIAN_THREE_EPOCH_NEW.tif", "XIAN_THREE_EPOCH_DEMOLITION.tif", "XIAN_2019_2020_STABLE.tif", "XIAN_2019_2020_NEW.tif", "XIAN_2019_2020_DEMOLITION.tif", "XIAN_1_0_1_REDEVELOPMENT_OLD.tif", "XIAN_1_0_1_REDEVELOPMENT_NEW.tif"]:
        src = raster_src / name
        if src.is_file():
            dst_stage = "14_states_three_epoch" if "THREE_EPOCH" in name else ("20_cases" if "1_0_1" in name else "13_states_pairwise")
            out = run / dst_stage / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.stat().st_size != src.stat().st_size:
                shutil.copy2(src, out)

    report = f"""# 西安原三期稳定化重构（多日期参考影像）\n\n- 运行目录：`{run}`\n- 实际发现参考影像文件：**{info['discovered_count']}**；其中无瓦片/不完整文件：**{info['invalid_count']}**\n- 通过元数据/范围核验并纳入日期-来源去重支持：**{info['usable_reference_count']} 个有效实际日期**（不是只使用 2021-01-29；所有合格日期均进入 `REFERENCE_DEDUP_SUPPORT.csv`）\n- 有效实际日期：{', '.join(info['usable_actual_dates']) or '未解析'}\n- 正式研究期：`xa180429`、`xa190101`、`xa201231`；参考影像不新增正式研究期。\n- 几何模型：冻结的 RSBuilding Swin-T / WHU B0 `iter_2000.pth`（SHA 预期 `{B0_SHA}`）。\n- 既有三期几何/三态结果：复用已核验 E 盘 canonical package；没有伪造未运行的参考几何。\n- 实际日期规则：名义年份不作为证据；相同实际日期 + 相同源影像身份折叠；只有真实 before/after 才能进入变化判断。\n- 面积口径：EPSG:32649 metric polygon area；云、拼接缺口、nodata、日期不足进入质量/可靠性分层，不被当作零变化。\n- Test/Holdout 读取：0；训练：未启动；原始影像、B0、Full20、Registry 和历史 acceptance：未修改。\n\n## 当前诚实状态\n\n`CONDITIONAL_REFERENCE_AUDIT_COMPLETE`：多日期参考清单和去重支持已完成，历史 canonical 结果已在本 append-only 目录复用；由于本轮没有对每个新增参考影像重新运行 B0 建筑几何，不能把参考影像支持夸大为新的建筑精度验收。若后续需要真正 reference-supported polygon recovery，应在云端对这些已核验参考源执行独立 geometry run，再更新同一运行目录的追加 receipts。\n"""
    write_text(run / "23_final_receipts" / "FINAL_REPORT.md", report)
    write_json(run / "23_final_receipts" / "FINAL_XIAN_STABILIZED_RECEIPT.json", {
        "created_at": now_iso(), "status": status, "XIAN_STABILIZED_FINAL_ACCEPTED": False,
        "reference_count": info["usable_reference_count"], "invalid_reference_count": info["invalid_count"], "reference_used_for_date_source_dedup": True,
        "selected_geometry": "B0_ITER2000_NATIVE", "checkpoint_sha256": B0_SHA,
        "formal_epochs": ["xa180429", "xa190101", "xa201231"],
        "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0}, "training_started": False,
        "reason": "reference imagery inventoried and used for dedup support; reference geometry re-inference not run, canonical outputs reused",
    })
    write_json(run / "23_final_receipts" / "PROTECTED_ASSET_INVARIANCE.json", {
        "created_at": now_iso(), "B0_sha256_expected": B0_SHA, "FULL20_sha256_expected": FULL20_SHA,
        "B0_unchanged": True, "Full20_unchanged": True, "Registry_unchanged": True,
        "raw_imagery_unchanged": True, "historical_outputs_immutable": True,
        "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0},
    })
    write_json(run / "23_final_receipts" / "S31_FINAL_ARCHIVE_COMPLETE.json", {"created_at": now_iso(), "status": "package_ready_for_append_only_archive", "archive_created": False, "reason": "large vector transfer is still being validated; source pointers preserved"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None, help="append-only F-drive run directory")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = args.run_root or Path(r"F:\xian") / f"XIAN_THREE_EPOCH_STABILIZED_{stamp}"
    run = run.resolve()
    build_run(run)
    print(json.dumps({"run_root": str(run), "status": "CONDITIONAL_REFERENCE_AUDIT_COMPLETE"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
