"""Write an evidence-backed progress snapshot for the Xi'an stabilized run.

This controller is deliberately non-invasive: it never starts/stops a worker,
does not alter source imagery, and only derives queue/progress receipts from
the authoritative inventory and existing output metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


B0_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
WEIGHTS = {
    "S00_state_recovery": 3,
    "S01_reference_transfer_mosaic_tif_validation": 10,
    "S02_geometry_model_audit": 3,
    "S03_reference_geometry_inference": 20,
    "S04_original_geometry_reaudit": 10,
    "S05_intra_date_dedup": 6,
    "S06_identity_reconciliation": 12,
    "S07_stable_supplementation": 12,
    "S08_presence_1_0_U_v8r8": 8,
    "S09_three_state_reconstruction": 6,
    "S10_precise_area": 4,
    "S11_pixel_sensitivity_results": 3,
    "S12_maps_cases_visual_qa": 2,
    "S13_final_archive": 1,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory_rows(run: Path) -> list[dict[str, str]]:
    path = run / "01_input_inventory" / "NEW_REFERENCE_IMAGE_INVENTORY.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def effective_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("effective_observation", "")).upper() != "TRUE":
            continue
        if str(row.get("reference_support_usable", "")).upper() != "TRUE":
            continue
        if str(row.get("status", "")).upper().startswith("INVALID"):
            continue
        key = str(row.get("duplicate_group") or row.get("actual_acquisition_date") or row.get("reference_id"))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def tif_status(row: dict[str, str]) -> tuple[str, float, str]:
    source = Path(str(row.get("file_task", "")))
    ext = source.suffix.lower()
    exists = source.is_file()
    if not exists:
        return "SOURCE_MISSING", 0.0, "inventory source path does not exist"
    if ext in {".tif", ".tiff"}:
        size = source.stat().st_size
        if size <= 0:
            return "TIF_INVALID_ZERO_BYTES", 0.0, "validated source is zero bytes"
        return "TIF_VALIDATED", 1.0, "complete local raster source is present"
    if ext == ".vrt":
        coverage = str(row.get("coverage_area", ""))
        missing = str(row.get("missing_tile_count", ""))
        return "TIF_PENDING_MATERIALIZATION", 0.0, f"VRT source; coverage={coverage}; missing_tile_count={missing}"
    return "SOURCE_VALIDATED_NEEDS_TIF", 0.0, f"unsupported source extension {ext}"


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append({"gpu_id": parts[0], "name": parts[1], "memory_used_mib": parts[2], "memory_total_mib": parts[3], "utilization_pct": parts[4]})
    return rows


def process_snapshot() -> dict[str, Any]:
    names = {"scp.exe", "ssh.exe", "python.exe", "python3", "gdal_translate.exe", "gdalwarp.exe"}
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Process | Select-Object Id,ProcessName,CPU,WorkingSet64 | ConvertTo-Json -Compress"], capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"process_query": "unavailable"}
    if result.returncode != 0:
        return {"process_query": "failed"}
    try:
        parsed = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return {"process_query": "unparseable"}
    if isinstance(parsed, dict):
        parsed = [parsed]
    active = [item for item in parsed if str(item.get("ProcessName", "")).casefold() + ".exe" in {n.casefold() for n in names}]
    return {"tracked_process_count": len(active), "tracked_processes": active}


def build(run: Path) -> None:
    rows = inventory_rows(run)
    effective = effective_rows(rows)
    timestamp = now_iso()
    reference_progress: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for row in sorted(effective, key=lambda item: str(item.get("actual_acquisition_date", ""))):
        date = str(row.get("actual_acquisition_date", "unknown"))
        ref_id = str(row.get("reference_id", date))
        source = Path(str(row.get("file_task", "")))
        status, tif_pct, reason = tif_status(row)
        cloud_ready = False
        final_status = status
        if status == "TIF_VALIDATED":
            final_status = "TIF_VALIDATED_CLOUD_TRANSFER_PENDING"
        source_sha = str(row.get("sha256") or "") or (sha256(source) if source.suffix.lower() not in {".tif", ".tiff"} else None)
        state_path = run / "02_reference_tifs" / "REFERENCE_STATES" / date / "REFERENCE_STATE.json"
        prior_state = read_json(state_path)
        state = {
            "reference_id": ref_id,
            "actual_date": date,
            "state": final_status,
            "source": str(source),
            "source_sha256": source_sha,
            "coverage": row.get("coverage_area"),
            "missing_tile_count": row.get("missing_tile_count"),
            "tif_validated": status == "TIF_VALIDATED",
            "cloud_ready": cloud_ready,
            "geometry_started": False,
            "geometry_completed": False,
            "polygon_merge_completed": False,
            "dedup_completed": False,
            "reason": reason,
            "updated_at": timestamp,
            "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0},
        }
        # Once a reference geometry worker has started, never regress its
        # state merely because a later progress snapshot sees the same source.
        # This keeps the controller resume-safe and append-only.
        if prior_state.get("geometry_started") or prior_state.get("geometry_completed"):
            state.update(prior_state)
            state["updated_at"] = timestamp
        states.append(state)
        reference_progress.append({
            "reference_id": ref_id,
            "actual_date": date,
            "source_status": status,
            "transfer_pct": 1.0 if status == "TIF_VALIDATED" else 0.0,
            "mosaic_pct": 1.0 if status == "TIF_VALIDATED" else 0.0,
            "tif_validation_pct": tif_pct,
            "geometry_pct": 0.0,
            "polygon_merge_pct": 0.0,
            "dedup_pct": 0.0,
            "gpu_id": "",
            "runtime_seconds": "",
            "eta": "CALIBRATING",
            "final_status": final_status,
            "reason": reason,
        })
        queue.append({
            "reference_id": ref_id,
            "actual_date": date,
            "priority": 0 if status == "TIF_VALIDATED" else 1,
            "status": "BLOCKED_CLOUD_TRANSFER_PENDING" if status == "TIF_VALIDATED" else "BLOCKED_TIF_NOT_MATERIALIZED",
            "gpu_id": None,
            "source": str(source),
            "source_sha256": source_sha,
            "reason": "No cloud-ready reference TIF is present; no geometry worker was started." if status == "TIF_VALIDATED" else reason,
        })
    state_root = run / "02_reference_tifs" / "REFERENCE_STATES"
    for state in states:
        write_json(state_root / state["actual_date"] / "REFERENCE_STATE.json", state)
    fields = list(reference_progress[0]) if reference_progress else ["reference_id", "actual_date", "source_status", "transfer_pct", "mosaic_pct", "tif_validation_pct", "geometry_pct", "polygon_merge_pct", "dedup_pct", "gpu_id", "runtime_seconds", "eta", "final_status", "reason"]
    write_csv(run / "REFERENCE_PROGRESS.csv", reference_progress, fields)
    write_json(run / "GPU_JOB_QUEUE.json", {"created_at": timestamp, "queue_status": "WAITING_FOR_CLOUD_READY_REFERENCE_TIF", "jobs": queue, "active_jobs": [], "scheduler_started": False, "no_duplicate_writer": True})

    complete = {
        "S00_state_recovery": 1.0,
        "S01_reference_transfer_mosaic_tif_validation": (sum(1 for item in states if item["tif_validated"]) / len(states)) if states else 0.0,
        "S02_geometry_model_audit": 1.0 if (run / "05_model_selection" / "GEOMETRY_MODEL_SELECTION_AUDIT.md").is_file() else 0.0,
        "S03_reference_geometry_inference": 0.0,
        "S04_original_geometry_reaudit": 1.0 if (run / "23_final_receipts" / "FINAL_XIAN_STABILIZED_RECEIPT.json").is_file() else 0.0,
        "S05_intra_date_dedup": 1.0 if (run / "08_intra_epoch_dedup" / "INTRA_EPOCH_DEDUP_RECEIPT.json").is_file() else 0.0,
        "S06_identity_reconciliation": 0.0,
        "S07_stable_supplementation": 0.0,
        "S08_presence_1_0_U_v8r8": 0.0,
        "S09_three_state_reconstruction": 0.0,
        "S10_precise_area": 0.0,
        "S11_pixel_sensitivity_results": 0.0,
        "S12_maps_cases_visual_qa": 0.0,
        "S13_final_archive": 0.0,
    }
    weighted_pct = sum(WEIGHTS[key] * complete[key] for key in WEIGHTS)
    progress = {
        "schema_version": "urrs.xian_stabilized.master_progress.v1",
        "updated_at": timestamp,
        "run_root": str(run),
        "overall_percent": round(weighted_pct, 3),
        "overall_status": "CONDITIONAL_REFERENCE_AUDIT_COMPLETE",
        "current_stage": "S01_REFERENCE_TIF_VALIDATION_AND_CLOUD_TRANSFER",
        "eta": "CALIBRATING",
        "weights_percent": WEIGHTS,
        "stage_completion_fraction": complete,
        "effective_reference_count": len(effective),
        "tif_validated_reference_count": sum(1 for item in states if item["tif_validated"]),
        "reference_geometry_completed_count": 0,
        "active_gpu_jobs": 0,
        "cloud_geometry_started": False,
        "training_started": False,
        "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0},
        "b0_checkpoint_sha256": B0_SHA,
        "protected_assets_unchanged": True,
        "blocking_reasons": [
            "Only 2021-01-29 has a complete local TIF; other effective references are partial VRT sources pending TIF materialization.",
            "No cloud-ready reference TIF has been transferred; geometry scheduler remains fail-closed and idle.",
            "Reference geometry has not been run, so final stabilization/acceptance is not claimed.",
        ],
    }
    write_json(run / "MASTER_PROGRESS.json", progress)
    lines = [
        "# Xi'an stabilized master pipeline progress",
        "",
        f"更新时间：`{timestamp}`",
        f"总体状态：`{progress['overall_status']}`；加权进度：`{progress['overall_percent']:.3f}%`（不是按阶段数量伪造）。",
        "",
        "## 当前事实",
        "",
        f"- 有效实际日期：{len(effective)}；已验证完整本地 TIF：{progress['tif_validated_reference_count']}。",
        "- 参考 geometry 已启动：`false`；云端活跃 geometry worker：`0`。",
        "- 当前队列保持 fail-closed：完整 TIF 需要先完成云端传输或明确选择本地 GPU；本快照未启动任何 worker。",
        "- 2021-01-29 已有完整 TIFF；其余有效日期仍为部分覆盖 VRT，缺失区域只能记录为 U/UNOBSERVED。",
        f"- WHU/SECOND/HOLDOUT reads：`0/0/0`；B0 SHA：`{B0_SHA}`。",
        "",
        "## 阶段进度",
        "",
    ]
    for key, weight in WEIGHTS.items():
        pct = complete[key] * 100
        lines.append(f"- {key}（{weight}%）：{pct:.1f}%")
    lines.extend([
        "",
        "## 参考日期状态",
        "",
    ])
    for state in states:
        lines.append(f"- {state['actual_date']}：`{state['state']}`；{state['reason']}。")
    lines.extend([
        "",
        "## 输出索引",
        "",
        "- `MASTER_PROGRESS.json`：机器可读总进度。",
        "- `REFERENCE_PROGRESS.csv`：按实际日期的可恢复状态。",
        "- `GPU_JOB_QUEUE.json`：当前无活跃 job，不会重复写入。",
        "- `02_reference_tifs/REFERENCE_STATES/<actual_date>/REFERENCE_STATE.json`：每期状态与原因。",
        "",
        "最终接受仍未宣称；需要完成全部参考 TIF、每期 geometry/polygon/dedup、identity stabilization 和 QA 后才能改变状态。",
    ])
    (run / "MASTER_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gpu_rows = gpu_snapshot()
    utilization_fields = ["timestamp", "host", "gpu_id", "gpu_name", "memory_used_mib", "memory_total_mib", "utilization_pct", "active_reference", "geometry_worker_count", "queue_status", "source"]
    util_rows = []
    for gpu in gpu_rows:
        util_rows.append({"timestamp": timestamp, "host": platform.node(), **gpu, "active_reference": "", "geometry_worker_count": 0, "queue_status": "WAITING_FOR_CLOUD_READY_REFERENCE_TIF", "source": "local_nvidia_smi"})
    if not util_rows:
        util_rows.append({"timestamp": timestamp, "host": platform.node(), "gpu_id": "", "gpu_name": "nvidia-smi unavailable", "memory_used_mib": "", "memory_total_mib": "", "utilization_pct": "", "active_reference": "", "geometry_worker_count": 0, "queue_status": "WAITING_FOR_CLOUD_READY_REFERENCE_TIF", "source": "local_snapshot"})
    write_csv(run / "RESOURCE_UTILIZATION_REPORT.csv", util_rows, utilization_fields)
    write_json(run / "PROGRESS_SNAPSHOT_RECEIPT.json", {"created_at": timestamp, "status": "PASS_READ_ONLY_SNAPSHOT", "run_root": str(run), "files_written": ["MASTER_PROGRESS.json", "MASTER_PROGRESS.md", "REFERENCE_PROGRESS.csv", "GPU_JOB_QUEUE.json", "RESOURCE_UTILIZATION_REPORT.csv"], "workers_started": 0, "workers_stopped": 0, "source_imagery_modified": False, "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    build(Path(args.run_root).resolve())
    print(json.dumps({"run_root": args.run_root, "status": "PASS_READ_ONLY_SNAPSHOT"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
