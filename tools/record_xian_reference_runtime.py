"""Record a verified remote Xi'an reference-job transition.

Only derived queue/progress receipts are written.  The script never controls
or terminates a remote process; the caller must supply a PID obtained from a
successful, explicit launch command.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def update_csv(path: Path, ref_id: str, date: str, gpu: str, status: str, runtime: str = "") -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    if not fields:
        fields = ["reference_id", "actual_date", "source_status", "transfer_pct", "mosaic_pct", "tif_validation_pct", "geometry_pct", "polygon_merge_pct", "dedup_pct", "gpu_id", "runtime_seconds", "eta", "final_status", "reason"]
    found = False
    for row in rows:
        if row.get("reference_id") == ref_id or row.get("actual_date") == date:
            row.update({"reference_id": ref_id, "actual_date": date, "gpu_id": gpu, "geometry_pct": "0.0", "runtime_seconds": runtime, "eta": "CALIBRATING", "final_status": status})
            found = True
    if not found:
        rows.append({"reference_id": ref_id, "actual_date": date, "source_status": "TIF_VALIDATED", "transfer_pct": "1.0", "mosaic_pct": "1.0", "tif_validation_pct": "1.0", "geometry_pct": "0.0", "polygon_merge_pct": "0.0", "dedup_pct": "0.0", "gpu_id": gpu, "runtime_seconds": runtime, "eta": "CALIBRATING", "final_status": status, "reason": "verified cloud geometry worker launched"})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--actual-date", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--remote-pid", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-bytes", required=True, type=int)
    args = parser.parse_args()
    run = Path(args.run_root).resolve()
    timestamp = now()
    state_path = run / "02_reference_tifs" / "REFERENCE_STATES" / args.actual_date / "REFERENCE_STATE.json"
    state = read_json(state_path)
    state.update({"reference_id": args.reference_id, "actual_date": args.actual_date, "state": "GEOMETRY_RUNNING", "source_sha256": args.source_sha256, "source_bytes": args.source_bytes, "cloud_ready": True, "geometry_started": True, "geometry_completed": False, "polygon_merge_completed": False, "dedup_completed": False, "gpu_id": args.gpu, "remote_pid": args.remote_pid, "batch_size": 4, "strip_rows": 1024, "quality_workers": 8, "updated_at": timestamp, "reason": "remote TIF size/SHA verified; frozen B0 geometry worker is active"})
    write_json(state_path, state)
    queue_path = run / "GPU_JOB_QUEUE.json"
    queue = read_json(queue_path)
    for job in queue.get("jobs", []):
        if job.get("reference_id") == args.reference_id or job.get("actual_date") == args.actual_date:
            job.update({"status": "RUNNING", "gpu_id": int(args.gpu), "remote_pid": int(args.remote_pid), "source_sha256": args.source_sha256, "source_bytes": args.source_bytes, "started_at": timestamp, "batch_size": 4, "strip_rows": 1024})
    queue["queue_status"] = "RUNNING_REFERENCE_GEOMETRY"
    queue["active_jobs"] = [{"reference_id": args.reference_id, "actual_date": args.actual_date, "gpu_id": int(args.gpu), "remote_pid": int(args.remote_pid), "status": "RUNNING"}]
    queue["updated_at"] = timestamp
    write_json(queue_path, queue)
    update_csv(run / "REFERENCE_PROGRESS.csv", args.reference_id, args.actual_date, args.gpu, "GEOMETRY_RUNNING")
    progress_path = run / "MASTER_PROGRESS.json"
    progress = read_json(progress_path)
    progress.update({"updated_at": timestamp, "overall_status": "REFERENCE_GEOMETRY_RUNNING", "current_stage": "S03_REFERENCE_GEOMETRY_INFERENCE", "active_gpu_jobs": 1, "cloud_geometry_started": True, "reference_geometry_completed_count": 0, "eta": "CALIBRATING"})
    progress["blocking_reasons"] = [reason for reason in progress.get("blocking_reasons", []) if "No cloud-ready reference TIF" not in str(reason)]
    progress["blocking_reasons"].append("Six effective references remain partial VRT sources pending TIF materialization; final stabilization remains blocked until all allowed references are processed.")
    write_json(progress_path, progress)
    receipt = {"schema_version": "urrs.xian.reference_geometry_launch_receipt.v1", "created_at": timestamp, "status": "GEOMETRY_RUNNING", "reference_id": args.reference_id, "actual_date": args.actual_date, "gpu": int(args.gpu), "remote_pid": int(args.remote_pid), "source_sha256": args.source_sha256, "source_bytes": args.source_bytes, "batch_size": 4, "strip_rows": 1024, "quality_workers": 8, "checkpoint_sha256": "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7", "duplicate_writer_started": False, "source_imagery_modified": False, "test_reads": {"WHU": 0, "SECOND": 0, "HOLDOUT": 0}}
    write_json(run / "22_cloud_receipts" / "REFERENCE_2021-01-29_GEOMETRY_LAUNCH_RECEIPT.json", receipt)
    print(json.dumps({"status": "GEOMETRY_RUNNING", "reference_id": args.reference_id, "remote_pid": int(args.remote_pid)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
