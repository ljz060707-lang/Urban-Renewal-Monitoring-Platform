"""Resume-safe queue for Xi'an sparse B0 geometry packages.

The queue consumes the date-aware source inventory produced by
``assemble_xian_temporal.py``.  It never trains a model, reads a locked test
set, rewrites a GEID source, or runs more than one GPU geometry worker at a
time.  Completed epoch packages are matched by their immutable source path and
are skipped; interrupted packages keep their run directory so the underlying
strip runner can resume from its last receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = Path(
    r"D:\Codex\projects\Documents-Codex\2026-07-18\urban-renewal-phase1-starter-v0-2"
    r"\outputs\urban-renewal-phase1-starter\outputs\v2_readiness\rsbuilding\whu\work_dirs"
    r"\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.parent / "Swin_T_whu_windows_pilot5000.py"
DEFAULT_METADATA_ROOTS = [Path(r"D:\Data Set\xiannew"), Path(r"E:\xiannew"), Path(r"E:\xian\xian")]
PROCESS_SCRIPT = Path(__file__).with_name("process_xian_single_epoch.py")
GPU_LOCK = Path(r"E:\URRS Output\runtime_locks\xian_r1_2020_geometry_gpu.lock")
QUEUE_LOCK = Path(r"E:\URRS Output\runtime_locks\xian_temporal_geometry_queue.lock")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def load_records(inventory_path: Path) -> list[dict[str, Any]]:
    payload = read_json(inventory_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise RuntimeError(f"invalid source inventory: {inventory_path}")
    records = [item for item in payload["records"] if isinstance(item, dict)]
    records.sort(key=lambda item: (int(item.get("nominal_year") or 9999), str(item.get("stem") or ""), str(item.get("source_tif") or "")))
    return records


def process_worker_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return False


def geometry_worker_running() -> bool:
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "cmdline"]):
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if proc.pid == os.getpid():
                continue
            if "process_xian_single_epoch.py" in cmdline and "--stage" in cmdline:
                return True
            if "infer_shenzhen_sparse_geometry.py" in cmdline and "xian" in cmdline.lower():
                return True
    except Exception:
        return False
    return False


def lock_owner(path: Path) -> int | None:
    payload = read_json(path, {}) if path.is_file() else {}
    try:
        pid = int(payload.get("pid"))
    except (TypeError, ValueError):
        pid = 0
    return pid or None


def lock_is_active(path: Path) -> bool:
    owner = lock_owner(path)
    return bool(owner and process_worker_alive(owner))


def acquire_queue_lock() -> None:
    QUEUE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    if QUEUE_LOCK.is_file() and lock_is_active(QUEUE_LOCK):
        owner = lock_owner(QUEUE_LOCK)
        raise RuntimeError(f"geometry queue already owned by PID {owner}: {QUEUE_LOCK}")
    if QUEUE_LOCK.is_file():
        QUEUE_LOCK.unlink(missing_ok=True)
    atomic_json(QUEUE_LOCK, {"pid": os.getpid(), "created_at": now_iso(), "purpose": "xian_temporal_geometry_queue"})


def release_queue_lock() -> None:
    if lock_owner(QUEUE_LOCK) == os.getpid():
        QUEUE_LOCK.unlink(missing_ok=True)


def source_roots(record: dict[str, Any], metadata_roots: list[Path]) -> tuple[Path, Path, Path]:
    source = Path(str(record.get("source_tif") or "")).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"source TIFF missing or empty: {source}")
    stem = str(record.get("stem") or source.parent.name.removesuffix("_combined"))
    data_root = source.parent.parent
    metadata_root = data_root
    if not (metadata_root / f"{stem}.geid").is_file() or not list(metadata_root.glob(f"{stem}_log.txt*")):
        for candidate in metadata_roots:
            if (candidate / f"{stem}.geid").is_file() and list(candidate.glob(f"{stem}_log.txt*")):
                metadata_root = candidate.resolve()
                break
    raw_root = data_root / stem / "20"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw tile root missing: {raw_root}")
    if not (metadata_root / f"{stem}.geid").is_file():
        raise FileNotFoundError(f"GEID metadata missing for {stem}: {metadata_root}")
    return source, data_root.resolve(), metadata_root.resolve()


def existing_successes(epoch_packages: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not epoch_packages.is_dir():
        return result
    for run in sorted(epoch_packages.iterdir()):
        if not run.is_dir():
            continue
        manifest = run / "04_geometry" / "GEOMETRY_MANIFEST.json"
        payload = read_json(manifest, {})
        if not isinstance(payload, dict) or payload.get("status") != "pass":
            continue
        scene = payload.get("scene") or {}
        source = scene.get("selected_source")
        if source:
            result[str(Path(source).resolve())] = {"run_dir": str(run.resolve()), "manifest": str(manifest.resolve()), "payload": payload}
    return result


def initial_state(records: list[dict[str, Any]], successes: dict[str, dict[str, Any]], state_path: Path) -> dict[str, dict[str, Any]]:
    prior = read_json(state_path, {})
    prior_tasks = prior.get("tasks", {}) if isinstance(prior, dict) else {}
    tasks: dict[str, dict[str, Any]] = {}
    for record in records:
        source_key = str(Path(str(record.get("source_tif") or "")).resolve())
        stem = str(record.get("stem") or "UNKNOWN")
        old = prior_tasks.get(stem, {}) if isinstance(prior_tasks, dict) else {}
        if source_key in successes:
            tasks[stem] = {
                "stem": stem,
                "source": source_key,
                "nominal_year": record.get("nominal_year"),
                "status": "SUCCESS_EXISTING",
                "run_dir": successes[source_key]["run_dir"],
                "building_manifest": successes[source_key]["manifest"],
                "updated_at": now_iso(),
            }
        elif (
            str(old.get("status")) in {"SUCCESS", "SUCCESS_EXISTING"}
            and Path(str(old.get("building_manifest") or "")).is_file()
            and read_json(Path(str(old["building_manifest"])), {}).get("status") == "pass"
        ):
            # A prior queue state may be the only local index for a completed
            # child package (for example after the queue is resumed with a
            # narrower epoch-packages directory).  Preserve it only when its
            # manifest is still present and explicitly successful; a stale
            # status without a valid manifest remains pending.
            tasks[stem] = {
                "stem": stem,
                "source": source_key,
                "nominal_year": record.get("nominal_year"),
                "status": "SUCCESS_EXISTING",
                "run_dir": old.get("run_dir"),
                "building_manifest": str(Path(str(old["building_manifest"])).resolve()),
                "updated_at": now_iso(),
            }
        else:
            prior_status = old.get("status")
            if prior_status not in {"FAILED", "BLOCKED_SOURCE_OR_METADATA", "WAITING_GPU_LOCK"}:
                prior_status = "PENDING"
            tasks[stem] = {
                "stem": stem,
                "source": source_key,
                "nominal_year": record.get("nominal_year"),
                "status": prior_status,
                "run_dir": old.get("run_dir"),
                "error": old.get("error"),
                "updated_at": old.get("updated_at"),
            }
    return tasks


def write_state(path: Path, run_dir: Path, tasks: dict[str, dict[str, Any]], current: str | None = None) -> None:
    counts: dict[str, int] = {}
    for task in tasks.values():
        status = str(task.get("status"))
        counts[status] = counts.get(status, 0) + 1
    atomic_json(
        path,
        {
            "schema_version": "urrs.xian.temporal.geometry_queue.v1",
            "updated_at": now_iso(),
            "run_dir": str(run_dir.resolve()),
            "current_task": current,
            "counts": counts,
            "tasks": tasks,
            "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
            "raw_inputs_modified": False,
            "checkpoint_policy": "frozen B0 only; no training",
        },
    )


def run_one(
    record: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    tasks: dict[str, dict[str, Any]],
    state_path: Path,
    run_dir: Path,
) -> int:
    stem = str(record.get("stem"))
    source, data_root, metadata_root = source_roots(record, args.metadata_root)
    child_run = Path(task.get("run_dir") or (run_dir / "03_geometry_runs" / f"{safe_stem(stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    task["run_dir"] = str(child_run.resolve())
    task["data_root"] = str(data_root)
    task["metadata_root"] = str(metadata_root)
    task["source"] = str(source)
    task["status"] = "RUNNING"
    task["started_at"] = now_iso()
    task["pid"] = os.getpid()
    task["error"] = None
    write_state(state_path, run_dir, tasks, current=stem)
    log_dir = run_dir / "logs" / "geometry_queue"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{safe_stem(stem)}.stdout.log"
    stderr_path = log_dir / f"{safe_stem(stem)}.stderr.log"
    command = [
        sys.executable,
        str(PROCESS_SCRIPT),
        "--stage",
        "all",
        "--data-root",
        str(data_root),
        "--metadata-root",
        str(metadata_root),
        "--task-stem",
        stem,
        "--output-root",
        str(run_dir / "03_geometry_runs"),
        "--run-dir",
        str(child_run),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--config",
        str(args.config.resolve()),
        "--source-override",
        str(source),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--strip-rows",
        str(args.strip_rows),
        "--quality-workers",
        str(args.quality_workers),
    ]
    task["command"] = command
    with stdout_path.open("a", encoding="utf-8", errors="replace") as stdout, stderr_path.open("a", encoding="utf-8", errors="replace") as stderr:
        stdout.write(f"[{now_iso()}] START {' '.join(command)}\n")
        stdout.flush()
        try:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=str(PROCESS_SCRIPT.parent), check=False)
            return_code = int(completed.returncode)
        except Exception as exc:
            return_code = 999
            task["error"] = f"{type(exc).__name__}: {exc}"
        stdout.write(f"[{now_iso()}] EXIT {return_code}\n")
    manifest = child_run / "04_geometry" / "GEOMETRY_MANIFEST.json"
    payload = read_json(manifest, {})
    if return_code == 0 and isinstance(payload, dict) and payload.get("status") == "pass":
        task["status"] = "SUCCESS"
        task["building_manifest"] = str(manifest.resolve())
        task["building_count"] = (payload.get("geometry_manifest") or {}).get("object_count")
        task["finished_at"] = now_iso()
        task["return_code"] = return_code
    else:
        task["status"] = "FAILED"
        task["return_code"] = return_code
        task["finished_at"] = now_iso()
        failure = read_json(child_run / "06_receipt" / "FAILURE_RECEIPT.json", {})
        task["error"] = (failure or {}).get("error") or task.get("error") or f"process returned {return_code}"
    task.pop("pid", None)
    write_state(state_path, run_dir, tasks, current=None)
    return return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch-packages", type=Path, default=Path(r"E:\URRS Output\xiannew\epoch_packages"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metadata-root", type=Path, action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--strip-rows", type=int, default=1024)
    parser.add_argument("--quality-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.inventory = args.inventory.resolve()
    args.run_dir = args.run_dir.resolve()
    args.epoch_packages = args.epoch_packages.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.config = args.config.resolve()
    metadata_roots = [item.resolve() for item in (args.metadata_root or DEFAULT_METADATA_ROOTS)]
    args.metadata_root = metadata_roots
    if not PROCESS_SCRIPT.is_file():
        raise FileNotFoundError(PROCESS_SCRIPT)
    if not args.checkpoint.is_file() or not args.config.is_file():
        raise FileNotFoundError(f"checkpoint/config missing: {args.checkpoint} / {args.config}")
    records = load_records(args.inventory)
    successes = existing_successes(args.epoch_packages)
    state_path = args.run_dir / "GEOMETRY_QUEUE_STATE.json"
    tasks = initial_state(records, successes, state_path)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_state(state_path, args.run_dir, tasks, current=None)
    pending = [record for record in records if tasks.get(str(record.get("stem")), {}).get("status") not in {"SUCCESS", "SUCCESS_EXISTING"}]
    summary = {
        "created_at": now_iso(),
        "inventory": str(args.inventory),
        "run_dir": str(args.run_dir),
        "record_count": len(records),
        "existing_success_count": len(successes),
        "pending_count": len(pending),
        "pending_stems": [record.get("stem") for record in pending],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7",
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "raw_inputs_modified": False,
    }
    atomic_json(args.run_dir / "receipts" / "GEOMETRY_QUEUE_PLAN.json", summary)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    acquire_queue_lock()
    completed = 0
    failed = 0
    try:
        for record in pending:
            if args.max_tasks and completed + failed >= args.max_tasks:
                break
            stem = str(record.get("stem"))
            task = tasks[stem]
            try:
                if geometry_worker_running() or lock_is_active(GPU_LOCK):
                    task["status"] = "WAITING_GPU_LOCK"
                    task["updated_at"] = now_iso()
                    write_state(state_path, args.run_dir, tasks, current=None)
                    # Do not contend with another worker.  A detached queue can
                    # be resumed later without changing any parameters.
                    continue
                rc = run_one(record, task, args, tasks, state_path, args.run_dir)
                if rc == 0 and task.get("status") == "SUCCESS":
                    completed += 1
                else:
                    failed += 1
            except Exception as exc:
                task["status"] = "BLOCKED_SOURCE_OR_METADATA"
                task["error"] = f"{type(exc).__name__}: {exc}"
                task["updated_at"] = now_iso()
                failed += 1
                write_state(state_path, args.run_dir, tasks, current=None)
    finally:
        release_queue_lock()
    final = {
        "schema_version": "urrs.xian.temporal.geometry_queue_receipt.v1",
        "created_at": now_iso(),
        "status": "pass" if failed == 0 else "completed_with_failures",
        "completed_this_run": completed,
        "failed_this_run": failed,
        "counts": read_json(state_path, {}).get("counts", {}),
        "run_dir": str(args.run_dir),
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "raw_inputs_modified": False,
        "source_policy": "source TIFFs and GEID inputs read-only; existing successful packages skipped",
    }
    atomic_json(args.run_dir / "receipts" / "GEOMETRY_QUEUE_RECEIPT.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
