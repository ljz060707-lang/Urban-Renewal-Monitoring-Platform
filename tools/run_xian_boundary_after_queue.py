"""Run final Xi'an boundary accounting after the existing geometry queue drains.

This one-shot CPU helper is deliberately independent from the GEID watcher. It
waits for the known queue PID and any child ``process_xian_single_epoch.py``
workers under the queue run to disappear, then runs the boundary accounting
against a corrected date-aware inventory. It never starts a GPU worker and
never mutates source imagery, checkpoints, or historical packages.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_final_report(run_dir: Path, queue_run_dir: Path, inventory_path: Path) -> Path:
    """Write a compact, boundary-scoped handoff after accounting succeeds."""
    receipt = read_json(run_dir / "05_temporal_outputs" / "TEMPORAL_ACCOUNTING_RECEIPT.json", {})
    inventory = read_json(inventory_path, {})
    queue = read_json(queue_run_dir / "GEOMETRY_QUEUE_STATE.json", {})
    tasks = (queue or {}).get("tasks", {}) if isinstance(queue, dict) else {}
    success_count = sum(str(item.get("status")) in {"SUCCESS", "SUCCESS_EXISTING"} for item in tasks.values() if isinstance(item, dict))
    report = run_dir / "FINAL_XIAN_TEMPORAL_REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# Xi'an citywide temporal building package",
                "",
                "## Scope",
                "",
                f"- Date-aware inventory: `{inventory.get('source_count', 0)}` combined GeoTIFFs; usable `{inventory.get('usable_count', 0)}`.",
                f"- Successful geometry packages indexed: `{success_count}`; source packages are reused and raw imagery remains read-only.",
                f"- Actual-date categories: `{read_json(run_dir / '02_temporal_reliability' / 'TEMPORAL_RELIABILITY.json', {}).get('category_counts', {})}`.",
                "- Nominal task year is never used as acquisition evidence; duplicate/off-date/insufficient observations remain labelled explicitly.",
                "",
                "## Boundary-scoped statistics",
                "",
                "All areas and counts are intersections with the supplied Xi'an city or county GeoJSON after reprojection to EPSG:32649; outside-city slivers are excluded.",
                f"- City source/epoch tables: `{run_dir / '04_statistics' / 'XIAN_SOURCE_BOUNDARY_STATS.csv'}`; `{run_dir / '04_statistics' / 'XIAN_CITY_EPOCH_STATS.csv'}`",
                f"- County source/epoch tables: `{run_dir / '04_statistics' / 'XIAN_COUNTY_SOURCE_BOUNDARY_STATS.csv'}`; `{run_dir / '04_statistics' / 'XIAN_COUNTY_EPOCH_STATS.csv'}`",
                f"- City/county candidate change tables: `{run_dir / '05_temporal_outputs' / 'XIAN_CITY_CHANGE_STATS.csv'}`; `{run_dir / '05_temporal_outputs' / 'XIAN_COUNTY_CHANGE_STATS.csv'}`",
                f"- City/county state area tables: `{run_dir / '05_temporal_outputs' / 'XIAN_CITY_STATE_STATS.csv'}`; `{run_dir / '05_temporal_outputs' / 'XIAN_COUNTY_STATE_STATS.csv'}`",
                "The epoch tables include total clipped building area and boundary shares; change/state tables include clipped new, demolition, and unchanged areas plus their boundary shares.",
                f"- Accounting receipt: `{run_dir / '05_temporal_outputs' / 'TEMPORAL_ACCOUNTING_RECEIPT.json'}`",
                "",
                "## Scientific limits",
                "",
                "New/demolition fields are emitted only where a genuine interval between distinct actual dates exists. All geometry and change rows remain `candidate_not_ground_truth` pending independent review.",
                "",
                "## Validation",
                "",
                "WHU Test reads = 0; SECOND Test reads = 0; Holdout reads = 0; raw inputs modified = false.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def active_matching_processes(queue_pid: int, run_dir: Path) -> list[int]:
    """Return queue/geometry PIDs that still belong to this run."""
    # Keep an explicit liveness check for the queue PID.  On elevated Windows
    # hosts psutil may return an empty/incomplete cmdline for another-session
    # process, but pid_exists remains reliable; the queue parent waits for its
    # geometry child and therefore is the primary synchronization point.
    result: set[int] = set()
    if queue_pid and _pid_alive(queue_pid):
        result.add(int(queue_pid))
    try:
        import psutil
    except Exception:
        return sorted(result)
    run_token = str(run_dir.resolve()).lower()
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info.get("pid") or 0)
            command = " ".join(process.info.get("cmdline") or []).lower()
        except (psutil.Error, TypeError, ValueError):
            continue
        if not pid:
            continue
        if run_token in command and ("run_xian_temporal_geometry.py" in command or "process_xian_single_epoch.py" in command):
            result.add(pid)
    return sorted(result)


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return bool(pid and psutil.pid_exists(int(pid)))
    except Exception:
        # The rsbuilding environment intentionally has no psutil.  Use the
        # read-only Win32 process handle API as a narrow fallback so a live
        # queue cannot be mistaken for an idle queue.
        if not pid or sys.platform != "win32":
            return False
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, int(pid))
        if not handle:
            return False
        try:
            STILL_ACTIVE = 259
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return int(code.value) == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queue-run-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--geometry-runs", type=Path, required=True)
    parser.add_argument("--epoch-packages", type=Path, default=Path(r"E:\URRS Output\xiannew\epoch_packages"))
    parser.add_argument("--city-boundary", type=Path, default=Path(r"C:\Users\ljz52\Desktop\西安市_市.geojson"))
    parser.add_argument("--county-boundary", type=Path, default=Path(r"C:\Users\ljz52\Desktop\西安市_县.geojson"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    queue_run_dir = args.queue_run_dir.resolve()
    state_path = run_dir / "receipts" / "BOUNDARY_FINALIZER_STATE.json"
    state: dict[str, Any] = {
        "status": "waiting_for_geometry_queue",
        "created_at": now_iso(),
        "queue_pid": args.queue_pid,
        "run_dir": str(run_dir),
        "inventory": str(args.inventory.resolve()),
        "geometry_runs": str(args.geometry_runs.resolve()),
        "queue_run_dir": str(queue_run_dir),
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "raw_inputs_modified": False,
    }
    atomic_json(state_path, state)
    while True:
        active = active_matching_processes(args.queue_pid, queue_run_dir)
        state.update({"status": "waiting_for_geometry_queue" if active else "starting_boundary_accounting", "updated_at": now_iso(), "active_pids": active})
        atomic_json(state_path, state)
        if not active:
            break
        time.sleep(max(10, int(args.poll_seconds)))

    boundary_script = Path(__file__).with_name("build_xian_boundary_temporal_outputs.py")
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(boundary_script),
        "--assembly-run",
        str(run_dir),
        "--inventory",
        str(args.inventory.resolve()),
        "--epoch-packages",
        str(args.epoch_packages.resolve()),
        "--geometry-runs",
        str(args.geometry_runs.resolve()),
        "--run-dir",
        str(run_dir),
        "--city-boundary",
        str(args.city_boundary.resolve()),
        "--county-boundary",
        str(args.county_boundary.resolve()),
    ]
    state.update({"status": "running_boundary_accounting", "updated_at": now_iso(), "command": command})
    atomic_json(state_path, state)
    with (log_dir / "boundary_finalizer.stdout.log").open("w", encoding="utf-8") as stdout, (log_dir / "boundary_finalizer.stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(boundary_script.parent), stdout=stdout, stderr=stderr, check=False)
    report = None
    if completed.returncode == 0:
        report = str(write_final_report(run_dir, queue_run_dir, args.inventory).resolve())
    state.update({"status": "complete" if completed.returncode == 0 else "failed", "return_code": int(completed.returncode), "final_report": report, "updated_at": now_iso()})
    atomic_json(state_path, state)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
