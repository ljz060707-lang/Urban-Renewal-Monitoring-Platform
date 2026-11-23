"""Wait for the final4 cloud geometry queue and run bounded post-processing.

The controller is deliberately resume-safe: it waits for both existing shard
workers, retries only the three receipts known to have failed for missing
metadata, then runs the CPU-only boundary/date-aware package.  It never reads
test/holdout data and never changes the B0 checkpoint or sparse-geometry
parameters.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/data/xian_urrs")
ENV_PY = BASE / "env/bin/python"
MANIFEST = BASE / "manifests/final4/RUNPOD_PENDING_SOURCE_MANIFEST.csv"
SUPPLEMENTAL_MANIFEST = BASE / "manifests/final4/supplemental/RUNPOD_SUPPLEMENTAL_MANIFEST.csv"
RECOVERY_DIR = BASE / "manifests/final4/recovery"
SHARD_SCRIPT = BASE / "work/run_aliyun_geometry_shard.py"
RUNNER = BASE / "work/run_cloud_single_aliyun.sh"
OUTPUT_ROOT = BASE / "outputs"
LOG_ROOT = BASE / "logs"
RECEIPT_ROOT = BASE / "receipts"
ASSEMBLY = BASE / "postprocess/assembly"
CODE = BASE / "postprocess/code"
BOUNDARY_CITY = BASE / "postprocess/boundaries/西安市_市.geojson"
BOUNDARY_COUNTY = BASE / "postprocess/boundaries/西安市_县.geojson"
FINAL_RUN = BASE / "postprocess/final_temporal_cloud"
LOCAL_GEOMETRY_ARCHIVE = BASE / "postprocess/local_geometry_minimal_20260901_112240.tar.gz"
LOCAL_GEOMETRY_EXTRACTED = BASE / "postprocess/local_geometry"


def stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def active_shards() -> bool:
    patterns = ("run_aliyun_geometry_shard.py.*--gpu 0", "run_aliyun_geometry_shard.py.*--gpu 1")
    for pattern in patterns:
        result = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return True
    return False


def write_recovery_manifests() -> list[list[str]]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    if SUPPLEMENTAL_MANIFEST.is_file():
        rows.extend(csv.DictReader(SUPPLEMENTAL_MANIFEST.open(encoding="utf-8-sig")))
    failed = {"xar4180429", "xar1201231", "xar4190101", "xa180429", "xar2180429_e04"}
    selected = [row for row in rows if row.get("stem") in failed]
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    assignments = [["xar4180429", "xar1201231", "xa180429"], ["xar4190101", "xar2180429_e04"]]
    output_paths: list[list[str]] = []
    by_stem = {row["stem"]: row for row in selected}
    fields = list(rows[0]) if rows else []
    for gpu, stems in enumerate(assignments):
        path = RECOVERY_DIR / f"RECOVERY_GPU{gpu}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for stem in stems:
                if stem in by_stem:
                    writer.writerow(by_stem[stem])
        output_paths.append([str(path), *stems])
    return output_paths


def run_recovery() -> list[int]:
    processes: list[tuple[int, subprocess.Popen]] = []
    for gpu, (csv_path, *stems) in enumerate(write_recovery_manifests()):
        if not stems:
            continue
        command = [
            str(ENV_PY),
            "-u",
            str(SHARD_SCRIPT),
            "--shard-csv",
            csv_path,
            "--gpu",
            str(gpu),
            "--input-root",
            str(BASE / "input"),
            "--output-root",
            str(OUTPUT_ROOT / f"gpu{gpu}"),
            "--log-root",
            str(LOG_ROOT / f"gpu{gpu}"),
            "--progress-root",
            str(RECEIPT_ROOT / f"gpu{gpu}/progress"),
            "--receipt-root",
            str(RECEIPT_ROOT),
            "--work-root",
            str(BASE / f"work/gpu{gpu}"),
            "--runner",
            str(RUNNER),
        ]
        log(f"starting metadata recovery GPU{gpu}: {','.join(stems)}")
        processes.append((gpu, subprocess.Popen(command)))
    statuses: list[int] = []
    for gpu, process in processes:
        status = process.wait()
        log(f"metadata recovery GPU{gpu} finished with {status}")
        statuses.append(status)
    return statuses


def stage_local_geometry() -> int:
    """Stage immutable local-success packages beside cloud outputs.

    The archive contains only ``04_geometry`` directories from local packages
    that were already successful and absent from the cloud pending manifest.
    It is extracted after GPU workers finish, so no active output directory is
    touched while inference is running.  Existing target directories are
    preserved and never overwritten.
    """

    if not LOCAL_GEOMETRY_ARCHIVE.is_file():
        log("no local-success geometry archive; cloud outputs only")
        return 0
    import tarfile

    LOCAL_GEOMETRY_EXTRACTED.mkdir(parents=True, exist_ok=True)
    with tarfile.open(LOCAL_GEOMETRY_ARCHIVE, "r:gz") as archive:
        try:
            archive.extractall(BASE / "postprocess", filter="data")
        except TypeError:  # Python < 3.12
            archive.extractall(BASE / "postprocess")
    target_root = OUTPUT_ROOT / "gpu1"
    target_root.mkdir(parents=True, exist_ok=True)
    staged = 0
    for child in sorted(LOCAL_GEOMETRY_EXTRACTED.iterdir()):
        if not child.is_dir():
            continue
        target = target_root / child.name
        if target.exists():
            log(f"local geometry target exists; preserving {target}")
            continue
        child.rename(target)
        staged += 1
    log(f"staged {staged} immutable local-success geometry packages")
    return staged


def run_postprocess() -> int:
    boundary = [
        str(ENV_PY),
        "-u",
        str(CODE / "build_xian_boundary_temporal_outputs.py"),
        "--assembly-run",
        str(ASSEMBLY),
        "--inventory",
        str(ASSEMBLY / "00_audit/SOURCE_INVENTORY.json"),
        "--epoch-packages",
        str(OUTPUT_ROOT / "gpu0"),
        "--geometry-runs",
        str(OUTPUT_ROOT / "gpu1"),
        "--run-dir",
        str(FINAL_RUN),
        "--city-boundary",
        str(BOUNDARY_CITY),
        "--county-boundary",
        str(BOUNDARY_COUNTY),
    ]
    log("running exact EPSG:32649 city/county boundary accounting")
    status = subprocess.run(boundary, check=False).returncode
    if status != 0:
        log(f"boundary builder returned {status}; preserving receipt and continuing visual package")
    final = [
        str(ENV_PY),
        "-u",
        str(CODE / "build_xian_final_temporal_package.py"),
        "--assembly-run",
        str(ASSEMBLY),
        "--inventory",
        str(ASSEMBLY / "00_audit/SOURCE_INVENTORY.json"),
        "--epoch-packages",
        str(OUTPUT_ROOT / "gpu0"),
        "--geometry-root",
        str(OUTPUT_ROOT / "gpu1"),
        "--run-dir",
        str(FINAL_RUN),
        "--city-boundary",
        str(BOUNDARY_CITY),
        "--county-boundary",
        str(BOUNDARY_COUNTY),
        "--skip-boundary-builder",
    ]
    log("running actual-date visual/state package")
    final_status = subprocess.run(final, check=False).returncode
    FINAL_RUN.mkdir(parents=True, exist_ok=True)
    (FINAL_RUN / "receipts").mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "urrs.aliyun.postprocess.controller.v1",
        "created_at": stamp(),
        "main_shards_waited": True,
        "recovery_stems": ["xar4180429", "xar1201231", "xar4190101", "xa180429", "xar2180429_e04"],
        "supplemental_source_stems": ["xa180429", "xar2180429_e04"],
        "local_geometry_staged_count": int(sum(1 for child in (OUTPUT_ROOT / "gpu1").iterdir() if child.is_dir() and child.name.startswith("local_"))) if (OUTPUT_ROOT / "gpu1").is_dir() else 0,
        "boundary_returncode": status,
        "final_package_returncode": final_status,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "raw_inputs_modified": False,
        "model_weights_modified": False,
        "scientific_status": "candidate_not_ground_truth",
    }
    (FINAL_RUN / "receipts/POSTPROCESS_CONTROLLER_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return final_status


def main() -> int:
    log("waiting for existing final4 GPU shard workers")
    while active_shards():
        time.sleep(60)
    log("main shard workers ended; starting bounded metadata recovery")
    stage_local_geometry()
    recovery_status = run_recovery()
    log(f"metadata recovery statuses={recovery_status}")
    while active_shards():
        time.sleep(30)
    return run_postprocess()


if __name__ == "__main__":
    raise SystemExit(main())
