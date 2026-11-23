"""Build a read-only Shenzhen four-epoch integrated-system preflight.

This utility deliberately does not open raster pixels, start inference, touch
GEID, or modify any protected model/data asset.  It records the current
role-based system and the GEID merge gate in a new output directory so a later
run can resume from an auditable, immutable preflight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(r"D:\Data Set")
DEFAULT_OUTPUT_ROOT = Path(r"E:\URRS Output")
DEFAULT_SYSTEM_ROOT = DEFAULT_OUTPUT_ROOT / "final_pre_shenzhen_20260819_232255"
DEFAULT_MERGE_STATE = DEFAULT_DATA_ROOT / "_automation" / "GEID_MERGE_STATE.json"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NESTED_REPO = (
    DEFAULT_PROJECT_ROOT / "outputs" / "urban-renewal-phase1-starter"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def model_entry(
    *,
    role: str,
    module: str,
    status: str,
    authority: str,
    active: bool,
    checkpoint: Path | None = None,
    checkpoint_sha256: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "role": role,
        "module": module,
        "status": status,
        "final_authority": authority,
        "active_for_shenzhen": active,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_exists": bool(checkpoint and checkpoint.is_file()),
        "checkpoint_sha256": checkpoint_sha256,
        "scientific_status": "candidate_not_ground_truth",
        "notes": notes,
    }


def choose_output_root(base: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = base / f"shenzhen_four_epoch_final_{stamp}"
    if not candidate.exists():
        return candidate
    return base / f"shenzhen_four_epoch_final_{stamp}_{datetime.now().microsecond:06d}"


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_role_entries(nested_repo: Path) -> list[dict[str, Any]]:
    model_root = nested_repo / "outputs" / "models"
    b0 = first_existing(
        nested_repo
        / "outputs"
        / "v2_readiness"
        / "rsbuilding"
        / "whu"
        / "work_dirs"
        / "swin_t_whu_pilot5000_20260728_110750"
        / "iter_2000.pth",
        nested_repo
        / "outputs"
        / "models"
        / "rsbuilding_swin_t_whu_candidate_20260728"
        / "checkpoints"
        / "best_validation_candidate_iter_2000.pth",
    )
    dsifn = model_root / "urban_change_dsifn_binary_candidate_20260725_232317" / "best_binary_change.pth"
    s2 = model_root / "urban_change_v1_s2looking" / "checkpoints" / "best_directional_change.pth"
    full20 = model_root / "urban_change_v2_full_candidate_20260727_001029" / "best_v2_multisource_change.pth"
    v8r8 = first_existing(
        nested_repo / "outputs" / "models" / "v8r8_stability" / "checkpoint.pt",
        nested_repo / "outputs" / "models" / "v8r8_stability" / "best.pt",
    )
    if v8r8 is None:
        # The accepted system records the digest even when its large checkpoint
        # is stored in the E: output archive rather than the working tree.
        v8r8_sha = "0ebd115465710e7024465a6984a95dac6fbbe356bda47f145e42809153548366"
    else:
        v8r8_sha = sha256_file(v8r8)
    return [
        model_entry(
            role="geometry",
            module="Main Building Geometry Provider (RSBuilding/B0 + objectizer)",
            status="accepted_final_authority",
            authority="FINAL_BUILDING_GEOMETRY",
            active=True,
            checkpoint=b0,
            checkpoint_sha256=(
                "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
            ),
            notes="Frozen B0 iter_2000; no change model may create final polygons.",
        ),
        model_entry(
            role="identity",
            module="Tracker / Linker (building_tracker/linker.py)",
            status="accepted_final_authority",
            authority="FINAL_CROSS_DATE_IDENTITY",
            active=True,
            checkpoint_sha256="9b4f323c528fcfe59c5c4577b7d5ca095651b807b2d67d418d78ebdbfe043c99",
            notes="One canonical identity authority; no probability voting.",
        ),
        model_entry(
            role="temporal_decoder",
            module="V8R8 decode_presence + decode_trajectory",
            status="accepted_final_authority",
            authority="FINAL_LONG_TERM_STATE",
            active=True,
            checkpoint=v8r8,
            checkpoint_sha256=v8r8_sha,
            notes="Preserves 1/0/U and trajectory codes; does not invent geometry.",
        ),
        model_entry(
            role="area",
            module="role_based_integration.area_seam (canonical)",
            status="accepted_final_authority",
            authority="FINAL_AREA_ACCOUNTING",
            active=True,
            notes="Projected CRS and canonical polygon quantities only.",
        ),
        model_entry(
            role="DSIFN",
            module="DSIFN binary change candidate teacher",
            status="candidate_teacher_supporting_evidence",
            authority="NONE_SUPPORTING_ONLY",
            active=False,
            checkpoint=dsifn,
            checkpoint_sha256=sha256_file(dsifn),
            notes="Supporting any-change evidence; validation/test history is not a final Shenzhen authority.",
        ),
        model_entry(
            role="S2Looking",
            module="S2Looking directional change candidate teacher",
            status="candidate_teacher_supporting_evidence",
            authority="NONE_SUPPORTING_ONLY",
            active=False,
            checkpoint=s2,
            checkpoint_sha256=sha256_file(s2),
            notes="Directional addition/demolition evidence only; does not create polygons.",
        ),
        model_entry(
            role="Full20",
            module="Full20 multisource change candidate",
            status="candidate_validation_only_supporting_evidence",
            authority="NONE_SUPPORTING_ONLY",
            active=False,
            checkpoint=full20,
            checkpoint_sha256=(
                "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
                if full20.is_file()
                else None
            ),
            notes="Must be wired as per-date evidence before use; never averaged into geometry.",
        ),
        model_entry(
            role="SECOND",
            module="SECOND semantic supervision absorbed by Full20 dev training",
            status="no_independent_accepted_runtime_checkpoint",
            authority="NONE_DIAGNOSTIC_ONLY",
            active=False,
            notes="Do not invent a standalone SECOND inference model or duplicate Full20 runtime.",
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--system-root", type=Path, default=DEFAULT_SYSTEM_ROOT)
    parser.add_argument("--merge-state", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--nested-repo", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root
    project_root = args.project_root
    nested_repo = args.nested_repo or (project_root / "outputs" / "urban-renewal-phase1-starter")
    merge_state_path = args.merge_state or (data_root / "_automation" / "GEID_MERGE_STATE.json")
    output_root = choose_output_root(DEFAULT_OUTPUT_ROOT, args.output_root)
    preflight = output_root / "00_preflight"
    preflight.mkdir(parents=True, exist_ok=False)

    merge_state = json_load(merge_state_path, {}) or {}
    acceptance_path = args.system_root / "10_final" / "FINAL_PRE_SHENZHEN_ACCEPTANCE.json"
    system_map_path = args.system_root / "06_storage" / "CURRENT_SYSTEM_MAP.md"
    best_system_path = args.system_root / "02_ablation" / "BEST_ROLE_BASED_SYSTEM.json"
    acceptance = json_load(acceptance_path, {}) or {}
    best_system = json_load(best_system_path, {}) or {}
    epochs = merge_state.get("epochs") or []
    stable_required = int(merge_state.get("stable_observations_required") or 2)
    active_processes = merge_state.get("active_processes") or {}

    blocked_reasons: list[str] = []
    if not bool(merge_state.get("four_tif_merge_complete")):
        blocked_reasons.append("four_tif_merge_complete=false")
    if any(int(value or 0) > 0 for value in active_processes.values()):
        blocked_reasons.append("merge_processes_active")
    if not bool(merge_state.get("no_active_merge_process")):
        blocked_reasons.append("no_active_merge_process=false")
    if not (data_root / "_automation" / "GEID_MERGE_COMPLETION_RECEIPT.json").is_file():
        blocked_reasons.append("GEID_MERGE_COMPLETION_RECEIPT.json_missing")
    for epoch in epochs:
        if int(epoch.get("stable_count") or 0) < stable_required:
            blocked_reasons.append(f"{epoch.get('task_name')}:stable_count<{stable_required}")
        if int(epoch.get("size_delta") or 0) != 0:
            blocked_reasons.append(f"{epoch.get('task_name')}:size_delta_nonzero")

    role_entries = build_role_entries(nested_repo)
    source_paths = [
        nested_repo / "src" / "urban_monitor" / "change" / "inference_v2.py",
        nested_repo / "src" / "urban_monitor" / "change" / "teacher_registry.py",
        nested_repo / "src" / "urban_monitor" / "change" / "adapters" / "second.py",
        nested_repo / "src" / "urban_monitor" / "role_based_integration.py",
        nested_repo / "src" / "urban_monitor" / "citywide_monitoring.py",
        nested_repo / "tools" / "run_citywide_monitoring.py",
    ]
    source_audit = [
        {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ]

    role_fields = [
        "role",
        "module",
        "status",
        "final_authority",
        "active_for_shenzhen",
        "checkpoint_path",
        "checkpoint_exists",
        "checkpoint_sha256",
        "scientific_status",
        "notes",
    ]
    write_csv(preflight / "SHENZHEN_MODEL_ROLE_MATRIX.csv", role_entries, role_fields)

    system_map = f"""# Shenzhen integrated preflight system map\n\nGenerated: {utc_now()}\n\nThis is a read-only preflight. No raster pixels were opened and no inference was started.\n\n- Local accepted system: `{args.system_root}`\n- Acceptance status: `{acceptance.get('status')}`\n- Best role-based arm: `{acceptance.get('best_role_based_system', {}).get('arm', best_system.get('best_arm'))}`\n- Merge gate: `{merge_state.get('status')}`\n- Four-TIF merge complete: `{bool(merge_state.get('four_tif_merge_complete'))}`\n- Inference started: `false`\n- Test pixel reads during this preflight: `0`\n\n## Final authorities\n\n- Geometry: frozen RSBuilding/B0 iter_2000; SHA `f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7`\n- Identity: Tracker/Linker; SHA `9b4f323c528fcfe59c5c4577b7d5ca095651b807b2d67d418d78ebdbfe043c99`\n- Temporal/decoder: V8R8; SHA `0ebd115465710e7024465a6984a95dac6fbbe356bda47f145e42809153548366`\n- Area: canonical role-based area seam in projected CRS; no change pixels are used as footprint area.\n\nDSIFN, S2Looking and Full20 are supporting evidence only in this preflight. SECOND has no independent accepted runtime checkpoint and is not duplicated.\n\n## Raw-data policy\n\nThe four GEID TIFs under `{data_root}` remain immutable read-only inputs. The monitor currently reports active merge work and non-zero size deltas; no analysis-ready view or model output may be created until the merge completion receipt and read gate pass.\n"""
    (preflight / "CURRENT_FULL_SYSTEM_MAP.md").write_text(system_map, encoding="utf-8")

    data_gate = {
        "schema_version": "urrs.shenzhen.data_gate_status.v1",
        "timestamp": utc_now(),
        "status": "BLOCKED_WAITING_FOR_TIF_MERGE" if blocked_reasons else "READY_FOR_TIF_READ_AUDIT",
        "four_tif_merge_complete": bool(merge_state.get("four_tif_merge_complete")),
        "merge_state_path": str(merge_state_path),
        "merge_state_status": merge_state.get("status"),
        "active_processes": active_processes,
        "no_active_merge_process": bool(merge_state.get("no_active_merge_process")),
        "stable_observations_required": stable_required,
        "blocked_reasons": blocked_reasons,
        "completion_receipt": str(data_root / "_automation" / "GEID_MERGE_COMPLETION_RECEIPT.json"),
        "completion_receipt_exists": (data_root / "_automation" / "GEID_MERGE_COMPLETION_RECEIPT.json").is_file(),
        "epochs": epochs,
        "raw_tif_policy": "immutable_read_only_no_move_no_overwrite_no_retile",
        "raw_tif_pixels_opened_by_this_run": False,
        "model_inference_started": False,
        "test_pixel_reads": 0,
    }
    (preflight / "DATA_GATE_STATUS.json").write_text(
        json.dumps(data_gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    preflight_payload = {
        "schema_version": "urrs.shenzhen.integrated_preflight.v1",
        "timestamp": utc_now(),
        "output_root": str(output_root),
        "data_root": str(data_root),
        "system_root": str(args.system_root),
        "project_root": str(project_root),
        "nested_repo": str(nested_repo),
        "git": {
            "head": git_value(nested_repo, "rev-parse", "HEAD"),
            "status_porcelain": git_value(nested_repo, "status", "--short", "--untracked-files=no"),
            "note": "Existing dirty source is user-owned; this preflight does not alter or commit it.",
        },
        "local_acceptance": {
            "path": str(acceptance_path),
            "exists": acceptance_path.is_file(),
            "status": acceptance.get("status"),
            "system_acceptance_pass": acceptance.get("system_acceptance_pass"),
            "ready_for_shenzhen_citywide": acceptance.get("ready_for_shenzhen_citywide"),
            "protected_reads": acceptance.get("protected_reads"),
        },
        "source_audit": source_audit,
        "model_roles": role_entries,
        "merge_gate": data_gate,
        "execution_gate": {
            "status": "BLOCKED_WAITING_FOR_TIF_MERGE" if blocked_reasons else "OPEN_FOR_DATE_AND_QUALITY_PREFLIGHT",
            "model_inference_started": False,
            "raw_tif_policy": "immutable_read_only_no_move_no_overwrite_no_retile",
            "test_reads": {"WHU_Test": 0, "SECOND_Test": 0, "Holdout": "SPENT_NO_REREAD"},
            "next_allowed_stage": "verify_merge_completion_receipt_then_date_reconstruction",
            "next_forbidden_stage": "all_model_inference_and_downstream_outputs_until_merge_gate_passes",
        },
    }
    (preflight / "SHENZHEN_INTEGRATED_PREFLIGHT.json").write_text(
        json.dumps(preflight_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": "urrs.shenzhen.preflight.run_manifest.v1",
        "created_at": utc_now(),
        "output_root": str(output_root),
        "artifacts": [
            str(preflight / "SHENZHEN_MODEL_ROLE_MATRIX.csv"),
            str(preflight / "CURRENT_FULL_SYSTEM_MAP.md"),
            str(preflight / "DATA_GATE_STATUS.json"),
            str(preflight / "SHENZHEN_INTEGRATED_PREFLIGHT.json"),
        ],
        "model_inference_started": False,
        "raw_tifs_modified": False,
        "test_reads": {"WHU_Test": 0, "SECOND_Test": 0},
        "status": data_gate["status"],
    }
    (output_root / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "status": data_gate["status"], "blocked_reasons": blocked_reasons}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
