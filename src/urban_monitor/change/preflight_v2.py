from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import hashlib

from .teacher_registry import TeacherStatus, build_teacher_registry


def run_v2_preflight(config: dict, output_path: str | Path | None = None) -> dict:
    registry = build_teacher_registry(config)
    teachers = registry.statuses()
    dsifn_ready = teachers["dsifn"]["status"] == TeacherStatus.READY.value
    s2_ready = teachers["s2looking"]["status"] == TeacherStatus.READY.value
    second_data = teachers["second"]["status"] in {
        TeacherStatus.READY.value,
        TeacherStatus.PARTIAL_READY.value,
    }
    second_mapping = bool(config.get("datasets", {}).get("second", {}).get("mapping_verified", False))
    rs_ready = teachers["rsbuilding"]["status"] in {TeacherStatus.READY.value, TeacherStatus.EXTERNAL_READY.value}
    holdout_path = Path(config.get("holdout", {}).get("second_protocol", "outputs/v2_readiness/V2_HOLDOUT_PROTOCOL.json"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.is_file() else {"status": "missing"}
    cuda: dict[str, object] = {"available": False}
    try:
        import torch
        cuda = {"available": bool(torch.cuda.is_available()), "torch_version": torch.__version__}
        if torch.cuda.is_available():
            cuda.update({"gpu_name": torch.cuda.get_device_name(0), "vram_bytes": int(torch.cuda.get_device_properties(0).total_memory)})
    except Exception as exc:  # preflight must remain diagnostic on minimal installs
        cuda["error"] = str(exc)
    readiness = {
        "READY_FOR_PARTIAL_INFERENCE": bool(dsifn_ready or s2_ready),
        "READY_FOR_V2_QUICK": bool(dsifn_ready and s2_ready),
        "READY_FOR_V2_FULL": bool(dsifn_ready and s2_ready and second_data and second_mapping),
        "READY_FOR_FULL_FUSION": bool(dsifn_ready and s2_ready and second_data and second_mapping and rs_ready),
    }
    report = {
        "status": "ready" if readiness["READY_FOR_PARTIAL_INFERENCE"] else "blocked",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "teachers": teachers,
        "readiness": readiness,
        "cuda": cuda,
        "holdout_protocol": holdout,
        "test_images_read": 0,
        "warnings": [],
    }
    if not second_data:
        report["warnings"].append("SECOND dataset missing; semantic outputs will be PENDING")
    elif not second_mapping:
        report["warnings"].append("SECOND mapping pending; semantic supervision is allowed, building direction is disabled")
    if not rs_ready:
        report["warnings"].append("RSBuilding export unavailable; building geometry will be PENDING")
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def format_v2_preflight(report: dict) -> str:
    lines = []
    for name, item in report["teachers"].items():
        lines += [
            name.upper(),
            f"  status: {item['status'].upper()}",
            f"  role: {item['role']}",
            f"  dataset: {item.get('dataset_root') or 'not configured'}",
            f"  checkpoint: {item.get('checkpoint') or 'not configured'}",
        ]
    lines.append("")
    lines.extend(f"{name} = {str(value).lower()}" for name, value in report["readiness"].items())
    lines.append(f"test_images_read = {report['test_images_read']}")
    if report["warnings"]:
        lines += ["", "Warnings:"] + [f"- {value}" for value in report["warnings"]]
    return "\n".join(lines)
