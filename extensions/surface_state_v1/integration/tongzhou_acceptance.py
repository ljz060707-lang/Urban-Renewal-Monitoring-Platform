from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")
REQUIRED_GEOMETRY = (
    "building_probability",
    "building_mask",
    "building_boundary",
)


def _scene_alignment(audit_csv: Path) -> dict:
    if not audit_csv.exists():
        return {
            "status": "missing_scene_audit",
            "alignment_status": "not_checked",
            "scenes": [],
        }
    rows = list(csv.DictReader(audit_csv.open(encoding="utf-8-sig")))
    fields = ("width", "height", "crs", "resolution_x", "resolution_y", "bounds")
    reference = rows[0] if rows else {}
    differences = []
    for row in rows[1:]:
        changed = {
            field: [reference.get(field), row.get(field)]
            for field in fields
            if reference.get(field) != row.get(field)
        }
        if changed:
            differences.append({"file": row.get("file") or row.get("filename"), "differences": changed})
    return {
        "status": "pass" if rows and not differences else "failed",
        "alignment_status": (
            "aligned"
            if rows and not differences
            else "registration_required_before_temporal_inference"
        ),
        "scene_count": len(rows),
        "differences": differences,
        "scenes": rows,
    }


def audit_tongzhou(project: str | Path, output: str | Path | None = None) -> dict:
    project = Path(project)
    geometry_root = project / "outputs" / "inference" / "beijing_tongzhou_building_geometry"
    source_root = project / "outputs" / "inference" / "beijing_tongzhou_to_latest"
    if output is None:
        output = (
            project
            / "outputs"
            / "inference"
            / "beijing_tongzhou_surface_state_v1"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    files = list(geometry_root.rglob("*")) if geometry_root.exists() else []
    relative = [str(path.relative_to(geometry_root)) for path in files if path.is_file()]
    readiness = {}
    for date in DATES:
        tokens = (date, date.replace("-", ""), date[2:].replace("-", ""))
        date_files = [name for name in relative if any(token in name for token in tokens)]
        readiness[date] = {
            key: any(key in name.casefold() for name in date_files)
            for key in REQUIRED_GEOMETRY
        }
        readiness[date]["gpkg"] = any(name.casefold().endswith(".gpkg") for name in date_files)
        readiness[date]["manifest"] = any("manifest" in name.casefold() for name in date_files)
    complete = all(all(values.values()) for values in readiness.values())
    alignment = _scene_alignment(source_root / "TONGZHOU_TIF_AUDIT.csv")
    status = (
        "ready"
        if complete and alignment["alignment_status"] == "aligned"
        else (
            "blocked_by_missing_geometry_outputs"
            if not complete
            else "registration_required_before_temporal_inference"
        )
    )
    manifest = {
        "tongzhou_overlay_status": status,
        "geometry_root": str(geometry_root.resolve()),
        "geometry_file_count": len(relative),
        "geometry_readiness": readiness,
        "dates": list(DATES),
        "alignment_status": alignment["alignment_status"],
        "checkpoint_selection_used_tongzhou": False,
        "threshold_selection_used_tongzhou": False,
        "mti_calibration_used_tongzhou": False,
        "predictions_are_ground_truth": False,
        "full20_retrained": False,
        "rsbuilding_retrained": False,
        "whu_test_reads": 0,
        "second_test_reads": 0,
        "visual_review_packet": "generated_blocking_audit",
        "human_visual_review": "pending_user_review",
        "inference_ready": False,
        "executed_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "TONGZHOU_ALIGNMENT_AUDIT.json").write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "TONGZHOU_SURFACE_STATE_NUMERICAL_AUDIT.json").write_text(
        json.dumps(
            {
                "status": "not_run",
                "reason": status,
                "all_empty_check": "not_run",
                "all_full_check": "not_run",
                "class_collapse_check": "not_run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "TONGZHOU_SURFACE_STATE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = "".join(
        f"<tr><td>{date}</td>"
        + "".join(f"<td>{'yes' if value else 'missing'}</td>" for value in values.values())
        + "</tr>"
        for date, values in readiness.items()
    )
    (output / "TONGZHOU_REVIEW_PACKET.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>通州 SurfaceState V1 验收</title>"
        "<style>body{font-family:system-ui;margin:32px}td,th{padding:8px;border:1px solid #aaa}"
        "table{border-collapse:collapse}.blocked{color:#b00020}</style>"
        f"<h1>通州 SurfaceState V1 验收</h1><p class='blocked'>状态：{status}</p>"
        "<p>没有进行 checkpoint、阈值或 MTI 参数选择；预测不是真值。</p>"
        "<table><tr><th>日期</th><th>probability</th><th>mask</th><th>boundary</th>"
        f"<th>GPKG</th><th>manifest</th></tr>{rows}</table>",
        encoding="utf-8",
    )
    return {"output": str(output.resolve()), **manifest}

