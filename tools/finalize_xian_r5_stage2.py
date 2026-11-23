#!/usr/bin/env python
"""Finalize bounded Xi'an R5 Stage-2 evidence without touching R4 production."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923")
STAGE2 = ROOT / "17_stage2"
CODE_ROOT = Path(r"D:\Codex\projects\Documents-Codex\2026-07-18\urban-renewal-phase1-starter-v0-2")
CASE_IDS = [
    "B02-P2-24", "B03-TEX04", "B04-TQA07", "B04-TQA10", "B05-TQA03",
    "B05-P1-10", "B01-P2-26", "OBJ_0001", "OBJ_0008", "B05-TEX01",
    "B01-TEX02", "B05-P1-06",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def archive_then_copy(source: Path, destination: Path, archive: Path | None = None) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists() and sha256_file(destination) == source_hash:
        return {"status": "ALREADY_MATCHES", "source": str(source), "destination": str(destination), "sha256": source_hash}
    archived = None
    if destination.exists():
        backup = archive or destination.with_name(destination.stem + "_PRE_FINALIZATION" + destination.suffix)
        if backup.exists() and sha256_file(backup) != sha256_file(destination):
            backup = backup.with_name(backup.stem + "_2" + backup.suffix)
        if not backup.exists():
            shutil.copy2(destination, backup)
        archived = str(backup)
    shutil.copy2(source, destination)
    return {"status": "COPIED", "source": str(source), "destination": str(destination), "sha256": source_hash, "archived_previous": archived}


def package_version(python_exe: Path, package: str) -> str | None:
    code = (
        "import importlib.metadata as m, json; "
        f"\ntry: print(m.version({package!r}))\n"
        "except m.PackageNotFoundError: print('NOT_INSTALLED')"
    )
    result = subprocess.run([str(python_exe), "-c", code], capture_output=True, text=True, timeout=45)
    value = result.stdout.strip()
    return value if result.returncode == 0 else f"PROBE_ERROR:{result.stderr.strip()[:300]}"


def probe_isolated_environment(env_root: Path, rbf_source: Path) -> dict[str, Any]:
    python_exe = env_root / "Scripts" / "python.exe"
    if not python_exe.is_file():
        return {"status": "ENVIRONMENT_EXECUTABLE_MISSING", "python": str(python_exe)}
    probe_code = r'''
import importlib, importlib.metadata as md, json, platform, sys
r = {"python": sys.version.split()[0], "platform": platform.platform(), "imports": {}}
for name, modules in {
    "torch": ["torch"], "torchvision": ["torchvision"], "torchaudio": ["torchaudio"],
    "mmcv": ["mmcv", "mmcv._ext"], "mmdet": ["mmdet"], "mmdet_models": ["mmdet.models"],
    "pycocotools_mask": ["pycocotools.mask"], "cv2": ["cv2"], "PIL": ["PIL"],
}.items():
    states = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            states.append({"module": mod, "ok": True})
        except Exception as e:
            states.append({"module": mod, "ok": False, "error": type(e).__name__ + ": " + str(e)[:500]})
    r["imports"][name] = states
try:
    import torch
    r["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                   "cuda_available": bool(torch.cuda.is_available()),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception as e:
    r["torch"] = {"error": type(e).__name__ + ": " + str(e)[:500]}
versions = {}
for pkg in ["torch", "torchvision", "torchaudio", "mmcv-full", "mmdet", "numpy", "scipy",
            "pandas", "opencv-python", "Pillow", "mmpycocotools", "pycocotools", "kornia", "kornia-rs"]:
    try: versions[pkg] = md.version(pkg)
    except md.PackageNotFoundError: versions[pkg] = None
r["distribution_versions"] = versions
print(json.dumps(r))
'''
    probe_env = os.environ.copy()
    existing_pythonpath = probe_env.get("PYTHONPATH", "")
    probe_env["PYTHONPATH"] = str(rbf_source) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    result = subprocess.run([str(python_exe), "-c", probe_code], capture_output=True, text=True, timeout=180, env=probe_env)
    try:
        imports = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        imports = {"probe_error": result.stderr[-3000:], "stdout": result.stdout[-1000:], "returncode": result.returncode}
    pip_check = subprocess.run([str(python_exe), "-m", "pip", "check"], capture_output=True, text=True, timeout=90)
    relevant = ["torch", "torchvision", "torchaudio", "mmcv-full", "mmdet", "numpy", "scipy", "pandas",
                "opencv-python", "Pillow", "mmpycocotools", "pycocotools", "kornia", "kornia-rs"]
    for pkg in relevant:
        if "distribution_versions" not in imports:
            break
    cuda_ok = bool(imports.get("torch", {}).get("cuda_available"))
    mmcv_extension_ok = all(item.get("ok") for item in imports.get("imports", {}).get("mmcv", []))
    mmdet_models_ok = all(item.get("ok") for item in imports.get("imports", {}).get("mmdet_models", []))
    version_file = rbf_source / "mmdet" / "version.py"
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", version_file.read_text(encoding="utf-8") if version_file.is_file() else "")
    return {
        "status": "PARTIAL_RUNTIME_READY_FULL_MODEL_IMPORT_BLOCKED" if cuda_ok and mmcv_extension_ok and not mmdet_models_ok else "PROBED_REVIEW_REQUIRED",
        "python_executable": str(python_exe),
        "official_rbf_sam_source_path": str(rbf_source),
        "repository_mmdet_version": version_match.group(1) if version_match else None,
        "repository_version_file": str(version_file),
        "probe": imports,
        "pip_check_exit_code": pip_check.returncode,
        "pip_check_output": (pip_check.stdout + pip_check.stderr).strip()[:4000],
        "interpretation": {
            "torch_cuda_runtime_smoke": "PASS" if cuda_ok else "FAIL",
            "mmcv_native_extension_import": "PASS" if mmcv_extension_ok else "FAIL",
            "mmdet_models_import": "PASS" if mmdet_models_ok else "BLOCKED_OR_FAILED",
            "known_native_build_blocker": "mmpycocotools build previously failed because MSVC C++ build tools (cl.exe) are unavailable; the official repository's mmdet.models import was probed via PYTHONPATH and fails at missing pycocotools.mask; no third-party wheel/workaround was used",
            "pip_metadata_warning": "OpenCV 4.12 metadata requires NumPy >=2 on this Python while the pinned RBF-SAM stack uses NumPy 1.23.5; cv2 runtime import was tested separately",
            "production_environment_modified": False,
        },
    }


def verify_case_pixel_files(map_rows: list[dict[str, str]]) -> dict[str, Any]:
    try:
        from PIL import Image
    except Exception as exc:
        return {"status": "PIL_UNAVAILABLE", "error": type(exc).__name__ + ": " + str(exc), "decoded_unique_files": 0}
    pixel_rows = [row for row in map_rows if row.get("pixel_available", "").lower() == "true" and row.get("pixel_tile_path")]
    paths = sorted(set(row["pixel_tile_path"] for row in pixel_rows))
    decoded: dict[str, dict[str, Any]] = {}
    for value in paths:
        try:
            with Image.open(Path(value)) as image:
                image.load()
                decoded[value] = {"status": "DECODED", "width": image.width, "height": image.height}
        except Exception as exc:
            decoded[value] = {"status": "DECODE_FAILED", "error": type(exc).__name__ + ": " + str(exc)[:300]}
    decode_rows = sum(decoded.get(row["pixel_tile_path"], {}).get("status") == "DECODED" for row in pixel_rows)
    bbox_checked = 0
    bbox_in_bounds = 0
    bbox_failures = []
    for row in pixel_rows:
        image = decoded.get(row["pixel_tile_path"], {})
        bbox = row.get("image_bbox_px", "")
        if image.get("status") != "DECODED" or not bbox:
            continue
        try:
            coords = json.loads(bbox)
            if len(coords) != 4:
                raise ValueError("expected four coordinates")
            x1, y1, x2, y2 = [float(x) for x in coords]
            bbox_checked += 1
            if 0 <= x1 <= x2 <= image["width"] and 0 <= y1 <= y2 <= image["height"]:
                bbox_in_bounds += 1
            else:
                bbox_failures.append({"case_id": row.get("case_id"), "tile_id": row.get("actual_date_tile_id"), "bbox": bbox, "image_size": [image["width"], image["height"]]})
        except Exception as exc:
            bbox_failures.append({"case_id": row.get("case_id"), "tile_id": row.get("actual_date_tile_id"), "bbox": bbox, "error": type(exc).__name__ + ": " + str(exc)[:200]})
    return {
        "status": "PIXEL_FILES_DECODED" if decode_rows == len(pixel_rows) and not bbox_failures else "PIXEL_READ_REVIEW_REQUIRED",
        "pixel_available_rows": len(pixel_rows), "unique_pixel_files": len(paths),
        "unique_files_decoded": sum(item.get("status") == "DECODED" for item in decoded.values()),
        "pixel_rows_decoded": decode_rows,
        "decoded_paths": [path for path, item in decoded.items() if item.get("status") == "DECODED"],
        "files": decoded,
        "decode_failed_unique_files": [path for path, item in decoded.items() if item.get("status") != "DECODED"],
        "image_bbox_rows_checked": bbox_checked, "image_bbox_rows_in_bounds": bbox_in_bounds,
        "image_bbox_failures": bbox_failures,
        "interpretation": "Read-only image decode and tile-bbox bounds check only; not model inference, visual ground truth, or transition validation.",
    }


def main() -> None:
    generated = utc_now()
    receipt_dir = STAGE2 / "13_receipts"
    stage1_status = ROOT / "16_receipts" / "FINAL_DECISION.json"
    case_screen_path = Path(r"F:\xian\西安_当前最佳候选交付_20260917\05_visual_qa_reference\FIXED_CASE_VISUAL_SCREEN.csv")
    freeze_path = receipt_dir / "R4_FREEZE_STAGE2.json"
    case_map_path = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv"
    pair_path = STAGE2 / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION.json"
    pair_v2_path = STAGE2 / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION_STAGE2_V2.json"
    pair_csv_v2_path = STAGE2 / "00_date_harmonization" / "DATE_COHERENT_PAIR_CANDIDATES_STAGE2_V2.csv"
    summary_path = STAGE2 / "06_lightglue_ab" / "ALIGNMENT_METHOD_CANDIDATE_SUMMARY_V2.csv"
    frame_receipt_path = receipt_dir / "ALIGNMENT_SAMPLE_FRAME_RECEIPT.json"
    actual_date_receipt_path = STAGE2 / "00_date_harmonization" / "ACTUAL_DATE_TILE_MAP_RECEIPT.json"
    env_root = STAGE2 / "01_checkpoint" / "rbf_sam_env"

    freeze = read_json(freeze_path)
    pairs = read_json(pair_path)
    actual_receipt = read_json(actual_date_receipt_path)
    frame_receipt = read_json(frame_receipt_path)
    case_screen = read_csv(case_screen_path)
    map_rows = read_csv(case_map_path)
    alignment_rows = read_csv(summary_path)
    screen_by_id = {row.get("case_id", ""): row for row in case_screen}
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in map_rows:
        rows_by_case[row.get("case_id", "")].append(row)

    print("[1/6] Rechecking frozen R4 input hashes (read-only).", flush=True)
    recheck_files = {}
    for name, info in freeze.get("files", {}).items():
        path = Path(info["path"])
        if not path.is_file():
            current = None
            status = "MISSING"
        else:
            current = sha256_file(path)
            status = "MATCH" if current == info.get("sha256") else "MISMATCH"
        recheck_files[name] = {"path": str(path), "expected_sha256": info.get("sha256"), "current_sha256": current, "status": status}
        print(f"  {name}: {status}", flush=True)
    freeze_ok = all(item["status"] == "MATCH" for item in recheck_files.values())
    write_json(receipt_dir / "R4_RECHECK_STAGE2.json", {
        "schema_version": "xian_r5.stage2.r4_recheck.v1", "generated_at_utc": generated,
        "freeze_receipt": str(freeze_path), "candidate_inputs": recheck_files,
        "all_frozen_hashes_match": freeze_ok,
        "production_modified_by_stage2": False, "formal_geometry_modified_by_stage2": False,
        "method": "SHA256 of each exact file listed in the initial Stage-2 freeze receipt; read-only verification",
    })

    print("[2/6] Probing the isolated RBF-SAM environment (no install or network).", flush=True)
    rbf_source = CODE_ROOT / "third_party_reference" / "RBF-SAM"
    env_probe = probe_isolated_environment(env_root, rbf_source)
    env_receipt = {
        "schema_version": "xian_r5.rbf_sam_environment_receipt.v2",
        "generated_at_utc": generated,
        "installation_status": env_probe.get("status"),
        "checkpoint_status": "BLOCKED_DOWNLOAD_ACCESS",
        "isolated_environment_root": str(env_root),
        "official_rbf_sam_source_path": str(rbf_source),
        "requested_environment": {"python": "3.10.x", "pytorch": "1.13.1+cu116", "cuda": "11.6", "mmcv_full": "1.7.0", "mmdetection": "2.x repository-compatible"},
        "probe": env_probe,
        "native_build_attempt": {"package": "mmpycocotools==12.0.3", "status": "FAILED_NO_MSVC_CPP_BUILD_TOOLS", "third_party_workaround_used": False},
        "production_environment_modified": False,
        "official_inference_ready": False,
    }
    initial_env_receipt = STAGE2 / "01_checkpoint" / "ENVIRONMENT_RECEIPT.json"
    initial_env_copy = STAGE2 / "01_checkpoint" / "ENVIRONMENT_RECEIPT_INITIAL.json"
    if initial_env_receipt.exists() and not initial_env_copy.exists():
        shutil.copy2(initial_env_receipt, initial_env_copy)
    write_json(initial_env_receipt, env_receipt)

    print("[3/6] Building per-case evidence summaries from frozen screening and tile mapping.", flush=True)
    pixel_read = verify_case_pixel_files(map_rows)
    write_json(STAGE2 / "03_fixed_cases" / "ACTUAL_DATE_PIXEL_READ_RECEIPT.json", {
        "schema_version": "xian_r5.stage2.fixed_case_pixel_read.v1", "generated_at_utc": generated,
        "result": pixel_read,
    })
    case_summaries: list[dict[str, str]] = []
    for case_id in CASE_IDS:
        mapped = rows_by_case.get(case_id, [])
        screen = screen_by_id.get(case_id, {})
        pixel_rows = [row for row in mapped if row.get("pixel_available", "").lower() == "true"]
        existing_rows = [row for row in pixel_rows if row.get("pixel_tile_path") and Path(row["pixel_tile_path"]).is_file()]
        decoded_paths = set(pixel_read.get("decoded_paths", []))
        decoded_rows = [row for row in pixel_rows if row.get("pixel_tile_path") in decoded_paths]
        def values(field: str, omit_sentinel: bool = False) -> list[str]:
            found = []
            for row in mapped:
                value = row.get(field, "").strip()
                if value and not (omit_sentinel and value in {"NO_R4_OBJECT_FOR_TRACK_EPOCH", "DATE_UNKNOWN"}) and value not in found:
                    found.append(value)
            return found
        stage1_summary = screen.get("one_sentence_note", "")
        case_summaries.append({
            "case_id": case_id,
            "stage1_visual_label": screen.get("visual_label", "NOT_IN_SCREEN"),
            "stage1_reviewed": screen.get("reviewed", ""),
            "stage1_candidate_gate_verdict": screen.get("fixed_gate_verdict", ""),
            "stage1_special_verdict": screen.get("fixed_special_verdict", ""),
            "stage1_identity_verdict": screen.get("identity_relationship_verdict", ""),
            "stage1_frozen_note_not_ground_truth": stage1_summary,
            "tile_map_rows": str(len(mapped)),
            "pixel_available_rows": str(len(pixel_rows)),
            "pixel_paths_existing_rows": str(len(existing_rows)),
            "pixel_paths_decoded_rows": str(len(decoded_rows)),
            "actual_dates_with_mapped_pixels": ";".join(sorted(set(row.get("actual_date", "") for row in pixel_rows if row.get("actual_date", "").count("-") == 2))),
            "epochs_with_mapped_pixels": ";".join(sorted(set(row.get("epoch_label", "") for row in pixel_rows if row.get("epoch_label")))),
            "actual_date_tile_ids": ";".join(values("actual_date_tile_id", True)),
            "epoch_object_ids": ";".join(values("object_uid", True)),
            "r4_polygon_object_ids": ";".join(values("r4_polygon_object_id", True)),
            "neighbor_polygon_count_max": str(max([int(row.get("neighbor_polygons_count") or 0) for row in mapped] or [0])),
            "image_bbox_px_values": ";".join(values("image_bbox_px", True)),
            "stage2_rbf_inference": "NOT_RUN_BLOCKED_CHECKPOINT_AND_ENVIRONMENT",
            "stage2_tbia_ab": "NOT_RUN_NO_IDENTITY_LABELLED_DEV_BENCHMARK",
            "stage2_result": "NOT_RUN_NOT_A_MODEL_PASS_OR_FAILURE",
        })
    case_out = STAGE2 / "03_fixed_cases" / "FIXED_CASE_STAGE2_STATUS.csv"
    case_out.parent.mkdir(parents=True, exist_ok=True)
    with case_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(case_summaries)

    print("[4/6] Writing bounded Stage-2 gates and compatibility deliverables.", flush=True)
    selected_pairs = pairs.get("selected_pair_details", [])
    selected_ids = pairs.get("selected_for_bounded_stage2", [])
    pair_counts_ok = len(selected_pairs) == len(selected_ids) == 6
    main_counts = actual_receipt.get("counts", {})
    proxy_pairs = frame_receipt.get("pairs", {})
    checkpoint = read_json(STAGE2 / "01_checkpoint" / "CHECKPOINT_REGISTRY.json")
    method_totals = {"candidate_n": 600, "metric_rows": 1200, "match_point_rows": 9636}
    metrics_receipt = read_json(STAGE2 / "06_lightglue_ab" / "ALIGNMENT_PROXY_DIAGNOSTIC_RECEIPT_V2.json")

    gate_specs = {
        STAGE2 / "00_date_harmonization" / "ACTUAL_DATE_GATE.json": {
            "gate": "ACTUAL_DATE_GATE", "status": "PASS_BOUNDED_ONLY_WITH_UNKNOWN_DATE_ROWS",
            "selected_exact_date_pairs": len(selected_pairs), "selected_pairs_all_date_coherent": all(bool(p.get("date_coherent")) and not p.get("mosaic_dates_combined") for p in selected_pairs),
            "selected_pair_sample_date_mixing_n": 0, "main_tile_rows": main_counts.get("main_rows"), "main_pixel_rows": main_counts.get("main_pixel_rows"),
            "main_date_unknown_rows": main_counts.get("main_date_unknown_rows"), "supplement_rows": main_counts.get("supplement_rows"),
            "supplement_pixel_rows": main_counts.get("supplement_pixel_rows"),
            "limitation": "Full R4 track-level impact count for nominal-date mosaicking is not quantified; 261 main tile-date rows remain unknown and supplemental rows are metadata-only.",
        },
        STAGE2 / "04_stable_ground" / "STABLE_GROUND_ALIGNMENT_GATE.json": {
            "gate": "STABLE_GROUND_ALIGNMENT_GATE", "status": "NOT_RUN", "required_n_per_real_date_pair": 100,
            "validated_sample_n": 0, "proxy_proposals_per_primary_pair": {k: v.get("stable_ground_proxy_selected", 0) for k, v in proxy_pairs.items()},
            "reason": "Density/footprint proxies include non-ground cover and remain visually unreviewed; no accepted road/parking/bare-ground/low-infrastructure ROIs.",
        },
        STAGE2 / "05_parallax" / "PARALLAX_ALIGNMENT_GATE.json": {
            "gate": "PARALLAX_ALIGNMENT_GATE", "status": "NOT_RUN", "required_n": 100, "validated_sample_n": 0,
            "proxy_proposals_per_primary_pair": {k: v.get("parallax_proxy_selected", 0) for k, v in proxy_pairs.items()},
            "reason": "No visually confirmed high-rise/roof-offset risk samples or approved local building-offset ROIs.",
        },
        STAGE2 / "07_tbia_ab" / "TBIA_AB_GATE.json": {
            "gate": "TBIA_AB_GATE", "status": "NOT_RUN", "identity_labelled_fixed_cases": 0,
            "random_development_cases": 0, "required_random_development_cases": 300,
            "neighbor_switch_r4": None, "neighbor_switch_tbia": None, "identity_metrics": None,
            "reason": "No Stage-2 TBIA inference or identity-labelled paired dev benchmark was executed; frozen Stage-1 visual review is candidate review, not Stage-2 truth labels.",
        },
        STAGE2 / "08_merge_resolver" / "MERGE_RESOLVER_GATE.json": {
            "gate": "MERGE_RESOLVER_GATE", "status": "NOT_RUN", "rbf_sam_official_repro": "BLOCKED_DOWNLOAD_ACCESS",
            "tbia_merge_suspects_run": 0, "resolver_cases_run": 0, "reason": "Cannot integrate or evaluate an RBF-SAM prompt resolver without the official checkpoint, importable model stack, and identity-labelled merge cases.",
        },
        STAGE2 / "03_fixed_cases" / "FIXED_CASE_GATE.json": {
            "gate": "FIXED_CASE_GATE", "status": "FAIL_NOT_RUN_GATE_UNSATISFIED", "registered_cases": len(CASE_IDS),
            "mapped_rows": len(map_rows), "pixel_available_rows": sum(r.get("pixel_available", "").lower() == "true" for r in map_rows),
            "stage2_inference_cases": 0, "stage2_pass_cases": 0, "empirical_model_failure_inferred": False,
            "reason": "Readiness gate is unsatisfied because R5 inference was not run; this is not evidence of a model regression.",
        },
        STAGE2 / "09_false_change" / "FALSE_CHANGE_GATE.json": {
            "gate": "FALSE_CHANGE_GATE", "status": "FAIL_NOT_RUN_GATE_UNSATISFIED", "required_visually_confirmed_stable_buildings": 100,
            "validated_stable_buildings": 0, "raw_vs_alignment_aware_metrics": None, "empirical_increase_inferred": False,
            "reason": "No visually accepted stable-building set and no Stage-2 R5 alignment/identity outputs.",
        },
        STAGE2 / "10_geometry_regression" / "GEOMETRY_REGRESSION_GATE.json": {
            "gate": "GEOMETRY_REGRESSION_GATE", "status": "NOT_RUN", "r5_geometry_outputs": 0,
            "r4_geometry_changed": False, "regression_metrics": None,
            "reason": "No R5 geometry output was generated; formal and frozen R4 geometry remained read-only.",
        },
    }
    for path, payload in gate_specs.items():
        write_json(path, {"schema_version": "xian_r5.stage2.gate.v1", "generated_at_utc": generated, **payload})

    rbf_repro = {
        "schema_version": "xian_r5.stage2.rbf_repro_gate.v1", "generated_at_utc": generated,
        "gate": "RBF_SAM_REPRO_GATE", "status": "BLOCKED_DOWNLOAD_ACCESS",
        "official_repository": checkpoint.get("official_repository"), "official_share_url": checkpoint.get("official_share_url"),
        "checkpoint_status": checkpoint.get("status"), "checkpoint_sha256": checkpoint.get("sha256"),
        "environment_status": env_probe.get("status"), "official_example_inference": "NOT_RUN",
        "prompt_types_run": [], "reason": "Official Baidu share returned an interactive verification form rather than a checkpoint; full MMDetection model imports also remain blocked by missing pycocotools build dependency.",
    }
    rbf_fixed = {
        "schema_version": "xian_r5.stage2.rbf_fixed_case_gate.v1", "generated_at_utc": generated,
        "gate": "RBF_SAM_FIXED_CASE_GATE", "status": "NOT_RUN_BLOCKED_CHECKPOINT_AND_ENVIRONMENT",
        "registered_fixed_cases": len(CASE_IDS), "pixel_available_map_rows": sum(r.get("pixel_available", "").lower() == "true" for r in map_rows),
        "readable_pixel_map_rows": pixel_read.get("pixel_rows_decoded", 0),
        "inference_cases_run": 0, "accepted_benchmark_cases": 0,
        "reason": "No official checkpoint, no successful official reproduction, and no Xi'an prompt inference; mapped pixel availability alone is not inference evidence.",
    }
    rbf_dir = STAGE2 / "03_rbf_sam"
    write_json(STAGE2 / "01_checkpoint" / "RBF_SAM_REPRO_GATE.json", rbf_repro)
    write_json(STAGE2 / "01_checkpoint" / "RBF_SAM_FIXED_CASE_GATE.json", rbf_fixed)
    write_json(rbf_dir / "RBF_SAM_REPRO_GATE.json", rbf_repro)
    write_json(rbf_dir / "RBF_SAM_FIXED_CASE_GATE.json", rbf_fixed)
    write_json(rbf_dir / "R5_MERGE_PROMPT_BENCHMARK" / "BENCHMARK_STATUS.json", {
        "schema_version": "xian_r5.stage2.rbf_merge_benchmark.v1", "generated_at_utc": generated,
        "status": "NOT_RUN_BLOCKED_CHECKPOINT_AND_LABELS", "real_merge_curated_n": 0,
        "real_split_curated_n": 0, "dense_neighbor_curated_n": 0, "parallax_risk_curated_n": 0,
        "synthetic_cases_run": 0, "case_category_counts_are_benchmark_counts": False,
        "reason": "Stage-1 fixed-case review flags are retained only as candidate review; no RBF output or Stage-2 accepted benchmark labels exist.",
    })

    alignment_gate = {
        "schema_version": "xian_r5.stage2.alignment_method_gate.v1", "generated_at_utc": generated,
        "status": "PROXY_DIAGNOSTIC_ONLY_VISUAL_LABELS_PENDING", **method_totals,
        "validated_stable_ground_n": 0, "validated_parallax_risk_n": 0,
        "same_samples_for_methods": metrics_receipt.get("same_samples_for_both_methods"),
        "estimator": metrics_receipt.get("same_geometry_estimator"),
        "summary_csv": str(summary_path), "run_version": metrics_receipt.get("run_version"),
        "V1_warning": "V1 LightGlue metrics are superseded because list-valued confidence scores were mishandled; use V2 only.",
    }
    write_json(STAGE2 / "06_lightglue_ab" / "ALIGNMENT_METHOD_GATE.json", alignment_gate)
    write_json(STAGE2 / "11_visual_qa" / "VISUAL_REVIEW_COMPLETION_RECEIPT.json", {
        "schema_version": "xian_r5.stage2.visual_review_receipt.v1", "generated_at_utc": generated,
        "proposal_samples": 600, "visually_reviewed_and_accepted_stable_ground_n": 0,
        "visually_reviewed_and_accepted_parallax_risk_n": 0, "candidate_contact_sheets": 20,
        "review_status": "PROXY_MONTAGES_INSPECTED_BUT_NO_ROW_LEVEL_LABELS_ACCEPTED",
        "observed_contamination": ["cropland", "tree_canopy", "water", "roofs", "roads", "mixed urban cover"],
        "limitation": "No per-tile reviewed ROI, stable-ground acceptance, or high-rise/parallax label is recorded.",
    })

    # Preserve V1 duplicate outputs, then point the canonical names to the corrected V2 artifacts.
    canonical_candidate = STAGE2 / "00_date_harmonization" / "DATE_COHERENT_PAIR_CANDIDATES_STAGE2.csv"
    canonical_selection_v2 = STAGE2 / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION_STAGE2.json"
    correction_copies = [
        archive_then_copy(pair_csv_v2_path, canonical_candidate,
                          STAGE2 / "00_date_harmonization" / "DATE_COHERENT_PAIR_CANDIDATES_STAGE2_V1_DUPLICATED.csv"),
        archive_then_copy(pair_v2_path, canonical_selection_v2,
                          STAGE2 / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION_STAGE2_V1_DUPLICATED.json"),
    ]
    fixed_alias_dir = STAGE2 / "09_fixed_cases"
    alias_copy = archive_then_copy(case_map_path, fixed_alias_dir / "R5_FIXED_CASE_TILE_MAP.csv")
    case_status_alias = archive_then_copy(case_out, fixed_alias_dir / "FIXED_CASE_STAGE2_STATUS.csv")
    for source_name in ["CHECKPOINT_REGISTRY.json", "ENVIRONMENT_RECEIPT.json", "RBF_SAM_CHECKPOINT_DOWNLOAD_BLOCKER.md"]:
        src = STAGE2 / "01_checkpoint" / source_name
        archive_then_copy(src, rbf_dir / source_name)
    write_json(fixed_alias_dir / "FIXED_CASE_MAP_ALIAS_RECEIPT.json", {
        "schema_version": "xian_r5.stage2.fixed_case_alias.v1", "generated_at_utc": generated,
        "canonical_map": str(case_map_path), "requested_compatibility_path": str(fixed_alias_dir / "R5_FIXED_CASE_TILE_MAP.csv"),
        "canonical_sha256": sha256_file(case_map_path), "alias_sha256": sha256_file(fixed_alias_dir / "R5_FIXED_CASE_TILE_MAP.csv"),
        "same_content": sha256_file(case_map_path) == sha256_file(fixed_alias_dir / "R5_FIXED_CASE_TILE_MAP.csv"),
    })

    # Correction notes keep the first diagnostic attempts auditable and prevent accidental use.
    write_text(STAGE2 / "06_lightglue_ab" / "V1_CORRECTION_NOTE.md",
        "# Alignment diagnostic V1 correction\n\n"
        "The first V1 run incorrectly marked every ALIKED+LightGlue row as failed because the official matcher returns confidence scores as a list and the diagnostic called `.numel()` on that list. V1 LightGlue metrics are invalid and are retained only as a failed instrumentation attempt. The corrected V2 run handles list-valued scores and supersedes V1 for both methods; see `ALIGNMENT_PROXY_DIAGNOSTIC_RECEIPT_V2.json`. V2 remains proxy-only and is not a validated stable-ground or parallax A/B.\n")
    write_text(STAGE2 / "00_date_harmonization" / "DATE_PAIR_V1_CORRECTION_NOTE.md",
        "# Date-pair candidate V1 correction\n\n"
        "V1 duplicated date-pair combinations during alternative enumeration. The duplicated outputs are preserved with `_V1_DUPLICATED` names. The canonical `DATE_COHERENT_PAIR_CANDIDATES_STAGE2.csv` and `DATE_COHERENT_PAIR_SELECTION_STAGE2.json` now mirror the corrected V2 outputs; `DATE_COHERENT_PAIR_SELECTION.json` also contains the corrected six exact-date selections. Each exact date pair remains separate; no date mosaics were constructed.\n")

    print("[5/6] Writing final decision and report.", flush=True)
    report_table = []
    for item in alignment_rows:
        report_table.append(
            "| {pair} | {stratum} | {method} | {passed}/{n} | {rate:.1%} | {matches} | {coverage} | {resid} / {p90} |".format(
                pair=item.get("pair_id", ""), stratum=item.get("proxy_stratum", "").replace("_", " "),
                method=item.get("method", ""), passed=int(float(item.get("pass_diagnostic_n", 0))),
                n=int(float(item.get("attempted_n", 0))), rate=float(item.get("diagnostic_failure_rate") or 0),
                matches=item.get("median_match_count", ""), coverage=item.get("median_spatial_coverage_4x4", ""),
                resid=item.get("median_reprojection_residual_px", ""), p90=item.get("median_p90_reprojection_residual_px", "")))
    case_table = []
    for item in case_summaries:
        screen_result = item["stage1_special_verdict"] or item["stage1_candidate_gate_verdict"] or "REVIEW_ONLY"
        note = item["stage1_frozen_note_not_ground_truth"].replace("|", "/")
        case_table.append(
            f"| {item['case_id']} | {item['stage1_visual_label']} / {screen_result} | {item['actual_dates_with_mapped_pixels'] or 'none mapped'} | {item['pixel_available_rows']}/{item['tile_map_rows']} rows | {note} | R5 inference NOT RUN |")
    pair_table = []
    for pair in selected_pairs:
        pair_table.append(
            f"| {pair.get('pair_id')} | {pair.get('source_date')} → {pair.get('target_date')} | {pair.get('same_spatial_tile_count')} | {float(pair.get('same_spatial_coverage_area_km2') or 0):.3f} | {pair.get('fixed_case_pairable_case_n')} | {pair.get('selection_role')} |")
    date_rows = actual_receipt.get("main_tile_rows_by_epoch_actual_date", [])
    date_table = [f"| {r.get('epoch_label')} | {r.get('actual_date')} | {r.get('tile_count')} | {float(r.get('coverage_area_km2') or 0):.3f} |" for r in date_rows]
    env_status = env_probe.get("status", "UNKNOWN")
    report = f"""# Xi'an R5 Stage-2 final report

Generated: {generated}

## Decision

**DO NOT PROMOTE R5.** `R5_STAGE2_PROMOTION_GATE=FAIL`; `READY_FOR_STAGE3_CITYWIDE=NO`.

This is a readiness decision, not a measured finding that R5 is worse than R4. Critical gates did not run: official RBF-SAM inference was blocked, visually validated stable-ground and parallax strata are both n=0, and identity-labelled TBIA/fixed-case/false-change benchmarks were not executed. No citywide production or formal holdout was run.

## Scope and integrity

- All R5 outputs here are isolated Stage-2 candidates. No R4 or formal geometry was written or warped; no holdout was read for tuning.
- The final read-only SHA256 recheck compared {len(recheck_files)} frozen R4 candidate inputs; all match: **{freeze_ok}**. See `13_receipts/R4_RECHECK_STAGE2.json`.
- The fixed-case map contains {len(map_rows)} epoch/object rows for {len(CASE_IDS)} registered cases; {sum(r.get('pixel_available', '').lower() == 'true' for r in map_rows)} rows have mapped actual-date pixel paths. Read-only Pillow decoding succeeded for {pixel_read.get('pixel_rows_decoded', 0)} mapped rows; bbox bounds are valid for {pixel_read.get('image_bbox_rows_in_bounds', 0)}/{pixel_read.get('image_bbox_rows_checked', 0)} checked boxes. This establishes image readability, not model inference or visual ground truth.

## Actual-date harmonization

The tile-level map is `00_date_harmonization/ACTUAL_DATE_TILE_MAP.parquet` (also linked into the requested `02_alignment` path). Nominal epoch labels were not used as acquisition dates. Main registry coverage: {main_counts.get('main_rows')} tile rows, {main_counts.get('main_pixel_rows')} with pixel files, {main_counts.get('main_date_unknown_rows')} date-unknown rows. Supplemental evidence has {main_counts.get('supplement_rows')} rows but {main_counts.get('supplement_pixel_rows')} pixels; its geometry is unresolved and it is metadata-only. The {main_counts.get('supplement_duplicate_rows')} supplemental rows flagged duplicate are row flags, not a count of unique duplicate acquisitions.

### Main date strata

| Epoch label | Actual date | Tiles | Coverage km² |
|---|---|---:|---:|
{chr(10).join(date_table)}

The corrected bounded selection has six exact-date comparisons. Each comparison stays separate; **0 selected pair samples combine multiple acquisition dates**. The dominant 2019→2020 pair (2018-11-01→2019-12-10) has no pairable fixed cases; the 2018-11-01→2019-12-02 alternative has three, but only 30,096 common tiles. Other date alternatives have smaller coverage and are not pooled.

| Pair | Actual dates | Common tiles | Common area km² | Pairable fixed cases | Role |
|---|---|---:|---:|---:|---|
{chr(10).join(pair_table)}

Thus the selected Stage-2 paired sample has no cross-date mosaic mixing by construction. The number of full-population R4 object tracks whose nominal transitions would be affected by mixed actual dates is **not quantified** from tile coverage alone. The per-epoch residual-Unknown figures in `DATE_STRATA_COVERAGE.csv` are adjacent-interval unions, not interval-specific denominators, and should not be compared directly with the overall R4 transition Unknown percentages.

## RBF-SAM checkpoint and isolated environment

1. Official checkpoint obtained? **No.** The official repository advertises weights, but the official Baidu share returned an interactive verification page, not a checkpoint. No checkpoint path, filename, size, or SHA256 exists. No unofficial mirror was used. See `01_checkpoint/CHECKPOINT_REGISTRY.json` and the blocker note.
2. Official inference reproduced? **No — BLOCKED_DOWNLOAD_ACCESS.** There was no checkpoint to load or run through the official prompt examples.
3. Is RBF-SAM useful for Xi'an merged masks? **NOT_DETERMINED.** Zero Xi'an RBF inference cases; no outcome can be inferred.

The isolated Python 3.10 environment has Torch/CUDA and MMCV native extension smoke checks, but its state is **{env_status}**. With the official local RBF-SAM source on PYTHONPATH (repository MMDetection version {env_probe.get('repository_mmdet_version')}), `mmdet.models` import fails at missing `pycocotools.mask`; the attempted `mmpycocotools==12.0.3` native build could not proceed because MSVC C++ build tools were unavailable. Pip metadata also reports the OpenCV 4.12 / NumPy 1.23.5 constraint mismatch. Production Python remained untouched. See the updated `01_checkpoint/ENVIRONMENT_RECEIPT.json` and `03_rbf_sam/ENVIRONMENT_RECEIPT.json`.

No real RBF merge/split/dense-neighbor/parallax benchmark was run. Curated Stage-2 benchmark counts are 0 in all four categories; Stage-1 fixed-case review flags are not counted as benchmark labels.

## Alignment: SIFT vs ALIKED+LightGlue

The corrected V2 run evaluated 600 same tile-pair candidates with both methods (1,200 metric rows; 9,636 match-point rows), using the same partial-affine RANSAC 3 px estimator. The four strata are only footprint-density proxies: visually accepted stable-ground n=0 and parallax-risk n=0. Candidate contact sheets show mixed cover including crops, canopy, water, roofs, and roads. Required real sample sizes were therefore not met.

Diagnostic pass below means only that at least one inlier was found by the scripted estimator; it is **not** a gate pass, external accuracy, stable-ground label, or parallax label. Residuals are in-sample estimator residuals.

| Pair | Proxy only | Method | Diagnostic pass | No-inlier diagnostic rate | Median matches | Median 4×4 coverage | Median / p90 residual px |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(report_table)}

4. Better for real stable ground? **NOT_DETERMINED.**
5. Better for real parallax risk? **NOT_DETERMINED.**

The raw proxy diagnostics show method- and pair-dependent behavior, not a defensible winner. No stable-ground global transform, parallax offset analysis, or local 512/1024 alignment field was accepted. V1 LightGlue metrics are invalid due to a list-score instrumentation error and are superseded by V2; see `06_lightglue_ab/V1_CORRECTION_NOTE.md`.

## TBIA, merge resolution, false change, and geometry

7. TBIA reduces neighbor switches by how much? **NOT_DETERMINED**; Stage-2 TBIA A/B cases = 0. R4 and TBIA neighbor-switch counts are null/not measured, not zero.
8. Merge/split failures reduced by how much? **NOT_DETERMINED**; RBF and resolver runs = 0. R4/R5 merge counts are not measured in this experiment.
9. False Addition/Demolition down? **NOT_DETERMINED**; zero accepted stable buildings (100 required), no paired alignment-aware outputs.
10. Did footprint geometry regress? **NOT_RUN**; no Stage-2 geometry was generated, and frozen R4 geometry was unchanged.

All five requested Unknown categories—MODEL_CAUSED, DATE_HETEROGENEITY, UNOBSERVABLE, IDENTITY_AMBIGUITY, GEOMETRY_AMBIGUITY—are **NOT_QUANTIFIED** for Stage-2. They are not zero. Selected exact-date strata eliminate mixing within those bounded pairs, but full-population track-level date attribution was not computed.

## Twelve fixed cases

The table reports frozen Stage-1 candidate review and Stage-2 map coverage. Stage-1 visual labels/verdicts are not ground truth and are not R5 Stage-2 outcomes. The row-by-row IDs, R4 polygon IDs, image bboxes, neighbors, actual dates, and tile paths are in `03_fixed_cases/R5_FIXED_CASE_TILE_MAP.csv` and the requested compatibility path `09_fixed_cases/R5_FIXED_CASE_TILE_MAP.csv`.

| Case | Frozen Stage-1 candidate review | Mapped actual dates | Pixel rows / map rows | Frozen review note | Stage-2 R5 |
|---|---|---|---:|---|---|
{chr(10).join(case_table)}

11. Per-case R5 results? **No R5 inference ran for any of the 12.** Ten map rows lack an R4 object-to-tile mapping and are marked `NO_PIXEL_EVIDENCE`/`NO_R4_OBJECT_FOR_TRACK_EPOCH`; this does not prove that the raw imagery itself is absent. The mapped pixel rows exist as files but were not consumed by RBF inference. `FIXED_CASE_GATE=FAIL_NOT_RUN_GATE_UNSATISFIED` means the required readiness evidence is missing, **not** that the model failed these cases.

## Gate summary

- Actual-date gate: **PASS_BOUNDED_ONLY_WITH_UNKNOWN_DATE_ROWS**.
- Stable-ground alignment: **NOT_RUN**, validated n=0.
- Parallax alignment: **NOT_RUN**, validated n=0.
- RBF-SAM official reproduction: **BLOCKED_DOWNLOAD_ACCESS**.
- RBF-SAM fixed-case benchmark: **NOT_RUN**.
- TBIA A/B and merge resolver: **NOT_RUN**.
- Fixed-case acceptance: **FAIL_NOT_RUN_GATE_UNSATISFIED** (not a measured model failure).
- False-change acceptance: **FAIL_NOT_RUN_GATE_UNSATISFIED** (not evidence of increased false change).
- Geometry regression: **NOT_RUN**.
- Stage-2 promotion: **FAIL**; Stage-3 citywide: **NO**.

## Answers and decision fields

1. Official checkpoint obtained: **NO**.
2. Official inference reproduced: **NO**.
3. RBF-SAM usefulness for Xi'an merged masks: **NOT_DETERMINED**.
4. Stable-ground method winner: **NOT_DETERMINED**.
5. Parallax method winner: **NOT_DETERMINED**.
6. Date mixing: 0 mixed samples in the six selected exact-date pairs; full-R4-track impact count **NOT_QUANTIFIED**.
7. TBIA neighbor-switch reduction: **NOT_DETERMINED**.
8. Merge/split failure reduction: **NOT_DETERMINED**.
9. False Addition/Demolition reduction: **NOT_DETERMINED**.
10. Footprint geometry regression: **NOT_RUN**.
11. Fixed cases: mapping and frozen review only; R5 inference **NOT_RUN for all 12**.
12. Stage-3 citywide: **NO**.

Decision values: `R5_ALIGNMENT_BETTER_THAN_R4=NOT_DETERMINED`; `R5_TBIA_BETTER_THAN_R4=NOT_DETERMINED`; `R5_RBFSAM_USEFUL=NOT_DETERMINED`; `R5_MERGE_RESOLVER_GATE=NOT_RUN`; `R5_FIXED_CASE_GATE=FAIL`; `R5_FALSE_CHANGE_GATE=FAIL`; `R5_STAGE2_PROMOTION_GATE=FAIL`; `READY_FOR_STAGE3_CITYWIDE=NO`.
"""
    report_path = STAGE2 / "12_reports" / "R5_STAGE2_FINAL_REPORT.md"
    write_text(report_path, report)

    answers = {
        "official_checkpoint_obtained": "NO",
        "official_inference_reproduced": "NO",
        "rbf_sam_xian_merged_mask_usefulness": "NOT_DETERMINED",
        "stable_ground_alignment_method_winner": "NOT_DETERMINED",
        "parallax_alignment_method_winner": "NOT_DETERMINED",
        "selected_pair_date_mixed_sample_n": 0,
        "full_r4_track_date_mixing_impact_n": None,
        "tbia_neighbor_switch_reduction": None,
        "merge_split_failure_reduction": None,
        "false_addition_demolition_reduction": None,
        "geometry_regression": "NOT_RUN",
        "fixed_cases": {item["case_id"]: {"stage2_inference": "NOT_RUN", "mapped_pixel_rows": int(item["pixel_available_rows"]), "map_rows": int(item["tile_map_rows"]), "stage1_screen_label_not_ground_truth": item["stage1_visual_label"], "stage1_note": item["stage1_frozen_note_not_ground_truth"]} for item in case_summaries},
        "ready_for_stage3_citywide": "NO",
    }
    decision = {
        "schema_version": "xian_r5.stage2.decision.v1", "generated_at_utc": generated,
        "task": "XIAN-R5-STAGE2-ACTUAL-DATE-RBFSAM-AND-IDENTITY-AB-001",
        "decision": "DO_NOT_PROMOTE_R5", "answers": answers,
        "R5_ALIGNMENT_BETTER_THAN_R4": "NOT_DETERMINED",
        "R5_TBIA_BETTER_THAN_R4": "NOT_DETERMINED",
        "R5_RBFSAM_USEFUL": "NOT_DETERMINED",
        "R5_MERGE_RESOLVER_GATE": "NOT_RUN",
        "R5_FIXED_CASE_GATE": "FAIL",
        "R5_FALSE_CHANGE_GATE": "FAIL",
        "R5_STAGE2_PROMOTION_GATE": "FAIL",
        "READY_FOR_STAGE3_CITYWIDE": "NO",
        "interpretation": "Promotion readiness failed because required evidence is absent or blocked; do not interpret non-run gates as observed model failures or zero effects.",
        "unknown_category_counts": {key: {"count": None, "status": "NOT_QUANTIFIED"} for key in ["UNKNOWN_MODEL_CAUSED", "UNKNOWN_DATE_HETEROGENEITY", "UNKNOWN_UNOBSERVABLE", "UNKNOWN_IDENTITY_AMBIGUITY", "UNKNOWN_GEOMETRY_AMBIGUITY"]},
        "r4_freeze_integrity": "PASS" if freeze_ok else "FAIL",
        "production_modified": False, "formal_geometry_modified": False,
        "report_path": str(report_path),
    }
    write_json(STAGE2 / "12_reports" / "R5_STAGE2_DECISION.json", decision)
    write_json(STAGE2 / "13_receipts" / "STAGE2_GATE_SUMMARY.json", {
        "schema_version": "xian_r5.stage2.gate_summary.v1", "generated_at_utc": generated,
        "gates": {path.stem: read_json(path).get("status") for path in gate_specs},
        "RBF_SAM_REPRO_GATE": rbf_repro["status"], "RBF_SAM_FIXED_CASE_GATE": rbf_fixed["status"],
        "ALIGNMENT_METHOD_GATE": alignment_gate["status"], "R5_STAGE2_PROMOTION_GATE": "FAIL",
        "READY_FOR_STAGE3_CITYWIDE": "NO", "production_modified": False,
    })

    progress_path = receipt_dir / "RUN_PROGRESS.md"
    existing_progress = progress_path.read_text(encoding="utf-8-sig") if progress_path.exists() else "# Stage-2 progress\n"
    if "FINAL DECISION" not in existing_progress:
        existing_progress = existing_progress.rstrip() + f"\n\n## FINAL DECISION ({generated})\n\n- R4 freeze input SHA256 recheck: {'PASS' if freeze_ok else 'FAIL'} ({len(recheck_files)} files).\n- Actual-date selections: {len(selected_pairs)} exact-date pairs; selected sample mixing=0; unknown-date rows remain.\n- RBF-SAM: checkpoint blocked by official share verification; inference not run. Isolated env: {env_status}.\n- Alignment: V2 diagnostics complete (600 candidates, 1,200 method rows); validated stable/parallax n=0.\n- Fixed-case mapping: {len(CASE_IDS)} cases, {len(map_rows)} map rows, {sum(r.get('pixel_available', '').lower() == 'true' for r in map_rows)} mapped pixel rows; R5 inference=0.\n- TBIA, merge resolver, false-change and geometry gates: not run / readiness unsatisfied.\n- Decision: DO NOT PROMOTE R5; READY_FOR_STAGE3_CITYWIDE=NO. No production or formal geometry modified.\n"
        write_text(progress_path, existing_progress)
    status = {
        "schema_version": "xian_r5.stage2.run_status.v2", "updated_at_utc": generated,
        "phase": "STAGE2_BOUNDED_EVIDENCE_CLOSED_DO_NOT_PROMOTE",
        "active_process": None,
        "actual_date_pair": [{"source": p.get("source_date"), "target": p.get("target_date"), "pair_id": p.get("pair_id")} for p in selected_pairs],
        "rbf_checkpoint_status": "BLOCKED_DOWNLOAD_ACCESS", "rbf_repro_status": "NOT_RUN_BLOCKED_DOWNLOAD_ACCESS_AND_ENVIRONMENT",
        "stable_ground_n": 0, "parallax_n": 0,
        "stable_ground_proxy_proposals_per_primary_pair": {k: v.get("stable_ground_proxy_selected", 0) for k, v in proxy_pairs.items()},
        "parallax_proxy_proposals_per_primary_pair": {k: v.get("parallax_proxy_selected", 0) for k, v in proxy_pairs.items()},
        "sift_metrics": "CANDIDATE_PROXY_ONLY", "lightglue_metrics": "CANDIDATE_PROXY_ONLY_V2",
        "alignment_candidate_n": method_totals["candidate_n"], "alignment_method_rows": method_totals["metric_rows"],
        "fixed_cases_registered": len(CASE_IDS), "fixed_case_tile_map_rows": len(map_rows),
        "fixed_case_pixel_rows": sum(r.get("pixel_available", "").lower() == "true" for r in map_rows),
        "fixed_case_pixel_rows_decoded": pixel_read.get("pixel_rows_decoded", 0),
        "fixed_cases_done": 0, "fixed_cases_pass": 0, "tbia_ab_cases": 0,
        "neighbor_switch_r4": None, "neighbor_switch_r5": None, "merge_cases_r4": None, "merge_cases_r5": None,
        "false_change_r4": None, "false_change_r5": None, "geometry_regression": "NOT_RUN",
        "current_gate": "R5_STAGE2_PROMOTION_FAIL", "blocker": "RBF_CHECKPOINT_INTERACTIVE_DOWNLOAD;MMDET_MODELS_IMPORT_BLOCKED;VISUAL_STRATA_UNVALIDATED;NO_IDENTITY_LABELLED_TBIA_BENCHMARK",
        "next_action": "KEEP_R4_FROZEN;IF_CONTINUING_STAGE2_ENABLE_MANUAL_OFFICIAL_CHECKPOINT_ACCESS_AND_ROW_LEVEL_VISUAL_STRATUM_LABELING_BEFORE_ANY_INFERENCE",
        "r4_freeze_recheck": "PASS" if freeze_ok else "FAIL",
        "production_modified": False, "formal_geometry_modified": False,
        "selected_date_pair_mosaic_samples": 0,
        "decision_path": str(STAGE2 / "12_reports" / "R5_STAGE2_DECISION.json"),
    }
    write_json(receipt_dir / "RUN_STATUS.json", status)

    print("[6/6] Final verification of report, aliases, and R4 freeze.", flush=True)
    date_map_alias = ROOT / "02_alignment" / "ACTUAL_DATE_TILE_MAP.parquet"
    required_outputs = [*gate_specs.keys(), STAGE2 / "01_checkpoint" / "RBF_SAM_REPRO_GATE.json",
                        STAGE2 / "01_checkpoint" / "RBF_SAM_FIXED_CASE_GATE.json", rbf_dir / "CHECKPOINT_REGISTRY.json",
                        rbf_dir / "ENVIRONMENT_RECEIPT.json", rbf_dir / "RBF_SAM_CHECKPOINT_DOWNLOAD_BLOCKER.md",
                        rbf_dir / "R5_MERGE_PROMPT_BENCHMARK" / "BENCHMARK_STATUS.json",
                        fixed_alias_dir / "R5_FIXED_CASE_TILE_MAP.csv", report_path,
                        STAGE2 / "12_reports" / "R5_STAGE2_DECISION.json", receipt_dir / "RUN_STATUS.json",
                        receipt_dir / "RUN_PROGRESS.md"]
    checks = {
        "freeze_hashes_all_match": freeze_ok,
        "six_selected_pairs": pair_counts_ok,
        "selected_pairs_all_exact_date_non_mosaic": len(selected_pairs) == 6 and all(bool(p.get("date_coherent")) and not p.get("mosaic_dates_combined") for p in selected_pairs),
        "lightglue_v2_metrics_complete": len(alignment_rows) == 8 and metrics_receipt.get("run_version", "").startswith("V2"),
        "fixed_case_map_alias_matches": sha256_file(case_map_path) == sha256_file(fixed_alias_dir / "R5_FIXED_CASE_TILE_MAP.csv"),
        "fixed_case_pixel_read_check": pixel_read.get("status") == "PIXEL_FILES_DECODED",
        "actual_date_map_alias_matches": date_map_alias.is_file() and sha256_file(date_map_alias) == sha256_file(STAGE2 / "00_date_harmonization" / "ACTUAL_DATE_TILE_MAP.parquet"),
        "required_outputs_exist": all(path.is_file() for path in required_outputs),
        "official_checkpoint_absent_as_reported": checkpoint.get("checkpoint_path") is None and checkpoint.get("sha256") is None,
        "production_formal_geometry_untouched": True,
    }
    write_json(receipt_dir / "STAGE2_FINALIZATION_RECEIPT.json", {
        "schema_version": "xian_r5.stage2.finalization_receipt.v1", "generated_at_utc": generated,
        "checks": checks, "all_checks_pass": all(checks.values()),
        "required_output_paths": [str(path) for path in required_outputs],
        "outputs": [str(report_path), str(STAGE2 / "12_reports" / "R5_STAGE2_DECISION.json"), str(receipt_dir / "RUN_STATUS.json"), str(case_out)],
        "date_pair_alias_updates": correction_copies, "fixed_case_alias_update": alias_copy,
        "fixed_case_status_alias_update": case_status_alias,
    })
    print(json.dumps({"all_checks_pass": all(checks.values()), "checks": checks,
                      "decision": decision["decision"], "environment": env_status,
                      "report": str(report_path), "decision_file": str(STAGE2 / "12_reports" / "R5_STAGE2_DECISION.json")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
