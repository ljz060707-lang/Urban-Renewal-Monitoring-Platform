#!/usr/bin/env python3
"""Create an append-only freeze receipt from bounded B03/B04 case outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_record(root: Path, expected_case: str) -> dict[str, Any]:
    evidence_path = root / "EVIDENCE_RECEIPT.json"
    manifest_path = root / "PRODUCT_MANIFEST.json"
    qa_path = root / "ENGINEERING_QA_FINAL.json"
    grid_path = root / "GRID_ALIGNMENT_RECEIPT.json"
    for path in (evidence_path, manifest_path, qa_path, grid_path,
                 root / "building_lineage.gpkg"):
        if not path.is_file():
            raise FileNotFoundError(path)
    evidence = _read_json(evidence_path)
    manifest = _read_json(manifest_path)
    qa = _read_json(qa_path)
    grid = _read_json(grid_path)
    if evidence.get("case_id") != expected_case:
        raise ValueError(f"Unexpected case in {root}: {evidence.get('case_id')!r}")
    if qa.get("structural_qa") != "PASS" or qa.get("errors"):
        raise ValueError(f"Final structural QA is not PASS for {expected_case}")
    if manifest.get("schema_version") != "xian_r5.grounded_products.v17":
        raise ValueError(f"Unexpected product schema for {expected_case}")
    if evidence.get("registration_candidate_run_status") != "NOT_RUN_BY_EXPLICIT_RECONSTRUCTION_FREEZE":
        raise ValueError(f"New registration candidates were not frozen for {expected_case}")
    if evidence.get("rbf_sam_status") != "NOT_RUN_NO_OFFICIAL_CHECKPOINT":
        raise ValueError(f"Unexpected RBF-SAM state for {expected_case}")
    if (grid.get("schema_version") != "xian_r5.grid_alignment_receipt.v1"
            or grid.get("fine_to_10m_nesting_pass") is not True
            or grid.get("fine_to_reference_alignment_pass") is not True):
        raise ValueError(f"Fine/base grid alignment receipt failed for {expected_case}")

    input_rechecks = []
    hashes = evidence.get("input_sha256", {})
    for name, raw_path in evidence.get("input_paths", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tracked input disappeared: {path}")
        current = _sha256(path)
        before = hashes.get(f"{name}_before", hashes.get(name))
        after = hashes.get(f"{name}_after", hashes.get(name))
        if before is not None and current != before:
            raise ValueError(f"Tracked input differs from pre-run hash: {name}")
        if after is not None and current != after:
            raise ValueError(f"Tracked input differs from post-run hash: {name}")
        input_rechecks.append({"name": name, "path": str(path),
                               "sha256_before": before, "sha256_after": after,
                               "sha256_final": current, "status": "PASS"})

    pixel_rechecks = []
    for item in evidence.get("pixel_inputs", []):
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Source image tile disappeared: {path}")
        current = _sha256(path)
        if current != item.get("sha256_before") or current != item.get("sha256_after"):
            raise ValueError(f"Source image tile hash changed: {path}")
        pixel_rechecks.append({"path": str(path), "sha256_before": item["sha256_before"],
                               "sha256_after": item["sha256_after"],
                               "sha256_final": current, "status": "PASS"})

    base_rechecks = []
    for item in evidence.get("10m_base_stack", {}).get("rasters", []):
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Reference raster disappeared: {path}")
        current = _sha256(path)
        if current != item.get("sha256"):
            raise ValueError(f"Reference raster hash changed: {path}")
        base_rechecks.append({"path": str(path), "sha256_recorded": item["sha256"],
                              "sha256_final": current, "status": "PASS"})

    product_inventory = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file():
            product_inventory.append({
                "name": path.name, "path": str(path.resolve()),
                "size_bytes": path.stat().st_size, "sha256": _sha256(path),
            })

    presence = manifest.get("presence_state_counts", {})
    dates = manifest.get("actual_date_order", [])
    return {
        "case_id": expected_case,
        "output_directory": str(root.resolve()),
        "schema_version": manifest["schema_version"],
        "structural_qa": qa["structural_qa"],
        "formal_ground_correction": qa.get("formal_ground_correction"),
        "formal_change_accuracy": qa.get("formal_change_accuracy"),
        "raster_count": qa.get("raster_count"),
        "observation_count": len(evidence.get("observations", [])),
        "actual_date_order": dates,
        "actual_date_count": len(dates),
        "accepted_registered_roof_count": manifest.get("roof_registered_count"),
        "reviewed_ground_footprint_count": manifest.get("ground_footprint_count"),
        "reviewed_physical_identity_count": manifest.get("confirmed_physical_identity_observations"),
        "canonical_geometry_resolved_count": manifest.get("canonical_geometry", {}).get("resolved_count"),
        "presence_table_counts": presence,
        "ground_change_transitions": {
            state: sum(row.get("state") == state for row in evidence.get("transitions", []))
            for state in ("STABLE", "ADDITION", "DEMOLITION", "UNRESOLVED")
        },
        "registration_candidate_run_status": evidence["registration_candidate_run_status"],
        "registration_diagnostic_count": len(evidence.get("registrations", [])),
        "rbf_sam_status": evidence["rbf_sam_status"],
        "presence_raster_diagnostics": manifest.get("presence_raster_diagnostics", {}),
        "grid_alignment_receipt": grid,
        "input_rechecks": input_rechecks,
        "pixel_input_rechecks": pixel_rechecks,
        "base_raster_rechecks": base_rechecks,
        "product_inventory": product_inventory,
        "artifact_hashes": {
            "EVIDENCE_RECEIPT.json": _sha256(evidence_path),
            "PRODUCT_MANIFEST.json": _sha256(manifest_path),
            "ENGINEERING_QA_FINAL.json": _sha256(qa_path),
            "GRID_ALIGNMENT_RECEIPT.json": _sha256(grid_path),
            "building_lineage.gpkg": _sha256(root / "building_lineage.gpkg"),
        },
    }


def finalize(b03: Path, b04: Path, output_dir: Path,
             unit_tests_passed: int, unit_tests_skipped: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite freeze report directory: {output_dir}")
    cases = {
        "B03-TEX04": _case_record(b03, "B03-TEX04"),
        "B04-TQA10": _case_record(b04, "B04-TQA10"),
    }
    all_input_checks = [item for case in cases.values() for item in case["input_rechecks"]]
    all_pixel_checks = [item for case in cases.values() for item in case["pixel_input_rechecks"]]
    all_base_checks = [item for case in cases.values() for item in case["base_raster_rechecks"]]
    if unit_tests_passed < 1 or unit_tests_skipped < 0:
        raise ValueError("Provide the observed bounded unit-test totals")
    complete = bool(
        all(case["structural_qa"] == "PASS" for case in cases.values())
        and all(case["grid_alignment_receipt"].get("fine_to_10m_nesting_pass") is True
                and case["grid_alignment_receipt"].get("fine_to_reference_alignment_pass") is True
                for case in cases.values())
        and all(item["status"] == "PASS" for item in all_input_checks + all_pixel_checks + all_base_checks)
        and all(case["registration_candidate_run_status"] ==
                "NOT_RUN_BY_EXPLICIT_RECONSTRUCTION_FREEZE" for case in cases.values()))
    r4_rows = [item for item in all_input_checks if item["name"] == "r4_status"]
    r4_unchanged = bool(r4_rows) and all(
        item["sha256_before"] == item["sha256_after"] == item["sha256_final"]
        for item in r4_rows)
    recheck = {
        "schema_version": "xian_r5.r4_final_hash_recheck.v1",
        "r4_status_files_unchanged": r4_unchanged,
        "tracked_input_count": len(all_input_checks),
        "source_image_tile_count": len(all_pixel_checks),
        "base_reference_raster_count": len(all_base_checks),
        "all_hash_rechecks_pass": all(
            item["status"] == "PASS" for item in all_input_checks + all_pixel_checks + all_base_checks),
        "cases": {case_id: {
            "input_rechecks": case["input_rechecks"],
            "pixel_input_rechecks": case["pixel_input_rechecks"],
            "base_raster_rechecks": case["base_raster_rechecks"],
        } for case_id, case in cases.items()},
    }
    if not r4_unchanged or not recheck["all_hash_rechecks_pass"]:
        raise ValueError("Final R4/source hash recheck did not pass")

    first_manifest = _read_json(b03 / "PRODUCT_MANIFEST.json")
    geometry_contract = first_manifest["geometry_layer_contract"]
    presence_contract = {
        "schema_version": "xian_r5.presence_state_contract.v1",
        "table_layer": "building_lineage.gpkg#building_presence",
        "table_codes": first_manifest["presence_table_codes"],
        "raster_codes": first_manifest["presence_raster_codes"],
        "table_semantics": first_manifest["presence_state_semantics"],
        "raster_authority": "TABLE_IS_AUTHORITATIVE;UNMAPPED_ROWS_REMAIN_IN_TABLE",
        "absence_policy": "ONLY_REVIEWED_FULL_FOOTPRINT_COVERAGE_CAN_CODE_0",
        "background_is_absence": False,
        "unresolved_is_absence": False,
        "counts_by_case": {key: value["presence_table_counts"]
                            for key, value in cases.items()},
    }
    grid_receipt = {
        "schema_version": "xian_r5.multi_case_grid_alignment_receipt.v1",
        "crs": "EPSG:32649",
        "reference_grid_policy": "FIXED_CASE_WINDOWS_NESTED_TO_THE_SAME_VERIFIED_10M_REFERENCE_GRID",
        "cases": {key: value["grid_alignment_receipt"] for key, value in cases.items()},
        "all_fine_to_10m_nesting_pass": all(
            value["grid_alignment_receipt"].get("fine_to_10m_nesting_pass") is True
            for value in cases.values()),
        "all_fine_to_reference_alignment_pass": all(
            value["grid_alignment_receipt"].get("fine_to_reference_alignment_pass") is True
            for value in cases.values()),
    }
    final_manifest = {
        "schema_version": "xian_r5.final_product_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_reconstruction_complete": "YES" if complete else "NO",
        "ground_footprint_correction": "UNRESOLVED_NOT_DEMONSTRATED",
        "ground_change_accuracy": "NOT_EVALUATED",
        "cases": cases,
        "geometry_layer_contract": geometry_contract,
        "presence_state_contract": "PRESENCE_STATE_CONTRACT.json",
        "grid_alignment_receipt": "GRID_ALIGNMENT_RECEIPT.json",
        "r4_hash_recheck": "R4_FINAL_HASH_RECHECK.json",
        "unit_regression": {"passed": unit_tests_passed, "skipped": unit_tests_skipped},
    }
    summary = {
        "schema_version": "xian_r5.reconstruction_freeze.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_reconstruction_complete": "YES" if complete else "NO",
        "completion_basis": (
            "UNIFIED_DATED_INPUTS_GEOMETRY_LAYERS_IDENTITY_PRESENCE_STATE_PRODUCTS_FIXED_GRIDS_AND_FINAL_QA;"
            "GROUND_FOOTPRINT_CORRECTION_REMAINS_UNRESOLVED_AND_IS_NOT_CLAIMED"),
        "registration_experiments": "NOT_RUN_BY_EXPLICIT_FREEZE_FLAG",
        "accepted_image_grid_registered_roofs": sum(
            case["accepted_registered_roof_count"] or 0 for case in cases.values()),
        "reviewed_ground_footprints": sum(
            case["reviewed_ground_footprint_count"] or 0 for case in cases.values()),
        "reviewed_physical_identity_observations": sum(
            case["reviewed_physical_identity_count"] or 0 for case in cases.values()),
        "rbf_sam": "NOT_RUN_NO_OFFICIAL_CHECKPOINT",
        "real_change_conclusion": (
            "NO_CONFIRMED_GROUND_CHANGE_LABELS;THIS_IS_NOT_EVIDENCE_OF_ZERO_REAL_CHANGE"),
        "ground_correction_conclusion": (
            "NOT_DEMONSTRATED_NO_INDEPENDENT_ORTHO_CONTROL_DEM_HEIGHT_OR_VIEW_METADATA"),
        "formal_change_accuracy": "NOT_EVALUATED",
        "r4_status_files_unchanged": r4_unchanged,
        "cases": {key: value for key, value in cases.items()},
        "unit_regression": {"passed": unit_tests_passed, "skipped": unit_tests_skipped},
    }
    report = [
        "# Xi'an R5 reconstruction freeze report", "",
        f"Created: {summary['created_at_utc']}", "",
        f"- `SYSTEM_RECONSTRUCTION_COMPLETE`: **{summary['system_reconstruction_complete']}**",
        "- Final ground-footprint correction: **UNRESOLVED / NOT DEMONSTRATED**.",
        "- Formal change accuracy: **NOT EVALUATED**.",
        "- New registration diagnostics: **NOT RUN** under the explicit freeze option.",
        "- RBF-SAM: **NOT_RUN_NO_OFFICIAL_CHECKPOINT**.",
        "- Confirmed ground-change labels: none; this is not evidence that real change is zero.",
        "- R4 status, source imagery tiles, and six reference rasters passed final SHA-256 rechecks.",
        "",
        "## Fixed cases", "",
        "| Case | Observations | Actual dates | 1 / 0 / U | Registered roofs | Ground footprints | QA | Rasters |",
        "|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for case_id, item in cases.items():
        counts = item["presence_table_counts"]
        date_text = ", ".join(item["actual_date_order"])
        report.append(
            f"| {case_id} | {item['observation_count']} | {date_text} | "
            f"{counts.get('1', 0)} / {counts.get('0', 0)} / {counts.get('U', 0)} | "
            f"{item['accepted_registered_roof_count']} | {item['reviewed_ground_footprint_count']} | "
            f"{item['structural_qa']} | {item['raster_count']} |")
    report.extend([
        "", "## Interpretation", "",
        "The engineering chain is wired and the two bounded fixed cases pass structural QA. "
        "Raw roofs, accepted registered roofs, reviewed ground footprints, physical identity, "
        "Presence, canonical registered-roof medoids, ground-change products, non-ground "
        "diagnostics, and quality layers remain distinct. Unresolved objects stay `U`; the "
        "roof outlines supplied for 2019 remain roof geometry, not ground truth. No inferred "
        "ground translation or accuracy claim is made.", "",
        "## Audit artifacts", "",
        "- `FINAL_PRODUCT_MANIFEST.json`", "- `GRID_ALIGNMENT_RECEIPT.json`",
        "- `GEOMETRY_LAYER_CONTRACT.json`", "- `PRESENCE_STATE_CONTRACT.json`",
        "- `R4_FINAL_HASH_RECHECK.json`", "- `R5_RECONSTRUCTION_FREEZE.json`",
        "- `ENGINEERING_REGRESSION_TEST_RECEIPT.json`", "",
    ])
    test_receipt = {
        "schema_version": "xian_r5.engineering_regression_receipt.v1",
        "scope": "FOCUSED_RECONSTRUCTION_IDENTITY_PRESENCE_ALIGNMENT_AND_QA_TESTS",
        "passed": unit_tests_passed, "skipped": unit_tests_skipped,
        "status": "PASS_WITH_RECORDED_SKIP",
        "full_city_or_accuracy_evaluation": False,
    }
    artifacts = {
        "R5_RECONSTRUCTION_FREEZE_REPORT.md": "\n".join(report),
        "R5_RECONSTRUCTION_FREEZE.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        "FINAL_PRODUCT_MANIFEST.json": json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n",
        "GRID_ALIGNMENT_RECEIPT.json": json.dumps(grid_receipt, ensure_ascii=False, indent=2) + "\n",
        "GEOMETRY_LAYER_CONTRACT.json": json.dumps(geometry_contract, ensure_ascii=False, indent=2) + "\n",
        "PRESENCE_STATE_CONTRACT.json": json.dumps(presence_contract, ensure_ascii=False, indent=2) + "\n",
        "R4_FINAL_HASH_RECHECK.json": json.dumps(recheck, ensure_ascii=False, indent=2) + "\n",
        "ENGINEERING_REGRESSION_TEST_RECEIPT.json": json.dumps(
            test_receipt, ensure_ascii=False, indent=2) + "\n",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b03", type=Path, required=True)
    parser.add_argument("--b04", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unit-tests-passed", type=int, required=True)
    parser.add_argument("--unit-tests-skipped", type=int, default=0)
    args = parser.parse_args()
    result = finalize(args.b03, args.b04, args.output_dir,
                      args.unit_tests_passed, args.unit_tests_skipped)
    print(json.dumps({"system_reconstruction_complete": result[
        "system_reconstruction_complete"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
