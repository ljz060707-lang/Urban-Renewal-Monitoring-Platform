"""Freeze explicit module handoffs, then replay them into a fresh R5 run.

No discovery of a changing 'latest' directory during execution. No model,
source artifact, old receipt, or module environment is modified.
"""
from __future__ import annotations

# Application dispatch precedes optional legacy Fiona imports. Historical paths remain intact.
import sys as _application_sys
if '--application' in _application_sys.argv:
    from pathlib import Path as _ApplicationPath
    _application_sys.path.insert(0,str(_ApplicationPath(__file__).resolve().parents[1]/'src'))
    from xian_r5.application import main as _application_main
    _application_main()
    raise SystemExit(0)

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import unittest

import fiona
import numpy as np
from affine import Affine
from shapely.geometry import shape
from shapely.ops import unary_union

from xian_r5.integration import (
    CONTRACT, adapt_native_components, adapt_sam_group, attach_single_date_links,
    build_gt10m_shadow, load_alignment_candidates, read_json, run_r5_state,
    sha256, verify_artifacts, verify_source_dates,
)
from xian_r5.alignment.reviewed_field import attach_reviewed_alignment_cells, recheck_reviewed_alignment_sources
from xian_r5.fraction_shadow import NODATA, QUALITY_BANDS
from xian_r5.label_transfer import Grid, export_raster, load_rasters

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("F:/xian/XIAN_R5_MODEL_UPGRADE_20260923")


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_lines(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def freeze_baseline(path, root, *, a_receipt=None, a_candidates=None, a_review=None):
    """One frozen, data-independent group per existing source region."""
    s = root / "21_source_partition_alignment"
    b = s / "SINGLE2019_FULLROI_ROOF_PROPOSALS_20261012_20"
    i = root / "23_building_object_identity/XIAN_R5_IDENTITY_MINIMAL_CLOSURE_20261015"
    pins = {}

    def pin(name, file, role, expected=None):
        file = Path(file).resolve()
        digest = sha256(file)
        if expected is not None and digest != expected:
            raise ValueError("Upstream receipt mismatch: " + name)
        pins[name] = {"path": str(file), "sha256": digest, "role": role}

    pin("a_receipt", a_receipt or s / "A_LOCAL_ALIGNMENT_NUMERICAL_ROBUSTNESS_20261013/FINAL_RECEIPT.json", "MODULE_A_RECEIPT")
    pin("b_receipt", b / "FULLROI_INFERENCE_RECEIPT.json", "MODULE_B_RECEIPT")
    pin("b_preparation", b / "FULLROI_PREPARATION_RECEIPT.json", "MODULE_B_PREPARATION")
    prep = read_json(b / "FULLROI_PREPARATION_RECEIPT.json")
    for name, file, role in (
        ("b_plan", "FROZEN_FULLROI_SAM_PLAN.json", "MODULE_B_PLAN"),
        ("b_packets", "FULLROI_NATIVE_SOURCE_PACKETS.json", "MODULE_B_PACKETS"),
        ("b_prompts", "FULLROI_SOURCE_PRIOR_PROMPTS.json", "MODULE_B_PROMPTS"),
    ):
        pin(name, b / file, role, prep["output_sha256"].get(file))
    pin("i_receipt", i / "FINAL_RECEIPT.json", "MODULE_I_RECEIPT")
    receipt_i = read_json(i / "FINAL_RECEIPT.json")
    for name, file, role in (
        ("i_graph", "IDENTITY_GRAPH_MINIMAL.gpkg", "MODULE_I_GRAPH"),
        ("i_links", "SOURCE_OBJECT_LINK_LEDGER.csv", "MODULE_I_LINKS"),
        ("i_proofs", "PLAUSIBLE_41_ADJUDICATION.json", "MODULE_I_ADJUDICATION"),
        ("i_recovery", "AMBIGUOUS_RECOVERY_DIAGNOSTIC.json", "MODULE_I_ADJUDICATION"),
        ("i_metrics", "MINIMAL_CLOSURE_METRICS.json", "MODULE_I_METRICS"),
    ):
        pin(name, i / file, role, receipt_i["artifact_sha256"][file])
    components = s / "GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg"
    pin("components", components, "OBSERVED_CANDIDATE", receipt_i["original_input_hashes"][str(components)])
    pin("source_registry", s / "XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_20261013/SOURCE_COMPONENT_LEDGER.json", "SOURCE_COVERAGE")
    pin("window_config", s / "TRAINROI_20261012_01/FROZEN_RUN_CONFIG.json", "WINDOW_CONFIG")
    window = read_json(pins["window_config"]["path"])
    pin("grid_template", window["reference"], "GRID_TEMPLATE_ONLY", window["reference_sha256"])
    pin("actual_date_registry", window["date_ledger"], "SOURCE_COVERAGE", window["date_ledger_sha256"])
    if a_candidates is not None:
        pin("a_candidates", a_candidates, "MODULE_A_CANDIDATES")
        a_manifest = read_json(a_candidates)
        pin("a_candidate_geometry", Path(a_candidates).parent / a_manifest["products"]["geopackage"], "MODULE_A_CANDIDATES")
        source_inputs = a_manifest["source_inputs"]
        sources = [source_inputs["direct_base_tie_diagnostic"], source_inputs["annual_reference_raster"],
                   *source_inputs["relative_diagnostics"]]
        for index, source in enumerate(sources):
            pin("a_candidate_source_" + str(index), source["path"], "MODULE_A_CANDIDATES", source["sha256_after"])
        fields = load_alignment_candidates(a_candidates, window["reference"], window["reference_sha256"])
        if a_review is not None:
            pin("a_review", a_review, "MODULE_A_REVIEW")
            _, review_report = attach_reviewed_alignment_cells(fields, Path(a_review))
            for index, (source, digest) in enumerate(review_report["integrity_files_sha256"].items()):
                pin("a_review_source_" + str(index), source, "MODULE_A_REVIEW", digest)
    elif a_review is not None:
        raise ValueError("Reviewed layer requires its frozen A candidates")
    packets = read_json(pins["b_packets"]["path"])["records"]
    prompts = read_json(pins["b_prompts"]["path"])["records"]
    receipt_b = read_json(pins["b_receipt"]["path"])
    plan = read_json(pins["b_plan"]["path"])
    pin("b_weights", plan["checkpoint"], "MODULE_B_WEIGHTS", plan["checkpoint_sha256"])
    groups = []
    for packet in sorted(packets, key=lambda r: r["region_id"]):
        prompt = min((r for r in prompts if r["source_packet_id"] == packet["region_id"]),
                     key=lambda r: r["group_id"])
        group = prompt["group_id"]
        groups.append(group)
        name = group + "_RECEIPT.json"
        pin(group + "_receipt", b / name, "MODULE_B_GROUP_RECEIPT", receipt_b["group_receipts_sha256"][name])
        group_receipt = read_json(b / name)
        pin(group + "_masks", b / group_receipt["npz"], "MODULE_B_NATIVE_MASK", group_receipt["npz_sha256"])
        pin(packet["region_id"] + "_rgb", packet["image_path"], "MODULE_B_NATIVE_RGB", packet["image_sha256"])
    # Every original tile used by the shared coverage registry is hash-checked.
    registry = read_json(pins["source_registry"]["path"])
    for tile in registry["tiles"]:
        pin("tile_" + tile["download_tile_id"], tile["source_JPG"], "SOURCE_COVERAGE", tile["source_JPG_sha256"])
    for file in sorted((REPO / "src/xian_r5").rglob("*.py")):
        pin("code_" + file.relative_to(REPO).as_posix(), file, "CODE")
    for file in [Path(__file__).resolve(), REPO / "docs/XIAN_R5_INTEGRATION_CONTRACT_V1.md",
                 REPO / "tests/xian_r5/test_reviewed_image_alignment.py",
                 *sorted((REPO / "tests").glob("test_xian_*.py"))]:
        pin("code_" + file.relative_to(REPO).as_posix(), file, "CODE")
    save(path, {
        "contract_version": CONTRACT, "project": "XIAN-R5-GROUNDREG-FREEZE-0929",
        "execution_mode": "CACHED_NATIVE_OUTPUT_INTEGRATION",
        "sam_scope": "LEXICOGRAPHIC_FIRST_EXISTING_GROUP_PER_REGION_FROZEN_BEFORE_REPLAY",
        "sam_group_ids": groups,
        "expected": {"observations": 1662, "single_date_links": 53, "source_tiles": 1433,
                     "source_regions": 16, "target_pixels": 65536},
        "manual_gt_required_for_inference": False, "inputs": pins,
    })
    return len(pins)


def regressions(output):
    sys.path.insert(0, str(REPO / "tests"))
    names = sorted(p.stem for p in (REPO / "tests").glob("test_xian_*.py")
                   if p.stem != "test_xian_additive_post_inference")
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    review_test = REPO / "tests/xian_r5/test_reviewed_image_alignment.py"
    spec = importlib.util.spec_from_file_location("xian_legacy_review_regressions", review_test)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    with (output / "REGRESSION_TEST_LOG.txt").open("x", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "successful": result.wasSuccessful(), "test_modules": names,
              "additional_review_regressions": str(review_test),
              "excluded_unrelated_test": "test_xian_additive_post_inference"}
    save(output / "REGRESSION_TESTS.json", report)
    if not result.wasSuccessful():
        raise RuntimeError("Cross-module regression failed; see REGRESSION_TEST_LOG.txt")
    return report


def run(manifest_path, output):
    manifest = read_json(manifest_path)
    if (manifest["contract_version"] != CONTRACT
            or manifest["execution_mode"] != "CACHED_NATIVE_OUTPUT_INTEGRATION"
            or manifest["manual_gt_required_for_inference"] is not False):
        raise ValueError("Unsupported inference contract/mode")
    if output.exists():
        raise FileExistsError("Append-only output must be new: " + str(output))
    pins = manifest["inputs"]
    if any(Path(p["path"]).resolve().is_relative_to(output) for p in pins.values()):
        raise ValueError("Output overlaps an input")
    before = verify_artifacts(pins)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "FROZEN_INTEGRATION_MANIFEST.json", manifest)
    with (output / "XIAN_R5_INTEGRATION_CONTRACT_V1.md").open("x", encoding="utf-8") as stream:
        stream.write((REPO / "docs/XIAN_R5_INTEGRATION_CONTRACT_V1.md").read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest_path)

    def path(name):
        return Path(pins[name]["path"])

    def document(name):
        return read_json(path(name))

    def progress(message):
        print(message, flush=True)

    started = time.monotonic()
    try:
        tests = regressions(output)
        progress("Regression passed: " + str(tests["tests_run"]))
        registry = document("source_registry")
        source_dates_checked = verify_source_dates(registry["tiles"], path("actual_date_registry"))
        packets = document("b_packets")["records"]
        plan = document("b_plan")
        if (plan["GT_or_target_pixels_or_predictions_used"] is not False
                or plan["automatic_mask_selection"] is not False):
            raise ValueError("Module B source-only policy changed")
        records = adapt_native_components(path("components"), registry["tiles"])
        expected = manifest["expected"]
        if (len(records) != expected["observations"] or len(registry["tiles"]) != expected["source_tiles"]
                or len(packets) != expected["source_regions"]):
            raise ValueError("Frozen real-window denominator changed")
        progress("Native source objects decoded: " + str(len(records)))
        links, closed = attach_single_date_links(records, path("i_links"), path("i_graph"),
                                                 [document("i_proofs"), document("i_recovery")])
        metrics_i = document("i_metrics")
        if closed != expected["single_date_links"] or closed != metrics_i["TRACEABLE_SINGLE_DATE_LINK_COUNT"]:
            raise ValueError("Frozen single-date links were lost")
        save_lines(output / "OBJECT_LINKS.jsonl", links)
        progress("Original single-date source links retained: " + str(closed))

        prompts = {r["group_id"]: r for r in document("b_prompts")["records"]}
        by_region = {r["region_id"]: r for r in packets}
        requested = manifest["sam_group_ids"]
        if (len(requested) != len(set(requested)) or len(requested) != len(packets)
                or {prompts[g]["source_packet_id"] for g in requested} != set(by_region)):
            raise ValueError("Frozen SAM group/region scope mismatch")
        sam = []
        for group in manifest["sam_group_ids"]:
            prompt = prompts[group]
            packet = by_region[prompt["source_packet_id"]]
            if prompt["component_key"] not in {r["observation_id"] for r in records}:
                raise ValueError("SAM parent is not an original component")
            with np.load(path(group + "_masks"), allow_pickle=False) as arrays:
                sam.extend(adapt_sam_group(packet, prompt, document(group + "_receipt"), arrays,
                                           plan["checkpoint_sha256"]))
            progress("Native SAM replay: " + group + " (all 3 alternatives retained)")
        save_lines(output / "SAM_NATIVE_CANDIDATES.jsonl", sam)
        if len(sam) != len(manifest["sam_group_ids"]) * 3:
            raise ValueError("SAM alternatives dropped")
        fields = (load_alignment_candidates(path("a_candidates"), path("grid_template"), pins["grid_template"]["sha256"])
                  if "a_candidates" in pins else [])
        state, alignment_report = run_r5_state(
            records, document("a_receipt"), [str(path("grid_template"))], candidate_fields=fields,
            reviewed_layer=path("a_review") if "a_review" in pins else None)
        applied = sum(r["registered_geometry"] is not None for r in records)
        if alignment_report is not None:
            save(output / "A_REVIEWED_FIELD_ADAPTER_RECEIPT.json", alignment_report)
        save_lines(output / "UNIFIED_ROOF_OBSERVATIONS.jsonl", records)
        save(output / "R5_STATE_HANDOFF.json", {
            "contract_version": CONTRACT, "engine": "xian_r5.reconstruction.core.reconstruct",
            "actual_date_order": state.actual_date_order,
            "identity_graph": state.identity_graph, "presence": state.presence,
            "transitions": state.transitions, "candidate_edges": state.candidate_edges,
            "temporal_interval_available": len(state.actual_date_order) > 1,
        })
        window = document("window_config")
        grid = Grid(Affine(*window["roi_affine"]), 256, 256, "EPSG:32649")
        if grid.width * grid.height != expected["target_pixels"]:
            raise ValueError("Target grid denominator differs")
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(path("grid_template")) as src:
            target = Window(*window["roi_window"])
            if src.window_transform(target) != grid.affine or str(src.crs) != grid.crs:
                raise ValueError("Affine/GDAL or CRS grid mismatch")
        support = unary_union([shape(t["support_geometry"]) for t in registry["tiles"]])
        progress("R5 Presence evaluated; exact 65,536-pixel fraction build starting")
        fractions, eligibility, supervision = build_gt10m_shadow(records, packets, support, grid, state)
        raster_arrays = {
            "ROOF_FRACTION_NATIVE_CANDIDATE.tif": (fractions["raw_fraction"][None], ["NATIVE_UNCORRECTED_CANDIDATE_ROOF_FRACTION"], NODATA),
            "ROOF_FRACTION_REGISTERED_QUALIFIED.tif": (fractions["candidate_fraction"][None], ["REGISTERED_ROOF_PROXY_FRACTION_OR_NODATA"], NODATA),
            "SOURCE_SUPPORT_FRACTION.tif": (fractions["scope_fraction"][:1], ["RECORDED_SOURCE_SCOPE_NOT_LABEL_VALIDITY"], None),
            "SOURCE_OBJECT_COUNT.tif": (fractions["component_count"][:1], ["ORIGINAL_COMPONENT_POSITIVE_AREA_COUNT"], None),
            "GT10M_QUALITY.tif": (fractions["quality"], QUALITY_BANDS, None),
            "GT10M_STATE_IGNORE.tif": (np.full((1, 256, 256), 255, np.uint8), ["IGNORE_NO_QUALIFIED_CHANGE_INTERVAL"], 255),
            "GT10M_TRAINING_WEIGHT.tif": (supervision["weight"][None], ["TRAINING_WEIGHT"], None),
            "GT10M_SUPERVISION_FRACTIONS.tif": (np.concatenate([supervision["target"], supervision["valid"][None], supervision["unknown"][None]]),
                                                ["STABLE_UNQUALIFIED", "ADDITION_UNQUALIFIED", "DEMOLITION_UNQUALIFIED", "VALID_SUPERVISION", "UNKNOWN"], None),
        }
        for name, (array, bands, nodata) in raster_arrays.items():
            export_raster(output / name, array, grid, bands, root=output, nodata=nodata)
        reloaded = load_rasters([output / name for name in raster_arrays])
        if reloaded["grid"] != grid:
            raise RuntimeError("GT10m dataloader readback grid drift")
        for actual, (array, _, _) in zip(reloaded["arrays"], raster_arrays.values()):
            if not np.array_equal(actual, array.astype(actual.dtype)):
                raise RuntimeError("GT10m exported values differ from unified inputs")
        if (any(row["physical_id"] is not None for row in state.presence)
                or any(row["presence"] != "U" for row in state.presence)
                or state.transitions
                or np.any(supervision["weight"]) or np.any(fractions["quality"][-1])):
            raise RuntimeError("Candidate evidence was promoted")
        save(output / "MODULE_HANDOFFS.json", {
            "contract_version": CONTRACT,
            "A": {"version": document("a_receipt")["adapter_method"], "receipt": pins["a_receipt"],
                  "accepted_regions": document("a_receipt")["ACCEPTED_LOCAL_REGIONS"],
                  "applied_objects": applied, "transform_m": None if not applied else "PER_OBJECT_IN_UNIFIED_RECORDS",
                  "strict_review_adapter_connected": True,
                  "position_semantics": "RELATIVE_IMAGE_GRID_ONLY"},
            "B": {"version": "OFFICIAL_SAM_VIT_B_NOT_RBF_SAM", "receipt": pins["b_receipt"],
                  "native_component_records": len(records), "sam_total_upstream_groups": 1396,
                  "sam_replayed_groups": len(manifest["sam_group_ids"]), "sam_replayed_alternatives": len(sam),
                  "selected_alternatives": 0, "fresh_model_inference_run": False,
                  "geographic_correction_performed": False},
            "I": {"version": "SINGLE_DATE_OBJECT_LINKAGE_PARTIALLY_CLOSED", "receipt": pins["i_receipt"],
                  "source_links": closed, "all_link_rows_preserved": len(links), "physical_ids": 0,
                  "historical_GT_identity_provenance_retained": True, "manual_GT_read_this_run": False},
            "R5": {"presence_U": len(state.presence), "actual_date_order": state.actual_date_order,
                   "change_intervals": len(state.transitions)},
            "GT10m": {"affine_order": "a,b,c,d,e,f", "affine": list(grid.affine)[:6],
                      "gdal_geotransform": list(grid.affine.to_gdal()), "crs": grid.crs,
                      "width": grid.width, "height": grid.height, "denominator": "FULL_TARGET_PIXEL",
                      "candidate_positive_pixels": int((fractions["raw_fraction"] > 0).sum()),
                      "candidate_area_m2": float(np.maximum(fractions["raw_fraction"], 0).sum() * 100),
                      "area_conservation_error_m2": fractions["area_conservation_error_m2"],
                      "eligibility": eligibility["counts"],
                      "exclusive_exclusion_causes": eligibility["exclusive_reason_counts"],
                      "training_eligible_pixels": 0, "threshold_0_43_applied": False},
        })
        issues = [
            {"module": "A", "issue": "REAL_ALIGNMENT_EVIDENCE_STATUS",
             "accepted_regions": document("a_receipt")["ACCEPTED_LOCAL_REGIONS"], "applied_roofs": applied},
            {"module": "A", "issue": "NONZERO_REVIEWED_FIELD_HANDOFF_NOT_EXERCISED_ON_REAL_WINDOW",
             "behavior": "Strict reader, V2 review recomputation and complete-instance application are connected; synthetic checks only."},
            {"module": "B", "issue": "SAM_ALTERNATIVES_UNSELECTED_ROOF_PURITY_NOT_VERIFIED",
             "replayed_groups": len(manifest["sam_group_ids"]), "upstream_groups": 1396},
            {"module": "B", "issue": "FRESH_RGB_INFERENCE_NOT_PART_OF_THIS_CACHED_OUTPUT_RUN"},
            {"module": "I", "issue": "SINGLE_DATE_SOURCE_ASSOCIATION_IS_NOT_PHYSICAL_IDENTITY", "closed_source_links": closed},
            {"module": "R5", "issue": "ONE_ACTUAL_DATE_NO_REAL_CHANGE_INTERVAL"},
            {"module": "GT10m", "issue": "TARGET_TRAINING_TIME_BINDING_AND_SPATIAL_STATE_ELIGIBILITY_UNRESOLVED"},
            {"module": "ALL", "issue": "INDEPENDENT_SCIENTIFIC_VALIDATION_NOT_AVAILABLE"},
        ]
        save(output / "UNRESOLVED_ISSUES.json", issues)
        if alignment_report is not None:
            recheck_reviewed_alignment_sources(alignment_report)
        after = verify_artifacts(pins)
        if before != after or sha256(manifest_path) != manifest_hash:
            raise RuntimeError("Frozen input changed during integration")
        for name in ("SAM_NATIVE_CANDIDATES.jsonl", "UNIFIED_ROOF_OBSERVATIONS.jsonl", "OBJECT_LINKS.jsonl"):
            with (output / name).open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            if any(r["physical_id"] is not None for r in rows):
                raise RuntimeError("NULL physical ID was not persisted")
        summary = {
            "contract_version": CONTRACT, "ENGINEERING_PASS": True,
            "engineering_scope": ("FROZEN_NATIVE_OUTPUT_REPLAY_FULL_256X256_REVIEWED_A_PATH" if applied else
                                  "FROZEN_NATIVE_OUTPUT_REPLAY_FULL_256X256_ZERO_ACCEPTED_A_PATH"),
            "SCIENTIFIC_VALIDATION_PASS": False, "AUTOMATIC_HANDOFF_RUN_READY": True,
            "FRESH_RGB_TO_QUALIFIED_GT_AUTORUN_READY": False,
            "native_observations": len(records), "single_date_source_links": closed,
            "sam_groups_replayed": len(manifest["sam_group_ids"]), "sam_alternatives_preserved": len(sam),
            "physical_ids": 0, "accepted_alignment_regions": document("a_receipt")["ACCEPTED_LOCAL_REGIONS"],
            "registered_roofs": applied, "strict_reviewed_field_adapter_connected": True,
            "ground_footprints": 0, "training_eligible_pixels": 0,
            "presence_U": len(state.presence), "target_pixels": 65536,
            "original_ids_and_geometries_preserved": True,
            "manual_GT_read_this_run": False, "student_predictions_read": False,
            "protected_files_checked": len(before), "protected_files_changed": 0,
            "source_actual_date_rows_checked": source_dates_checked,
            "regression_tests": tests["tests_run"], "ABSOLUTE_POSITION_ACCURACY": "UNKNOWN",
            "READY_FOR_GT10M_TRAINING": False, "READY_FOR_CITYWIDE": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        report = (
            "# XIAN-R5 工程集成运行\n\n"
            "ENGINEERING_PASS = YES；SCIENTIFIC_VALIDATION_PASS = NO。\n\n"
            "已用单一入口复演真实窗口的模块 B/A/I → R5 Presence → GT10m fraction/quality/ignore/weight → dataloader。"
            "本轮工程通过覆盖冻结模块输出的交接、真实窗口零接受位移分支、原ID与来源关联保持、固定目标网格及读取回归。\n\n"
            f"1662个原组件全部读取，53条单期来源关联及{len(links)}条关系行保持；"
            f"16个区域各冻结1组既有SAM输出，复演{len(sam)}个原生候选，全部保持未选择。"
            "完整上游SAM有1396组，本轮未重新运行分割网络。\n\n"
            f"原生候选屋顶面积分数覆盖{int((fractions['raw_fraction']>0).sum())}个正值像元，"
            f"面积守恒误差{fractions['area_conservation_error_m2']['raw_area_m2']:.9g} m²。"
            "原生fraction为未纠偏候选诊断。"
            + (f"{applied}个对象具有严格review证据并产生registered屋顶代理。" if applied else
               "无获准变换的registered fraction保持NoData，未将未知变换写成零向量。") + "\n\n"
            "实际日期为2018-11-01；仅一个日期，Presence全部U，未构造变化区间。"
            "GT10m监督valid=0、unknown=1、weight=0、state=255(ignore)，合格训练像元为0。"
            "SOURCE_SUPPORT_FRACTION仅表示有来源支持；原生fraction中的零值不等于已确认背景。\n\n"
            "53条来源关联只属于原组件，SAM子候选不继承；所有physical_id持久化为NULL。"
            "未读取人工GT几何或10m模型预测，历史GT参与identity的provenance仍保留。\n\n"
            "已复用既有sam_source_candidate、single_date_identity、apply_accepted_roof_grid_alignment、"
            "reconstruct、build_fraction_shadow、supervision_eligibility及load_rasters。"
            "按codebase-design和ponytail原则，仅增加合同、适配入口和回归测试，未建立第二套数据库或任务框架。\n\n"
            "非零位移分支已连接既有严格spatial-composition reader、reviewed-field adapter、"
            "V2 uncertainty复算和完整实例应用，再将结果交到registered fraction。"
            "该分支使用明确标识的合成测试验证，本次真实窗口仍无获准向量。"
            "后续缺口见UNRESOLVED_ISSUES.json：真实独立FIT/CHECK、实例资格、跨期身份/时间及"
            "新鲜RGB到合格GT自动运行尚未完成。工程PASS不代表系统科学验证完成。\n\n"
            f"{tests['tests_run']}项回归通过，{len(before)}项输入/源码SHA256前后相同。"
            "所有输出写入新目录。ABSOLUTE_POSITION_ACCURACY = UNKNOWN。\n\n"
            "复跑：使用相同FROZEN_INTEGRATION_MANIFEST.json，传入另一个不存在的--output目录；"
            "任何文件哈希变化、日期冲突、身份晋升、输入角色违规或目标网格不一致均停止。\n"
        )
        with (output / "FINAL_REPORT.md").open("x", encoding="utf-8") as stream:
            stream.write(report)
        hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
        save(output / "FINAL_RECEIPT.json", {**summary, "input_manifest_sha256": manifest_hash,
                                            "verified_input_sha256": after, "output_sha256": hashes})
        with (output / "FINAL_RECEIPT.sha256").open("x", encoding="ascii") as stream:
            stream.write(sha256(output / "FINAL_RECEIPT.json") + "  FINAL_RECEIPT.json\n")
        progress(json.dumps(summary, ensure_ascii=False))
    except Exception as error:
        save(output / "FAILED_RECEIPT.json", {"ENGINEERING_PASS": False,
             "error": repr(error), "traceback": traceback.format_exc(), "inputs": before})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--freeze-live-manifest", type=Path,
                        help="Freeze original-B0 RGB-only shadow inputs; no model run")
    parser.add_argument("--freeze-candidate-manifest", type=Path,
                        help="Freeze sealed B0/V2 products and existing I/source manifest; no model run")
    parser.add_argument("--candidate-products", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--a-receipt", type=Path)
    parser.add_argument("--a-candidates", type=Path, help="Existing spatial-composition manifest, not a newly estimated vector")
    parser.add_argument("--a-review", type=Path, help="Existing signed reviewed-cell GPKG; never generated here")
    args = parser.parse_args()
    if any((args.freeze_candidate_manifest, args.candidate_products, args.source_manifest)):
        if (not all((args.freeze_candidate_manifest, args.candidate_products, args.source_manifest))
                or any((args.freeze_manifest, args.freeze_live_manifest, args.manifest, args.output,
                        args.a_receipt, args.a_candidates, args.a_review))):
            parser.error("Use --freeze-candidate-manifest PATH --candidate-products DIR --source-manifest EXISTING_I_MANIFEST")
        from xian_r5.cached_roof_products import freeze_candidate_manifest
        print("Frozen cached candidate inputs:", freeze_candidate_manifest(
            args.freeze_candidate_manifest, args.candidate_products, args.source_manifest))
    elif args.freeze_live_manifest and not any((args.freeze_manifest, args.manifest, args.output,
                                             args.a_receipt, args.a_candidates, args.a_review)):
        from xian_r5.shadow_pipeline import freeze_live_manifest
        print("Frozen live inputs:", freeze_live_manifest(args.freeze_live_manifest, args.root))
    elif args.freeze_manifest and not args.freeze_live_manifest and not args.manifest and not args.output:
        print("Frozen artifacts:", freeze_baseline(args.freeze_manifest, args.root, a_receipt=args.a_receipt,
                                                  a_candidates=args.a_candidates, a_review=args.a_review))
    elif args.manifest and args.output and not any((args.freeze_manifest, args.freeze_live_manifest,
                                                  args.a_receipt, args.a_candidates, args.a_review)):
        mode = read_json(args.manifest).get("execution_mode")
        if mode == "CACHED_B0_V2_CANDIDATE_INTEGRATION_V1":
            from xian_r5.cached_roof_products import replay_candidate_products
            print(json.dumps(replay_candidate_products(args.manifest, args.output), ensure_ascii=False))
        elif mode == "LIVE_B0_RGB_SHADOW_V1":
            from xian_r5.shadow_pipeline import run_shadow
            run_shadow(args.manifest.resolve(), args.output.resolve())
        else:
            run(args.manifest.resolve(), args.output.resolve())
    else:
        parser.error("Choose --freeze-manifest PATH or --manifest PATH --output NEW_DIRECTORY")


if __name__ == "__main__":
    main()
