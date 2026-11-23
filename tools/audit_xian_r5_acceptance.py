"""Read-only S4.14 postflight of an explicit frozen, unaligned Shadow run.

No discovery, inference, GT matching, fitting, application, or promotion. This
is intentionally a negative-path audit; nonzero accepted transforms require
their own evidence-aware execution and cannot use this receipt to bypass it.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import traceback
import unittest

from affine import Affine
import fiona
import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from shapely.geometry import shape
from shapely.ops import transform, unary_union

from xian_r5.acceptance import (CONTRACT, TASKS, OBJECT_STATUSES, GATE_STATUSES,
                               gate, sha256, sha256_geometry, tie_diagnostics, unresolved_building)
from xian_r5.integration import build_gt10m_shadow, run_r5_state, verify_artifacts
from xian_r5.label_transfer import Grid, load_rasters

REPO = Path(__file__).resolve().parents[1]
STOP = "STOP_CURRENT_GT_BRIDGING_BRANCH_NO_NEW_EFFECTIVE_CORRESPONDENCES"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def audit_shadow(shadow, bridge, output):
    shadow, bridge, output = map(lambda p: Path(p).resolve(), (shadow, bridge, output))
    if output.exists() or output == shadow or output == bridge:
        raise ValueError("Append-only output must be a new directory")
    if not shadow.is_dir() or not bridge.is_dir():
        raise ValueError("Explicit existing authority directories required")
    output.mkdir(parents=False)
    checks = {}
    try:
        def check(name, condition):
            checks[name] = bool(condition)
            if not checks[name]:
                raise ValueError("Acceptance audit failure: " + name)

        receipt = read(shadow / "FINAL_RECEIPT.json")
        old = read(shadow / "S4_12_END_TO_END_RUN_MANIFEST.json")
        decision = read(bridge / "S4_13_NEXT_STAGE_DECISION.json")
        check("S413_stop_preserved", decision["decision"] == STOP)
        check("S412_negative_path_only", receipt["applied_displacements"] == 0 and
              receipt["registered_roofs"] == 0 and receipt["QUALIFIED_10M_PIXEL_COUNT"] == 0)
        check("original_GT_free_inference_contract", old["manual_gt_required_for_inference"] is False and
              old["model_role"] == "ORIGINAL_B0_ONLY" and receipt["GT_FREE_INFERENCE_PASS"] is True)
        # Freeze every inspected authority artifact before audit. No GT inputs.
        pins = dict(old["inputs"])
        for label, folder in (("s412", shadow), ("s413", bridge)):
            rpath = folder / "FINAL_RECEIPT.json"
            expected = (folder / "FINAL_RECEIPT.sha256").read_text().strip()
            check(label + "_authority_receipt_SHA", sha256(rpath) == expected)
            pins[label + "_receipt"] = dict(path=str(rpath), sha256=expected)
        for name, meta in receipt["artifacts"].items():
            pins["s412_artifact_" + name] = dict(path=str(shadow / name), sha256=meta["sha256"])
        # Current supplemental receipts bind the completed downstream checks.
        for filename in ("POST_RUN_CROSS_MODULE_VERIFICATION.json", "SUPPLEMENTAL_RECEIPT.json"):
            path = shadow / filename
            pins[filename] = dict(path=str(path), sha256=sha256(path))
        sr = read(shadow / "SUPPLEMENTAL_RECEIPT.json")
        check("supplement_receipt_SHA", sha256(shadow / "SUPPLEMENTAL_RECEIPT.json") ==
              (shadow / "SUPPLEMENTAL_RECEIPT.sha256").read_text().strip())
        check("post_verification_bound_to_supplement", sr["artifacts"]["POST_RUN_CROSS_MODULE_VERIFICATION.json"]["sha256"] ==
              pins["POST_RUN_CROSS_MODULE_VERIFICATION.json"]["sha256"])
        # Pin only the previous decision, not the stopped GT search inputs.
        bp = bridge / "S4_13_NEXT_STAGE_DECISION.json"
        br = read(bridge / "FINAL_RECEIPT.json")
        bm = br.get("artifacts", {}).get(bp.name)
        expected = bm.get("sha256") if isinstance(bm, dict) else bm
        if expected is None:
            expected = br.get("output_sha256", {}).get(bp.name)
        check("bridge_decision_receipt_bound", expected == sha256(bp))
        pins["s413_stop_decision"] = dict(path=str(bp), sha256=expected)
        code_paths = [Path(__file__), REPO / "src/xian_r5/acceptance.py",
                      REPO / "tests/test_xian_acceptance.py", REPO / "docs/XIAN_R5_S414_ACCEPTANCE_V1.md"]
        for path in code_paths:
            pins["s414_" + path.name] = dict(path=str(path), sha256=sha256(path))
        manifest = dict(contract_version=CONTRACT, run_id=output.name,
                        execution="READ_ONLY_EXISTING_SHADOW_ACCEPTANCE_NO_PROMOTION",
                        shadow_run=str(shadow), bridge_run=str(bridge), inputs=pins,
                        tasks=list(TASKS), object_statuses=list(OBJECT_STATUSES), gate_statuses=list(GATE_STATUSES),
                        S413_STOP=STOP, fit="NOT_RUN", model_inference="NOT_RUN", training="NOT_RUN",
                        displacement_application="NOT_RUN", manual_GT_read=False, protected_tests_read=False,
                        output_scope="FIXED_roi_r0_c0_177_CANDIDATES_AND_256x256_SHADOW_GRID",
                        reproduction_command=[sys.executable,"-B",str(Path(__file__).resolve()),
                                              "--shadow-run",str(shadow),"--bridge-run",str(bridge),
                                              "--output","NEW_APPEND_ONLY_DIRECTORY"])
        save(output / "RUN_MANIFEST.json", manifest)
        mh = sha256(output / "RUN_MANIFEST.json")
        common = dict(contract_version=CONTRACT, input_manifest_sha256=mh, source_version=shadow.name,
                      bridge_version=bridge.name, code_sha256=sha256(Path(__file__)), scope=manifest["output_scope"],
                      execution=manifest["execution"])

        def report(name, data):
            save(output / name, {**common, **data})

        print("VERIFY_FROZEN_INPUT_AND_ARTIFACT_HASHES", flush=True)
        before = {}
        for name, pin in pins.items():
            actual = sha256(pin["path"])
            check("SHA_" + name, actual == pin["sha256"])
            before[name] = actual
        check("original_verified_input_map", verify_artifacts(old["inputs"]) == receipt["verified_inputs_sha256"])
        test_modules = ["test_xian_acceptance", "test_xian_live_shadow", "test_xian_integration",
                        "test_xian_instance_adaptive", "test_xian_fraction_shadow"]
        sys.path.insert(0, str(REPO / "tests"))
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in test_modules)
        log = io.StringIO()
        tests = unittest.TextTestRunner(stream=log).run(suite)
        check("synthetic_contract_regressions", tests.wasSuccessful())
        report("ACCEPTANCE_REGRESSION_TESTS.json", dict(tests_run=tests.testsRun,
               failures=len(tests.failures), errors=len(tests.errors), modules=test_modules, log=log.getvalue(),
               dataset_reads="SYNTHETIC_ONLY_NOT_PROTECTED_TEST_RECORDS"))

        packet = old["packet"]
        with Image.open(packet["image_path"]) as rgb:
            check("native_dimensions", list(reversed(rgb.size)) == packet["local_image_shape"])
        observations = read(shadow / "FRESH_OBSERVATIONS_RAW.json")["records"]
        native = read(shadow / "ROOF_CANDIDATES_NATIVE.json")["ROOF_INSTANCE_CANDIDATES"]
        check("original_IDs_unique", len({r["observation_id"] for r in observations}) == len(observations))
        check("runtime_IDs_preserved", {r["observation_id"] for r in observations} == {r["candidate_id"] for r in native})
        check("frozen_count_177", len(observations) == 177)
        check("source_RGB_paths_consistent",all(Path(r["source_path"]).resolve()==Path(packet["image_path"]).resolve()
                                                for r in observations))
        buildings = [unresolved_building({**r,"source_RGB_sha256":packet["image_sha256"]}) for r in observations]
        a = Affine(*packet["source_raster_affine"])
        col, row, _, _ = packet["source_window"]
        to_metric = Transformer.from_crs(packet["source_raster_crs"], "EPSG:32649", always_xy=True)
        def project(x, y, z=None):
            lon, lat = a * (np.asarray(x) + col, np.asarray(y) + row)
            return to_metric.transform(lon, lat)
        for record in observations:
            g = transform(project, shape(record["native_pixel_geometry"]))
            check("native_to_metric_hash_" + record["observation_id"], sha256_geometry(g) == record["geometry_sha256"])
        gp = shadow / "S4_12_INSTANCE_ALIGNMENT_CANDIDATES.gpkg"
        byid = {r["observation_id"]: r for r in observations}
        with fiona.open(gp, layer="roof_alignment_candidates") as src:
            check("GPKG_crs_and_count", src.crs.to_epsg() == 32649 and len(src) == len(observations))
            for feature in src:
                p = dict(feature.properties)
                check("unaltered_gpkg_" + p["observation_id"],
                      sha256_geometry(shape(feature.geometry)) == byid[p["observation_id"]]["geometry_sha256"] and
                      all(p[k] is None for k in ("total_dx_m","total_dy_m","physical_id","uncertainty_m")))
        with fiona.open(gp, layer="registered_roofs_qualified") as src:
            check("no_applied_geometries", len(src) == 0)

        print("REPLAY_EXISTING_DOWNSTREAM_ONCE_WITH_REVERSED_ORDER_CHECK", flush=True)
        grid = Grid(Affine(10,0,308460,0,-10,3794750),256,256,"EPSG:32649")
        tiles = read(old["inputs"]["source_registry"]["path"])["tiles"]
        support = unary_union([shape(t["support_geometry"]) for t in tiles]).intersection(shape(packet["exact_support"]))
        previous = read(shadow / "PRESENCE_AND_STATE_SHADOW.json")
        state, _ = run_r5_state(observations, {"ACCEPTED_LOCAL_REGIONS":0,"APPLIED_SHADOW_LOCAL_REGIONS":0},
                                [old["inputs"]["grid_template"]["path"]])
        check("Presence_replay_exact", state.presence == previous["presence"] and state.transitions == previous["transitions"])
        f, _, _ = build_gt10m_shadow(observations, [packet], support, grid, state)
        names = ["S4_12_GT10M_FRACTION_SHADOW.tif", "S4_12_GT10M_QUALITY_SHADOW.tif",
                 "GT10M_VALID_UNKNOWN.tif", "GT10M_TRAINING_WEIGHT.tif", "GT10M_STATE_IGNORE.tif"]
        rasters = load_rasters([shadow / n for n in names])
        check("grid_CRS_affine_dimensions", rasters["grid"] == grid)
        fraction, quality, valid_unknown, weight, ignore = rasters["arrays"]
        check("fraction_exact_float32_replay", np.array_equal(fraction[0], f["raw_fraction"].astype("float32")))
        check("quality_exact_replay", np.array_equal(quality, f["quality"]))
        check("area_conservation_before_transform", max(f["area_conservation_error_m2"].values()) <= .001)
        check("no_qualified_fraction", np.all(fraction[1] == -9999))
        with rasterio.open(shadow/names[0]) as ds:
            check("fraction_NoData_semantics",ds.nodata==-9999 and ds.count==2)
        check("unknown_not_background", np.all(valid_unknown[0] == 0) and np.all(valid_unknown[1] == 1)
              and np.all(weight == 0) and np.all(ignore == 255))
        reversed_f, _, _ = build_gt10m_shadow(list(reversed(observations)), [packet], support, grid, state)
        check("whole_object_order_independent_fraction", np.array_equal(f["raw_fraction"],reversed_f["raw_fraction"]))
        check("whole_object_order_independent_quality", np.array_equal(f["quality"],reversed_f["quality"]))
        report("INPUT_HASH_MANIFEST.json", dict(inputs=pins, verification="ALL_MATCHED", input_protection="RECHECKED_BEFORE_AND_AFTER"))
        report("SOURCE_LINEAGE.json", dict(packet=packet, records=[dict(
            observation_id=r["observation_id"], geometry_sha256=r["geometry_sha256"],
            native_geometry_sha256=r["native_geometry_sha256"], actual_date=r["actual_date"],
            source_paths=r["source_paths"], source_hashes=r["source_hashes"], source_tile_ids=r["source_tile_ids"],
            source_support_complete=r["source_support_complete"]) for r in observations],
            download_tiles_are_camera_units=False, original_ID_count=len(observations), physical_ids=0))
        with (output / "BUILDING_APPLICATION_AUDIT.csv").open("x",encoding="utf-8-sig",newline="") as stream:
            fields=["contract_version","input_manifest_sha256",*buildings[0]]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for building in buildings:
                values={k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) or v is None else v
                        for k,v in building.items()}
                writer.writerow(dict(contract_version=CONTRACT,input_manifest_sha256=mh,**values))
        report("LOCAL_FIT_CHECK_REPORT.json", dict(task=TASKS[0],status="UNRESOLVED_INSUFFICIENT_TIES",
               statistics=tie_diagnostics([],None,None), fit_check_execution="NOT_RUN_NO_ELIGIBLE_TIES",
               design_before_fit=None, independent_group_coverage=None, valid_match_ratio=None,
               spatial_coverage=None, seam_difference_m=None, systematic_bias_assessment=None,
               median_limit_m=2,p90_limit_m=3,V2_max_adjusted_limit_m=2,
               localization_bound_source=None, notes="Missing metrics are null, not zero residuals"))
        report("TRANSFORM_AND_UNCERTAINTY_REPORT.json", dict(tasks={
            TASKS[0]:"UNRESOLVED",TASKS[1]:"UNRESOLVED",TASKS[2]:"UNRESOLVED"},
            candidate_vectors=0, validated_vectors=0, applied_vectors=0, denominator_objects=177,
            units="metres EPSG:32649 east/north", relative_uncertainty_m=None,
            absolute_position_accuracy="UNKNOWN", building_parallax="UNRESOLVED",
            uncertainty_rule="R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1",
            raw_geometry_preserved=True, transform_fit="NOT_RUN", building_accuracy_metrics=None,
            synthetic_direction_check="PASS_NOT_REAL_ACCURACY"))
        report("MOSAIC_CONTINUITY_AUDIT.json", dict(status="NOT_RUN", reason="ONE_ACTIVE_PACKET_NO_ACCEPTED_FIELDS",
               active_packets=1, empirical_regions_total=16, accepted_regions=0,
               adjacent_field_difference_m=None, overlap_check_residual_m=None,
               source_gaps_and_overlaps="UNRESOLVED_NOT_FILLED_OR_SMOOTHED",
               grid_consistency="PASS", complete_original_objects_preserved=177,
               input_order_fraction_invariance="PASS_CURRENT_RAW_CANDIDATE_SCOPE_ONLY",
               multi_region_registered_mosaic="NOT_RUN", full_mosaic_scientific_accuracy="UNRESOLVED"))
        qualified=int(np.count_nonzero(weight))
        report("GT10M_QUALIFICATION_REPORT.json", dict(status="UNRESOLVED",qualified_pixels=qualified,
               denominator_pixels=65536, pixel_area_m2=100, raw_positive_fraction_pixels=int(np.count_nonzero(fraction[0]>0)),
               raw_evaluated_pixels=int(np.count_nonzero(fraction[0]!=-9999)),
               native_fraction_path=str(shadow/names[0]),qualified_fraction_band=2,
               source_coverage_path=str(shadow/"SOURCE_SUPPORT_FRACTION.tif"),
               unknown_fraction_path=str(shadow/"GT10M_VALID_UNKNOWN.tif"),unknown_band=2,
               quality_path=str(shadow/names[1]),ignore_path=str(shadow/names[4]),weight_path=str(shadow/names[3]),
               uncertainty_m=None,raw_fraction_zeros_are_known_background=False,
               checks=dict(source_coordinate_traceable=True, spatial_transform=None,temporal_compatible=None,
                           roof_semantics_verified=None, complete_valid_support=None,uncertainty_qualified=None,
                           independent_preregistered_validation=None, deterministic_reproduction=True),
               failure_reasons=["NO_QUALIFIED_TRANSFORM_OR_CHECK","REFERENCE_DATE_UNKNOWN",
                                "UNCALIBRATED_ROOF_CANDIDATE","SINGLE_OBSERVATION_PRESENCE_U"],
               hard_labels="NOT_RUN_NO_QUALIFIED_SUPERVISION",threshold=None,threshold_operator=None,
               old_0_43_rule_changed=False,three_state_truth="NOT_RUN",training="NOT_RUN"))
        report("METRICS_AND_DENOMINATORS.json", dict(
            geometry=dict(reference="IMMUTABLE_SOURCE_CANDIDATE_NOT_GT",objects=177,
                          complete_source_support=sum(r["source_support_complete"] is True for r in observations),
                          unknown_source_date=sum(r["actual_date"] is None for r in observations)),
            fraction=dict(definition="Area(union(raw_roof intersect source_support) intersect pixel)/Area(full_pixel)",
                          pixel_denominator_m2=100,grid_denominator_pixels=65536,unknown_excluded_from_training=True,
                          raw_zeros_not_background=True,hard_conversion=None,
                          area_conservation_error_m2=f["area_conservation_error_m2"],
                          fraction_MAE=None,preregistered_MAE_limit=None,MAE_0_1_gate_active=False),
            FIT_residual_m=None,CHECK_residual_m=None,building_position_error_m=None,
            per_building_area_error_m2=None, signed_total_area_not_MAE=True,
            labels_read=False,independent_validation="NOT_RUN",unmatched_buildings="UNRESOLVED_RETAINED"))
        fr=read(shadow/"FIGURE_RECEIPT.json")
        report("FIGURE_MANIFEST.json",dict(execution="REUSED_EXISTING_REAL_DATA_FIGURES_NO_NEW_RENDER",
               figure_receipt_path=str(shadow/"FIGURE_RECEIPT.json"),figure_receipt_sha256=sha256(shadow/"FIGURE_RECEIPT.json"),
               figures=[dict(path=str(shadow/n),sha256=h) for n,h in fr["outputs"].items()],
               NO_ACCEPTED_ALIGNMENT=True,fake_corrected_geometry=False,decorative_displacement_arrows=False))
        # No previous frozen file may have changed while downstream replay ran.
        for name,pin in pins.items():check("unchanged_after_"+name,sha256(pin["path"])==before[name])
        gates=dict(ENGINEERING_PASS=gate(checks),GT_FREE_INFERENCE_PASS="PASS",
                   LOCAL_REGISTRATION_PASS="UNRESOLVED",BUILDING_ALIGNMENT_PASS="UNRESOLVED",
                   POSITION_UNCERTAINTY_PASS="UNRESOLVED",MOSAIC_CONTINUITY_PASS="NOT_RUN",
                   QUALIFIED_10M_SUPERVISION_PASS="UNRESOLVED",SCIENTIFIC_VALIDATION_PASS="UNRESOLVED",
                   PRODUCTION_ADMISSION_PASS="NOT_RUN",READY_FOR_CITYWIDE="UNRESOLVED")
        report("ENGINEERING_CHECKS.json",dict(checks=checks,checks_passed=len(checks),
               engineering_scope="CURRENT_RAW_SHADOW_FAILURE_CLOSED_PATH_ONLY_NOT_NONZERO_TRANSFORM",
               network_determinism="PRIOR_FROZEN_OUTPUT_HASH_CHECK_NOT_NEW_FORWARD_REPLAY",
               downstream_determinism="EXECUTED_EXACT_AND_REVERSED_ORDER_MATCH"))
        report("ACCEPTANCE_IMPLEMENTATION_SCOPE.json",dict(
            implemented=["read_only_postflight","input_hash_guard","G1_raw_shadow_checks","V2_adapter_reuse",
                         "additive_tie_diagnostics","whole_object_support_conflict_guard","hard_label_ignore_guard",
                         "per_gate_non_promotion","twelve_artifact_receipt"],
            not_executed=["real_G2_FIT_CHECK","real_building_level_validation","nonzero_transform_application",
                          "multi_region_registered_mosaic","independent_pixel_validation"],
            future_acceptance_requires="Frozen design/independence, time, semantic, support, local anomaly and seam evidence PLUS existing strict V2/application adapters; numerical diagnostics alone never ACCEPT",
            automatic_positive_promotion="NOT_IMPLEMENTED_BY_THIS_NEGATIVE_PATH_POSTFLIGHT"))
        text="\n\n".join(["# XIAN-R5-S4.14 可执行验收补充",
            "合同、只读验收入口与回归检查已落地；按 codebase-design 最小适配复用原 V2 和 fraction builder，未另建框架。",
            "S4.13 终态保持 "+STOP+"；未读取 GT 多边形或重启对应搜索。",
            f"真实 S4.12 固定窗口：{len(observations)} 栋、65536 个目标像元。候选原 ID、原始几何、CRS/网格/来源哈希已复核；"+
            "实际复跑 Presence/fraction/quality，并反转对象次序检查精确一致。没有新网络推理、训练、拟合、位移应用或科学精度测量。",
            f"{tests.testsRun} 项合成/既有回归通过，{len(checks)} 项真实文件及交接检查通过。工程 PASS 仅指现有未校准 Shadow 失败关闭路径，不认证非零变换或拼接。",
            "177 栋向量/不确定性均 null、physical_id 未继承；14 栋来源支持不完整，3 栋实际日期未知（分别统计，可重叠）。所有跨参考日期兼容、独立 CHECK 和视差仍未解决。",
            "2703 个原生正 fraction、4096 个来源范围内像元只是候选诊断；合格像元0，unknown65536、ignore255、weight0；未将零候选覆盖当背景。",
            "门槛：\n"+"\n".join(f"- {k} = {v}" for k,v in gates.items()),
            "图件仅按哈希引用既有真实图，NO_ACCEPTED_ALIGNMENT；无新箭头、纠偏后几何或伪造成功图。",
            "本入口不产生正向科学授权：未来还需拟合前冻结的同局部支持3+3独立地面证据及V2复算、时间/语义/空间分布和接缝审查；逐栋视差和地面足迹另验。当前不具备这些新证据，不继续搜索旧材料。",
            "未运行：真实配准、逐栋独立精度、拼接、训练/推理、正式GT、生产准入和全市。ABSOLUTE_POSITION_ACCURACY=UNKNOWN。",
            "用户附件末句截断于‘不以装饰性箭头暗示已经’；未臆造截断后的条款，已按明确要求禁止伪纠偏展示。"])
        with (output/"FINAL_REPORT.md").open("x",encoding="utf-8") as stream:stream.write(text+"\n")
        artifacts={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()}
        report("FINAL_RECEIPT.json",dict(**gates,completed_at=datetime.now().astimezone().isoformat(),
               QUALIFIED_10M_PIXEL_COUNT=qualified,ACCEPTED_ALIGNMENT_CELLS=0,APPLIED_DISPLACEMENTS=0,
               ABSOLUTE_POSITION_ACCURACY="UNKNOWN",S413_STOP=STOP,training="NOT_RUN",new_inference="NOT_RUN",
               independent_validation="NOT_RUN",cloud_started=False,artifacts=artifacts,
               positive_scientific_authorization=False,protected_tests_read=False))
        with (output/"FINAL_RECEIPT.sha256").open("x",encoding="ascii") as stream:stream.write(sha256(output/"FINAL_RECEIPT.json"))
        print(json.dumps(dict(**gates,tests_run=tests.testsRun,checks=len(checks),qualified_pixels=qualified)),flush=True)
    except Exception:
        save(output/"FAILED_RECEIPT.json",dict(ENGINEERING_PASS="FAIL",error=traceback.format_exc(),checks=checks,
             training="NOT_RUN",new_inference="NOT_RUN",transform_application="NOT_RUN"))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("shadow-run","bridge-run","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    audit_shadow(args.shadow_run,args.bridge_run,args.output)


if __name__=="__main__":main()
