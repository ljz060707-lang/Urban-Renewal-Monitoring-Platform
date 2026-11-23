"""Bounded B04 zero-acceptance closure; consumes existing candidates only."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import fiona
import numpy as np
import rasterio
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from run_xian_r5_grounded_reconstruction import run
from qa_xian_r5_grounded_reconstruction import check

BASE = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction")
PACKET = BASE / "B04_TQA10_alignment_review_grid2_4x4_20260929_02"
PRIOR = BASE / "B04_TQA10_PIPELINE_RECONSTRUCTION_FREEZE_FIXEDCASE_20260929_02"
OUTPUT = BASE / "B04_TQA10_REVIEWED_REGISTRATION_CLOSURE_20260929_02"
STAGING = BASE / "B04_TQA10_REVIEWED_REGISTRATION_CLOSURE_INPUT_20260929_02"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as sink:
        json.dump(obj, sink, ensure_ascii=False, indent=2)
        sink.write("\n")

def features(path, layer):
    with fiona.open(path, layer=layer) as src:
        return list(src)

def main():
    if OUTPUT.exists() or STAGING.exists():
        raise FileExistsError("Closure output or input staging exists; refusing overwrite")
    manifest = read(PACKET / "REVIEW_PACKET_MANIFEST.json")
    packet_hashes = {str(p): digest(p) for p in PACKET.iterdir() if p.is_file()}
    for name, expected in manifest["created_files_sha256"].items():
        assert digest(PACKET / name) == expected, name
    source = manifest["spatial_composition_review"]
    spatial_path = Path(source["candidate_source_geopackage"])
    assert digest(spatial_path) == source["candidate_source_geopackage_sha256"]
    spatial = features(spatial_path, "spatial_alignment_candidates")
    template = PACKET / "absolute_base_review_template.gpkg"
    with fiona.open(template, layer="reviewed_image_alignment_cells") as src:
        rows = list(src)
        schema = src.schema.copy()
    assert len(rows) == manifest["base_grid_alignment_review"]["numeric_candidate_count"]
    assert all(not r["properties"]["reviewed"] and not r["properties"]["accepted"] for r in rows)
    raw = features(PRIOR / "building_lineage.gpkg", "roof_raw")
    target = unary_union([shape(r["geometry"]) for r in raw])
    # A transparent prioritization neighborhood, not a registration search window.
    neighbor = target.buffer(20).difference(target)
    priority = []
    for row in rows:
        p = dict(row["properties"])
        g = shape(row["geometry"])
        overlap = g.intersection(target).area
        near = g.intersection(neighbor).area
        rank = "P1" if overlap > 0 else "P2" if near > 0 else "P3"
        priority.append({
            "field_id": p["field_id"], "cell_id": p["cell_id"],
            "moving_date": p["moving_date"], "reference_source": p["reference_source"],
            "method": p["method"], "support_area": g.area,
            "target_overlap": overlap, "neighbor_overlap": near,
            "priority_rank": rank, "review_status": "UNRESOLVED_NO_COMPLETED_TIE_EVIDENCE",
            "review_scope": "DOCUMENTARY_ELIGIBILITY_AUDIT_NOT_HUMAN_TIE_REVIEW",
            "fit_pairs_verified": 0, "holdout_pairs_verified": 0,
            "reason": "No completed independently checked FIT/HOLDOUT ties; no absolute control; cycle closure is not positional accuracy",
        })
    priority.sort(key=lambda r: (r["priority_rank"], -r["target_overlap"], r["moving_date"], r["cell_id"]))
    counts = Counter(r["priority_rank"] for r in priority)
    statuses = Counter(str(r["properties"].get("candidate_status", r["properties"].get("status"))) for r in spatial)
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "direct_candidate_count": sum("SPATIAL" not in r["properties"]["method"] and not r["properties"]["anchor_count"] for r in rows),
        "multi_anchor_candidate_count": sum(bool(r["properties"]["anchor_count"]) for r in rows),
        "spatial_composition_candidate_count": sum("SPATIAL" in r["properties"]["method"] for r in rows),
        "cycle_screen_only_count": sum("CYCLE_SCREEN_ONLY" in r["properties"]["candidate_status"] for r in rows),
        "cycle_screen_failed_count": sum(v for k, v in statuses.items() if "CYCLE_SCREEN_FAILED" in k),
        "source_spatial_candidate_count": len(spatial), "source_spatial_status_counts": dict(statuses),
        "reviewable_total_count": len(rows), "pending_count": len(rows),
        "accepted_count": 0, "rejected_count": 0, "unresolved_count": len(rows),
        "ACCEPTED_COUNT": 0, "status": "NO_REVIEWABLE_ALIGNMENT_EVIDENCE",
        "stop_reason": "GE_TO_SENTINEL_REGISTRATION_EVIDENCE_INSUFFICIENT",
        "count_relationship": "18 numeric = same 18 spatial cycle-screen-only cells; not 36. Three failed source cells excluded, not human rejections.",
        "priority_counts": dict(counts), "priority_review_candidates": counts["P1"] + counts["P2"],
        "documentary_audit_counts": dict(counts), "completed_human_tie_review_counts": {"P1": 0, "P2": 0, "P3": 0},
        "priority_geometry_source": str(PRIOR / "building_lineage.gpkg") + "#roof_raw",
        "priority_geometry_sha256": digest(PRIOR / "building_lineage.gpkg"),
        "priority_geometry_semantics": "All 13 frozen B04 roof observations; 20m exterior buffer for prioritization only, not verified stable ground or a new matching window",
        "independent_ground_control_available": False, "relative_uncertainty_measured": False,
        "human_review_fabricated": False, "new_registration_search": False,
        "historical_packet_sha256": packet_hashes,
        "relative_date_layer_status": manifest["relative_date_pair_diagnostics"]["status"],
    }
    STAGING.mkdir()
    accepted = STAGING / "B04_ACCEPTED_ALIGNMENT_CELLS.gpkg"
    with fiona.open(accepted, "w", driver="GPKG", layer="reviewed_image_alignment_cells", schema=schema, crs="EPSG:32649"):
        pass
    write(STAGING / "AUDIT_BEFORE_RUN.json", audit)
    print(json.dumps({"audit": audit, "next": "existing frozen pipeline"}, ensure_ascii=False), flush=True)
    reference = Path(source["reference_raster"])
    evidence = run("B04-TQA10", OUTPUT, base_raster=reference,
                   reviewed_image_alignment_layer=accepted,
                   spatial_alignment_composition_manifest=Path(source["candidate_source_manifest"]),
                   freeze_registration_candidates=True)
    shutil.copy2(accepted, OUTPUT / accepted.name)
    assert digest(accepted) == digest(OUTPUT / accepted.name)
    audit["adapter_receipt"] = evidence["reviewed_image_alignment_source"]
    assert audit["adapter_receipt"]["accepted_cell_count"] == 0
    assert all(digest(p) == h for p, h in packet_hashes.items())
    audit["historical_packet_unchanged"] = True
    write(OUTPUT / "ALIGNMENT_REVIEW_RECEIPT.json", audit)
    with (OUTPUT / "B04_PRIORITY_ALIGNMENT_REVIEW.csv").open("x", newline="", encoding="utf-8-sig") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(priority[0]))
        writer.writeheader()
        writer.writerows(priority)
    lineage = OUTPUT / "building_lineage.gpkg"
    for layer in ("canonical_building_geometry", "canonical_geometry_evidence"):
        with fiona.open(lineage, layer=layer) as src:
            kwargs = {"crs_wkt": src.crs_wkt} if src.crs_wkt else {}
            with fiona.open(OUTPUT / "canonical_buildings.gpkg", "w", driver="GPKG", layer=layer, schema=src.schema, **kwargs) as dst:
                dst.writerecords(src)
    old_raw = {r["properties"]["uid"]: shape(r["geometry"]).wkb_hex for r in raw}
    new_raw = {r["properties"]["uid"]: shape(r["geometry"]).wkb_hex for r in features(lineage, "roof_raw")}
    assert old_raw == new_raw, "Raw roofs changed"
    registration_states = []
    for obs in evidence["observations"]:
        registration_states.append({"observation_id": obs["uid"], "actual_date": obs["actual_date"],
            "raw_roof_id": obs["uid"], "raw_geometry_hash": hashlib.sha256(bytes.fromhex(new_raw[obs["uid"]])).hexdigest(),
            "registration_status": "UNRESOLVED", "canonical_status": "UNRESOLVED_NO_CONFIRMED_PHYSICAL_ID",
            "registration_field_id": None, "cell_id": None, "dx": None, "dy": None,
            "transform_type": None, "review_basis": None, "evidence_path": None, "evidence_sha256": None,
            "relative_registration_uncertainty_m": None, "positional_uncertainty_m": None})
    write(OUTPUT / "OBSERVATION_REGISTRATION_STATUS.json", registration_states)
    m = read(OUTPUT / "PRODUCT_MANIFEST.json")
    # Extend only this newly created manifest, retaining the pipeline's full contract.
    m["alignment_review_audit"] = audit
    m["canonical_standalone_geopackage"] = "canonical_buildings.gpkg"
    m["observation_registration_status"] = "OBSERVATION_REGISTRATION_STATUS.json"
    m["registration_status_products"] = [p.name for p in OUTPUT.glob("*_registration_status.tif")]
    (OUTPUT / "PRODUCT_MANIFEST.json").write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    r4 = Path(evidence["input_paths"]["r4_status"])
    r4_hash = digest(r4)
    r4_ok = r4_hash == evidence["input_sha256"]["r4_status_before"] == evidence["input_sha256"]["r4_status_after"]
    write(OUTPUT / "R4_HASH_RECHECK.json", {"path": str(r4), "before": evidence["input_sha256"]["r4_status_before"], "after": r4_hash, "unchanged": r4_ok, "scope": "R4 RUN_STATUS.json plus pipeline tracked source hashes; not a recursive full-R4 inventory"})
    qa = check(OUTPUT, "ENGINEERING_QA.json")
    histograms = {}
    for filename in m["non_ground_change_products"]["registered"]:
        with rasterio.open(OUTPUT / filename) as src:
            values, frequencies = np.unique(src.read(1), return_counts=True)
            histograms[filename] = dict(zip(map(str, values), map(int, frequencies)))
    states = {label: sum(h.get(str(code), 0) for h in histograms.values()) for code, label in [(1, "stable"), (2, "addition"), (3, "demolition"), (4, "unresolved")]}
    counts_out = {
        "alignment_candidates_total": len(rows), "priority_candidates": counts["P1"] + counts["P2"],
        "accepted_cells": 0, "rejected_cells": 0, "unresolved_cells": len(rows),
        "roof_raw": len(new_raw), "registered_accepted": len(features(lineage, "roof_registered")),
        "registered_unresolved": len(new_raw), "identity_candidate_edges": m["candidate_identity_edge_count"],
        "identity_candidate_tracks": m["candidate_track_id_codebook"]["candidate_identity_label_count"],
        "physical_ids": m["canonical_geometry"]["physical_id_codebook"]["count"],
        "canonical_geometries": m["canonical_geometry"]["resolved_count"],
        "presence": m["presence_state_counts"], "registered_state_pixels_summed_over_intervals": states,
        "registered_change_histograms": histograms,
        "confirmed_ground_footprints": evidence["ground_footprints_confirmed"],
        "confirmed_ground_changes": evidence["confirmed_changes"],
        "object_transition_count": len(evidence["transitions"]),
    }
    complete = qa["structural_qa"] == "PASS" and r4_ok
    status = {"ALIGNMENT_REVIEW_PACKET_AUDITED": "YES", "ACCEPTED_ALIGNMENT_CELLS": 0,
        **{key: "COMPLETE" if complete else "INCOMPLETE" for key in ["REVIEWED_REGISTRATION_PIPELINE", "REGISTERED_ROOF_PIPELINE", "CANONICAL_GEOMETRY_PIPELINE", "PRESENCE_LAYER", "REGISTERED_CHANGE_PIPELINE"]},
        "GROUND_FOOTPRINT_RECOVERY": "UNRESOLVED", "GROUND_CHANGE_PRODUCT": "UNRESOLVED", "READY_FOR_CITYWIDE": "NO"}
    write(OUTPUT / "CLOSURE_STATUS.json", {"counts": counts_out, "status": status, "stop_reason": audit["stop_reason"], "raw_geometry_unchanged": True})
    summary = "# Alignment review summary\n\n18 numeric candidates are the same 18 spatial-composition cycle-screen-only rows, not 36. Source spatial layer contains 21 rows, of which 3 failed cycle screening and are excluded.\n\n" + json.dumps(audit, ensure_ascii=False, indent=2)
    (OUTPUT / "ALIGNMENT_REVIEW_SUMMARY.md").write_text(summary, encoding="utf-8")
    answers = [
        "审核层实际18个候选；来源空间组合层21个，其中3个cycle-screen-failed未进入审核层。",
        "18 numeric和18 spatial-composition是同一批记录，不是36。",
        f"完成文档资格核对 P1={counts['P1']}、P2={counts['P2']}；完成人工地面tie审核 P1=0、P2=0。空间覆盖不代表已识别稳定地面。",
        "accepted=0。", "rejected=0；未把缺少证据当成已证伪。", "unresolved=18；历史packet仍保留PENDING。",
        "没有任何actual date取得accepted registration。", "没有首次生成实际经审核的registered roof；本轮完成零接受分支。",
        f"accepted registered roof=0；unresolved observations={len(new_raw)}。",
        "raw roof全部保留，按uid逐一比较WKB，与冻结输入完全相同。",
        f"physical ID={counts_out['physical_ids']}。", f"canonical geometry={counts_out['canonical_geometries']}；未确认身份的观测逐一标为UNRESOLVED。",
        "复用BEST_OBSERVATION_MEDOID：最小平均边界Hausdorff距离，随后低不确定性和UID；不使用union/intersection，不当作ground。",
        f"Presence={m['presence_state_counts']}；0仅允许reviewed absence；背景0不是absence。",
        f"registered变化像元（两区间合计）={states}；对象transition={len(evidence['transitions'])}，不能把像元数当建筑数。",
        f"confirmed ground footprint={evidence['ground_footprints_confirmed']}。", "ground change仍unresolved，没有复制roof/canonical填充。",
        f"未修改R4，RUN_STATUS和受跟踪输入哈希复核={r4_ok}；未声称全R4递归文件清单审计。",
        "未修改原始GE；本轮实际读取影像哈希由pipeline和QA复核。", "未运行citywide，未启动新配准搜索或模型训练。",
        f"零接受/未解决工程闭环={complete}，不等于实际偏移已成功校正。",
        "地面足迹恢复仍待可靠地面依据；无独立控制，绝对精度未知。",
    ]
    report = "# B04 reviewed registration closure\n\nGE_TO_SENTINEL_REGISTRATION_EVIDENCE_INSUFFICIENT\n\n" + "\n\n".join(f"{i}. {answer}" for i, answer in enumerate(answers, 1))
    report += "\n\n## Counts\n\n```json\n" + json.dumps(counts_out, ensure_ascii=False, indent=2) + "\n```\n\n## Status\n\n```json\n" + json.dumps(status, indent=2) + "\n```\n\nCOMPLETE仅指允许零接受的工程分支闭环。无真实FIT/HOLDOUT证据，未编造人工审核或2m精度。到此停止，不进入全市。\n"
    (OUTPUT / "B04_REVIEWED_REGISTRATION_CLOSURE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": counts_out, "status": status, "qa_errors": qa["errors"]}, ensure_ascii=False), flush=True)
    if not complete:
        raise RuntimeError("Closure QA failed")

if __name__ == "__main__":
    main()
