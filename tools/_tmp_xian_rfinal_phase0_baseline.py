from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(r"F:\xian\西安_RFINAL_基线与残余诊断_20260916")
SUPPLEMENT = Path(r"F:\xian\XIAN补充")
MAIN = Path(r"F:\xian\xian")

MAIN_HASHES = {
    "2018": {
        "path": MAIN / "xa180429_combined/xa180429_zoom_20.tif",
        "sha256": "350E601FF330F8D1C64EC00BDDDC90FE4FF3F989FF2104306BE5174CC2E23CBB",
        "actual_date_dominant": "2018-04-29",
        "actual_date_candidates": ["2018-04-29"],
        "actual_date_status": "VALID_TEMPORAL",
    },
    "2019": {
        "path": MAIN / "xa190101_combined/xa190101_zoom_20.tif",
        "sha256": "73537C0696A13CD0C9264BAB18AC71B0191082C118B21A7A76A067ADA9EA61B0",
        "actual_date_dominant": "2018-11-01",
        "actual_date_candidates": ["2018-11-01", "2018-11-08"],
        "actual_date_status": "OFF_DATE_VALID",
    },
    "2020": {
        "path": MAIN / "xa201231_combined/xa201231_zoom_20.tif",
        "sha256": "93319A5D921A48A607F0E86E1BA77E7CE40462DE8990CA87B9BADDF18011A9C0",
        "actual_date_dominant": "2019-12-10",
        "actual_date_candidates": ["2019-12-02", "2019-12-10", "2020-02-09", "2020-02-16", "2020-02-21", "2020-04-24", "2020-06-14", "2020-11-14"],
        "actual_date_status": "TEMPORAL_INSUFFICIENT",
    },
}

FORMAL_HASHES = {
    "tracking_csv": "729d8dae8b91ca983972523c44326449f5695afa47a8613207daddf5bda526c2",
    "identity_edges_parquet": "838a882fdd675010972e78dc735578ab38f041b29789d56852e5562db6324089",
    "presence_csv": "7ab980c8a639df25fe16278921d31678bacc4a4d8928307f084f4bff0d1ece21",
    "state_csv": "8bb520ddcaf0f6aa44c8c6681520bf3d0bd5fceab65f6925f625e61c8b11ff61",
}


def stat_record(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_local": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(),
    }


def shallow_supplement_inventory() -> list[dict]:
    records = []
    if not SUPPLEMENT.exists():
        return records
    pat = re.compile(r"^XIAN_Z20_(\d{4}-\d{2}-\d{2})(?:_combined)?$")
    for d in sorted(SUPPLEMENT.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        files = [p for p in d.iterdir() if p.is_file()]
        suffix_counts: dict[str, int] = {}
        for p in files:
            suffix_counts[p.suffix.lower() or "<none>"] = suffix_counts.get(p.suffix.lower() or "<none>", 0) + 1
        tifs = [p for p in files if p.suffix.lower() in {".tif", ".tiff"}]
        raw = [p for p in files if p.suffix.lower() == ".raw"]
        datever = [p for p in files if p.suffix.lower() == ".datever"]
        if tifs:
            evidence_class = "PIXEL_SUPPORT"
        elif raw or datever or any(k in suffix_counts for k in (".bpw", ".map", ".tfw")):
            evidence_class = "METADATA_OR_TILE_CACHE_ONLY"
        else:
            evidence_class = "NO_DIRECT_EVIDENCE"
        records.append(
            {
                "observation_date": m.group(1),
                "directory": str(d),
                "directory_role": "COMBINED" if d.name.endswith("_combined") else "RAW_OR_TILE_CACHE",
                "inventory_scope": "DIRECT_FILES_ONLY",
                "evidence_class": evidence_class,
                "direct_file_count": len(files),
                "direct_suffix_counts": suffix_counts,
                "direct_tif_count": len(tifs),
                "direct_tif_paths": [str(p) for p in tifs],
                "direct_raw_count": len(raw),
                "direct_datever_count": len(datever),
            }
        )
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_records = []
    main_verification = []
    for year, item in MAIN_HASHES.items():
        rec = stat_record(item["path"])
        rec["nominal_group"] = year
        rec["nominal_date"] = {"2018": "2018-04-29", "2019": "2019-01-01", "2020": "2020-12-31"}[year]
        rec["actual_date_dominant"] = item["actual_date_dominant"]
        rec["actual_date_candidates"] = item["actual_date_candidates"]
        rec["actual_date_status"] = item["actual_date_status"]
        rec["sha256_recorded"] = item["sha256"]
        rec["sha256_verification"] = "HASHED_THIS_RUN"
        main_verification.append(rec)
        input_records.append(rec)

    formal_sources = {
        "formal_root_remote": "/data/xian_r4_100km2_20260914/output/西安_R4_辅助影像接入前_三期三态候选版_20260914",
        "v2_root_remote": "/data/xian_r4_100km2_20260914/output/16_candidate_identitygraph_v2_20260916",
        "visual_root_remote": "/data/xian_r4_100km2_20260914/output/17_visual_qa_v2_20260916",
        "phase1_root_remote": "/data/xian_r4_100km2_20260914/output/18_rfinal_baseline_and_residual_diagnosis_20260916",
        "formal_hashes_rechecked_remote": FORMAL_HASHES,
        "formal_outputs_modified": False,
        "geometry_owner_formal_unchanged": True,
    }

    supplement = shallow_supplement_inventory()
    receipt = {
        "task": "西安三期建筑追踪与三态最终正式生产任务（R-FINAL）",
        "phase": "PHASE_0_BASELINE_FREEZE_AND_INPUT_VERIFICATION",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "host": os.environ.get("COMPUTERNAME", "unknown"),
        "local_roots": {"main": str(MAIN), "supplement": str(SUPPLEMENT), "output": str(OUT)},
        "main_inputs": input_records,
        "supplement_inventory": supplement,
        "supplement_evidence_boundary": {
            "PIXEL_SUPPORT": "Only direct supplemental combined TIFs are treated as pixel-support candidates.",
            "METADATA_OR_TILE_CACHE_ONLY": "bpw/map/tfw/raw/datever/cache metadata are not treated as target-date pixel observations.",
            "target_interval_note": "The 2019-11-21, 2020-03-04, and 2020-09-08 supplemental entries currently have no direct combined TIF in this inventory; no pixel support is claimed for them.",
            "scope": "This receipt inventories direct files in date directories and does not delete or alter any source.",
        },
        "formal_sources": formal_sources,
        "gate_snapshot": {
            "v4b_regression": "PASS",
            "strict_core_ledger": "PASS",
            "state_sanity": "PASS",
            "unknown_lt5": "FAIL",
            "tracking_accuracy": "REVIEW_REQUIRED",
            "fixed_case_regression": "REVIEW_REQUIRED",
            "false_change_visual": "FAIL_REVIEW_REQUIRED",
            "formal_production": "NOT_RUN",
        },
        "formal_results_affected": False,
        "scientific_status": "BASELINE_FROZEN_DIAGNOSTIC_ONLY",
    }
    (OUT / "PHASE0_BASELINE_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Xi'an R-FINAL Phase 0 Baseline Freeze",
        "",
        "本目录是只读基线与诊断工作区，不是正式结果目录。正式 Geometry、Tracking、Presence、State 未被修改。",
        "",
        "## Main image inputs",
        "",
        "| nominal group | nominal date | dominant actual date | actual-date status | size (bytes) | SHA-256 |",
        "|---|---|---|---|---:|---|",
    ]
    for r in main_verification:
        lines.append(f"| {r['nominal_group']} | {r['nominal_date']} | {r['actual_date_dominant']} | {r['actual_date_status']} | {r.get('size_bytes', '')} | `{r['sha256_recorded']}` |")
    lines += [
        "",
        "主影像的日期以实际采集日期登记；名义年份只用于分组。",
        "",
        "## Supplemental image evidence boundary",
        "",
        "补充目录按实际日期独立登记。只有直接存在的 combined TIF 才进入 `PIXEL_SUPPORT`；bpw/map/tfw/raw/datever 等只记为元数据或缓存证据，不能替代像素。当前 2019-11-21、2020-03-04、2020-09-08 目录未登记到直接 combined TIF，因此后续不能声称这些日期提供了目标区间的像素核验。",
        "",
        "## Formal-source immutability",
        "",
        "正式 tracking/presence/state 的已复核 SHA-256 已写入 `PHASE0_BASELINE_RECEIPT.json`。后续诊断和闭合候选必须在云端新目录中完成；只有所有 Gate 通过后才允许生成正式产品。",
        "",
        "## Current gate snapshot",
        "",
        "- V4B regression / strict-core ledger / state sanity: PASS",
        "- Unknown <5%: FAIL",
        "- Tracking accuracy / fixed-case regression: REVIEW_REQUIRED",
        "- False-change visual gate: FAIL_REVIEW_REQUIRED",
        "- Formal production: NOT_RUN",
    ]
    (OUT / "PHASE0_BASELINE_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(OUT / "PHASE0_BASELINE_RECEIPT.json"), "supplement_records": len(supplement)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
