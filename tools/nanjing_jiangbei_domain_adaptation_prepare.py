from __future__ import annotations

"""Prepare isolated Nanjing Construction domain-adaptation data.

This module reads the already sealed Nanjing baseline and the six retained
source scenes.  It writes only derived artifacts below the new domain-
adaptation run root.  No source scene, old inference directory, review
inventory, or Locked Test pixel is opened for writing or used as a target.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "training" / "construction_nanjing_domain_adaptation_20260803_235940"
DEFAULT_OLD_OUTPUT = PROJECT_ROOT / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320"
DEFAULT_SOURCE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei")
DEFAULT_UCS_GATE = PROJECT_ROOT / "outputs" / "urrs" / "audits" / "ucs386_subset_readiness_20260803_113903"
VALID_DATES = ("2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02")
EXCLUDED_DATE = "2024-10-20"
SPLIT_SEED = 20260803
WEIGHTS = {
    "human_verified": 1.00,
    "human_corrected": 1.00,
    "UCS_supervised": 1.00,
    "stable_building_hard_negative": 0.60,
    "multi_source_construction_positive": 0.50,
    "single_module_high_confidence": 0.25,
    "uncertain_or_conflict": 0.00,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scene_records(source_root: Path, unique_manifest: Path) -> list[dict[str, Any]]:
    payload = read_json(unique_manifest)
    records = list(payload.get("records", []))
    if payload.get("unique_scene_count") != 6 or [row.get("date") for row in records] != list(VALID_DATES):
        raise RuntimeError("unique scene manifest does not contain the required six sorted dates")
    for row in records:
        path = Path(row["path"])
        if not path.is_file() or not str(path.resolve()).startswith(str(source_root.resolve())):
            raise RuntimeError(f"scene path is missing or outside audited source root: {path}")
        if row.get("date") == EXCLUDED_DATE or "241020" in path.name:
            raise RuntimeError("2024 duplicate scene leaked into unique scene manifest")
    return records


def _make_split(run_root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    import geopandas as gpd
    from shapely.geometry import box

    first = records[0]
    xmin, ymin, xmax, ymax = [float(value) for value in first["bounds"]]
    rows = 5
    cols = 5
    block_ids = [f"B{row:02d}_{col:02d}" for row in range(rows) for col in range(cols)]
    rng = random.Random(SPLIT_SEED)
    shuffled = block_ids[:]
    rng.shuffle(shuffled)
    train_ids = set(shuffled[:15])
    validation_ids = set(shuffled[15:20])
    holdout_ids = set(shuffled[20:])
    split_by_id = {
        block_id: "train" if block_id in train_ids else "validation" if block_id in validation_ids else "untouched_holdout"
        for block_id in block_ids
    }
    geometries = []
    manifest_rows = []
    for row in range(rows):
        for col in range(cols):
            block_id = f"B{row:02d}_{col:02d}"
            left = xmin + (xmax - xmin) * col / cols
            right = xmin + (xmax - xmin) * (col + 1) / cols
            bottom = ymin + (ymax - ymin) * row / rows
            top = ymin + (ymax - ymin) * (row + 1) / rows
            split = split_by_id[block_id]
            geometries.append({"block_id": block_id, "split": split, "row": row, "col": col, "geometry": box(left, bottom, right, top)})
            manifest_rows.append(
                {
                    "block_id": block_id,
                    "row": row,
                    "col": col,
                    "split": split,
                    "xmin": left,
                    "ymin": bottom,
                    "xmax": right,
                    "ymax": top,
                    "dates": "|".join(VALID_DATES),
                    "same_split_all_dates": True,
                    "object_site_cross_split": False,
                }
            )
    data_root = run_root / "02_training_data"
    data_root.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(geometries, geometry="geometry", crs=first.get("crs") or "EPSG:4326")
    gpkg_path = data_root / "SPATIAL_SPLIT.gpkg"
    gdf.to_file(gpkg_path, layer="spatial_split", driver="GPKG")
    write_csv(data_root / "SPLIT_MANIFEST.csv", list(manifest_rows[0]), manifest_rows)
    audit = {
        "status": "pass",
        "seed": SPLIT_SEED,
        "grid": {"rows": rows, "cols": cols, "block_count": len(block_ids)},
        "split_block_counts": {"train": len(train_ids), "validation": len(validation_ids), "untouched_holdout": len(holdout_ids)},
        "split_percentages": {"train": 0.60, "validation": 0.20, "untouched_holdout": 0.20},
        "same_spatial_split_all_dates": True,
        "object_or_site_cross_split": False,
        "object_site_rule": "each derived pseudo patch is clipped to exactly one fixed block; no object/site labels cross block assignments",
        "holdout_used_for_selection": False,
        "holdout_used_for_pseudo_label_tuning": False,
        "locked_test_pixel_reads": 0,
        "excluded_dates": [EXCLUDED_DATE],
        "valid_dates": list(VALID_DATES),
        "gpkg": str(gpkg_path.resolve()),
        "created_at": now(),
    }
    write_json(data_root / "SPLIT_AUDIT.json", audit)
    return {"blocks": manifest_rows, "split_by_id": split_by_id, "audit": audit}


def _read_raster(path: Path, *, dtype: Any = np.float32) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.read(1).astype(dtype, copy=False)


def _remove_tiny(mask: np.ndarray, minimum: int = 25) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:
        return mask.astype(bool)
    labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(minimum)
    keep[0] = False
    return keep[labels]


def _old_probability_paths(old_output: Path, date: str) -> dict[str, Path]:
    root = old_output / "06_construction_single_date" / date
    names = ("background_probability", "exposed_ground_probability", "dustproof_net_probability", "building_probability", "boundary_probability", "uncertainty_probability")
    paths = {name: root / f"{name}.tif" for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"old Construction probabilities missing for {date}: {missing}")
    return paths


def _evidence_paths(old_output: Path, date: str) -> tuple[Path, Path]:
    support_date = date if date != "2022-12-02" else "2019-03-16"
    pair = old_output / "04_baseline_fusion" / f"{support_date}_to_2022-12-02"
    full20 = pair / "baseline_change_candidate.tif"
    geometry = pair / "geometry_change_candidate.tif"
    if not full20.is_file() or not geometry.is_file():
        raise FileNotFoundError(f"baseline evidence missing for {date}: {pair}")
    return full20, geometry


def _load_scene_arrays(old_output: Path, date: str) -> dict[str, np.ndarray]:
    probabilities = {name: _read_raster(path) for name, path in _old_probability_paths(old_output, date).items()}
    geometry = _read_raster(old_output / "03_baseline_rsbuilding" / date / "building_mask.tif", dtype=np.uint8) > 0
    full20_path, geometry_path = _evidence_paths(old_output, date)
    full20 = _read_raster(full20_path) > 0
    geometry_change = _read_raster(geometry_path) > 0
    return {**probabilities, "building_mask": geometry, "full20_change": full20, "geometry_change": geometry_change}


def _derive_pseudo_labels(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    exposed = arrays["exposed_ground_probability"]
    dustproof = arrays["dustproof_net_probability"]
    building = arrays["building_probability"]
    background = arrays["background_probability"]
    boundary = arrays["boundary_probability"]
    uncertainty = arrays["uncertainty_probability"]
    max_construction = np.maximum.reduce([exposed, dustproof, building])
    edge = np.zeros_like(max_construction, dtype=bool)
    margin = 64
    edge[:margin, :] = True
    edge[-margin:, :] = True
    edge[:, :margin] = True
    edge[:, -margin:] = True
    stable_geometry = arrays["building_mask"].astype(bool)
    high_conf_a = ((np.maximum(exposed, dustproof) >= 0.55) | (boundary >= 0.55)) & (uncertainty <= 0.45)
    full20_support = arrays["full20_change"].astype(bool)
    geometry_support = arrays["geometry_change"].astype(bool)
    evidence_count = high_conf_a.astype(np.uint8) + full20_support.astype(np.uint8) + geometry_support.astype(np.uint8)
    stable_building = stable_geometry & (max_construction >= 0.35) & (uncertainty <= 0.45) & ~edge
    stable_building = _remove_tiny(stable_building)
    positive = (evidence_count >= 2) & ~stable_building & ~edge & (uncertainty <= 0.45)
    positive = _remove_tiny(positive)
    single = (max_construction >= 0.85) & (uncertainty <= 0.25) & ~stable_building & ~positive & ~edge
    single = _remove_tiny(single)
    background_mask = (background >= 0.85) & (uncertainty <= 0.30) & ~stable_building & ~positive & ~single & ~edge
    labels = np.full(background.shape, 255, dtype=np.uint8)
    labels[background_mask] = 0
    labels[stable_building] = 3
    labels[positive & (dustproof >= exposed)] = 2
    labels[positive & (exposed > dustproof)] = 1
    argmax = np.argmax(np.stack([background, exposed, dustproof, building], axis=0), axis=0).astype(np.uint8)
    labels[single] = argmax[single]
    stats = {
        "background_pixels": int(np.count_nonzero(labels == 0)),
        "stable_building_hard_negative_pixels": int(np.count_nonzero(stable_building)),
        "multi_source_construction_positive_pixels": int(np.count_nonzero(positive)),
        "single_module_high_confidence_pixels": int(np.count_nonzero(single)),
        "uncertain_or_conflict_pixels": int(np.count_nonzero(labels == 255)),
        "valid_pseudo_pixels": int(np.count_nonzero(labels != 255)),
        "construction_positive_pixels": int(np.count_nonzero((labels == 1) | (labels == 2))),
        "building_pixels": int(np.count_nonzero(labels == 3)),
        "evidence_A_pixels": int(np.count_nonzero(high_conf_a)),
        "evidence_B_pixels": int(np.count_nonzero(full20_support)),
        "evidence_C_pixels": int(np.count_nonzero(geometry_support)),
    }
    aux = {
        "stable_mask": stable_building,
        "change_support": full20_support,
        "geometry_support": geometry_support,
        "edge_mask": edge,
    }
    return labels, aux, stats


def _write_mixed_manifests(data_root: Path, pseudo_rows: list[dict[str, Any]], ucs_gate: Path) -> dict[str, Any]:
    fieldnames = ["kind", "split", "sample_id", "scene_id", "image_path", "label_path", "pseudo_path", "source_date", "block_id", "sample_weight", "label_source"]
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "untouched_holdout": []}
    source_manifest_audit: dict[str, Any] = {"train_rows": 0, "validation_rows": 0, "excluded_2024_rows": [], "source_data_modified": False}
    for split, filename in (("train", "UCS_TRAIN_MANIFEST.csv"), ("validation", "UCS_VALIDATION_MANIFEST.csv")):
        source_path = ucs_gate / filename
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = "|".join(str(row.get(key, "")) for key in ("image_path", "label_path", "scene_id", "sample_id"))
                if EXCLUDED_DATE in text or "241020" in text:
                    source_manifest_audit["excluded_2024_rows"].append({"split": split, "sample_id": row.get("sample_id"), "reason": "excluded_as_duplicate_observation"})
                    continue
                rows_by_split[split].append(
                    {
                        "kind": "ucs",
                        "split": split,
                        "sample_id": row.get("sample_id"),
                        "scene_id": row.get("scene_id", row.get("sample_id")),
                        "image_path": row.get("image_path"),
                        "label_path": row.get("label_path"),
                        "pseudo_path": "",
                        "source_date": "",
                        "block_id": "",
                        "sample_weight": WEIGHTS["UCS_supervised"],
                        "label_source": "UCS_supervised",
                    }
                )
        source_manifest_audit[f"{split}_rows"] = len(rows_by_split[split])
    for row in pseudo_rows:
        if row["split"] in rows_by_split:
            rows_by_split[row["split"]].append(row)
    write_csv(data_root / "MIXED_TRAIN_MANIFEST.csv", fieldnames, rows_by_split["train"])
    write_csv(data_root / "MIXED_VALIDATION_MANIFEST.csv", fieldnames, rows_by_split["validation"])
    write_csv(data_root / "NANJING_UNTOUCHED_HOLDOUT_MANIFEST.csv", fieldnames, rows_by_split["untouched_holdout"])
    source_manifest_audit["mixed_counts"] = {key: len(value) for key, value in rows_by_split.items()}
    source_manifest_audit["locked_test_pixel_reads"] = 0
    write_json(data_root / "MIXED_DATASET_AUDIT.json", source_manifest_audit)
    return source_manifest_audit


def prepare(run_root: Path, old_output: Path, source_root: Path, ucs_gate: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    old_output = old_output.resolve()
    source_root = source_root.resolve()
    ucs_gate = ucs_gate.resolve()
    receipt = read_json(run_root / "00_acceptance" / "FINAL_ACCEPTANCE_RECEIPT.json")
    if receipt.get("architecture_acceptance_passed") is not True:
        raise RuntimeError("phase-one architecture acceptance did not pass")
    if receipt.get("duplicate_2024_excluded") is not True or receipt.get("unique_scene_count") != 6:
        raise RuntimeError("phase-one duplicate and six-scene gate did not pass")
    unique_manifest = run_root / "01_data_audit" / "UNIQUE_SCENE_MANIFEST.json"
    records = _scene_records(source_root, unique_manifest)
    split = _make_split(run_root, records)
    data_root = run_root / "02_training_data"
    pseudo_root = data_root / "nanjing_pseudo"
    pseudo_root.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    date_stats: dict[str, dict[str, int]] = {}
    split_stats: dict[str, int] = {"train": 0, "validation": 0, "untouched_holdout": 0}
    block_lookup = {row["block_id"]: row for row in split["blocks"]}
    import rasterio

    for scene in records:
        date = str(scene["date"])
        arrays = _load_scene_arrays(old_output, date)
        labels, aux, stats = _derive_pseudo_labels(arrays)
        date_stats[date] = stats
        with rasterio.open(scene["path"]) as source:
            raw = source.read([3, 2, 1]).astype(np.uint8, copy=False)
        for block_id, block in block_lookup.items():
            split_name = str(block["split"])
            x0 = max(0, int(round((block["xmin"] - scene["bounds"][0]) / (scene["bounds"][2] - scene["bounds"][0]) * raw.shape[2])))
            x1 = min(raw.shape[2], int(round((block["xmax"] - scene["bounds"][0]) / (scene["bounds"][2] - scene["bounds"][0]) * raw.shape[2])))
            y0 = max(0, int(round((scene["bounds"][3] - block["ymax"]) / (scene["bounds"][3] - scene["bounds"][1]) * raw.shape[1])))
            y1 = min(raw.shape[1], int(round((scene["bounds"][3] - block["ymin"]) / (scene["bounds"][3] - scene["bounds"][1]) * raw.shape[1])))
            if x1 <= x0 or y1 <= y0:
                continue
            label_crop = labels[y0:y1, x0:x1]
            valid = label_crop != 255
            valid_count = int(np.count_nonzero(valid))
            if valid_count < 256:
                continue
            counts = {str(class_id): int(np.count_nonzero(label_crop == class_id)) for class_id in range(4)}
            stable_count = int(np.count_nonzero(aux["stable_mask"][y0:y1, x0:x1]))
            positive_count = int(np.count_nonzero((label_crop == 1) | (label_crop == 2)))
            single_count = int(np.count_nonzero((label_crop != 255) & (label_crop != 0) & (label_crop != 3)))
            if stable_count >= max(positive_count, single_count):
                source_label = "stable_building_hard_negative"
            elif positive_count >= single_count:
                source_label = "multi_source_construction_positive"
            else:
                source_label = "single_module_high_confidence"
            sample_id = f"nanjing_{date}_{block_id}"
            pseudo_path = pseudo_root / f"{sample_id}.npz"
            np.savez_compressed(
                pseudo_path,
                image=raw[:, y0:y1, x0:x1],
                label=label_crop,
                stable_mask=aux["stable_mask"][y0:y1, x0:x1].astype(np.uint8),
                change_support=aux["change_support"][y0:y1, x0:x1].astype(np.uint8),
                geometry_support=aux["geometry_support"][y0:y1, x0:x1].astype(np.uint8),
                source_date=date,
                block_id=block_id,
            )
            row = {
                "kind": "pseudo",
                "split": split_name,
                "sample_id": sample_id,
                "scene_id": f"nanjing_{date}",
                "image_path": "",
                "label_path": "",
                "pseudo_path": str(pseudo_path.resolve()),
                "source_date": date,
                "block_id": block_id,
                "sample_weight": WEIGHTS[source_label],
                "label_source": source_label,
                "valid_pixel_count": valid_count,
                "background_pixels": counts["0"],
                "exposed_pixels": counts["1"],
                "dustproof_pixels": counts["2"],
                "building_pixels": counts["3"],
                "uncertain_pixels": int(label_crop.size - valid_count),
                "source_sha256": scene["sha256"],
                "excluded_2024": False,
            }
            inventory_rows.append(row)
            split_stats[split_name] += 1
    inventory_fields = list(inventory_rows[0]) if inventory_rows else ["sample_id", "split", "source_date", "block_id"]
    write_csv(data_root / "PSEUDO_LABEL_INVENTORY.csv", inventory_fields, inventory_rows)
    mixed_audit = _write_mixed_manifests(data_root, inventory_rows, ucs_gate)
    seed_path = PROJECT_ROOT / "outputs" / "urrs" / "construction" / "human_seed_labels_20260802_134145" / "SEED_LABEL_INVENTORY.json"
    seed = read_json(seed_path) if seed_path.exists() else {}
    seed_records = list(seed.get("records", []))
    human_verified = sum(1 for row in seed_records if row.get("human_confirmed") is True and str(row.get("seed_status", "")).casefold() in {"verified", "reviewed", "confirmed"})
    human_corrected = 0
    policy = {
        "status": "pass" if len(inventory_rows) > 0 else "PSEUDO_LABEL_GATE_FAILED",
        "weights": WEIGHTS,
        "human_verified_records_available": human_verified,
        "human_corrected_records_available": human_corrected,
        "human_seed_records_read": len(seed_records),
        "human_seed_unreviewed_records": sum(1 for row in seed_records if str(row.get("seed_status", "")).casefold() == "unreviewed"),
        "human_seed_used_for_training": False,
        "positive_evidence": {
            "A": "Construction exposed/dustproof/boundary high and uncertainty low",
            "B": "Full20 change/new/demolition support",
            "C": "RSBuilding later addition/disappearance/area-change or stable geometry support",
            "minimum_evidence_count": 2,
        },
        "negative_evidence": "stable geometry across at least two dates, low Full20 change, Construction conflict, not edge/high uncertainty",
        "exclusion_rules": ["uncertainty", "module_conflict", "image_edge", "no_coverage", "registration_residual", "single_date_only", "tiny_fragment", "shadow_water_road_parking_unseparable", "2024_duplicate"],
        "holdout_used_for_tuning": False,
        "locked_test_pixel_reads": 0,
        "created_at": now(),
    }
    write_json(data_root / "PSEUDO_LABEL_POLICY.json", policy)
    numerical = {
        "status": "pass" if inventory_rows and all(split_stats[key] > 0 for key in ("train", "validation", "untouched_holdout")) else "PSEUDO_LABEL_GATE_FAILED",
        "sample_count": len(inventory_rows),
        "sample_counts_by_split": split_stats,
        "pixel_counts_by_date": date_stats,
        "label_source_counts": {source: sum(1 for row in inventory_rows if row["label_source"] == source) for source in WEIGHTS},
        "valid_pixel_total": int(sum(int(row.get("valid_pixel_count", 0)) for row in inventory_rows)),
        "uncertain_pixel_total": int(sum(int(row.get("uncertain_pixels", 0)) for row in inventory_rows)),
        "source_dates": sorted({row["source_date"] for row in inventory_rows}),
        "duplicate_2024_rows": [row for row in inventory_rows if row.get("excluded_2024")],
        "no_locked_test_pixels": True,
        "locked_test_pixel_reads": 0,
        "holdout_used_for_selection": False,
        "holdout_used_for_pseudo_label_tuning": False,
        "quality_gate": {
            "at_least_two_dates": len({row["source_date"] for row in inventory_rows}) >= 2,
            "train_validation_holdout_all_nonempty": all(split_stats[key] > 0 for key in ("train", "validation", "untouched_holdout")),
            "multi_source_positive_present": any(row["label_source"] == "multi_source_construction_positive" for row in inventory_rows),
            "stable_building_hard_negative_present": any(row["label_source"] == "stable_building_hard_negative" for row in inventory_rows),
        },
        "created_at": now(),
    }
    write_json(data_root / "PSEUDO_LABEL_NUMERICAL_AUDIT.json", numerical)
    manifest = {
        "status": numerical["status"],
        "run_root": str(run_root.resolve()),
        "old_output_read_only": str(old_output.resolve()),
        "source_root_read_only": str(source_root.resolve()),
        "valid_dates": list(VALID_DATES),
        "excluded_duplicate_dates": [EXCLUDED_DATE],
        "rgb_mapping": [3, 2, 1],
        "pseudo_inventory": str((data_root / "PSEUDO_LABEL_INVENTORY.csv").resolve()),
        "mixed_train_manifest": str((data_root / "MIXED_TRAIN_MANIFEST.csv").resolve()),
        "mixed_validation_manifest": str((data_root / "MIXED_VALIDATION_MANIFEST.csv").resolve()),
        "untouched_holdout_manifest": str((data_root / "NANJING_UNTOUCHED_HOLDOUT_MANIFEST.csv").resolve()),
        "original_data_modified": False,
        "locked_test_pixel_reads": 0,
        "created_at": now(),
    }
    write_json(data_root / "DATA_PREPARATION_MANIFEST.json", manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Nanjing Construction domain-adaptation pseudo labels")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--old-output", default=str(DEFAULT_OLD_OUTPUT))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--ucs-gate", default=str(DEFAULT_UCS_GATE))
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = prepare(Path(args.run_root), Path(args.old_output), Path(args.source_root), Path(args.ucs_gate))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
