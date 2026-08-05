from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))

from urrs_lite_review.core import (  # noqa: E402
    EXPORT_ROOT,
    HOLDOUT_SPLIT,
    ReviewDB,
    TRAINING_IMPORTABLE,
    atomic_write_json,
    now_iso,
    write_csv,
    write_geopackage,
    write_geojson,
)


def _verified(db: ReviewDB, project: str | None = None) -> list[dict]:
    clauses = ["h.annotation_status IN ('human_verified','human_corrected')", "s.split<>?"]
    params: list = [HOLDOUT_SPLIT]
    if project:
        clauses.append("s.project_id=?"); params.append(project)
    sql = "SELECT h.*,s.module,s.split,s.chip_path,s.source_hash,s.reference_json,s.metadata_json,s.scene_date AS sample_scene_date FROM human_annotations h JOIN samples s ON s.sample_id=h.sample_id WHERE " + " AND ".join(clauses) + " ORDER BY s.project_id,s.split,s.spatial_block_id,s.sample_id"
    rows = []
    for r in db.conn.execute(sql, params):
        d = dict(r); d["geometry"] = json.loads(d.pop("geometry_json")) if d.get("geometry_json") else None
        d["label_source"] = "human_verified" if d["annotation_status"] == "human_verified" else "human_corrected"
        d["candidate_not_ground_truth"] = False
        rows.append(d)
    return rows


def _coco(rows: list[dict]) -> dict:
    images = []; annotations = []
    for i, row in enumerate(rows, 1):
        images.append({"id": i, "file_name": row.get("chip_path"), "sample_id": row.get("sample_id"), "split": row.get("split")})
        geom = row.get("geometry")
        annotations.append({"id": i, "image_id": i, "category_id": 1, "segmentation": [sum((list(p) for ring in geom.get("coordinates", []) for p in ring), [])] if geom and geom.get("type") == "Polygon" else [], "iscrowd": 0, "area": 0})
    return {"info": {"description": "URRS Lite human verified labels", "generated_at": now_iso()}, "licenses": [], "categories": [{"id": 1, "name": "building"}], "images": images, "annotations": annotations}


def export_verified(project: str | None = None) -> dict:
    db = ReviewDB()
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = EXPORT_ROOT / stamp
        out.mkdir(parents=True, exist_ok=False)
        rows = _verified(db, project)
        building = [r for r in rows if r["project_id"] == "nanjing_building"]
        construction = [r for r in rows if r["project_id"] == "tongzhou_construction"]
        features = [{"geometry": r.get("geometry"), "sample_id": r.get("sample_id"), "project_id": r.get("project_id"), "scene_date": r.get("scene_date"), "spatial_block_id": r.get("spatial_block_id"), "split": r.get("split"), "class_name": r.get("class_name"), "annotation_status": r.get("annotation_status"), "label_source": r.get("label_source"), "candidate_not_ground_truth": r.get("candidate_not_ground_truth")} for r in rows]
        write_geojson(out / "building_annotations.geojson", [f for f in features if f["project_id"] == "nanjing_building"])
        write_geojson(out / "construction_site_labels.geojson", [f for f in features if f["project_id"] == "tongzhou_construction"])
        write_geopackage(out / "building_annotations.gpkg", [f for f in features if f["project_id"] == "nanjing_building"], "building_annotations")
        write_geopackage(out / "construction_site_labels.gpkg", [f for f in features if f["project_id"] == "tongzhou_construction"], "construction_site_labels")
        write_geopackage(out / "valid_area_masks.gpkg", [], "valid_area_masks")
        write_geopackage(out / "ignore_masks.gpkg", [], "ignore_masks")
        manifests = {"building_training_manifest.csv": [r for r in building if r["split"] == "train"], "building_validation_manifest.csv": [r for r in building if r["split"] == "validation"], "building_holdout_manifest.csv": [], "site_classifier_train.csv": [r for r in construction if r["split"] == "train"], "site_classifier_validation.csv": [r for r in construction if r["split"] == "validation"], "site_classifier_holdout.csv": []}
        for name, data in manifests.items():
            write_csv(out / name, [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items() if k not in {"geometry"}} for r in data])
        atomic_write_json(out / "building_coco_train.json", _coco([r for r in building if r["split"] == "train"]))
        atomic_write_json(out / "building_coco_validation.json", _coco([r for r in building if r["split"] == "validation"]))
        write_csv(out / "construction_candidate_decisions.csv", [{k: v for k, v in r.items() if k != "geometry"} for r in construction])
        write_csv(out / "construction_negative_types.csv", [{"sample_id": r["sample_id"], "negative_type": r.get("negative_type"), "split": r["split"]} for r in construction])
        receipt = {"export_id": stamp, "created_at": now_iso(), "project_filter": project, "export_path": str(out), "rows": {"total": len(rows), "building": len(building), "construction": len(construction)}, "training_importable_statuses": sorted(TRAINING_IMPORTABLE), "holdout_exported": False, "model_predictions_promoted": False, "status": "human_verified_only", "sha256_manifest": {}}
        for p in sorted(out.iterdir()):
            if p.is_file():
                from urrs_lite_review.core import stable_hash
                receipt["sha256_manifest"][p.name] = stable_hash(p.read_bytes())
        atomic_write_json(out / "EXPORT_RECEIPT.json", receipt)
        db.conn.execute("INSERT INTO export_history(export_id,project_id,export_path,status,manifest_json,created_at) VALUES(?,?,?,?,?,?)", (stamp, project, str(out), "human_verified_only", json.dumps(receipt, ensure_ascii=False), now_iso())); db.conn.commit()
        return {"success": True, **receipt}
    finally: db.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--project", default=None); p.add_argument("-ExportLabels", "--export", action="store_true")
    args = p.parse_args(argv); print(json.dumps(export_verified(args.project), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

