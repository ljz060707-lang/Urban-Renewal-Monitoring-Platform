from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from urrs_lite_review.core import (  # noqa: E402
    CHIP_ROOT,
    DATA_ROOT,
    NANJING_BOUNDS,
    NANJING_DATES,
    NANJING_IMAGE_ROOT,
    NANJING_RUN,
    ROOT,
    TONGZHOU_DATES,
    TONGZHOU_IMAGE_ROOT,
    TONGZHOU_RUN,
    ReviewDB,
    atomic_write_json,
    compute_spatial_block,
    geometry_centroid,
    geometry_bbox,
    frozen_split,
    image_for_date,
    make_chip,
    project_id,
    read_gpkg_rows,
    rebuild_review_decisions_from_jsonl,
    stable_hash,
    transform_geometry,
)


def _fallback_geometry(row: dict[str, Any], bounds: tuple[float, float, float, float] | None) -> dict[str, Any]:
    x = row.get("centroid_x")
    y = row.get("centroid_y")
    if x is None or y is None:
        g = row.get("geometry")
        if g:
            x, y = geometry_centroid(g)
        elif bounds:
            x, y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        else:
            x, y = 0.0, 0.0
    return transform_geometry({"type": "Point", "coordinates": [float(x), float(y)]})


def _row_geometry(row: dict[str, Any], bounds=None) -> dict[str, Any]:
    return transform_geometry(row.get("geometry")) if row.get("geometry") else _fallback_geometry(row, bounds)


def _display_bounds(bounds):
    if not bounds:
        return None
    if max(abs(float(v)) for v in bounds) < 1000:
        return tuple(float(v) for v in bounds)
    from urrs_lite_review.core import utm_to_wgs84
    x1, y1 = utm_to_wgs84(float(bounds[0]), float(bounds[1]))
    x2, y2 = utm_to_wgs84(float(bounds[2]), float(bounds[3]))
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _candidate_gpkg(project: str) -> Path | None:
    if project == "nanjing_building":
        p = NANJING_RUN / "02_building" / "single_date" / NANJING_DATES[0] / "building_objects.gpkg"
        return p if p.exists() else None
    p = TONGZHOU_RUN / "12_tests" / "objectization_attempt_2" / "10_vectors" / "construction_candidates_ranked.gpkg"
    return p if p.exists() else None


def _register_assets(db: ReviewDB, project: str, dates: list[str], bounds=None) -> dict[str, Path | None]:
    assets = {}
    for date in dates:
        asset = image_for_date(project, date)
        assets[date] = asset
        if asset:
            db.add_asset(project, date, asset, bounds)
    return assets


def _prepare_nanjing(db: ReviewDB, max_candidates: int | None = None) -> dict[str, Any]:
    project = "nanjing_building"
    db.add_project(project, "Nanjing Building v1", NANJING_DATES, "Six-date Building review; model output remains reference-only until human action.")
    db.add_queue(project, "building_block", "Building Block Review", 0.25)
    db.add_queue(project, "building_polygon", "Building Polygon Review", 0.50)
    db.add_queue(project, "building_temporal", "Building Temporal Review", 0.25)
    db.add_queue(project, "building_hard_negative", "Building Hard Negative Review", 0.10)
    assets = _register_assets(db, project, NANJING_DATES, NANJING_BOUNDS)
    created = 0
    candidate_count = 0
    bounds = NANJING_BOUNDS

    # Six-date background blocks. The six image chips are attached to one
    # block sample, so the reviewer sees the same spatial block as a group.
    grid = 8
    for ix in range(grid):
        for iy in range(grid):
            cx = bounds[0] + (ix + 0.5) / grid * (bounds[2] - bounds[0])
            cy = bounds[3] - (iy + 0.5) / grid * (bounds[3] - bounds[1])
            geom = {"type": "Point", "coordinates": [cx, cy]}
            block = compute_spatial_block(cx, cy, project, grid=grid)
            sid = f"njb_block_{ix:02d}_{iy:02d}"
            date_chips = {}
            for date, asset in assets.items():
                if asset:
                    chip = CHIP_ROOT / project / sid / f"{date}.png"
                    date_chips[date] = make_chip(asset, geom, chip, project, bounds)
            split = frozen_split(block, project)
            created += int(db.add_sample(sample_id=sid, project=project, module="building_block", scene_date="multi_date", block=block, source_asset=None, chip_path=None, source_geometry=geom, reference={"model_reference_only": True, "six_dates": NANJING_DATES}, metadata={"review_mode": "six_date_grid", "date_chips": date_chips, "complete_background_required": True}, blind_first=split in {"validation", "holdout"}, model_visible=split == "train", bounds=bounds))
            negative_type = ["shadow_confusion", "vegetation_confusion", "industrial_roof_confusion", "bare_land_confusion", "parking_confusion", "metal_roof_confusion"][((ix * grid) + iy) % 6]
            first_chip = (date_chips.get(NANJING_DATES[0]) or {}).get("path")
            hard_sid = f"njb_hard_negative_{ix:02d}_{iy:02d}"
            created += int(db.add_sample(sample_id=hard_sid, project=project, module="building_hard_negative", scene_date=NANJING_DATES[0], block=block, source_asset=assets.get(NANJING_DATES[0]), chip_path=Path(first_chip) if first_chip else None, source_geometry=geom, reference={"model_reference_only": True, "candidate_not_ground_truth": True}, metadata={"hard_negative_type": negative_type, "review_mode": "full_background_negative"}, negative_type=negative_type, blind_first=split in {"validation", "holdout"}, model_visible=split == "train", bounds=bounds))

    for date in NANJING_DATES:
        gpkg = NANJING_RUN / "02_building" / "single_date" / date / "building_objects.gpkg"
        if not gpkg.exists():
            continue
        _, rows, row_bounds = read_gpkg_rows(gpkg, limit=max_candidates)
        for index, row in enumerate(rows):
            geom = _row_geometry(row, row_bounds or bounds)
            cx, cy = geometry_centroid(geom)
            block = compute_spatial_block(cx, cy, project)
            obj = row.get("object_id") or row.get("fid") or index
            sid = f"njb_polygon_{date}_{obj}"
            asset = assets.get(date)
            chip_path = CHIP_ROOT / project / f"{sid}.png" if asset else None
            if asset:
                make_chip(asset, geom, chip_path, project, bounds)
            ref = {k: row.get(k) for k in ("object_id", "state", "area_m2", "mean_probability", "uncertainty", "source_model", "threshold")}
            ref.update({"model_reference_only": True, "candidate_not_ground_truth": True, "source_gpkg": str(gpkg)})
            split = frozen_split(block, project)
            created += int(db.add_sample(sample_id=sid, project=project, module="building_polygon", scene_date=date, block=block, source_asset=asset, chip_path=chip_path, source_geometry=geom, reference=ref, metadata={"six_date_group": block, "full20_reference": True, "object_id": obj}, proposal={"proposal_id": sid + ":proposal", "class_name": "building", "confidence": row.get("mean_probability"), "source": "frozen_model_reference"}, bounds=bounds, blind_first=split in {"validation", "holdout"}, model_visible=split == "train"))
            candidate_count += 1

    return {"project": project, "created": created, "candidate_samples": candidate_count, "background_blocks": grid * grid, "dates": NANJING_DATES}


def _deterministic_sample(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if len(rows) <= n: return rows
    rows = sorted(rows, key=lambda r: str(r.get("candidate_id") or r.get("fid") or stable_hash(r)))
    step = len(rows) / float(n)
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]


def _prepare_tongzhou(db: ReviewDB, max_total: int = 900) -> dict[str, Any]:
    project = "tongzhou_construction"
    db.add_project(project, "Tongzhou Construction v1", TONGZHOU_DATES, "Candidate truth review and Site Classifier preparation; no automatic acceptance.")
    db.add_queue(project, "construction_candidate", "Construction Candidate Review", 0.40)
    db.add_queue(project, "construction_hard_negative", "Construction Hard Negative Review", 0.35)
    db.add_queue(project, "construction_background", "Construction Background Review", 0.25)
    assets = _register_assets(db, project, TONGZHOU_DATES)
    ranked = TONGZHOU_RUN / "12_tests" / "objectization_attempt_2" / "10_vectors" / "construction_candidates_ranked.gpkg"
    confusions = TONGZHOU_RUN / "12_tests" / "objectization_attempt_2" / "10_vectors" / "construction_likely_confusions.gpkg"
    _, candidates, bounds = read_gpkg_rows(ranked, limit=None) if ranked.exists() else ("", [], None)
    _, negatives, n_bounds = read_gpkg_rows(confusions, limit=None) if confusions.exists() else ("", [], None)
    bounds = _display_bounds(bounds or n_bounds)
    target = max(120, int(max_total))
    n_candidate = max(1, int(target * 0.40)); n_negative = max(1, int(target * 0.35)); n_background = max(1, target - n_candidate - n_negative)
    created = 0
    for role, rows, number, module, default_negative in [
        ("candidate", _deterministic_sample(candidates, n_candidate), n_candidate, "construction_candidate", None),
        ("hard_negative", _deterministic_sample(negatives, n_negative), n_negative, "construction_hard_negative", "other"),
    ]:
        for index, row in enumerate(rows):
            date = str(row.get("scene_date") or TONGZHOU_DATES[index % len(TONGZHOU_DATES)])
            if date not in assets: date = TONGZHOU_DATES[index % len(TONGZHOU_DATES)]
            geom = _row_geometry(row, bounds)
            cx, cy = geometry_centroid(geom); block = compute_spatial_block(cx, cy, project)
            ident = row.get("candidate_id") or row.get("fid") or index
            sid = f"tzc_{role}_{date}_{ident}"
            asset = assets.get(date); chip = CHIP_ROOT / project / f"{sid}.png" if asset else None
            if asset: make_chip(asset, geom, chip, project, bounds)
            neg = row.get("primary_confusion_type") or row.get("confusion_labels") or default_negative
            ref = {k: row.get(k) for k in ("candidate_id", "seed_type", "area_m2", "candidate_tier", "review_priority", "overall_evidence_score", "uncertainty", "stable_building_overlap", "primary_confusion_type", "confusion_labels")}
            ref.update({"model_reference_only": True, "candidate_not_ground_truth": True, "source_gpkg": str(ranked if role == "candidate" else confusions)})
            split = frozen_split(block, project)
            created += int(db.add_sample(sample_id=sid, project=project, module=module, scene_date=date, block=block, source_asset=asset, chip_path=chip, source_geometry=geom, reference=ref, metadata={"queue_role": role, "construction_types": ["exposed", "dustproof_net_covered", "mixed", "other_construction", "uncertain"], "missed_site_drawing": True}, negative_type=neg, proposal={"proposal_id": sid + ":proposal", "class_name": "construction_candidate", "confidence": row.get("overall_evidence_score") or row.get("element_evidence_score"), "source": "frozen_model_reference"}, bounds=bounds, blind_first=split in {"validation", "holdout"}, model_visible=split == "train"))
    # Deterministic spatial background grid, separate from model candidates.
    bg_bounds = bounds or (116.90, 39.72, 117.15, 39.95)
    for i in range(n_background):
        frac = (i + 0.5) / n_background
        cx = bg_bounds[0] + frac * (bg_bounds[2] - bg_bounds[0])
        cy = bg_bounds[1] + ((i * 37) % n_background + 0.5) / n_background * (bg_bounds[3] - bg_bounds[1])
        date = TONGZHOU_DATES[i % len(TONGZHOU_DATES)]
        geom = {"type": "Point", "coordinates": [cx, cy]}; block = compute_spatial_block(cx, cy, project)
        sid = f"tzc_background_{i:04d}"
        asset = assets.get(date); chip = CHIP_ROOT / project / f"{sid}.png" if asset else None
        if asset: make_chip(asset, geom, chip, project, bg_bounds)
        split = frozen_split(block, project)
        created += int(db.add_sample(sample_id=sid, project=project, module="construction_background", scene_date=date, block=block, source_asset=asset, chip_path=chip, source_geometry=geom, reference={"model_candidate_present": False, "model_reference_only": True}, metadata={"queue_role": "random_background", "missed_site_drawing": True}, negative_type="random_background", proposal=None, bounds=bg_bounds, blind_first=split in {"validation", "holdout"}, model_visible=split == "train"))
    return {"project": project, "created": created, "candidate_samples": n_candidate, "hard_negative_samples": n_negative, "background_samples": n_background, "dates": TONGZHOU_DATES, "source_bounds": bounds}


def prepare(max_nanjing: int | None = None, max_tongzhou: int = 900) -> dict[str, Any]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True); CHIP_ROOT.mkdir(parents=True, exist_ok=True)
    db = ReviewDB()
    try:
        nj = _prepare_nanjing(db, max_nanjing)
        tz = _prepare_tongzhou(db, max_tongzhou)
        progress = db.progress()
        manifest = {"generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "app_version": "urrs-lite-review/0.1.0", "projects": [nj, tz], "policy": {"model_predictions_are_reference_only": True, "auto_acceptance": False, "external_network": "disabled", "holdout_sealed": True, "split_frozen": True}, "database": str(db.path), "event_log": str(__import__("urrs_lite_review.core", fromlist=["EVENTS_PATH"]).EVENTS_PATH)}
        atomic_write_json(DATA_ROOT / "REVIEW_QUEUE_MANIFEST.json", manifest)
        atomic_write_json(DATA_ROOT / "REVIEW_PROGRESS.json", progress)
        return {"success": True, "manifest": manifest, "progress": progress}
    finally:
        db.close()


def status() -> dict[str, Any]:
    db = ReviewDB()
    try:
        return {"success": True, "database": str(db.path), "database_exists": db.path.exists(), "event_log": str(__import__("urrs_lite_review.core", fromlist=["EVENTS_PATH"]).EVENTS_PATH), "queue": db.queue(page=1, page_size=1), "progress": db.progress(), "projects": [dict(r) for r in db.conn.execute("SELECT * FROM projects ORDER BY project_id")], "security": {"bind_address": "127.0.0.1", "external_network": "disabled", "holdout_sealed": db.setting("holdout_sealed", "1") == "1"}}
    finally: db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and inspect URRS Lite Review queues")
    parser.add_argument("-PrepareQueues", "--prepare-queues", action="store_true")
    parser.add_argument("-Status", "--status", action="store_true")
    parser.add_argument("-Backup", "--backup", action="store_true")
    parser.add_argument("-RepairDatabase", "--repair-database", action="store_true")
    parser.add_argument("--max-nanjing", type=int, default=None)
    parser.add_argument("--max-tongzhou", type=int, default=900)
    args = parser.parse_args(argv)
    if args.prepare_queues:
        print(json.dumps(prepare(args.max_nanjing, args.max_tongzhou), ensure_ascii=False, indent=2, default=str)); return 0
    db = ReviewDB()
    try:
        if args.backup: print(json.dumps(db.backup(), ensure_ascii=False, indent=2)); return 0
        if args.repair_database: print(json.dumps(rebuild_review_decisions_from_jsonl(db), ensure_ascii=False, indent=2)); return 0
        print(json.dumps(status(), ensure_ascii=False, indent=2, default=str)); return 0
    finally: db.close()


if __name__ == "__main__":
    raise SystemExit(main())
