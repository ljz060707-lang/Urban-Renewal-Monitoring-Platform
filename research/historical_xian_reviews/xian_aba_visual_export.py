"""Export a deterministic 100+ ABA candidate visual-QA set on the cloud.

This is a candidate-only export.  It reuses the already-audited common-bbox
crop helpers and never edits formal tracking or geometry products.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


REPAIR = Path("/data/xian_r4_100km2_20260914/code/repair_20260916/visual_asset_repair.py")
OUT = Path("/data/xian_r4_100km2_20260914/output/13_visual_asset_repair_and_obj0008_identity_closure_20260916/aba_visual_qa")
SAFE = Path("/data/xian_r4_100km2_20260914/output/13_visual_asset_repair_and_obj0008_identity_closure_20260916/identity_candidate/ABA_SAFE_CLOSURE_CANDIDATES.csv")


def load_module():
    spec = importlib.util.spec_from_file_location("xian_repair_helpers", REPAIR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load repair helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_hash(image) -> str:
    h = hashlib.sha256()
    h.update(str(image.mode).encode("utf-8"))
    h.update(str(image.size).encode("utf-8"))
    h.update(image.tobytes())
    return h.hexdigest()


def read_candidates() -> tuple[list[dict[str, str]], dict[str, str] | None]:
    with SAFE.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Deterministic high-confidence selection, with a stable key so the QA
    # sample can be reproduced without changing the formal graph.
    rows.sort(key=lambda r: (-float(r.get("direct_score") or -1), r.get("source_2018", ""), r.get("middle_2019", ""), r.get("target_2020", "")))
    focal = next((r for r in rows if r.get("source_2018") == "xa180429::180396" and r.get("middle_2019") == "xa190101::169127" and r.get("target_2020") == "xa201231::264190"), None)
    general = [r for r in rows if r is not focal]
    # Stratify the manual set across all bridge relation families before
    # filling the remaining slots by direct confidence.  This prevents the
    # QA set from validating only easy 1:1 -> 1:1 cases.
    selected: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str, str]] = set()
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in general:
        groups.setdefault((row.get("bridge_ab_relation", ""), row.get("bridge_bc_relation", "")), []).append(row)
    for relation_key in sorted(groups):
        for row in groups[relation_key][:5]:
            key = (row.get("source_2018", ""), row.get("middle_2019", ""), row.get("target_2020", ""))
            selected.append(row)
            selected_keys.add(key)
    for row in general:
        if len(selected) >= 100:
            break
        key = (row.get("source_2018", ""), row.get("middle_2019", ""), row.get("target_2020", ""))
        if key not in selected_keys:
            selected.append(row)
            selected_keys.add(key)
    return selected[:100], focal


def main() -> None:
    mod = load_module()
    OUT.mkdir(parents=True, exist_ok=True)
    selected, focal = read_candidates()
    all_rows = list(selected)
    if focal is not None:
        # Keep 100 general candidates and add OBJ_0008 as an extra fixed-case
        # row, even if its metrics would place it among the general top 100.
        all_rows.append(focal)

    uids = set()
    for row in all_rows:
        uids.update(x for x in (row.get("source_2018"), row.get("middle_2019"), row.get("target_2020")) if x)
    attrs = mod.load_tracking_uids(uids)
    geoms = mod.load_geometries(uids)
    datasets = {scene: mod.rasterio.open(path) for scene, path in mod.RAW.items()}
    audit: list[dict[str, object]] = []
    try:
        for ordinal, row in enumerate(all_rows, 1):
            is_obj = row.get("source_2018") == "xa180429::180396" and row.get("middle_2019") == "xa190101::169127" and row.get("target_2020") == "xa201231::264190"
            qa_id = "ABA_QA_OBJ0008" if is_obj else f"ABA_QA_{ordinal:04d}"
            folder = OUT / qa_id
            folder.mkdir(parents=True, exist_ok=False)
            ids = [row.get("source_2018", ""), row.get("middle_2019", ""), row.get("target_2020", "")]
            missing = [uid for uid in ids if uid not in geoms]
            if missing:
                raise RuntimeError(f"VISUAL_ASSET_MISSING {qa_id}: {missing}")
            bbox = mod.make_utm_bbox([geoms[uid] for uid in ids])
            raw_images = {}
            overlay_images = {}
            windows = {}
            bounds = {}
            intersects = {}
            visible = {}
            for scene in mod.SCENES:
                window, actual = mod.common_window(datasets[scene], bbox)
                windows[scene] = window
                bounds[scene] = actual
                raw = mod.read_rgb(datasets[scene], window)
                overlay, hit = mod.draw_overlay(datasets[scene], window, raw.copy(), scene, ids, geoms, attrs, qa_id)
                raw_images[scene] = raw
                overlay_images[scene] = overlay
                intersects[scene] = hit
                crop_poly = mod.box(*actual)
                visible[scene] = any(uid.startswith(scene + "::") and mod.transform_geom(geoms[uid], "EPSG:32649", datasets[scene].crs).intersects(crop_poly) for uid in ids)
            sheet = mod.contact_sheet(raw_images, overlay_images, f"{qa_id} | ABA candidate | common geographic bbox")
            # Keep the cloud candidate light: formal rasters and polygons are
            # immutable sources; the review artifact is a labelled contact
            # sheet, downscaled only for transport/storage.
            sheet.thumbnail((1200, 900))
            sheet.save(folder / "contact_sheet_v2.png", "PNG", optimize=True)
            summary = {
                "qa_id": qa_id,
                "candidate_class": "OBJ0008_REPAIR" if is_obj else "SAFE_ABA_IDENTITY_CLOSURE_CANDIDATE",
                "source_2018": row.get("source_2018"),
                "middle_2019": row.get("middle_2019"),
                "target_2020": row.get("target_2020"),
                "source_2018_track_id": row.get("source_2018_track_id"),
                "middle_2019_track_id": row.get("middle_2019_track_id"),
                "target_2020_track_id": row.get("target_2020_track_id"),
                "direct_score": row.get("direct_score"),
                "direct_buffered_iou": row.get("direct_buffered_iou"),
                "direct_area_ratio": row.get("direct_area_ratio"),
                "direct_displacement_m": row.get("direct_displacement_m"),
                "direct_neighborhood_consistency": row.get("direct_neighborhood_consistency"),
                "bridge_ab_relation": row.get("bridge_ab_relation"),
                "bridge_ab_score": row.get("bridge_ab_score"),
                "bridge_bc_relation": row.get("bridge_bc_relation"),
                "bridge_bc_score": row.get("bridge_bc_score"),
                "same_utm_bbox_epsg32649": list(bbox),
                "window_pixels": {scene: {"col_off": w.col_off, "row_off": w.row_off, "width": w.width, "height": w.height} for scene, w in windows.items()},
                "target_visible": {"2018": visible["xa180429"], "2019": visible["xa190101"], "2020": visible["xa201231"]},
                "overlay_intersects": {"2018": intersects["xa180429"], "2019": intersects["xa190101"], "2020": intersects["xa201231"]},
                "raw_sha256": {mod.LABELS[scene]: image_hash(raw_images[scene]) for scene in mod.SCENES},
                "visual_label": "NOT_REVIEWED",
                "formal_writeback": False,
            }
            (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            audit.append(summary)
    finally:
        for ds in datasets.values():
            ds.close()

    keys = [
        "qa_id", "candidate_class", "source_2018", "middle_2019", "target_2020",
        "direct_score", "direct_buffered_iou", "direct_area_ratio", "direct_displacement_m",
        "direct_neighborhood_consistency", "bridge_ab_relation", "bridge_ab_score", "bridge_bc_relation", "bridge_bc_score",
        "same_utm_bbox_epsg32649", "target_visible", "overlay_intersects", "raw_sha256", "visual_label", "formal_writeback",
    ]
    with (OUT / "ABA_VISUAL_QA_EXPORT.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in audit:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in keys})
    receipt = {
        "task": "XIAN-VISUAL-ASSET-REPAIR-AND-OBJ0008-IDENTITY-CLOSURE-001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_general_safe_aba_count": len(selected),
        "exported_count": len(audit),
        "obj0008_exported": any(row["candidate_class"] == "OBJ0008_REPAIR" for row in audit),
        "formal_outputs_modified": False,
        "visual_label_status": "NOT_REVIEWED",
    }
    (OUT / "ABA_VISUAL_QA_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
