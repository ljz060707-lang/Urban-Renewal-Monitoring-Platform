from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import Window

from .transition_validation import validate_preview
from .transition_preview_v2 import (
    diagnose_transition_previews as _canonical_diagnose_transition_previews,
    regenerate_transition_previews as _canonical_regenerate_transition_previews,
)


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")
LEGACY_TRANSITIONS = {
    "20180506_to_20230716": ("2018-05-06", "2023-07-16", True),
    "20200723_to_20230716": ("2020-07-23", "2023-07-16", True),
    "20220320_to_20230716": ("2022-03-20", "2023-07-16", True),
}


_SOURCE_HASH_CACHE: dict[str, str] = {}


def _sha256(path: Path) -> str:
    key = str(path.resolve())
    if key not in _SOURCE_HASH_CACHE:
        _SOURCE_HASH_CACHE[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _SOURCE_HASH_CACHE[key]

def _stretch_rgb(data: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    data = np.asarray(data, dtype=np.float64)
    output = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
    ranges: list[list[float]] = []
    for band in range(3):
        values = data[band][valid & np.isfinite(data[band])]
        if values.size == 0:
            low, high = 0.0, 1.0
        else:
            low, high = (float(x) for x in np.percentile(values, (2, 98)))
            if high <= low:
                low, high = float(values.min()), float(values.max())
            if high <= low:
                high = low + 1.0
        output[..., band] = np.clip((data[band] - low) / (high - low), 0, 1) * 255
        ranges.append([low, high])
    output[~valid] = 128
    return output, ranges


def _read_scene_window(path: Path, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        col, row = index % 4, index // 4
        x0, x1 = int(col * src.width / 4), int((col + 1) * src.width / 4)
        y0, y1 = int(row * src.height / 4), int((row + 1) * src.height / 4)
        window = Window(x0, y0, x1 - x0, y1 - y0)
        if src.count < 3:
            raise ValueError(f"Transition preview requires three RGB bands: {path}")
        data = src.read((1, 2, 3), window=window)
        valid = np.all(src.read_masks((1, 2, 3), window=window) > 0, axis=0)
        rgb, ranges = _stretch_rgb(data, valid)
        metadata = {
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "source_dtype": list(src.dtypes[:3]),
            "rgb_bands": [1, 2, 3],
            "percentile_stretch": "2-98",
            "stretch_ranges": ranges,
            "source_window": [x0, y0, x1, y1],
            "source_width": src.width,
            "source_height": src.height,
            "source_crs": str(src.crs) if src.crs else None,
            "source_nodata": src.nodata,
        }
    return rgb, valid, metadata


def _annotated_panel(rgb: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    image.thumbnail((720, 560), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (image.width, image.height + 54), (238, 239, 235))
    panel.paste(image, (0, 54))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), title, fill=(18, 23, 22), font=ImageFont.load_default())
    draw.text((12, 28), subtitle, fill=(65, 72, 69), font=ImageFont.load_default())
    return panel


def diagnose_transition_previews(review_dir: Path, output_dir: Path) -> dict:
    review_dir, output_dir = Path(review_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((review_dir / "REVIEW_INVENTORY.json").read_text(encoding="utf-8"))
    records = [row for row in inventory.get("records", []) if row.get("review_group") == "transition"]
    stats, hashes_by_suffix = [], {}
    for row in records:
        preview = review_dir / row["preview_paths"]["tile"]
        array = np.asarray(Image.open(preview).convert("RGB"), dtype=np.uint8)
        quality = validate_preview(array).to_dict()
        suffix = row["sample_id"].rsplit("_", 1)[-1]
        hashes_by_suffix.setdefault(suffix, []).append(_sha256(preview))
        stats.append({
            "sample_id": row["sample_id"],
            "preview_path": str(preview),
            "preview_sha256": _sha256(preview),
            "source_paths": row.get("source_paths", {}),
            **quality,
        })
    duplicate_cross_transition = sum(len(set(values)) == 1 for values in hashes_by_suffix.values())
    source_overviews = sorted({row.get("source_paths", {}).get("overview", "") for row in records})
    result = {
        "status": "diagnosed",
        "transition_preview_count": len(records),
        "valid_under_display_gate": sum(row["preview_status"] == "valid" for row in stats),
        "invalid_under_display_gate": sum(row["preview_status"] != "valid" for row in stats),
        "identical_same_window_across_transitions": duplicate_cross_transition,
        "unique_transition_source_overviews": source_overviews,
        "root_causes": [
            "transition previews were cropped from a generated overlay_overview.png instead of date-specific source RGB rasters",
            "the same overview was reused for all three transitions, producing identical same-window previews",
            "no percentile stretch or raw-image validity gate was applied before review",
            "labels and generated overview content replaced the required before/after evidence layout",
        ],
        "transition_review_status": "deferred_due_to_invalid_visualization",
        "transition_model_status": "not_human_verified",
        "transition_review_failure": False,
        "records": stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "TRANSITION_PREVIEW_ROOT_CAUSE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def regenerate_transition_previews(project_root: Path, review_dir: Path, output_dir: Path) -> dict:
    project_root, review_dir, output_dir = Path(project_root), Path(review_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_manifest = project_root / "outputs" / "inference" / "beijing_tongzhou_building_geometry" / "TONGZHOU_SCENES.json"
    scenes = json.loads(scene_manifest.read_text(encoding="utf-8"))
    by_date = {row["scene_date"]: Path(row["selected_source"]) for row in scenes.get("scenes", [])}
    missing = [date for date in DATES if date not in by_date or not by_date[date].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source scenes for transition preview: {missing}")
    original_inventory = json.loads((review_dir / "REVIEW_INVENTORY.json").read_text(encoding="utf-8"))
    original_rows = {row["sample_id"]: row for row in original_inventory.get("records", [])}
    raw_dir, pair_dir = output_dir / "raw", output_dir / "pairs"
    raw_dir.mkdir(exist_ok=True); pair_dir.mkdir(exist_ok=True)
    raw_assets: dict[tuple[str, int], dict] = {}
    for date in DATES:
        for index in range(16):
            rgb, valid, metadata = _read_scene_window(by_date[date], index)
            quality = validate_preview(rgb, valid_mask=valid).to_dict()
            target = raw_dir / f"raw_{date}_{index:02d}.jpg"
            Image.fromarray(rgb).save(target, quality=94)
            raw_assets[(date, index)] = {"path": str(target), "quality": quality, "metadata": metadata}
    records = []
    for transition, (before, after, full20_supported) in LEGACY_TRANSITIONS.items():
        for index in range(16):
            sid = f"transition_{transition}_{index:02d}"
            before_asset, after_asset = raw_assets[(before, index)], raw_assets[(after, index)]
            before_rgb = np.asarray(Image.open(before_asset["path"]).convert("RGB"))
            after_rgb = np.asarray(Image.open(after_asset["path"]).convert("RGB"))
            left = _annotated_panel(before_rgb, f"BEFORE {before}", "RGB bands 1/2/3 | 2-98 percentile stretch")
            right = _annotated_panel(after_rgb, f"AFTER {after}", "RGB bands 1/2/3 | 2-98 percentile stretch")
            height, width = max(left.height, right.height), left.width + right.width
            canvas = Image.new("RGB", (width, height), (222, 224, 220)); canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
            pair_path = pair_dir / f"{sid}.jpg"; canvas.save(pair_path, quality=94)
            pair_quality = validate_preview(np.asarray(canvas)).to_dict()
            visually_valid = (
                before_asset["quality"]["preview_status"] == "valid"
                and after_asset["quality"]["preview_status"] == "valid"
                and pair_quality["preview_status"] == "valid"
            )
            construction_evidence_ready = False
            records.append({
                "sample_id": sid,
                "supersedes_sample_ids": [sid] if sid in original_rows else [],
                "transition": transition,
                "before_date": before,
                "after_date": after,
                "track_id": f"construction_review_track_{index:02d}",
                "object_type": "construction_track",
                "full20_supported": full20_supported,
                "transition_evidence_scope": "full20_and_geometry" if full20_supported else "construction_model_and_geometry_only",
                "scientific_status": "experimental_candidate",
                "raw_before_path": before_asset["path"],
                "raw_after_path": after_asset["path"],
                "pair_preview_path": str(pair_path),
                "source_metadata": {"before": before_asset["metadata"], "after": after_asset["metadata"]},
                "preview_quality": pair_quality,
                "visual_preview_valid": visually_valid,
                "construction_evidence_ready": construction_evidence_ready,
                "review_allowed": visually_valid and construction_evidence_ready,
                "review_status": "deferred_pending_construction_evidence",
                "transition_review_failure": False,
            })
    payload = {
        "status": "success",
        "source_inventory_unchanged": True,
        "source_inventory_path": str(review_dir / "REVIEW_INVENTORY.json"),
        "source_inventory_sha256": _sha256(review_dir / "REVIEW_INVENTORY.json"),
        "records": records,
        "valid_transition_preview_count": sum(row["visual_preview_valid"] for row in records),
        "review_allowed_count": sum(row["review_allowed"] for row in records),
        "review_gate_reason": "formal Construction element/TFR evidence is not yet available",
        "transition_review_status": "deferred_due_to_missing_formal_construction_evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "TRANSITION_REVIEW_INVENTORY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

# Canonical Transition preview implementation. The legacy definitions above remain
# import-compatible, but all callers use the manifest-backed implementation.
diagnose_transition_previews = _canonical_diagnose_transition_previews
regenerate_transition_previews = _canonical_regenerate_transition_previews
