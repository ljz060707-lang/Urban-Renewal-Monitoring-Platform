from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import Window


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")
TRANSITIONS = {
    # Full20-supported evidence classes. These remain candidate evidence, not GT.
    "20180506_to_20230716": ("2018-05-06", "2023-07-16", True),
    "20200723_to_20230716": ("2020-07-23", "2023-07-16", True),
    "20220320_to_20230716": ("2022-03-20", "2023-07-16", True),
    # Construction-model-supported adjacent candidates; never called Full20.
    "20180506_to_20200723": ("2018-05-06", "2020-07-23", False),
    "20200723_to_20220320": ("2020-07-23", "2022-03-20", False),
    "20220320_to_20230716_construction": ("2022-03-20", "2023-07-16", False),
}

_HASH_CACHE: dict[str, str] = {}


def _sha256(path: Path) -> str:
    key = str(path.resolve())
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def _manifest_path(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "outputs"
        / "inference"
        / "beijing_tongzhou_building_geometry"
        / "TONGZHOU_SCENES.json"
    )


def _load_scenes(project_root: Path) -> dict[str, dict]:
    manifest_path = _manifest_path(project_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("scene_discovery_status") != "pass":
        raise RuntimeError(f"scene manifest is not pass: {manifest_path}")
    result = {}
    for row in payload.get("scenes", []):
        date = row.get("scene_date")
        source = Path(row.get("selected_source", ""))
        bands = row.get("rgb_bands")
        # Existing three-band manifest rows are unambiguous: the complete
        # source is RGB. Do not infer from filenames or generated previews.
        if not bands and int(row.get("bands", 0)) == 3:
            bands = [1, 2, 3]
            band_source = "manifest_band_count_three_default"
        else:
            band_source = "manifest_rgb_bands"
        if not isinstance(bands, list) or len(bands) != 3:
            raise ValueError(f"manifest has no usable RGB band selection for {date}")
        if not source.is_file():
            raise FileNotFoundError(f"manifest source does not exist for {date}: {source}")
        result[date] = {
            "path": source,
            "rgb_bands": tuple(int(value) for value in bands),
            "rgb_band_selection": band_source,
            "manifest_source_sha256": row.get("source_sha256", ""),
            "manifest_row": row,
        }
    missing = [date for date in DATES if date not in result]
    if missing:
        raise RuntimeError(f"scene manifest missing dates: {missing}")
    return result


def _window(index: int, width: int, height: int) -> tuple[Window, bool]:
    column, row = index % 4, index // 4
    x0, x1 = int(column * width / 4), int((column + 1) * width / 4)
    y0, y1 = int(row * height / 4), int((row + 1) * height / 4)
    inside = 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
    return Window(x0, y0, x1 - x0, y1 - y0), inside


def _read_rgb_window(scene: dict, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
    path = scene["path"]
    bands = scene["rgb_bands"]
    with rasterio.open(path) as source:
        window, inside = _window(index, source.width, source.height)
        if not inside:
            raise ValueError(f"window outside source bounds: {path} index={index}")
        if any(band < 1 or band > source.count for band in bands):
            raise ValueError(f"manifest RGB bands {bands} exceed {source.count}: {path}")
        data = source.read(bands, window=window)
        valid = np.all(source.read_masks(bands, window=window) > 0, axis=0)
        valid &= np.all(np.isfinite(data), axis=0)
        display = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        stretch_ranges = []
        for band_index in range(3):
            values = data[band_index][valid]
            if values.size:
                low, high = (float(value) for value in np.percentile(values, (2, 98)))
                if high <= low:
                    low, high = float(values.min()), float(values.max())
            else:
                low, high = 0.0, 1.0
            if high <= low:
                high = low + 1.0
            display[..., band_index] = np.clip((data[band_index] - low) / (high - low), 0.0, 1.0) * 255.0
            stretch_ranges.append([low, high])
        display[~valid] = 128
        normalized = display.astype(np.float64) / 255.0
        luminance = (
            0.2126 * normalized[..., 0]
            + 0.7152 * normalized[..., 1]
            + 0.0722 * normalized[..., 2]
        )
        valid_luminance = luminance[valid]
        quality = {
            "valid_pixel_fraction": float(valid.mean()) if valid.size else 0.0,
            "nonzero_fraction": float(np.any(display > 0, axis=2)[valid].mean()) if valid.any() else 0.0,
            "luminance_mean": float(valid_luminance.mean()) if valid_luminance.size else 0.0,
            "luminance_std": float(valid_luminance.std()) if valid_luminance.size else 0.0,
            "display_dynamic_range": float(np.percentile(valid_luminance, 98) - np.percentile(valid_luminance, 2)) if valid_luminance.size else 0.0,
            "source_read_success": True,
            "crop_inside_bounds": inside,
            "rgb_band_selection": list(bands),
            "rgb_band_selection_source": scene["rgb_band_selection"],
            "source_dtype": list(source.dtypes),
            "source_nodata": source.nodata,
            "source_crs": str(source.crs) if source.crs else None,
            "source_window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
            "stretch_percentiles": [2, 98],
            "stretch_ranges": stretch_ranges,
        }
        metadata = {
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "manifest_source_sha256": scene["manifest_source_sha256"],
            "manifest_sha256_matches_source": not scene["manifest_source_sha256"] or scene["manifest_source_sha256"] == _sha256(path),
            **quality,
        }
    reasons = []
    if not quality["source_read_success"]:
        reasons.append("source_read_failed")
    if not quality["crop_inside_bounds"]:
        reasons.append("crop_outside_bounds")
    if quality["valid_pixel_fraction"] <= 0:
        reasons.append("no_valid_pixels")
    if quality["display_dynamic_range"] <= 0.08:
        reasons.append("insufficient_display_dynamic_range")
    if quality["luminance_mean"] <= 0.03:
        reasons.append("preview_too_dark")
    if quality["luminance_mean"] >= 0.97:
        reasons.append("preview_too_bright")
    metadata["preview_status"] = "valid" if not reasons else "invalid"
    metadata["review_allowed"] = metadata["preview_status"] == "valid"
    metadata["invalid_reasons"] = reasons
    return display, valid, metadata


def _panel(rgb: np.ndarray, title: str, metadata: dict) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    image.thumbnail((720, 560), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (image.width, image.height + 76), (238, 239, 235))
    panel.paste(image, (0, 76))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(18, 23, 22), font=font)
    draw.text((10, 25), f"RGB {metadata['rgb_band_selection']} · 2-98% stretch", fill=(65, 72, 69), font=font)
    draw.text((10, 42), f"mean={metadata['luminance_mean']:.3f} range={metadata['display_dynamic_range']:.3f}", fill=(65, 72, 69), font=font)
    draw.text((10, 59), str(metadata["source_path"]), fill=(65, 72, 69), font=font)
    return panel


def _existing_paths(value) -> list[str]:
    paths = []
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    for item in values:
        if isinstance(item, str) and Path(item).is_file():
            paths.append(item)
    return paths


def _evidence(original_rows: dict, sample_id: str, index: int, before: str, after: str, full20_supported: bool) -> dict:
    geometry = {
        date: original_rows.get(f"geometry_{date}_{index:02d}", {}).get("source_paths", {})
        for date in DATES
    }
    building_paths = sorted({path for sources in geometry.values() for path in _existing_paths(sources)})
    transition_sources = original_rows.get(sample_id, {}).get("source_paths", {})
    full20_sources = {key: value for key, value in transition_sources.items() if any(token in str(key).lower() for token in ("full20", "change", "new", "demolition"))}
    full20_paths = _existing_paths(full20_sources)
    return {
        "raw_before_after": {"status": "available", "dates": [before, after]},
        "building_geometry": {"status": "available" if building_paths else "unavailable", "paths": building_paths},
        "construction_elements": {"status": "unavailable", "paths": [], "reason": "formal Construction pixel model is blocked by onboarding Gate"},
        "full20": {"status": "available" if full20_supported and full20_paths else ("unavailable" if full20_supported else "not_applicable"), "paths": full20_paths},
        "tfr": {"status": "unavailable", "paths": [], "reason": "human temporal labels and formal TFR validation are not available"},
        "parcel_boundary": {"status": "unavailable", "paths": []},
        "building_boundary": {"status": "available" if building_paths else "unavailable", "paths": building_paths},
        "proposed_state_sequence": {"status": "candidate_only", "label_source": "model_prediction", "human_review": "pending"},
    }


def _write_validation(output_dir: Path, records: list[dict]) -> None:
    fields = [
        "sample_id", "transition", "before_date", "after_date", "preview_status",
        "review_allowed", "valid_pixel_fraction", "nonzero_fraction", "luminance_mean",
        "luminance_std", "display_dynamic_range", "source_read_success", "crop_inside_bounds",
        "rgb_band_selection", "invalid_reasons", "evidence_missing",
    ]
    with (output_dir / "TRANSITION_PREVIEW_VALIDATION.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            quality = row["preview_quality"]
            writer.writerow({
                **row,
                **quality,
                "rgb_band_selection": json.dumps(quality["rgb_band_selection"]),
                "invalid_reasons": json.dumps(quality["invalid_reasons"], ensure_ascii=False),
                "evidence_missing": json.dumps(row["evidence_missing"], ensure_ascii=False),
            })
    invalid = [row for row in records if row["preview_quality"]["preview_status"] != "valid"]
    with (output_dir / "INVALID_PREVIEW_SAMPLES.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "preview_status", "invalid_reasons"], extrasaction="ignore")
        writer.writeheader()
        for row in invalid:
            writer.writerow({"sample_id": row["sample_id"], "preview_status": row["preview_quality"]["preview_status"], "invalid_reasons": json.dumps(row["preview_quality"]["invalid_reasons"], ensure_ascii=False)})


def diagnose_transition_previews(review_dir: Path, output_dir: Path) -> dict:
    review_dir, output_dir = Path(review_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((review_dir / "REVIEW_INVENTORY.json").read_text(encoding="utf-8"))
    records = [row for row in inventory.get("records", []) if row.get("review_group") == "transition"]
    source_overviews = sorted({row.get("source_paths", {}).get("overview", "") for row in records})
    hashes_by_index: dict[str, list[str]] = {}
    stats = []
    for row in records:
        path = Path(row.get("preview_paths", {}).get("tile", ""))
        if not path.is_file():
            stats.append({"sample_id": row["sample_id"], "preview_status": "invalid", "invalid_reasons": ["preview_missing"]})
            continue
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        index = row["sample_id"].rsplit("_", 1)[-1]
        hashes_by_index.setdefault(index, []).append(_sha256(path))
        luminance = array.astype(np.float64).mean(axis=2) / 255.0
        stats.append({
            "sample_id": row["sample_id"],
            "preview_status": "invalid" if float(luminance.mean()) <= 0.03 else "valid",
            "preview_sha256": _sha256(path),
            "preview_path": str(path),
            "luminance_mean": float(luminance.mean()),
        })
    result = {
        "status": "diagnosed",
        "transition_preview_count": len(records),
        "valid_under_display_gate": sum(row["preview_status"] == "valid" for row in stats),
        "invalid_under_display_gate": sum(row["preview_status"] != "valid" for row in stats),
        "identical_same_window_across_transitions": sum(len(set(values)) == 1 for values in hashes_by_index.values()),
        "unique_transition_source_overviews": source_overviews,
        "root_causes": [
            "legacy previews were cropped from overlay_overview.png instead of date-specific source RGB rasters",
            "the same overview was reused for multiple transitions",
            "legacy preview generation had no raw-image percentile stretch and validity gate",
        ],
        "transition_review_status": "deferred_due_to_invalid_visualization",
        "transition_model_status": "not_human_verified",
        "transition_review_failure": False,
        "records": stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "TRANSITION_PREVIEW_ROOT_CAUSE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def regenerate_transition_previews(project_root: Path, review_dir: Path, output_dir: Path) -> dict:
    project_root, review_dir, output_dir = Path(project_root), Path(review_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    scenes = _load_scenes(project_root)
    original_inventory = json.loads((review_dir / "REVIEW_INVENTORY.json").read_text(encoding="utf-8"))
    original_rows = {row["sample_id"]: row for row in original_inventory.get("records", [])}
    raw_assets: dict[tuple[str, int], dict] = {}
    for scene_date in DATES:
        for index in range(16):
            rgb, valid, metadata = _read_rgb_window(scenes[scene_date], index)
            path = output_dir / "raw" / f"raw_{scene_date}_{index:02d}.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            raw_assets[(scene_date, index)] = {"path": str(path), "rgb": rgb, "valid": valid, "metadata": metadata}
    records = []
    for transition, (before, after, full20_supported) in TRANSITIONS.items():
        for index in range(16):
            sample_id = f"transition_{transition}_{index:02d}"
            before_asset, after_asset = raw_assets[(before, index)], raw_assets[(after, index)]
            left = _panel(before_asset["rgb"], f"BEFORE {before}", before_asset["metadata"])
            right = _panel(after_asset["rgb"], f"AFTER {after}", after_asset["metadata"])
            canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (222, 224, 220))
            canvas.paste(left, (0, 0)); canvas.paste(right, (left.width, 0))
            pair_path = output_dir / "previews" / f"{sample_id}.png"
            canvas.save(pair_path)
            visual_quality = {
                "preview_status": "valid" if before_asset["metadata"]["preview_status"] == "valid" and after_asset["metadata"]["preview_status"] == "valid" else "invalid",
                "review_allowed": False,
                "valid_pixel_fraction": min(before_asset["metadata"]["valid_pixel_fraction"], after_asset["metadata"]["valid_pixel_fraction"]),
                "nonzero_fraction": min(before_asset["metadata"]["nonzero_fraction"], after_asset["metadata"]["nonzero_fraction"]),
                "luminance_mean": (before_asset["metadata"]["luminance_mean"] + after_asset["metadata"]["luminance_mean"]) / 2,
                "luminance_std": (before_asset["metadata"]["luminance_std"] + after_asset["metadata"]["luminance_std"]) / 2,
                "display_dynamic_range": min(before_asset["metadata"]["display_dynamic_range"], after_asset["metadata"]["display_dynamic_range"]),
                "source_read_success": True,
                "crop_inside_bounds": True,
                "rgb_band_selection": before_asset["metadata"]["rgb_band_selection"],
                "invalid_reasons": sorted(set(before_asset["metadata"]["invalid_reasons"] + after_asset["metadata"]["invalid_reasons"])),
            }
            evidence = _evidence(original_rows, sample_id, index, before, after, full20_supported)
            missing = [key for key, value in evidence.items() if value.get("status") == "unavailable"]
            visual_valid = visual_quality["preview_status"] == "valid"
            review_allowed = visual_valid and not missing
            visual_quality["review_allowed"] = review_allowed
            records.append({
                "sample_id": sample_id,
                "supersedes_sample_ids": [sample_id] if sample_id in original_rows else [],
                "transition": transition,
                "before_date": before,
                "after_date": after,
                "track_id": f"construction_review_track_{index:02d}",
                "object_type": "construction_track",
                "full20_supported": full20_supported,
                "transition_evidence_scope": "full20_and_geometry" if full20_supported else "construction_model_and_geometry_only",
                "scientific_status": "candidate_not_ground_truth" if full20_supported else "experimental_candidate",
                "raw_before_path": before_asset["path"],
                "raw_after_path": after_asset["path"],
                "pair_preview_path": str(pair_path),
                "source_metadata": {"before": before_asset["metadata"], "after": after_asset["metadata"]},
                "preview_quality": visual_quality,
                "evidence": evidence,
                "evidence_missing": missing,
                "visual_preview_valid": visual_valid,
                "construction_evidence_ready": not missing,
                "review_allowed": review_allowed,
                "review_status": "pending_human_transition_review" if review_allowed else "deferred_pending_formal_evidence",
                "transition_review_failure": False,
            })
    build = {
        "status": "success",
        "source_inventory_unchanged": True,
        "source_inventory_path": str(review_dir / "REVIEW_INVENTORY.json"),
        "source_inventory_sha256": _sha256(review_dir / "REVIEW_INVENTORY.json"),
        "transition_count": len(records),
        "valid_transition_preview_count": sum(row["visual_preview_valid"] for row in records),
        "invalid_transition_preview_count": sum(not row["visual_preview_valid"] for row in records),
        "review_allowed_count": sum(row["review_allowed"] for row in records),
        "transition_review_status": "PENDING HUMAN TRANSITION REVIEW" if any(row["review_allowed"] for row in records) else "PENDING FORMAL CONSTRUCTION EVIDENCE",
        "construction_status": "experimental_in_progress",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "TRANSITION_PREVIEW_BUILD.json").write_text(json.dumps(build, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "TRANSITION_REVIEW_INVENTORY.json").write_text(json.dumps({**build, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_validation(output_dir, records)
    return {**build, "records": records, "review_gate_reason": "formal Construction element/TFR/parcel evidence is not available" if build["review_allowed_count"] == 0 else "visual and evidence Gates passed; human review pending"}
