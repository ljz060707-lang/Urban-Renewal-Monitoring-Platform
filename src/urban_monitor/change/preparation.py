from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import csv
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .adapters.base import read_binary
from .adapters.second import semantic_ids
from .registry import enabled_adapters, load_change_config
from .schema import CHANGE_INDEX_COLUMNS, ChangeSample, validate_sample


def create_change_run(output_root: str | Path = "outputs/pretraining/change", run_id: str | None = None) -> Path:
    root = Path(output_root)
    run = root / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    suffix = 1
    while run.exists():
        run = root / f"{run.name}_{suffix:02d}"
        suffix += 1
    for name in ("prepared", "dataset_audit", "checkpoints", "metrics", "predictions", "previews", "errors", "visualizations", "review", "reports", "logs"):
        (run / name).mkdir(parents=True, exist_ok=True)
    return run


def _mask(path: Path | None) -> np.ndarray | None:
    return read_binary(path) if path else None


def _rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _colorize(array: np.ndarray, *, semantic: bool = False) -> Image.Image:
    array = np.asarray(array)
    if not semantic:
        output = np.zeros((*array.shape, 3), dtype=np.uint8)
        output[array != 0] = (255, 77, 91)
        return Image.fromarray(output)
    palette = np.array([
        [20, 24, 31], [255, 191, 71], [93, 173, 226], [103, 194, 58],
        [39, 174, 214], [166, 124, 82], [196, 92, 224], [235, 235, 235],
    ], dtype=np.uint8)
    return Image.fromarray(palette[np.asarray(array, dtype=np.int64) % len(palette)])


def _preview(path: Path, panels: list[tuple[str, Image.Image]]) -> None:
    if not panels:
        return
    width = max(image.width for _, image in panels)
    height = max(image.height for _, image in panels)
    thumb_w = min(width, 256)
    thumb_h = min(height, 256)
    canvas = Image.new("RGB", (thumb_w * len(panels), thumb_h + 28), "#11151b")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        display = image.copy()
        display.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
        canvas.paste(display, (index * thumb_w, 28))
        draw.text((index * thumb_w + 6, 7), title, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _s2_audit(records: list[ChangeSample], audit: Path) -> None:
    rows, inconsistent = [], []
    for sample in records:
        cd, new, dem = _mask(sample.gt_cd_path), _mask(sample.gt_new_path), _mask(sample.gt_dem_path)
        mismatch = float(sample.notes.rsplit("=", 1)[-1]) if "mismatch=" in sample.notes else float(np.mean(cd != np.logical_or(new, dem)))
        rows.append({
            "split": sample.split, "sample_id": sample.sample_id,
            "new_pixel_ratio": float(new.mean()), "demolition_pixel_ratio": float(dem.mean()),
            "empty_change": int(not cd.any()), "logic_mismatch_ratio": mismatch,
        })
        if mismatch > 0:
            inconsistent.append(rows[-1])
    pd.DataFrame(rows).to_csv(audit / "s2looking_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inconsistent, columns=list(rows[0]) if rows else []).to_csv(
        audit / "s2looking_inconsistent.csv", index=False, encoding="utf-8-sig"
    )


def _second_audit(records: list[ChangeSample], audit: Path, config: dict, preview_count: int) -> None:
    distribution: Counter[tuple[str, int]] = Counter()
    transitions: Counter[tuple[int, int]] = Counter()
    for sample in records:
        t1 = semantic_ids(sample.gt_t1_path)
        t2 = semantic_ids(sample.gt_t2_path)
        for value, count in zip(*np.unique(t1, return_counts=True)):
            distribution[("t1", int(value))] += int(count)
        for value, count in zip(*np.unique(t2, return_counts=True)):
            distribution[("t2", int(value))] += int(count)
        pairs, counts = np.unique(np.stack([t1.ravel(), t2.ravel()], axis=1), axis=0, return_counts=True)
        for pair, count in zip(pairs, counts):
            transitions[(int(pair[0]), int(pair[1]))] += int(count)
    pd.DataFrame([
        {"date": date, "class_id": class_id, "pixel_count": count}
        for (date, class_id), count in sorted(distribution.items())
    ]).to_csv(audit / "second_class_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"class_t1": t1, "class_t2": t2, "pixel_count": count}
        for (t1, t2), count in sorted(transitions.items())
    ]).to_csv(audit / "second_transition_matrix.csv", index=False, encoding="utf-8-sig")
    mapping = {
        "building_class_id": config.get("building_class_id"),
        "class_mapping_file": config.get("class_mapping_file"),
        "class_mapping": config.get("class_mapping", {}),
        "verified_by_user": bool(config.get("mapping_verified", False)),
    }
    (audit / "second_color_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_root = audit / "second_previews"
    for index, sample in enumerate(records[:preview_count]):
        _preview(preview_root / f"{index:03d}_{sample.sample_id.rsplit(':', 1)[-1]}.jpg", [
            ("T1", _rgb(sample.t1_path)), ("T2", _rgb(sample.t2_path)),
            ("GT_T1", _colorize(semantic_ids(sample.gt_t1_path), semantic=True)),
            ("GT_T2", _colorize(semantic_ids(sample.gt_t2_path), semantic=True)),
            ("GT_CD", _colorize(_mask(sample.gt_cd_path))),
            ("GT_NEW", _colorize(_mask(sample.gt_new_path))),
            ("GT_DEM", _colorize(_mask(sample.gt_dem_path))),
        ])


def _dsifn_audit(records: list[ChangeSample], audit: Path, preview_count: int) -> None:
    rows = []
    for sample in records:
        cd = _mask(sample.gt_cd_path)
        rows.append({"split": sample.split, "sample_id": sample.sample_id, "change_pixel_ratio": float(cd.mean()), "empty_change": int(not cd.any())})
    pd.DataFrame(rows).to_csv(audit / "dsifn_summary.csv", index=False, encoding="utf-8-sig")
    for index, sample in enumerate(records[:preview_count]):
        _preview(audit / "dsifn_previews" / f"{index:03d}_{sample.sample_id.rsplit(':', 1)[-1]}.jpg", [
            ("T1", _rgb(sample.t1_path)), ("T2", _rgb(sample.t2_path)), ("GT_CD", _colorize(_mask(sample.gt_cd_path))),
        ])


def prepare_public_change(config_path: str | Path, *, output_root: str | Path | None = None, run_id: str | None = None) -> Path:
    config = load_change_config(config_path)
    run = create_change_run(output_root or config.get("output_root", "outputs/pretraining/change"), run_id)
    normalized_root = run / "prepared" / "normalized_labels"
    normalized: list[ChangeSample] = []
    raw_by_dataset: dict[str, list[ChangeSample]] = {}
    adapters = enabled_adapters(config)
    for adapter in adapters:
        records = adapter.discover()
        raw_by_dataset[adapter.dataset_name] = records
        for record in records:
            item = adapter.normalize(record, normalized_root / adapter.dataset_name)
            validate_sample(item)
            normalized.append(item)
    if not normalized:
        raise ValueError("No public change samples were prepared")
    frame = pd.DataFrame([sample.to_row() for sample in normalized], columns=CHANGE_INDEX_COLUMNS)
    frame.to_csv(run / "prepared" / "public_change_index.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby(["dataset_name", "split"], dropna=False).agg(
        samples=("sample_id", "count"), binary_samples=("has_binary", "sum"),
        directional_samples=("has_directional", "sum"), semantic_samples=("has_semantic", "sum"),
    ).reset_index()
    summary.to_csv(run / "dataset_audit" / "public_change_summary.csv", index=False, encoding="utf-8-sig")
    by_name = {adapter.dataset_name: adapter for adapter in adapters}
    if "s2looking" in raw_by_dataset:
        _s2_audit(normalized_for(normalized, "s2looking"), run / "dataset_audit")
    if "second" in raw_by_dataset:
        _second_audit(
            normalized_for(normalized, "second"), run / "dataset_audit",
            by_name["second"].config, int(config.get("audit", {}).get("second_preview_count", 30)),
        )
    if "dsifn" in raw_by_dataset:
        _dsifn_audit(normalized_for(normalized, "dsifn"), run / "dataset_audit", int(config.get("audit", {}).get("preview_count", 30)))
    manifest = {
        "run_id": run.name, "status": "prepared", "config": str(Path(config_path).resolve()),
        "sample_count": len(normalized), "datasets": sorted(raw_by_dataset),
        "index": str((run / "prepared" / "public_change_index.csv").resolve()),
    }
    (run / "RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run


def normalized_for(samples: list[ChangeSample], dataset: str) -> list[ChangeSample]:
    return [sample for sample in samples if sample.dataset_name == dataset]
