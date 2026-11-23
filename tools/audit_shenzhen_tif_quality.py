"""Read-only quality and analysis-ready contract for the four Shenzhen TIFs.

The source mosaics remain untouched.  This audit samples a deterministic grid
of windows, records valid/brightness/blur proxies, and writes explicit
sampled-quality masks plus analysis-ready passthrough manifests.  It never
claims a full-raster cloud or seamline mask when only sampled evidence exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box, mapping


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grid_windows(width: int, height: int, grid: int, size: int):
    for row in range(grid):
        for col in range(grid):
            x = min(max(0, int((col + 0.5) * width / grid - size / 2)), max(0, width - size))
            y = min(max(0, int((row + 0.5) * height / grid - size / 2)), max(0, height - size))
            yield row, col, Window(x, y, min(size, width), min(size, height))


def sample_epoch(epoch: dict, grid: int, size: int) -> tuple[dict, list[dict]]:
    path = Path(epoch["tif_path"])
    samples: list[dict] = []
    with rasterio.open(path) as ds:
        for row, col, window in grid_windows(ds.width, ds.height, grid, size):
            record = {
                "task_name": epoch["task_name"],
                "row": row,
                "col": col,
                "window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
                "status": "PASS",
                "error": None,
            }
            try:
                data = ds.read([1, 2, 3], window=window, boundless=False).astype(np.float32)
                finite = np.isfinite(data).all(axis=0)
                luminance = 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]
                gray = luminance[finite]
                if gray.size == 0:
                    raise ValueError("no finite pixels")
                # A conservative local blur proxy: gradient energy, not a
                # classifier.  It is only used to flag samples for review.
                gx = np.diff(luminance, axis=1)
                gy = np.diff(luminance, axis=0)
                grad = float((np.abs(gx).mean() + np.abs(gy).mean()) / 2.0)
                record.update(
                    {
                        "valid_pixel_fraction": float(finite.mean()),
                        "mean_luminance_0_255": float(gray.mean()),
                        "std_luminance_0_255": float(gray.std()),
                        "dynamic_range_p02_p98": float(np.percentile(gray, 98) - np.percentile(gray, 2)),
                        "gradient_energy_proxy": grad,
                        "zero_pixel_fraction": float((data == 0).all(axis=0).mean()),
                        "bounds": list(window_bounds(window, ds.transform)),
                        "crs": str(ds.crs) if ds.crs else None,
                    }
                )
                if record["valid_pixel_fraction"] <= 0:
                    record["status"] = "FAIL_NO_VALID_PIXELS"
                elif record["dynamic_range_ p02_p98".replace(" ", "")] <= 1.0:
                    record["status"] = "WARN_LOW_DYNAMIC_RANGE"
            except Exception as exc:  # noqa: BLE001 - persisted as audit evidence
                record["status"] = "FAIL_READ"
                record["error"] = f"{type(exc).__name__}: {exc}"
            samples.append(record)
    passed = [row for row in samples if row["status"] == "PASS"]
    failed = [row for row in samples if row["status"].startswith("FAIL")]
    summary = {
        "task_name": epoch["task_name"],
        "tif_path": str(path),
        "source_sha256": epoch.get("source_sha256") or sha256(path),
        "sample_count": len(samples),
        "passed_samples": len(passed),
        "failed_samples": len(failed),
        "sample_valid_pixel_fraction_mean": float(np.mean([r.get("valid_pixel_fraction", 0.0) for r in samples])),
        "sample_zero_pixel_fraction_mean": float(np.mean([r.get("zero_pixel_fraction", 1.0) for r in samples])),
        "sampled_quality_status": "PASS" if not failed else "BLOCKED_SAMPLE_READ_FAILURE",
        "quality_scope": "deterministic_grid_sample_not_full_raster_mask",
        "raw_tif_modified": False,
        "analysis_ready_view": "immutable_source_passthrough",
        "scientific_status": "candidate_not_ground_truth",
    }
    return summary, samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=512)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    quality_root = args.output_root / "02_quality"
    ready_root = args.output_root / "03_analysis_ready"
    quality_root.mkdir(parents=True, exist_ok=True)
    ready_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    samples: list[dict] = []
    for idx, epoch in enumerate(manifest["epochs"]):
        summary, rows = sample_epoch(epoch, args.grid, args.window_size)
        summary["epoch_index"] = idx
        summaries.append(summary)
        samples.extend(rows)
        ready_payload = {
            "schema_version": "urrs.shenzhen.analysis_ready_passthrough.v1",
            "created_at": now(),
            "epoch_index": idx,
            "task_name": epoch["task_name"],
            "source_tif": epoch["tif_path"],
            "source_sha256": summary["source_sha256"],
            "view_path": epoch["tif_path"],
            "view_method": "immutable_source_passthrough",
            "quality_audit": "02_quality/SHENZHEN_TIF_QUALITY_AUDIT.json",
            "quality_mask_scope": "sampled_windows_only",
            "full_cloud_shadow_haze_mask": "not_claimed",
            "registration_status": "not_yet_audited",
            "raw_tif_modified": False,
            "ready_for_geometry_inference": summary["sampled_quality_status"] == "PASS",
            "scientific_status": "candidate_not_ground_truth",
        }
        (ready_root / f"ANALYSIS_READY_T{idx}.json").write_text(
            json.dumps(ready_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    fields = sorted({key for row in summaries for key in row})
    with (quality_root / "SHENZHEN_TIF_QUALITY_SUMMARY.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    features = []
    for row in samples:
        if "bounds" in row:
            geom = box(*row["bounds"])
            features.append({"type": "Feature", "geometry": mapping(geom), "properties": row})
    (quality_root / "QUALITY_SAMPLE_MASKS.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "urrs.shenzhen.tif_quality_audit.v1",
        "created_at": now(),
        "manifest": str(args.manifest.resolve()),
        "grid": args.grid,
        "window_size": args.window_size,
        "epochs": summaries,
        "samples": samples,
        "quality_mask_scope": "sampled_windows_not_full_raster_masks",
        "analysis_ready_policy": "immutable_source_passthrough_until_registration_and_quality_masks_are_explicitly audited",
        "raw_tifs_modified": False,
        "test_pixel_reads": 0,
        "scientific_status": "candidate_not_ground_truth",
    }
    (quality_root / "SHENZHEN_TIF_QUALITY_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema_version": "urrs.shenzhen.quality_receipt.v1",
        "created_at": now(),
        "status": "PASS_SAMPLED_READ_GATE" if all(s["sampled_quality_status"] == "PASS" for s in summaries) else "BLOCKED_SAMPLE_READ_FAILURE",
        "full_raster_quality_masks": False,
        "analysis_ready_views": [str(p) for p in sorted(ready_root.glob("ANALYSIS_READY_T*.json"))],
        "raw_tifs_modified": False,
        "test_pixel_reads": 0,
        "scientific_status": "candidate_not_ground_truth",
    }
    (quality_root / "QUALITY_AUDIT_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS_SAMPLED_READ_GATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
