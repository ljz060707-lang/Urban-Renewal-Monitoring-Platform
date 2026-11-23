"""Finalize the Xi'an two-period GE3/GE5 package without CC rescans.

The accelerator writes the missing masks and component ledgers.  This script
is a deterministic, fail-closed finalizer: it verifies every expected raster
and exact source grid, derives statistics from the component table where
available, performs only one large-window exact raster QA per unique product,
and writes the acceptance/manifest/report artifacts atomically.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from build_primary_two_period_results import (
        EXPECTED_B0_SHA,
        FILTERS,
        METRIC_CRS,
        PERIODS,
        STATES,
        assert_same_grid,
        grid_meta,
        row_pixel_areas,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:
    from tools.build_primary_two_period_results import (
        EXPECTED_B0_SHA,
        FILTERS,
        METRIC_CRS,
        PERIODS,
        STATES,
        assert_same_grid,
        grid_meta,
        row_pixel_areas,
        sha256_file,
        write_json,
    )


def _valid_raster(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0 or path.with_suffix(path.suffix + ".part").exists():
        return False
    try:
        import rasterio

        with rasterio.open(path) as ds:
            return ds.count == 1 and ds.width > 0 and ds.height > 0 and ds.crs is not None
    except Exception:
        return False


def raster_stats_fast(path: Path, row_areas: np.ndarray, block: int = 2048) -> tuple[int, float]:
    """Exact nonzero count/area using large windows rather than 256px blocks."""
    import rasterio
    from rasterio.windows import Window

    pixels, area = 0, 0.0
    with rasterio.open(path) as src:
        for row0 in range(0, src.height, block):
            for col0 in range(0, src.width, block):
                h, w = min(block, src.height - row0), min(block, src.width - col0)
                arr = src.read(1, window=Window(col0, row0, w, h))
                counts = (arr != 0).sum(axis=1)
                pixels += int(counts.sum())
                area += float(np.dot(counts.astype(np.float64), row_areas[row0 : row0 + h]))
    return pixels, area


def _read_ledger(target: Path, period: str, state: str) -> dict[str, Any] | None:
    p = target / "04_statistics" / f"COMPONENT_LEDGER_{period}_{state}.json"
    if not p.is_file():
        return None


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy an immutable source into the package without exposing a partial file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.unlink(missing_ok=True)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _encode_three_state(paths: dict[str, Path], out: Path, block: int) -> None:
    """Build one encoded three-state raster for an unfiltered period only."""
    import rasterio
    from rasterio.windows import Window

    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(paths["STABLE"]) as stable, rasterio.open(paths["NEW"]) as new, rasterio.open(paths["DEMOLITION"]) as demolition:
        profile = stable.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", predictor=2,
                       tiled=True, BIGTIFF="IF_SAFER", SPARSE_OK=True)
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.unlink(missing_ok=True)
        with rasterio.open(tmp, "w", **profile) as dst:
            for row0 in range(0, stable.height, block):
                for col0 in range(0, stable.width, block):
                    h, w = min(block, stable.height - row0), min(block, stable.width - col0)
                    win = Window(col0, row0, w, h)
                    a = stable.read(1, window=win) != 0
                    b = new.read(1, window=win) != 0
                    c = demolition.read(1, window=win) != 0
                    dst.write((a.astype(np.uint8) + 2 * b.astype(np.uint8) + 3 * c.astype(np.uint8)), 1, window=win)
        os.replace(tmp, out)


def _materialize_unfiltered(builder: Path, target: Path, periods: Iterable[str], block: int) -> None:
    """Ensure the package has immutable UNFILTERED copies for both periods.

    The builder owns the canonical source rasters.  Earlier interrupted runs
    materialized 2018-2019 only; filling the missing 2019-2020 package paths is
    a packaging repair, not a model or filtering operation.
    """
    for period in periods:
        src = {state: builder / "02_rasters" / f"XIAN_{period}_{state}.tif" for state in STATES}
        out_dir = target / "02_rasters" / period / "UNFILTERED"
        out = {state: out_dir / f"XIAN_{period}_{state}.tif" for state in STATES}
        for state in STATES:
            if not _valid_raster(out[state]):
                _atomic_copy(src[state], out[state])
        combined = out_dir / f"XIAN_{period}_THREE_STATE.tif"
        if not _valid_raster(combined):
            _encode_three_state(out, combined, block)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _canonical_stats(builder: Path, target: Path) -> list[dict[str, Any]]:
    import pyogrio

    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        for state in STATES:
            gpkg = builder / "01_vectors" / f"{period}_{state}.gpkg"
            if not gpkg.is_file():
                continue
            info = pyogrio.read_info(gpkg)
            try:
                tab = pyogrio.read_dataframe(gpkg, read_geometry=False, columns=["area_m2"])
                area = float(tab["area_m2"].sum()) if "area_m2" in tab.columns else float("nan")
            except Exception:
                area = float("nan")
            rows.append({
                "period": period,
                "state": state,
                "object_count": int(info.get("features", 0)),
                "canonical_object_area_m2": area,
                "source_gpkg": str(gpkg.resolve()),
                "candidate_not_ground_truth": True,
            })
    out = target / "04_statistics" / "CANONICAL_OBJECT_AREA_STATISTICS.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(rows[0]) if rows else ["period", "state", "object_count", "canonical_object_area_m2", "source_gpkg", "candidate_not_ground_truth"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--builder-output", type=Path, required=True)
    ap.add_argument("--target-root", type=Path, required=True)
    ap.add_argument("--city-boundary", type=Path, default=None)
    ap.add_argument("--block", type=int, default=2048)
    args = ap.parse_args()
    builder = args.builder_output.resolve()
    target = args.target_root.resolve()

    source: dict[tuple[str, str], Path] = {(p, s): builder / "02_rasters" / f"XIAN_{p}_{s}.tif" for p in PERIODS for s in STATES}
    missing_sources = [str(x) for x in source.values() if not _valid_raster(x)]
    if missing_sources:
        raise RuntimeError("missing source rasters: " + "; ".join(missing_sources))
    grid = assert_same_grid(source.values())
    row_areas = row_pixel_areas(next(iter(source.values())))

    # Keep a complete, self-contained package.  Some earlier runs wrote the
    # 2018-2019 UNFILTERED copies but never materialized the second period;
    # restore those paths directly from the immutable builder sources before
    # checking the final product gate.
    _materialize_unfiltered(builder, target, PERIODS, args.block)

    expected: list[tuple[str, str, str, Path]] = []
    for period in PERIODS:
        for filt in FILTERS:
            for state in STATES:
                expected.append((period, filt, state, target / "02_rasters" / period / filt / f"XIAN_{period}_{state}.tif"))
            if filt != "UNFILTERED":
                for state in ("NEW", "DEMOLITION"):
                    expected.append((period, filt, state + "_FILTERED_OUT", target / "02_rasters" / period / filt / f"XIAN_{period}_{state}_FILTERED_OUT.tif"))
            expected.append((period, filt, "THREE_STATE", target / "02_rasters" / period / filt / f"XIAN_{period}_THREE_STATE.tif"))
    missing_products = [str(p) for _, _, _, p in expected if not _valid_raster(p)]
    if missing_products:
        write_json(target / "14_acceptance" / "PRIMARY_FINALIZER_BLOCKED.json", {"status": "BLOCKED_MISSING_PRODUCTS", "missing_products": missing_products, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}})
        raise RuntimeError("missing expected products: " + "; ".join(missing_products))

    # Verify every product against the immutable source grid before any
    # statistics are promoted.
    for _, _, _, p in expected:
        assert_same_grid((next(iter(source.values())), p))

    # One exact scan per unique mask.  Stable/UNFILTERED and any reused masks
    # are cached by SHA, while filtered-out area is obtained by subtraction
    # from disjoint source and retained masks whenever possible.
    stat_cache: dict[str, tuple[int, float]] = {}
    def stats(path: Path) -> tuple[int, float]:
        key = sha256_file(path)
        if key not in stat_cache:
            stat_cache[key] = raster_stats_fast(path, row_areas, args.block)
        return stat_cache[key]

    rows: list[dict[str, Any]] = []
    for period in PERIODS:
        for filt in FILTERS:
            for state in STATES:
                p = source[(period, state)] if filt == "UNFILTERED" else target / "02_rasters" / period / filt / f"XIAN_{period}_{state}.tif"
                pixels, area = stats(p)
                ledger = _read_ledger(target, period, state) if filt in {"GE3", "GE5"} and state in {"NEW", "DEMOLITION"} else None
                row: dict[str, Any] = {
                    "period": period,
                    "filter": filt,
                    "state": state,
                    "pixels": pixels,
                    "area_m2": area,
                    "area_km2": area / 1e6,
                    "component_count_before": None,
                    "component_count_after": None,
                    "components_removed": None,
                    "pixels_removed": None,
                    "retention_rate": None,
                    "source_grid_exact": True,
                    "sha256": sha256_file(p),
                    "candidate_not_ground_truth": True,
                }
                if ledger is not None:
                    row["component_count_before"] = ledger.get("total_components")
                    key = "components_ge3" if filt == "GE3" else "components_ge5"
                    pix_key = "pixels_ge3" if filt == "GE3" else "pixels_ge5"
                    row["component_count_after"] = ledger.get(key)
                    row["components_removed"] = (ledger.get("total_components", 0) - ledger.get(key, 0))
                    row["pixels_removed"] = ledger.get("total_nonzero_pixels", 0) - ledger.get(pix_key, 0)
                    row["retention_rate"] = ledger.get("retention_rate_ge3" if filt == "GE3" else "retention_rate_ge5")
                rows.append(row)
            if filt != "UNFILTERED":
                for state in ("NEW", "DEMOLITION"):
                    src_pixels, src_area = stats(source[(period, state)])
                    retained = target / "02_rasters" / period / filt / f"XIAN_{period}_{state}.tif"
                    kept_pixels, kept_area = stats(retained)
                    removed = target / "02_rasters" / period / filt / f"XIAN_{period}_{state}_FILTERED_OUT.tif"
                    # Exact disjoint-area check.  The output is still opened
                    # once below if its hash is not already cached, providing
                    # a final nonzero verification without another full scan.
                    removed_pixels, removed_area = stats(removed)
                    if removed_pixels != src_pixels - kept_pixels or not math.isclose(removed_area, src_area - kept_area, rel_tol=0.0, abs_tol=1e-6):
                        raise RuntimeError(f"filtered-out complement mismatch: {period} {filt} {state}")

    stats_path = target / "04_statistics" / "TWO_PERIOD_THREE_STATE_STATISTICS.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    city_area_m2 = float("nan")
    if args.city_boundary and args.city_boundary.is_file():
        try:
            import geopandas as gpd

            city_area_m2 = float(gpd.read_file(args.city_boundary).to_crs(METRIC_CRS).geometry.union_all().area)
        except Exception:
            city_area_m2 = float("nan")
    for row in rows:
        row["study_area_m2"] = city_area_m2
        row["share_of_city_pct"] = (row["area_m2"] / city_area_m2 * 100.0) if math.isfinite(city_area_m2) and city_area_m2 > 0 else float("nan")
    fields = ["period", "filter", "state", "pixels", "area_m2", "area_km2", "study_area_m2", "share_of_city_pct", "component_count_before", "component_count_after", "components_removed", "pixels_removed", "retention_rate", "source_grid_exact", "sha256", "candidate_not_ground_truth"]
    with stats_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    write_json(target / "04_statistics" / "TWO_PERIOD_THREE_STATE_STATISTICS.json", {"rows": rows, "grid": grid, "pixel_area_method": "row-wise EPSG:32649 corner transform", "exact_unique_raster_scans": len(stat_cache), "candidate_not_ground_truth": True})
    canonical = _canonical_stats(builder, target)

    products_manifest = []
    for period, filt, state, p in expected:
        products_manifest.append({"period": period, "filter": filt, "state": state, "path": str(p.resolve()), "sha256": sha256_file(p), "bytes": p.stat().st_size, "grid": grid})
    write_json(target / "00_provenance" / "PRIMARY_PROVENANCE.json", {"builder_output": str(builder), "source_rasters": {f"{p}_{s}": {"path": str(source[(p, s)].resolve()), "sha256": sha256_file(source[(p, s)]), "grid": grid} for p in PERIODS for s in STATES}, "b0_checkpoint_sha256": EXPECTED_B0_SHA, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}, "raw_inputs_modified": False, "candidate_not_ground_truth": True})
    write_json(target / "00_provenance" / "RASTER_OUTPUT_MANIFEST.json", {"products": products_manifest, "grid": grid, "filters": {"connectivity": 8, "min_component_pixels": [3, 5], "stable_policy": "unchanged", "filtered_change_policy": "not_relabeled_STABLE"}, "component_ledger_dir": str((target / "04_statistics").resolve())})
    write_json(target / "14_acceptance" / "PRIMARY_OUTPUT_SHA256.json", {"products": [{"path": x["path"], "sha256": x["sha256"], "bytes": x["bytes"]} for x in products_manifest], "source_grid": grid})
    write_json(target / "14_acceptance" / "TWO_PERIOD_DEDUP_THREE_STATE_ACCEPTANCE.json", {"PRIMARY_TWO_PERIOD_RESULT_COMPLETE": True, "TWO_PERIOD_DEDUP_THREE_STATE_COMPLETE": True, "period_state_pass_gate": {"2018_2019_STABLE_PASS": True, "2018_2019_ADDITION_PASS": True, "2018_2019_DEMOLITION_PASS": True, "2019_2020_STABLE_PASS": True, "2019_2020_ADDITION_PASS": True, "2019_2020_DEMOLITION_PASS": True, "source_grid_exact": True}, "products_complete": True, "statistics_complete": True, "canonical_object_area_statistics": str((target / "04_statistics" / "CANONICAL_OBJECT_AREA_STATISTICS.csv").resolve()), "exact_unique_raster_scans": len(stat_cache), "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}, "candidate_not_ground_truth": True, "FUSION_STARTED": False})
    report = [
        "# 两期三态去重主结果（加速后）",
        "",
        "- 2018–2019 与 2019–2020 六张源栅格均核验为可用；所有目标栅格与源网格宽高、CRS、transform 完全一致。",
        "- GE3/GE5 使用 8 邻域、组件像元数 ≥3/≥5；STABLE 原样保留，FILTERED_OUT 不改标为 STABLE。",
        "- 尚未完成的 period×state 由一次组件分析同时产出 GE3、GE5 及两层 FILTERED_OUT；组件统计台账用于保留/剔除计数。",
        f"- 最终精确非零/面积 QA 使用 {len(stat_cache)} 个唯一栅格哈希，窗口大小 {args.block}；未读取 WHU/SECOND/Holdout。",
        "- 结果为候选证据，不是人工 Ground Truth；FUSION_STARTED=false。",
        "",
    ]
    target.joinpath("FINAL_TWO_PERIOD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(target / "00_provenance" / "ACCELERATED_FINALIZER_RECEIPT.json", {"status": "PASS", "products_complete": True, "statistics_complete": True, "exact_unique_raster_scans": len(stat_cache), "component_ledgers_used": [str(p.resolve()) for p in (target / "04_statistics").glob("COMPONENT_LEDGER_*.json")], "source_grid": grid, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}})
    print(f"PRIMARY_FINALIZE_PASS products={len(products_manifest)} unique_scans={len(stat_cache)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
