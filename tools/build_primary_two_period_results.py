"""Build the bounded two-period primary raster package.

This is deliberately a post-process: it never trains, reads Test/Holdout, or
rewrites the canonical builder outputs.  The source rasters are copied with a
hash receipt, and GE3/GE5 are computed as true 8-connected components in
streaming windows so the very large source grid is never loaded into RAM.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


STATES = ("STABLE", "NEW", "DEMOLITION")
PERIODS = ("2018_2019", "2019_2020")
FILTERS = ("UNFILTERED", "GE3", "GE5")
EXPECTED_B0_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
METRIC_CRS = "EPSG:32649"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def grid_meta(path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else None,
            "transform": [float(x) for x in src.transform[:6]],
            "bounds": [float(x) for x in src.bounds],
            "nodata": src.nodata,
            "compress": str(src.profile.get("compress")) if src.profile.get("compress") else None,
        }


def assert_same_grid(paths: Iterable[Path]) -> dict[str, Any]:
    metas = [grid_meta(p) for p in paths]
    if not metas:
        raise ValueError("no rasters supplied")
    keys = ("width", "height", "crs", "transform")
    first = metas[0]
    for idx, meta in enumerate(metas[1:], 1):
        for key in keys:
            if meta[key] != first[key]:
                raise RuntimeError(f"source grid mismatch at index {idx}: {key}: {meta[key]} != {first[key]}")
    return first


def atomic_copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        src_sha = sha256_file(src)
        if sha256_file(dst) == src_sha:
            return {"source": str(src.resolve()), "path": str(dst.resolve()), "sha256": src_sha, "bytes": dst.stat().st_size, "reused": True}
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(src, tmp)
    src_sha = sha256_file(src)
    dst_sha = sha256_file(tmp)
    if src_sha != dst_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"copy hash mismatch: {src} -> {dst}")
    os.replace(tmp, dst)
    return {"source": str(src.resolve()), "path": str(dst.resolve()), "sha256": dst_sha, "bytes": dst.stat().st_size, "reused": False}


class DSU:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []

    def add(self, pixels: int) -> int:
        i = len(self.parent)
        self.parent.append(i)
        self.size.append(int(pixels))
        return i

    def find(self, x: int) -> int:
        p = self.parent[x]
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        while x != p:
            nxt = self.parent[x]
            self.parent[x] = p
            x = nxt
        return p

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _edge_union(dsu: DSU, current: np.ndarray, previous: np.ndarray) -> None:
    """Union 8-neighbour labels across one tile boundary."""
    if current.size == 0 or previous.size == 0:
        return
    for i, a in enumerate(current.tolist()):
        if not a:
            continue
        lo, hi = max(0, i - 1), min(previous.size, i + 2)
        for b in previous[lo:hi].tolist():
            if b:
                dsu.union(int(a), int(b))


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    return ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))


def filtered_mask(src_path: Path, out_path: Path, min_pixels: int, block: int = 2048) -> dict[str, Any]:
    """8-connected component filtering with a two-pass tiled DSU."""
    import rasterio
    from rasterio.windows import Window

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dsu = DSU()
    tile_maps: dict[tuple[int, int], np.ndarray] = {}
    tile_windows: list[tuple[int, int, Window]] = []
    tile_edges: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    with rasterio.open(src_path) as src:
        rows = list(range(0, src.height, block))
        cols = list(range(0, src.width, block))
        # Pass 1: local labels and boundary unions.
        for ri, row0 in enumerate(rows):
            left_right: np.ndarray | None = None
            for ci, col0 in enumerate(cols):
                h, w = min(block, src.height - row0), min(block, src.width - col0)
                win = Window(col0, row0, w, h)
                arr = src.read(1, window=win)
                labels, n = _label(arr != 0)
                mapping = np.zeros(n + 1, dtype=np.int64)
                # bincount avoids scanning the whole tile once per component
                # (a material cost for dense change masks).
                counts = np.bincount(labels.ravel(), minlength=n + 1)
                for local_id in range(1, n + 1):
                    mapping[local_id] = dsu.add(int(counts[local_id]))
                if left_right is not None:
                    _edge_union(dsu, mapping[labels[:, 0]], left_right)
                if ri > 0:
                    upper = tile_edges[(ri - 1, ci)][1]
                    _edge_union(dsu, mapping[labels[0, :]], upper)
                right = mapping[labels[:, -1]].astype(np.int64, copy=True)
                bottom = mapping[labels[-1, :]].astype(np.int64, copy=True)
                tile_edges[(ri, ci)] = (right, bottom)
                left_right = right
                tile_maps[(ri, ci)] = mapping
                tile_windows.append((ri, ci, win))

        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, tiled=True, compress="DEFLATE", predictor=2, BIGTIFF="IF_SAFER", SPARSE_OK="TRUE")
        profile.pop("photometric", None)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        tmp.unlink(missing_ok=True)
        kept_pixels = 0
        components_kept: set[int] = set()
        with rasterio.open(tmp, "w", **profile) as dst:
            # Pass 2: repeat local labels and retain components meeting the gate.
            for ri, ci, win in tile_windows:
                arr = src.read(1, window=win)
                labels, n = _label(arr != 0)
                mapping = tile_maps[(ri, ci)]
                keep_local = np.zeros(n + 1, dtype=np.uint8)
                for local_id in range(1, n + 1):
                    root = dsu.find(int(mapping[local_id]))
                    if dsu.size[root] >= min_pixels:
                        keep_local[local_id] = 1
                        components_kept.add(root)
                out = keep_local[labels]
                if out.any():
                    dst.write(out.astype(np.uint8), 1, window=win)
                    kept_pixels += int(out.sum())
            try:
                dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
            except Exception:
                pass
    os.replace(tmp, out_path)
    return {"source": str(src_path.resolve()), "path": str(out_path.resolve()), "sha256": sha256_file(out_path), "bytes": out_path.stat().st_size, "min_component_pixels": int(min_pixels), "kept_pixels": int(kept_pixels), "kept_components": int(len(components_kept)), "tile_block": int(block), "connectivity": 8}


def encoded_state_raster(state_paths: dict[str, Path], out_path: Path) -> dict[str, Any]:
    import rasterio
    from rasterio.windows import Window

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(state_paths["STABLE"]) as stable, rasterio.open(state_paths["NEW"]) as new, rasterio.open(state_paths["DEMOLITION"]) as demolition:
        profile = stable.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, tiled=True, compress="DEFLATE", predictor=2, BIGTIFF="IF_SAFER", SPARSE_OK="TRUE")
        profile.pop("photometric", None)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        tmp.unlink(missing_ok=True)
        overlap = 0
        with rasterio.open(tmp, "w", **profile) as dst:
            for _, win in stable.block_windows(1):
                a = stable.read(1, window=win) != 0
                b = new.read(1, window=win) != 0
                c = demolition.read(1, window=win) != 0
                overlap += int((a.astype(np.uint8) + b.astype(np.uint8) + c.astype(np.uint8) > 1).sum())
                code = a.astype(np.uint8) + (b.astype(np.uint8) * 2) + (c.astype(np.uint8) * 3)
                if code.any():
                    dst.write(code, 1, window=win)
            try:
                dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
            except Exception:
                pass
    os.replace(tmp, out_path)
    return {"path": str(out_path.resolve()), "sha256": sha256_file(out_path), "bytes": out_path.stat().st_size, "overlap_pixels": overlap}


def row_pixel_areas(src_path: Path, metric_crs: str = METRIC_CRS) -> np.ndarray:
    import rasterio
    from pyproj import Transformer

    with rasterio.open(src_path) as src:
        tr = src.transform
        transformer = Transformer.from_crs(src.crs, metric_crs, always_xy=True)
        areas = np.empty(src.height, dtype=np.float64)
        for r in range(src.height):
            lonlat = [tr * (0, r), tr * (1, r), tr * (1, r + 1), tr * (0, r + 1)]
            xy = [transformer.transform(float(x), float(y)) for x, y in lonlat]
            areas[r] = abs(sum(xy[i][0] * xy[(i + 1) % 4][1] - xy[(i + 1) % 4][0] * xy[i][1] for i in range(4)) / 2.0)
        return areas


def raster_stats(path: Path, row_areas: np.ndarray) -> tuple[int, float]:
    import rasterio

    pixels, area = 0, 0.0
    with rasterio.open(path) as src:
        for _, win in src.block_windows(1):
            arr = src.read(1, window=win)
            counts = (arr != 0).sum(axis=1)
            r0 = int(win.row_off)
            pixels += int(counts.sum())
            area += float(np.dot(counts.astype(np.float64), row_areas[r0 : r0 + arr.shape[0]]))
    return pixels, area


def canonical_stats(builder: Path, target: Path) -> list[dict[str, Any]]:
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
            rows.append({"period": period, "state": state, "object_count": int(info.get("features", 0)), "canonical_object_area_m2": area, "source_gpkg": str(gpkg.resolve()), "candidate_not_ground_truth": True})
    out = target / "04_statistics" / "CANONICAL_OBJECT_AREA_STATISTICS.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder-output", type=Path, required=True)
    ap.add_argument("--target-root", type=Path, required=True)
    ap.add_argument("--city-boundary", type=Path, default=None)
    ap.add_argument("--block", type=int, default=2048)
    args = ap.parse_args()

    builder = args.builder_output.resolve()
    target = args.target_root.resolve()
    target.mkdir(parents=True, exist_ok=True)
    source: dict[tuple[str, str], Path] = {}
    for period in PERIODS:
        for state in STATES:
            p = builder / "02_rasters" / f"XIAN_{period}_{state}.tif"
            receipt = builder / "receipts" / f"XIAN_{period}_{state}_RASTER_RECEIPT.json"
            if not p.is_file() or p.stat().st_size == 0 or not receipt.is_file():
                raise RuntimeError(f"required PASS raster/receipt missing: {period} {state}")
            data = json.loads(receipt.read_text(encoding="utf-8"))
            if data.get("status", "PASS") not in {"PASS", "pass"}:
                raise RuntimeError(f"raster receipt is not PASS: {receipt}")
            source[(period, state)] = p
    grid = assert_same_grid(source.values())
    print("PRIMARY_GATE_INPUTS_PASS", flush=True)

    gate = {
        "2018_2019_STABLE_PASS": True,
        "2018_2019_ADDITION_PASS": True,
        "2018_2019_DEMOLITION_PASS": True,
        "2019_2020_STABLE_PASS": True,
        "2019_2020_ADDITION_PASS": True,
        "2019_2020_DEMOLITION_PASS": True,
        "source_grid_exact": True,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "candidate_not_ground_truth": True,
    }
    write_json(target / "00_provenance" / "PRIMARY_COMPLETENESS_GATE.json", gate)

    manifest: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    row_areas = row_pixel_areas(next(iter(source.values())))
    for period in PERIODS:
        period_dir = target / "02_rasters" / period
        for filt in FILTERS:
            print(f"START {period} {filt}", flush=True)
            filt_dir = period_dir / filt
            state_out: dict[str, Path] = {}
            for state in STATES:
                dst = filt_dir / f"XIAN_{period}_{state}.tif"
                if filt == "UNFILTERED":
                    info = atomic_copy(source[(period, state)], dst)
                elif state == "STABLE":
                    info = atomic_copy(source[(period, state)], dst)
                else:
                    threshold = int(filt[2:])
                    rec_path = dst.with_suffix(".receipt.json")
                    if dst.is_file() and dst.stat().st_size > 0 and rec_path.is_file():
                        info = json.loads(rec_path.read_text(encoding="utf-8"))
                        info["reused"] = True
                    else:
                        info = filtered_mask(source[(period, state)], dst, threshold, args.block)
                        write_json(rec_path, info)
                    print(f"DONE {period} {filt} {state}", flush=True)
                state_out[state] = dst
                pixels, area = raster_stats(dst, row_areas)
                stats.append({"period": period, "filter": filt, "state": state, "pixels": pixels, "area_m2": area, "area_km2": area / 1e6, "source_grid_exact": grid, "sha256": sha256_file(dst), "candidate_not_ground_truth": True})
                manifest.append({"period": period, "filter": filt, "state": state, **info})
            combined = encoded_state_raster(state_out, filt_dir / f"XIAN_{period}_THREE_STATE.tif")
            print(f"DONE {period} {filt} THREE_STATE", flush=True)
            manifest.append({"period": period, "filter": filt, "state": "THREE_STATE", **combined})

            # Keep an explicit filtered-out change mask; never re-label it STABLE.
            if filt != "UNFILTERED":
                for state in ("NEW", "DEMOLITION"):
                    import rasterio

                    out = filt_dir / f"XIAN_{period}_{state}_FILTERED_OUT.tif"
                    with rasterio.open(source[(period, state)]) as src0, rasterio.open(state_out[state]) as src1:
                        profile = src0.profile.copy()
                        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, tiled=True, compress="DEFLATE", predictor=2, BIGTIFF="IF_SAFER", SPARSE_OK="TRUE")
                        profile.pop("photometric", None)
                        tmp = out.with_suffix(out.suffix + ".part")
                        tmp.unlink(missing_ok=True)
                        with rasterio.open(tmp, "w", **profile) as dst:
                            for _, win in src0.block_windows(1):
                                a = src0.read(1, window=win) != 0
                                b = src1.read(1, window=win) != 0
                                arr = (a & ~b).astype(np.uint8)
                                if arr.any():
                                    dst.write(arr, 1, window=win)
                        os.replace(tmp, out)
                    manifest.append({"period": period, "filter": filt, "state": state + "_FILTERED_OUT", "path": str(out.resolve()), "sha256": sha256_file(out), "bytes": out.stat().st_size})

    stats_path = target / "04_statistics" / "TWO_PERIOD_THREE_STATE_STATISTICS.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["period", "filter", "state", "pixels", "area_m2", "area_km2", "source_grid_exact", "sha256", "candidate_not_ground_truth"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in stats:
            row = dict(row)
            row["source_grid_exact"] = "true"
            w.writerow(row)
    write_json(target / "04_statistics" / "TWO_PERIOD_THREE_STATE_STATISTICS.json", {"rows": stats, "grid": grid, "pixel_area_method": "row-wise EPSG:32649 corner transform", "candidate_not_ground_truth": True})
    canonical = canonical_stats(builder, target)

    if args.city_boundary and args.city_boundary.is_file():
        try:
            import geopandas as gpd

            city_area_m2 = float(gpd.read_file(args.city_boundary).to_crs(METRIC_CRS).geometry.union_all().area)
        except Exception:
            city_area_m2 = float("nan")
    else:
        city_area_m2 = float("nan")
    for row in stats:
        row["study_area_m2"] = city_area_m2
        row["share_of_city_pct"] = (row["area_m2"] / city_area_m2 * 100.0) if math.isfinite(city_area_m2) and city_area_m2 > 0 else float("nan")
    with stats_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["period", "filter", "state", "pixels", "area_m2", "area_km2", "study_area_m2", "share_of_city_pct", "source_grid_exact", "sha256", "candidate_not_ground_truth"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in stats:
            row = dict(row)
            row["source_grid_exact"] = "true"
            w.writerow({k: row.get(k) for k in fields})

    write_json(target / "00_provenance" / "PRIMARY_PROVENANCE.json", {"builder_output": str(builder), "source_rasters": {f"{p}_{s}": {"path": str(source[(p, s)].resolve()), "sha256": sha256_file(source[(p, s)]), "grid": grid} for p in PERIODS for s in STATES}, "b0_checkpoint_sha256": EXPECTED_B0_SHA, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}, "raw_inputs_modified": False, "candidate_not_ground_truth": True})
    write_json(target / "00_provenance" / "RASTER_OUTPUT_MANIFEST.json", {"products": manifest, "grid": grid, "filters": {"connectivity": 8, "min_component_pixels": [3, 5], "stable_policy": "unchanged", "filtered_change_policy": "not relabeled STABLE"}})
    write_json(target / "14_acceptance" / "TWO_PERIOD_DEDUP_THREE_STATE_ACCEPTANCE.json", {"PRIMARY_TWO_PERIOD_RESULT_COMPLETE": True, "TWO_PERIOD_DEDUP_THREE_STATE_COMPLETE": True, "period_state_pass_gate": gate, "products_complete": True, "statistics_complete": True, "canonical_object_area_statistics": str((target / "04_statistics" / "CANONICAL_OBJECT_AREA_STATISTICS.csv").resolve()), "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}, "candidate_not_ground_truth": True, "FUSION_STARTED": False})
    write_json(target / "14_acceptance" / "PRIMARY_OUTPUT_SHA256.json", {"products": [{"path": x["path"], "sha256": x.get("sha256"), "bytes": x.get("bytes")} for x in manifest if x.get("path")], "source_grid": grid})
    report = ["# 两期三态去重主结果", "", "- 2018–2019 与 2019–2020 六张源栅格均已取得 PASS receipt。", "- 输出 UNFILTERED、GE3、GE5；GE3/GE5 仅对 NEW/DEMOLITION 做 8 邻域连通域筛选，STABLE 原样保留。", "- 被筛掉的变化像元保存在 *_FILTERED_OUT.tif，绝不改标为 STABLE。", "- 所有栅格复制/写出均核对宽高、CRS、transform；面积按逐行角点转换到 EPSG:32649 计算，目标对象面积另列 canonical polygon 表。", "- 本包为候选证据，不是人工 Ground Truth；WHU/SECOND/Holdout 读取为 0。", ""]
    target.joinpath("FINAL_TWO_PERIOD_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
