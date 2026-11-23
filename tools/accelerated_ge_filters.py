"""Resume-safe one-pass GE3/GE5 filtering for the Xi'an primary package.

The older primary builder intentionally keeps a very simple implementation,
but it scans each change raster once for GE3 and once again for GE5.  This
post-process is an additive, lossless accelerator: one tiled 8-connected
component analysis writes both retained masks and both filtered-out masks.
It never changes the source rasters, the state definitions, or the exact
source grid, and it skips products that already have a valid artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:  # direct script execution (tools/ is on sys.path)
    from build_primary_two_period_results import (
        DSU,
        PERIODS,
        STATES,
        assert_same_grid,
        atomic_copy,
        encoded_state_raster,
        grid_meta,
        row_pixel_areas,
        raster_stats,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:  # package-style imports in tests
    from tools.build_primary_two_period_results import (
        DSU,
        PERIODS,
        STATES,
        assert_same_grid,
        atomic_copy,
        encoded_state_raster,
        grid_meta,
        row_pixel_areas,
        raster_stats,
        sha256_file,
        write_json,
    )


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    return ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))


def _edge_union(dsu: DSU, current: np.ndarray, previous: np.ndarray) -> None:
    """Union 8-neighbour labels across a tile boundary."""
    if current.size == 0 or previous.size == 0:
        return
    for i, a in enumerate(current.tolist()):
        if not a:
            continue
        for b in previous[max(0, i - 1) : min(previous.size, i + 2)].tolist():
            if b:
                dsu.union(int(a), int(b))


def _profile(src: Any) -> dict[str, Any]:
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        tiled=True,
        compress="DEFLATE",
        predictor=2,
        BIGTIFF="IF_SAFER",
        SPARSE_OK="TRUE",
    )
    profile.pop("photometric", None)
    return profile


def _component_ledger(
    dsu: DSU,
    *,
    component_sizes: Counter[int],
    ge3_components: int,
    ge3_pixels: int,
    ge5_components: int,
    ge5_pixels: int,
    source: Path,
    block: int,
) -> dict[str, Any]:
    ordered = sorted((int(k), int(v)) for k, v in component_sizes.items())
    total_components = int(sum(v for _, v in ordered))
    total_pixels = int(sum(k * v for k, v in ordered))

    def q(p: float) -> float:
        if not ordered or total_components <= 0:
            return 0.0
        # Weighted nearest-rank quantile: keeps memory bounded even if the
        # change mask contains millions of one-pixel components.
        rank = max(1, int(math.ceil(float(p) * total_components)))
        seen = 0
        for size, count in ordered:
            seen += count
            if seen >= rank:
                return float(size)
        return float(ordered[-1][0])

    by_size = {str(k): int(v) for k, v in sorted(component_sizes.items())}
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "connectivity": 8,
        "tile_block": int(block),
        "total_components": int(total_components),
        "total_nonzero_pixels": total_pixels,
        "component_size_min": int(ordered[0][0]) if ordered else 0,
        "component_size_median": q(0.5),
        "component_size_p95": q(0.95),
        "component_size_p99": q(0.99),
        "component_size_max": int(ordered[-1][0]) if ordered else 0,
        "components_ge3": int(ge3_components),
        "pixels_ge3": int(ge3_pixels),
        "components_ge5": int(ge5_components),
        "pixels_ge5": int(ge5_pixels),
        "pixels_filtered_out_ge3": int(total_pixels - ge3_pixels),
        "pixels_filtered_out_ge5": int(total_pixels - ge5_pixels),
        "retention_rate_ge3": (float(ge3_pixels) / total_pixels) if total_pixels else 0.0,
        "retention_rate_ge5": (float(ge5_pixels) / total_pixels) if total_pixels else 0.0,
        "component_size_histogram": by_size,
    }


def _write_component_ledger(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "source_sha256",
        "connectivity",
        "tile_block",
        "total_components",
        "total_nonzero_pixels",
        "component_size_min",
        "component_size_median",
        "component_size_p95",
        "component_size_p99",
        "component_size_max",
        "components_ge3",
        "pixels_ge3",
        "components_ge5",
        "pixels_ge5",
        "pixels_filtered_out_ge3",
        "pixels_filtered_out_ge5",
        "retention_rate_ge3",
        "retention_rate_ge5",
    ]
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: data.get(k) for k in fields})
    os.replace(tmp, path)


def _valid_product(path: Path, receipt: Path | None = None) -> bool:
    """A completed product is non-empty, readable, and has no active .part."""
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if path.with_suffix(path.suffix + ".part").exists():
        return False
    try:
        import rasterio

        with rasterio.open(path) as src:
            if src.width <= 0 or src.height <= 0 or src.count != 1:
                return False
            _ = src.transform
    except Exception:
        return False
    if receipt is not None and receipt.exists():
        try:
            data = json.loads(receipt.read_text(encoding="utf-8"))
            if str(data.get("status", "PASS")).lower() not in {"pass", "ok", "complete"}:
                return False
        except Exception:
            return False
    return True


def _write_raster_windows(dst: Any, arr: np.ndarray, win: Any) -> None:
    if arr.any():
        dst.write(arr, 1, window=win)


def filtered_masks_dual(
    src_path: Path,
    ge3_path: Path,
    ge5_path: Path,
    filtered_out_ge3_path: Path,
    filtered_out_ge5_path: Path,
    *,
    block: int = 2048,
    preserve_existing: set[Path] | None = None,
) -> dict[str, Any]:
    """One DSU component scan, writing GE3/GE5 and both removed masks.

    Four ``.part`` files are opened together and atomically renamed only after
    every writer is closed.  Thus a crash cannot leave a product that looks
    complete, and a resume can safely process only the missing source state.
    """
    import rasterio
    from rasterio.windows import Window

    for p in (ge3_path, ge5_path, filtered_out_ge3_path, filtered_out_ge5_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    dsu = DSU()
    tile_maps: dict[tuple[int, int], np.ndarray] = {}
    tile_edges: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    tile_windows: list[tuple[int, int, Window]] = []
    component_sizes: Counter[int] = Counter()

    with rasterio.open(src_path) as src:
        rows = list(range(0, src.height, block))
        cols = list(range(0, src.width, block))
        for ri, row0 in enumerate(rows):
            left_right: np.ndarray | None = None
            for ci, col0 in enumerate(cols):
                h, w = min(block, src.height - row0), min(block, src.width - col0)
                win = Window(col0, row0, w, h)
                arr = src.read(1, window=win)
                labels, n = _label(arr != 0)
                mapping = np.zeros(n + 1, dtype=np.int64)
                counts = np.bincount(labels.ravel(), minlength=n + 1)
                for local_id in range(1, n + 1):
                    mapping[local_id] = dsu.add(int(counts[local_id]))
                if left_right is not None:
                    _edge_union(dsu, mapping[labels[:, 0]], left_right)
                if ri > 0:
                    _edge_union(dsu, mapping[labels[0, :]], tile_edges[(ri - 1, ci)][1])
                right = mapping[labels[:, -1]].astype(np.int64, copy=True)
                bottom = mapping[labels[-1, :]].astype(np.int64, copy=True)
                tile_edges[(ri, ci)] = (right, bottom)
                tile_maps[(ri, ci)] = mapping
                tile_windows.append((ri, ci, win))
                left_right = right

        # Collapse local DSU roots once; all later writes use these immutable
        # component sizes, so no second CC scan is needed for GE5.
        for i in range(len(dsu.parent)):
            root = dsu.find(i)
            if dsu.size[i] == dsu.size[root]:
                # A root is counted exactly once.  Non-roots have a stale size
                # equal to their own local count, hence the parent check.
                if root == i:
                    component_sizes[int(dsu.size[root])] += 1

        ge3_components = sum(v for k, v in component_sizes.items() if k >= 3)
        ge5_components = sum(v for k, v in component_sizes.items() if k >= 5)
        total_pixels = int(sum(k * v for k, v in component_sizes.items()))
        ge3_pixels = sum(k * v for k, v in component_sizes.items() if k >= 3)
        ge5_pixels = sum(k * v for k, v in component_sizes.items() if k >= 5)
        profile = _profile(src)
        paths = (ge3_path, ge5_path, filtered_out_ge3_path, filtered_out_ge5_path)
        tmps = tuple(p.with_suffix(p.suffix + ".part") for p in paths)
        for t in tmps:
            t.unlink(missing_ok=True)
        try:
            with rasterio.open(tmps[0], "w", **profile) as out3, rasterio.open(tmps[1], "w", **profile) as out5, rasterio.open(tmps[2], "w", **profile) as removed3, rasterio.open(tmps[3], "w", **profile) as removed5:
                for ri, ci, win in tile_windows:
                    arr = src.read(1, window=win)
                    labels, n = _label(arr != 0)
                    mapping = tile_maps[(ri, ci)]
                    roots = np.zeros(n + 1, dtype=np.int64)
                    sizes = np.zeros(n + 1, dtype=np.int64)
                    keep3 = np.zeros(n + 1, dtype=np.uint8)
                    keep5 = np.zeros(n + 1, dtype=np.uint8)
                    for local_id in range(1, n + 1):
                        root = dsu.find(int(mapping[local_id]))
                        roots[local_id] = root
                        sizes[local_id] = dsu.size[root]
                        keep3[local_id] = 1 if sizes[local_id] >= 3 else 0
                        keep5[local_id] = 1 if sizes[local_id] >= 5 else 0
                    mask3 = keep3[labels]
                    mask5 = keep5[labels]
                    _write_raster_windows(out3, mask3.astype(np.uint8), win)
                    _write_raster_windows(out5, mask5.astype(np.uint8), win)
                    _write_raster_windows(removed3, ((arr != 0) & (mask3 == 0)).astype(np.uint8), win)
                    _write_raster_windows(removed5, ((arr != 0) & (mask5 == 0)).astype(np.uint8), win)
                for dst in (out3, out5, removed3, removed5):
                    try:
                        dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)
                        dst.update_tags(ns="rio_overview", resampling="nearest")
                    except Exception:
                        pass
        except Exception:
            for t in tmps:
                t.unlink(missing_ok=True)
            raise
    preserve_existing = preserve_existing or set()
    for t, p in zip(tmps, paths):
        # A previously accepted product is never replaced merely because a
        # second threshold is being materialized.  The fresh temporary output
        # is discarded after the single dual pass; its values are still used
        # for the new threshold products and are covered by the receipt.
        if p in preserve_existing and _valid_product(p):
            t.unlink(missing_ok=True)
        else:
            os.replace(t, p)
    ledger = _component_ledger(
        dsu,
        component_sizes=component_sizes,
        ge3_components=int(ge3_components),
        ge3_pixels=int(ge3_pixels),
        ge5_components=int(ge5_components),
        ge5_pixels=int(ge5_pixels),
        source=src_path,
        block=block,
    )
    return {
        "source": str(src_path.resolve()),
        "source_sha256": ledger["source_sha256"],
        "ge3": {"path": str(ge3_path.resolve()), "sha256": sha256_file(ge3_path), "bytes": ge3_path.stat().st_size},
        "ge5": {"path": str(ge5_path.resolve()), "sha256": sha256_file(ge5_path), "bytes": ge5_path.stat().st_size},
        "filtered_out_ge3": {"path": str(filtered_out_ge3_path.resolve()), "sha256": sha256_file(filtered_out_ge3_path), "bytes": filtered_out_ge3_path.stat().st_size},
        "filtered_out_ge5": {"path": str(filtered_out_ge5_path.resolve()), "sha256": sha256_file(filtered_out_ge5_path), "bytes": filtered_out_ge5_path.stat().st_size},
        "component_ledger": ledger,
    }


def derive_filtered_out(
    src_path: Path,
    retained_paths: dict[int, Path],
    output_paths: dict[int, Path],
    *,
    block: int = 2048,
) -> dict[str, Any]:
    """Derive missing filtered-out masks without relabeling or CC rescanning.

    This is used only when an earlier completed NEW pass already produced a
    retained GE3/GE5 mask but stopped before writing its explicit removed-mask
    companion.  It reads source and retained masks once per window and never
    performs connected-component labeling.
    """
    import rasterio
    from rasterio.windows import Window

    missing = {k: p for k, p in output_paths.items() if not _valid_product(p)}
    if not missing:
        return {"reused": True, "outputs": {str(k): {"path": str(p.resolve()), "sha256": sha256_file(p), "bytes": p.stat().st_size} for k, p in output_paths.items()}}
    with rasterio.open(src_path) as src:
        handles: list[tuple[int, Path, Path, Any]] = []
        retained_handles: dict[int, Any] = {}
        try:
            for threshold in missing:
                retained_handles[threshold] = rasterio.open(retained_paths[threshold])
            for threshold, out in missing.items():
                tmp = out.with_suffix(out.suffix + ".part")
                tmp.unlink(missing_ok=True)
                handles.append((threshold, out, tmp, rasterio.open(tmp, "w", **_profile(src))))
            for row0 in range(0, src.height, block):
                for col0 in range(0, src.width, block):
                    h, w = min(block, src.height - row0), min(block, src.width - col0)
                    win = Window(col0, row0, w, h)
                    src_arr = src.read(1, window=win) != 0
                    for threshold, _, _, dst in handles:
                        retained = retained_handles[threshold].read(1, window=win) != 0
                        removed = (src_arr & ~retained).astype(np.uint8)
                        _write_raster_windows(dst, removed, win)
            for _, _, _, dst in handles:
                try:
                    dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)
                    dst.update_tags(ns="rio_overview", resampling="nearest")
                except Exception:
                    pass
        except Exception:
            for _, _, tmp, dst in handles:
                dst.close()
                tmp.unlink(missing_ok=True)
            raise
        finally:
            for _, _, _, dst in handles:
                if not dst.closed:
                    dst.close()
            for keep in retained_handles.values():
                if not keep.closed:
                    keep.close()
    for _, out, tmp, _ in handles:
        os.replace(tmp, out)
    return {"reused": False, "outputs": {str(k): {"path": str(p.resolve()), "sha256": sha256_file(p), "bytes": p.stat().st_size} for k, p in output_paths.items()}}


def encoded_state_raster_fast(state_paths: dict[str, Path], out_path: Path, *, block: int = 2048) -> dict[str, Any]:
    """Encode one threshold's three state masks with large windows."""
    import rasterio
    from rasterio.windows import Window

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(state_paths["STABLE"]) as stable, rasterio.open(state_paths["NEW"]) as new, rasterio.open(state_paths["DEMOLITION"]) as demolition:
        profile = _profile(stable)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        tmp.unlink(missing_ok=True)
        overlap = 0
        with rasterio.open(tmp, "w", **profile) as dst:
            for row0 in range(0, stable.height, block):
                for col0 in range(0, stable.width, block):
                    h, w = min(block, stable.height - row0), min(block, stable.width - col0)
                    win = Window(col0, row0, w, h)
                    a = stable.read(1, window=win) != 0
                    b = new.read(1, window=win) != 0
                    c = demolition.read(1, window=win) != 0
                    overlap += int((a.astype(np.uint8) + b.astype(np.uint8) + c.astype(np.uint8) > 1).sum())
                    code = a.astype(np.uint8) + 2 * b.astype(np.uint8) + 3 * c.astype(np.uint8)
                    _write_raster_windows(dst, code, win)
            try:
                dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
            except Exception:
                pass
    os.replace(tmp, out_path)
    return {"path": str(out_path.resolve()), "sha256": sha256_file(out_path), "bytes": out_path.stat().st_size, "overlap_pixels": overlap, "window_block": int(block)}


def _receipt(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("status", "PASS")
    payload.setdefault("accelerator", "accelerated_ge_filters")
    write_json(path, payload)


def _progress(target: Path, payload: dict[str, Any]) -> None:
    write_json(target / "00_provenance" / "ACCELERATED_PROGRESS.json", payload)


def process_period(
    builder: Path,
    target: Path,
    period: str,
    *,
    block: int,
    row_areas: np.ndarray,
    grid: dict[str, Any],
    source_override: Path | None = None,
) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    source = {}
    for state in STATES:
        canonical = builder / "02_rasters" / f"XIAN_{period}_{state}.tif"
        override = (source_override / f"XIAN_{period}_{state}.tif") if source_override else None
        source[state] = override if override is not None and override.is_file() else canonical
    for state, path in source.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing source raster: {path}")
    period_dir = target / "02_rasters" / period
    # STABLE is unchanged for both filters; copy only when the target product
    # is not already present and readable.
    stable_outputs: dict[str, Path] = {}
    for filt in ("GE3", "GE5"):
        out = period_dir / filt / f"XIAN_{period}_STABLE.tif"
        if not _valid_product(out):
            products.append({"period": period, "filter": filt, "state": "STABLE", **atomic_copy(source["STABLE"], out)})
        else:
            assert_same_grid((source["STABLE"], out))
            products.append({"period": period, "filter": filt, "state": "STABLE", "path": str(out.resolve()), "sha256": sha256_file(out), "bytes": out.stat().st_size, "reused": True})
        stable_outputs[filt] = out

    for state in ("NEW", "DEMOLITION"):
        _progress(target, {"status": "RUNNING", "period": period, "state": state, "stage": "DUAL_COMPONENT_SCAN", "block": block, "source_grid": grid})
        ge3 = period_dir / "GE3" / f"XIAN_{period}_{state}.tif"
        ge5 = period_dir / "GE5" / f"XIAN_{period}_{state}.tif"
        removed3 = period_dir / "GE3" / f"XIAN_{period}_{state}_FILTERED_OUT.tif"
        removed5 = period_dir / "GE5" / f"XIAN_{period}_{state}_FILTERED_OUT.tif"
        rec3 = ge3.with_suffix(".receipt.json")
        rec5 = ge5.with_suffix(".receipt.json")
        retained_ready = _valid_product(ge3) and _valid_product(ge5)
        if retained_ready:
            # A worker may have completed a retained mask before it was
            # interrupted while writing the companion filtered-out mask.  In
            # that case derive only the missing complement; never redo the
            # connected-component analysis for this already completed NEW or
            # DEMOLITION state.
            missing_removed = {k: p for k, p in ((3, removed3), (5, removed5)) if not _valid_product(p)}
            if missing_removed:
                assert_same_grid((source[state], ge3, ge5))
                derived = derive_filtered_out(source[state], {3: ge3, 5: ge5}, missing_removed, block=block)
                for threshold, item in derived["outputs"].items():
                    products.append({"period": period, "state": state, "filter": f"GE{threshold}_FILTERED_OUT", **item, "derived_without_cc_rescan": True})
                if not rec3.is_file():
                    _receipt(rec3, {"period": period, "state": state, "filter": "GE3", "path": str(ge3.resolve()), "sha256": sha256_file(ge3), "bytes": ge3.stat().st_size, "reused_existing": True, "component_scan_not_repeated": True})
                if not rec5.is_file():
                    _receipt(rec5, {"period": period, "state": state, "filter": "GE5", "path": str(ge5.resolve()), "sha256": sha256_file(ge5), "bytes": ge5.stat().st_size, "reused_existing": True, "component_scan_not_repeated": True})
                _receipt(target / "00_provenance" / f"ACCELERATED_{period}_{state}.receipt.json", {"period": period, "state": state, "status": "PASS", "mode": "filtered_out_derivation_only", "component_scan_not_repeated": True, "source": str(source[state].resolve()), "outputs": {"GE3": str(ge3.resolve()), "GE5": str(ge5.resolve()), "FILTERED_OUT_GE3": str(removed3.resolve()), "FILTERED_OUT_GE5": str(removed5.resolve())}})
                _progress(target, {"status": "PASS", "period": period, "state": state, "stage": "STATE_COMPLETE", "mode": "filtered_out_derivation_only"})
                continue
            if not rec3.is_file():
                _receipt(rec3, {"period": period, "state": state, "filter": "GE3", "path": str(ge3.resolve()), "sha256": sha256_file(ge3), "bytes": ge3.stat().st_size, "reused_existing": True, "component_scan_not_repeated": True})
            if not rec5.is_file():
                _receipt(rec5, {"period": period, "state": state, "filter": "GE5", "path": str(ge5.resolve()), "sha256": sha256_file(ge5), "bytes": ge5.stat().st_size, "reused_existing": True, "component_scan_not_repeated": True})
            _receipt(target / "00_provenance" / f"ACCELERATED_{period}_{state}.receipt.json", {"period": period, "state": state, "status": "PASS", "mode": "reused_existing", "component_scan_not_repeated": True, "source": str(source[state].resolve()), "outputs": {"GE3": str(ge3.resolve()), "GE5": str(ge5.resolve()), "FILTERED_OUT_GE3": str(removed3.resolve()), "FILTERED_OUT_GE5": str(removed5.resolve())}})
            _progress(target, {"status": "PASS", "period": period, "state": state, "stage": "STATE_COMPLETE", "mode": "reused_existing"})
            continue
        # If all four outputs are already valid, do not even reopen the source.
        if all(_valid_product(p) for p in (ge3, ge5, removed3, removed5)) and rec3.is_file() and rec5.is_file():
            assert_same_grid((source[state], ge3, ge5, removed3, removed5))
            payload = json.loads(rec5.read_text(encoding="utf-8"))
            products.append({"period": period, "state": state, "reused": True, "component_ledger": payload.get("component_ledger")})
            continue
        preserve = {p for p in (ge3, removed3, ge5, removed5) if _valid_product(p)}
        info = filtered_masks_dual(source[state], ge3, ge5, removed3, removed5, block=block, preserve_existing=preserve)
        ledger = info["component_ledger"]
        ledger_base = target / "04_statistics" / f"COMPONENT_LEDGER_{period}_{state}"
        _write_component_ledger(ledger_base.with_suffix(".csv"), ledger)
        write_json(ledger_base.with_suffix(".json"), ledger)
        # Never overwrite an existing accepted receipt (it may be part of the
        # historical audit trail).  A receipt is written only for a newly
        # materialized product or where the old worker left no receipt.
        if not rec3.is_file():
            _receipt(rec3, {"period": period, "state": state, "filter": "GE3", "path": info["ge3"]["path"], "sha256": info["ge3"]["sha256"], "bytes": info["ge3"]["bytes"], "component_ledger": ledger})
        if not rec5.is_file():
            _receipt(rec5, {"period": period, "state": state, "filter": "GE5", "path": info["ge5"]["path"], "sha256": info["ge5"]["sha256"], "bytes": info["ge5"]["bytes"], "component_ledger": ledger})
        _receipt(target / "00_provenance" / f"ACCELERATED_{period}_{state}.receipt.json", {"period": period, "state": state, "status": "PASS", "mode": "one_cc_scan_dual_threshold", "component_ledger": ledger, "source": str(source[state].resolve()), "outputs": {"GE3": info["ge3"], "GE5": info["ge5"], "FILTERED_OUT_GE3": info["filtered_out_ge3"], "FILTERED_OUT_GE5": info["filtered_out_ge5"]}})
        _progress(target, {"status": "PASS", "period": period, "state": state, "stage": "STATE_COMPLETE", "mode": "one_cc_scan_dual_threshold", "component_ledger": ledger})
        products.extend([
            {"period": period, "state": state, "filter": "GE3", **info["ge3"]},
            {"period": period, "state": state, "filter": "GE5", **info["ge5"]},
            {"period": period, "state": state, "filter": "GE3_FILTERED_OUT", **info["filtered_out_ge3"]},
            {"period": period, "state": state, "filter": "GE5_FILTERED_OUT", **info["filtered_out_ge5"]},
        ])

    # Combined state rasters are built only after all three state masks exist,
    # once per threshold.  This avoids the old per-filter/per-state repeats.
    for filt in ("GE3", "GE5"):
        state_paths = {
            "STABLE": period_dir / filt / f"XIAN_{period}_STABLE.tif",
            "NEW": period_dir / filt / f"XIAN_{period}_NEW.tif",
            "DEMOLITION": period_dir / filt / f"XIAN_{period}_DEMOLITION.tif",
        }
        combined = period_dir / filt / f"XIAN_{period}_THREE_STATE.tif"
        if not _valid_product(combined):
            products.append({"period": period, "filter": filt, "state": "THREE_STATE", **encoded_state_raster_fast(state_paths, combined, block=block)})
        else:
            assert_same_grid((state_paths["STABLE"], state_paths["NEW"], state_paths["DEMOLITION"], combined))
            products.append({"period": period, "filter": filt, "state": "THREE_STATE", "path": str(combined.resolve()), "sha256": sha256_file(combined), "bytes": combined.stat().st_size, "reused": True})
    _progress(target, {"status": "PASS", "period": period, "stage": "PERIOD_COMPLETE", "source_grid": grid})
    return products


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--builder-output", type=Path, required=True)
    ap.add_argument("--target-root", type=Path, required=True)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--period", choices=PERIODS, action="append", dest="periods")
    ap.add_argument("--fast-source-root", type=Path, default=None, help="optional read-only SSD copies named XIAN_<period>_<state>.tif")
    args = ap.parse_args()
    builder = args.builder_output.resolve()
    target = args.target_root.resolve()
    periods = tuple(args.periods or PERIODS)
    source_paths = [builder / "02_rasters" / f"XIAN_{p}_{s}.tif" for p in periods for s in STATES]
    if any(not p.is_file() or p.stat().st_size <= 0 for p in source_paths):
        missing = [str(p) for p in source_paths if not p.is_file() or p.stat().st_size <= 0]
        raise RuntimeError("missing source rasters: " + ", ".join(missing))
    grid = assert_same_grid(source_paths)
    if args.fast_source_root:
        fast_root = args.fast_source_root.resolve()
        for p in source_paths:
            q = fast_root / p.name
            if q.is_file():
                assert_same_grid((p, q))
    row_areas = row_pixel_areas(source_paths[0])
    target.mkdir(parents=True, exist_ok=True)
    all_products: list[dict[str, Any]] = []
    for period in periods:
        print(f"ACCELERATED_START {period}", flush=True)
        all_products.extend(process_period(builder, target, period, block=args.block, row_areas=row_areas, grid=grid, source_override=args.fast_source_root.resolve() if args.fast_source_root else None))
        print(f"ACCELERATED_DONE {period}", flush=True)

    # Additive audit artifacts make the handoff explicit without overwriting
    # the historical primary manifest while the old worker is still running.
    audit = {
        "status": "PASS",
        "periods": list(periods),
        "products": all_products,
        "source_grid": grid,
        "connectivity": 8,
        "thresholds": {"GE3": 3, "GE5": 5},
        "stable_policy": "unchanged",
        "filtered_change_policy": "not_relabeled_STABLE",
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "source_rasters_modified": False,
    }
    write_json(target / "00_provenance" / "ACCELERATED_GE_FILTERS_RECEIPT.json", audit)
    # Stats are computed from the retained products only after the component
    # pass.  Existing primary stats remain untouched; this is a resumable
    # accelerator-side table that can be merged after the worker exits.
    rows: list[dict[str, Any]] = []
    for period in periods:
        for filt in ("GE3", "GE5"):
            for state in STATES:
                p = target / "02_rasters" / period / filt / f"XIAN_{period}_{state}.tif"
                if not _valid_product(p):
                    continue
                pixels, area = raster_stats(p, row_areas)
                rows.append({"period": period, "filter": filt, "state": state, "pixels": pixels, "area_m2": area, "area_km2": area / 1e6, "sha256": sha256_file(p), "source_grid_exact": True})
    stat_path = target / "04_statistics" / "ACCELERATED_GE_FILTER_STATS.csv"
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    with stat_path.open("w", newline="", encoding="utf-8-sig") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write_json(target / "04_statistics" / "ACCELERATED_GE_FILTER_STATS.json", {"rows": rows, "source_grid": grid, "component_ledger_policy": "component ledgers are authoritative for retained/filtered pixel counts"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
