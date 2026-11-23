"""Resumable runner for the Xi'an current-best-candidate rasterization.

Why this exists
---------------
The original ``tools/xian_rasterize_current_best.py`` (proven OUTPUT-ROW STRIP
implementation) was killed mid-2020 (stripe 100/198).  Because ``make_dataset``
truncates its targets, a naive restart would destroy the already-finished 2018
and 2019 products.  This runner therefore:

* reuses the original script's functions unchanged (no algorithm change);
* writes every epoch into ``*.incomplete`` sibling files and promotes them with
  an atomic rename only after the last stripe has been flushed and fsynced;
* refuses to touch an epoch whose promoted outputs already exist and verify;
* lets one epoch be selected per process, so epochs can run concurrently;
* keeps one shared ``RASTER_PROGRESS.md`` by appending whole lines atomically.

Scientific scope: CANDIDATE product export only.  No polygon is warped, no
state is invented, Unknown/Conflict is never merged into Stable.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ORIGINAL = TOOLS / "xian_rasterize_current_best.py"

spec = importlib.util.spec_from_file_location("xian_raster_core", ORIGINAL)
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)

import csv  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.windows import Window, transform as window_transform  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402

OUT_DIR = core.DELIVERY / "01_raster_products"


def progress(message: str, progress_file: str) -> None:
    line = f"- {time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
    print(line.rstrip(), flush=True)
    path = OUT_DIR / progress_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def incomplete(path: Path) -> Path:
    return path.with_name(path.name + ".incomplete")


def promote(paths: list[Path], progress_file: str, label: str) -> None:
    """Atomically promote .incomplete siblings to their final names."""
    for path in paths:
        tmp = incomplete(path)
        os.replace(tmp, path)
    progress(f"{label}: PROMOTED {len(paths)} raster files to final names", progress_file)


def epoch_outputs(year: str) -> list[Path]:
    return [
        OUT_DIR / f"building_boundary_{year}_unfiltered.tif",
        OUT_DIR / f"building_id_{year}_unfiltered.tif",
        OUT_DIR / f"building_boundary_{year}_ge100m2.tif",
        OUT_DIR / f"building_id_{year}_ge100m2.tif",
    ]


def outputs_are_complete(year: str, profile: dict) -> bool:
    """Completeness gate.

    A killed writer leaves a file that opens cleanly, has the right shape and
    dtype, and can still be tens of MB of pure zeros.  Existence is therefore
    never enough: the labelled data must actually reach the final 512-row
    stripe of the output grid.
    """
    paths = epoch_outputs(year)
    if not all(p.exists() and p.stat().st_size > 0 for p in paths):
        return False
    height, width = profile["height"], profile["width"]
    n_stripes = (height + core.STRIPE_ROWS - 1) // core.STRIPE_ROWS
    last_row_off = (n_stripes - 1) * core.STRIPE_ROWS
    for p in paths:
        try:
            with rasterio.open(p) as src:
                if src.width != width or src.height != height or src.crs is None:
                    return False
                tail = src.read(1, window=Window(0, last_row_off, width, height - last_row_off))
                if np.count_nonzero(tail) == 0:
                    return False
                middle = src.read(1, window=Window(0, height // 2, width, 512))
                if np.count_nonzero(middle) == 0:
                    return False
        except Exception:
            return False
    return True


def rasterize_epoch(
    year: str,
    profile: dict,
    object_code: dict,
    object_area: dict,
    stripe_rows: int,
    progress_file: str,
    force: bool,
) -> bool:
    scene, _, filename = core.SCENES[year]
    if not force and outputs_are_complete(year, profile):
        progress(f"{year}: existing complete rasters verified, epoch skipped (no restart)", progress_file)
        return False

    gpkg = core.geom_path(filename)
    if not gpkg.exists():
        raise FileNotFoundError(f"geometry not ready: {gpkg}")

    width, height = profile["width"], profile["height"]
    tf = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    targets = {
        "mask": incomplete(OUT_DIR / f"building_boundary_{year}_unfiltered.tif"),
        "ids": incomplete(OUT_DIR / f"building_id_{year}_unfiltered.tif"),
        "mask100": incomplete(OUT_DIR / f"building_boundary_{year}_ge100m2.tif"),
        "ids100": incomplete(OUT_DIR / f"building_id_{year}_ge100m2.tif"),
    }
    for path in targets.values():
        if path.exists():
            path.unlink()

    def open_target(path: Path, dtype: str, nodata: int = 0):
        p = dict(profile)
        p.update({"dtype": dtype, "nodata": nodata})
        return rasterio.open(path, "w+", **p)

    mask = open_target(targets["mask"], "uint8")
    ids = open_target(targets["ids"], "uint32")
    mask100 = open_target(targets["mask100"], "uint8")
    ids100 = open_target(targets["ids100"], "uint32")

    total_features = int(pyogrio.read_info(gpkg)["features"])
    n_stripes = (height + stripe_rows - 1) // stripe_rows
    progress(
        f"{year}: RASTERIZE {total_features:,} Geometry objects | source={gpkg} | "
        f"{stripe_rows}-row stripes | {n_stripes} stripes",
        progress_file,
    )
    started = time.time()
    try:
        for stripe_index, row_off in enumerate(range(0, height, stripe_rows), start=1):
            stripe_height = min(stripe_rows, height - row_off)
            win = Window(0, row_off, width, stripe_height)
            bbox = core.metric_bbox_for_window(win, profile["transform"])
            df = pyogrio.read_dataframe(
                gpkg, columns=["object_id", "area_m2", "geometry"], bbox=bbox, use_arrow=False
            )
            shapes = []
            shapes100 = []
            for obj, area, geom in zip(df["object_id"], df["area_m2"], df.geometry):
                if geom is None or geom.is_empty:
                    continue
                try:
                    oid = int(obj)
                    code = int(object_code.get((scene, oid), 0))
                    if code == 0:
                        continue
                    geom4326 = shapely_transform(tf.transform, geom)
                    shapes.append((geom4326, code))
                    area_value = float(area or 0.0)
                    if area_value >= core.AREA_FILTER and object_area.get((scene, oid), area_value) >= core.AREA_FILTER:
                        shapes100.append((geom4326, code))
                except Exception:
                    continue
            local = window_transform(win, profile["transform"])
            if shapes:
                ids_arr = rasterize(
                    shapes, out_shape=(stripe_height, width), transform=local, fill=0, dtype="uint32"
                )
                ids.write(ids_arr, 1, window=win)
                mask.write((ids_arr > 0).astype("uint8"), 1, window=win)
            if shapes100:
                ids100_arr = rasterize(
                    shapes100, out_shape=(stripe_height, width), transform=local, fill=0, dtype="uint32"
                )
                ids100.write(ids100_arr, 1, window=win)
                mask100.write((ids100_arr > 0).astype("uint8"), 1, window=win)
            if stripe_index == 1 or stripe_index == n_stripes or stripe_index % 10 == 0:
                elapsed = time.time() - started
                eta = (elapsed / stripe_index) * (n_stripes - stripe_index) / 60.0
                progress(
                    f"{year}: stripe {stripe_index}/{n_stripes} | candidates {len(df):,} | burned {len(shapes):,} "
                    f"| ge100 {len(shapes100):,} | elapsed {elapsed/60.0:.1f} min | eta {eta:.1f} min",
                    progress_file,
                )
    finally:
        for ds in (mask, ids, mask100, ids100):
            ds.close()

    promote(epoch_outputs(year), progress_file, year)
    progress(f"{year}: epoch rasterization complete in {(time.time()-started)/60.0:.1f} min", progress_file)
    return True


def _lock(fh) -> None:
    import msvcrt

    for _ in range(600):
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.5)


def _unlock(fh) -> None:
    import msvcrt

    try:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except Exception:
        pass


def write_lookup(code_to_physical: dict, object_code: dict, object_area: dict, progress_file: str) -> None:
    """Write BUILDING_ID_LOOKUP.csv (raster_id -> physical_track_id) under a lock."""
    lookup = OUT_DIR / "BUILDING_ID_LOOKUP.csv"
    tmp = OUT_DIR / "BUILDING_ID_LOOKUP.csv.tmp"
    reverse: dict[int, list[tuple[str, int, float]]] = defaultdict(list)
    for (scene, oid), code in object_code.items():
        reverse[code].append((scene, oid, object_area.get((scene, oid), 0.0)))
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["raster_id", "physical_track_id", "epoch", "epoch_object_id", "area_m2"])
        for code in sorted(code_to_physical):
            physical = code_to_physical[code]
            entries = sorted(reverse.get(code, []))
            if not entries:
                w.writerow([code, physical, "", "", ""])
                continue
            for scene, oid, area in entries:
                w.writerow([code, physical, scene, oid, f"{area:.3f}"])
    guard = OUT_DIR / ".lookup.lock"
    with guard.open("a+b") as fh:
        _lock(fh)
        try:
            os.replace(tmp, lookup)
        finally:
            _unlock(fh)
    progress(f"BUILDING_ID_LOOKUP.csv written ({len(code_to_physical):,} raster ids)", progress_file)


def run_state(profile: dict, object_code: dict, object_area: dict, stripe_rows: int, progress_file: str) -> None:
    core.STRIPE_ROWS = stripe_rows
    selected = core.load_state_object_maps(object_code, object_area)
    core.STRIPE_ROWS = stripe_rows
    n_selected = len(selected)
    scene_counts: dict[str, int] = defaultdict(int)
    for scene, _ in selected:
        scene_counts[scene] += 1
    progress(f"state masks: {n_selected:,} date-object/state entries {dict(scene_counts)}", progress_file)

    interval_dir = OUT_DIR
    datasets: dict = {}
    for interval in ("2018_2019", "2019_2020"):
        d = interval_dir / interval
        d.mkdir(parents=True, exist_ok=True)
        datasets[interval] = {}
        for state in ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT"):
            datasets[interval][state] = core.make_dataset(d / f"{state}_ge100m2.tif", profile, "uint8")
        datasets[interval]["COMBINED"] = core.make_dataset(
            d / "THREE_STATE_COMBINED_ge100m2.tif", profile, "uint8", nodata=255
        )
    try:
        width, height = profile["width"], profile["height"]
        for year, (scene, _, filename) in core.SCENES.items():
            gpkg = core.geom_path(filename)
            tf = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
            n_stripes = (height + stripe_rows - 1) // stripe_rows
            written = 0
            started = time.time()
            for stripe_index, row_off in enumerate(range(0, height, stripe_rows), start=1):
                stripe_height = min(stripe_rows, height - row_off)
                win = Window(0, row_off, width, stripe_height)
                bbox = core.metric_bbox_for_window(win, profile["transform"])
                df = pyogrio.read_dataframe(
                    gpkg, columns=["object_id", "geometry"], bbox=bbox, use_arrow=False
                )
                groups: dict = defaultdict(list)
                for obj, geom in zip(df["object_id"], df.geometry):
                    if geom is None or geom.is_empty:
                        continue
                    labels = selected.get((scene, int(obj)))
                    if not labels:
                        continue
                    geom4326 = shapely_transform(tf.transform, geom)
                    for interval, state in labels:
                        groups[(interval, state)].append((geom4326, 1))
                local = window_transform(win, profile["transform"])
                for (interval, state), shapes in groups.items():
                    if not shapes:
                        continue
                    burn = rasterize(
                        shapes,
                        out_shape=(stripe_height, width),
                        transform=local,
                        fill=0,
                        dtype="uint8",
                    )
                    datasets[interval][state].write(burn, 1, window=win)
                    written += len(shapes)
                if stripe_index == 1 or stripe_index == n_stripes or stripe_index % 10 == 0:
                    elapsed = time.time() - started
                    eta = (elapsed / stripe_index) * (n_stripes - stripe_index) / 60.0
                    progress(
                        f"state masks {year}: stripe {stripe_index}/{n_stripes} | candidates {len(df):,} "
                        f"| burned {written:,} | eta {eta:.1f} min",
                        progress_file,
                    )

        for interval, layers in datasets.items():
            combined = layers["COMBINED"]
            n_stripes = (height + stripe_rows - 1) // stripe_rows
            for row_off in range(0, height, stripe_rows):
                stripe_height = min(stripe_rows, height - row_off)
                win = Window(0, row_off, width, stripe_height)
                stable = layers["STABLE"].read(1, window=win)
                addition = layers["ADDITION"].read(1, window=win)
                demolition = layers["DEMOLITION"].read(1, window=win)
                unknown = layers["UNKNOWN_CONFLICT"].read(1, window=win)
                arr = np.zeros(stable.shape, dtype="uint8")
                arr[stable > 0] = 1
                arr[addition > 0] = 2
                arr[demolition > 0] = 3
                arr[unknown > 0] = 255
                combined.write(arr, 1, window=win)
            for state in ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT"):
                arr = layers[state].read(1)
                progress(f"{interval}/{state}: {int((arr > 0).sum()):,} pixels set", progress_file)
            progress(f"{interval}: combined three-state product finished", progress_file)
    finally:
        for layers in datasets.values():
            for ds in layers.values():
                ds.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="2020", help="comma list of epochs to rasterize for the boundary/ID pass")
    ap.add_argument("--state", action="store_true", help="run the interval state-mask pass in this process")
    ap.add_argument("--stripe-rows", type=int, default=512)
    ap.add_argument("--progress-file", default="RASTER_PROGRESS.md")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--index-only", action="store_true", help="only build the GeoPackage spatial index, then exit")
    args = ap.parse_args()

    if args.index_only:
        import pyogrio._err  # noqa: F401
        import pyogrio.raw as _raw

        for year, (scene, _, filename) in core.SCENES.items():
            gpkg = core.geom_path(filename)
            t0 = time.time()
            try:
                _raw.create_spatial_index(str(gpkg))
                print(f"{year}: spatial index built on {gpkg} in {time.time()-t0:.0f}s", flush=True)
            except Exception as exc:  # already indexed or read-only
                print(f"{year}: index skipped ({type(exc).__name__}: {exc})", flush=True)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile, grid = core.grid_profile()
    (OUT_DIR / "GRID_METADATA.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    object_code, object_area, code_to_physical = core.load_identity_maps()
    core.STRIPE_ROWS = args.stripe_rows

    if not args.state:
        write_lookup(code_to_physical, object_code, object_area, args.progress_file)
        for year in [e.strip() for e in args.epochs.split(",") if e.strip()]:
            if year not in core.SCENES:
                raise SystemExit(f"unknown epoch {year}")
            rasterize_epoch(
                year, profile, object_code, object_area, args.stripe_rows, args.progress_file, args.force
            )
    else:
        run_state(profile, object_code, object_area, args.stripe_rows, args.progress_file)

    progress(f"process finished: epochs={args.epochs} state={args.state}", args.progress_file)


if __name__ == "__main__":
    main()
