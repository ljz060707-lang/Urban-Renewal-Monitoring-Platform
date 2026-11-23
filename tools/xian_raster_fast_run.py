"""In-memory geometry rasterizer for the Xi'an current-best candidate delivery.

Semantics are identical to ``tools/xian_rasterize_current_best.py`` (the proven
row-stripe implementation): the same objects, the same numeric labels, the same
``rasterize(..., all_touched=False)`` burn into the same full-city grid, the
same +100 m2 product filter, and the same "last write wins on overlap" ordering
inside a stripe.  What changes is only *how* geometries reach the rasterizer:

  old: for every 512-row stripe, re-query the GeoPackage with a metric bbox.
       The GPKG is unordered and has no R-tree, so that is a 46 s full scan per
       stripe -> ~26 min per epoch of pure re-reading.
  new: read the GeoPackage exactly once, transform once, and stream geometries
       to the stripe that actually contains them.

Peak memory is bounded by one epoch of polygons (~300 MB for 2020).

Scientific scope: CANDIDATE product export only.  No polygon is warped, no
state is invented, and Unknown/Conflict is never merged into Stable.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "work" / "python-packages"))

spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from rasterio.windows import Window, transform as window_transform  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402

OUT_DIR = core.DELIVERY / "01_raster_products"
EXPECTED_COUNTS = {"2018": 195_140, "2019": 182_193, "2020": 281_417}
STATE_LABELS = ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT")
SCENE_PAIR = {"2018_2019": ("xa180429", "xa190101"), "2019_2020": ("xa190101", "xa201231")}


def progress(message: str, progress_file: str) -> None:
    line = f"- {time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
    print(line.rstrip(), flush=True)
    path = OUT_DIR / progress_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def incomplete(path: Path) -> Path:
    return path.with_name(path.name + ".incomplete")


def epoch_outputs(year: str) -> list[Path]:
    return [
        OUT_DIR / f"building_boundary_{year}_unfiltered.tif",
        OUT_DIR / f"building_id_{year}_unfiltered.tif",
        OUT_DIR / f"building_boundary_{year}_ge100m2.tif",
        OUT_DIR / f"building_id_{year}_ge100m2.tif",
    ]


def outputs_are_complete(year: str, profile: dict, stripe_rows: int) -> tuple[bool, str]:
    """Existence is never enough: a killed writer leaves valid, all-zero rasters."""
    paths = epoch_outputs(year)
    if not all(p.exists() and p.stat().st_size > 0 for p in paths):
        return False, "missing output"
    height, width = profile["height"], profile["width"]
    n_stripes = (height + stripe_rows - 1) // stripe_rows
    last_row_off = (n_stripes - 1) * stripe_rows
    for p in paths:
        try:
            with rasterio.open(p) as src:
                if src.width != width or src.height != height or src.crs is None:
                    return False, f"grid mismatch in {p.name}"
                tail = src.read(1, window=Window(0, last_row_off, width, height - last_row_off))
                if np.count_nonzero(tail) == 0:
                    return False, f"{p.name} has no data in the final stripe"
                middle = src.read(1, window=Window(0, height // 2, width, stripe_rows))
                if np.count_nonzero(middle) == 0:
                    return False, f"{p.name} has no data mid-grid"
        except Exception as exc:
            return False, f"{p.name} unreadable ({type(exc).__name__})"
    return True, "verified"


def load_epoch_geometries(year: str, object_code: dict, object_area: dict, progress_file: str) -> dict:
    """Read one epoch once and return arrays ready for stripe streaming."""
    scene, _, filename = core.SCENES[year]
    gpkg = core.geom_path(filename)
    if not gpkg.exists():
        raise FileNotFoundError(f"geometry not ready: {gpkg}")
    info = pyogrio.read_info(gpkg)
    total = int(info["features"])
    expected = EXPECTED_COUNTS[year]
    if total != expected:
        raise SystemExit(
            f"HARD STOP: {year} geometry object count {total:,} != expected {expected:,} ({gpkg})"
        )
    t0 = time.time()
    df = pyogrio.read_dataframe(gpkg, columns=["object_id", "area_m2", "geometry"], use_arrow=False)
    read_s = time.time() - t0
    tf = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    codes, keep_ge100 = [], []
    geoms = []
    skipped_no_label = 0
    for obj, area, geom in zip(df["object_id"], df["area_m2"], df.geometry):
        if geom is None or geom.is_empty:
            continue
        try:
            oid = int(obj)
        except Exception:
            continue
        code = int(object_code.get((scene, oid), 0))
        if code == 0:
            skipped_no_label += 1
            continue
        area_value = float(area or 0.0)
        codes.append(code)
        keep_ge100.append(
            area_value >= core.AREA_FILTER and object_area.get((scene, oid), area_value) >= core.AREA_FILTER
        )
        geoms.append(geom)
    del df
    n = len(geoms)
    t2 = time.time()
    # NOTE: shapely_transform returns a NEW geometry, so it must be written back
    # into the list.  The bounds array below is therefore built only AFTER the
    # projection loop -- building it earlier yields EPSG:32649 metres and the
    # degree-based stripe test then silently selects zero geometries.
    for i in range(n):
        geoms[i] = shapely_transform(tf.transform, geoms[i])
    proj_s = time.time() - t2
    arr = np.empty(n, dtype=object)
    arr[:] = geoms
    bnd = shapely.bounds(arr)
    ymax = bnd[:, 3]
    xmin = bnd[:, 0]
    xmax = bnd[:, 2]
    codes_arr = np.asarray(codes, dtype=np.uint32)
    ge100_arr = np.asarray(keep_ge100, dtype=bool)
    # North-first order, and stable inside a row band, keeps burn precedence
    # identical to the row-stripe original.
    order = np.lexsort((np.arange(n), -ymax))
    progress(
        f"{year}: LOADED {total:,} objects | labelled {n:,} | unlabelled-skipped {skipped_no_label:,} "
        f"| ge100 {int(ge100_arr.sum()):,} | gpkg-read {read_s:.1f}s | project {proj_s:.1f}s "
        f"| wgs84_y[{ymax.min():.6f},{ymax.max():.6f}] wgs84_x[{xmin.min():.6f},{xmax.max():.6f}] "
        f"| source={gpkg}",
        progress_file,
    )
    return {
        "scene": scene,
        "geoms": geoms,
        "codes": codes_arr,
        "ge100": ge100_arr,
        "order": order,
        "ymax_sorted": ymax[order],
        "xmin": xmin,
        "xmax": xmax,
        "total": total,
    }


def stripe_row_ranges(height: int, stripe_rows: int):
    for idx, row_off in enumerate(range(0, height, stripe_rows), start=1):
        yield idx, row_off, min(stripe_rows, height - row_off)


def window_y_extent(win: Window, transform) -> tuple[float, float]:
    """Return (top, bottom) in map Y for a full-width output window.

    rasterio 1.5 dropped Window.top/.bottom, so derive the extent from the
    affine transform instead.
    """
    top = transform.f + transform.e * float(win.row_off)
    bottom = transform.f + transform.e * float(win.row_off + win.height)
    return (top, bottom) if top >= bottom else (bottom, top)


def rasterize_epoch_fast(
    year: str,
    profile: dict,
    object_code: dict,
    object_area: dict,
    stripe_rows: int,
    progress_file: str,
    force: bool,
) -> bool:
    ok, why = outputs_are_complete(year, profile, stripe_rows) if not force else (False, "forced")
    if ok:
        progress(f"{year}: existing rasters PASS completeness gate, epoch skipped (no restart)", progress_file)
        return False
    progress(f"{year}: completeness gate -> {why}; rasterizing", progress_file)

    data = load_epoch_geometries(year, object_code, object_area, progress_file)
    geoms = data["geoms"]
    codes = data["codes"]
    ge100 = data["ge100"]
    order = data["order"]
    ymax_sorted = data["ymax_sorted"]
    xmin, xmax = data["xmin"], data["xmax"]

    width, height = profile["width"], profile["height"]
    grid_left = profile["transform"].c
    grid_right = profile["transform"].c + profile["transform"].a * width
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

    n_stripes = (height + stripe_rows - 1) // stripe_rows
    progress(
        f"{year}: IN-MEMORY RASTERIZE {data['total']:,} objects over {n_stripes} stripes of {stripe_rows} rows",
        progress_file,
    )
    started = time.time()
    cursor = 0
    burned_total = 0
    burned100_total = 0
    try:
        for stripe_index, row_off, stripe_height in stripe_row_ranges(height, stripe_rows):
            win = Window(0, row_off, width, stripe_height)
            top, bottom = window_y_extent(win, profile["transform"])
            # Objects cannot start above the top of their own bbox; because the
            # order is descending in ymax, everything before `cursor` is north of
            # this stripe for good.
            while cursor < len(order) and ymax_sorted[cursor] > top:
                cursor += 1
            hits = []
            j = cursor
            while j < len(order):
                yi = order[j]
                if ymax_sorted[j] <= bottom:
                    break
                if xmax[yi] >= grid_left and xmin[yi] <= grid_right:
                    hits.append(yi)
                j += 1
            local = window_transform(win, profile["transform"])
            if hits:
                shapes = [(geoms[i], int(codes[i])) for i in hits]
                ids_arr = rasterize(
                    shapes, out_shape=(stripe_height, width), transform=local, fill=0, dtype="uint32"
                )
                ids.write(ids_arr, 1, window=win)
                mask.write((ids_arr > 0).astype("uint8"), 1, window=win)
                burned_total += len(shapes)
                shapes100 = [(geoms[i], int(codes[i])) for i in hits if ge100[i]]
                if shapes100:
                    ids100_arr = rasterize(
                        shapes100, out_shape=(stripe_height, width), transform=local, fill=0, dtype="uint32"
                    )
                    ids100.write(ids100_arr, 1, window=win)
                    mask100.write((ids100_arr > 0).astype("uint8"), 1, window=win)
                    burned100_total += len(shapes100)
            if stripe_index == 1 or stripe_index == n_stripes or stripe_index % 20 == 0:
                elapsed = time.time() - started
                eta = (elapsed / stripe_index) * (n_stripes - stripe_index) / 60.0
                progress(
                    f"{year}: stripe {stripe_index}/{n_stripes} | hits {len(hits):,} | burned {burned_total:,} "
                    f"| ge100 {burned100_total:,} | elapsed {elapsed/60.0:.1f} min | eta {eta:.1f} min",
                    progress_file,
                )
    finally:
        for ds in (mask, ids, mask100, ids100):
            ds.close()
    for path in epoch_outputs(year):
        os.replace(incomplete(path), path)
    progress(
        f"{year}: PROMOTED 4 rasters in {(time.time()-started)/60.0:.1f} min "
        f"(burned {burned_total:,} instances, {burned100_total:,} at >= {core.AREA_FILTER:g} m2)",
        progress_file,
    )
    return True


def _lock(fh) -> None:
    import msvcrt

    for _ in range(2400):
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.25)


def _unlock(fh) -> None:
    import msvcrt

    try:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except Exception:
        pass


def write_lookup(code_to_physical: dict, object_code: dict, object_area: dict, progress_file: str) -> None:
    lookup = OUT_DIR / "BUILDING_ID_LOOKUP.csv"
    tmp = OUT_DIR / "BUILDING_ID_LOOKUP.csv.tmp"
    reverse: dict[int, list[tuple[str, int, float]]] = defaultdict(list)
    for (scene, oid), code in object_code.items():
        reverse[code].append((scene, oid, object_area.get((scene, oid), 0.0)))
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["raster_id", "physical_track_id", "epoch", "epoch_object_id", "area_m2"])
        for code in sorted(code_to_physical):
            entries = sorted(reverse.get(code, []))
            if not entries:
                w.writerow([code, code_to_physical[code], "", "", ""])
                continue
            for scene, oid, area in entries:
                w.writerow([code, code_to_physical[code], scene, oid, f"{area:.3f}"])
    guard = OUT_DIR / ".lookup.lock"
    with guard.open("a+b") as fh:
        _lock(fh)
        try:
            os.replace(tmp, lookup)
        finally:
            _unlock(fh)
    progress(
        f"BUILDING_ID_LOOKUP.csv written ({len(code_to_physical):,} raster ids -> physical_track_id)",
        progress_file,
    )


def load_state_selection(object_code: dict, object_area: dict, progress_file: str) -> dict:
    """date-object -> set of (interval, state), exactly as the original state pass."""
    state_csv = core.LATEST / "CANDIDATE_STATE_EVENTS_FROM_SCRATCH.csv"
    tracking_csv = core.LATEST / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv"
    if not state_csv.exists() or not tracking_csv.exists():
        raise FileNotFoundError("latest candidate state/tracking CSVs are not ready")
    track_objects: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with tracking_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scene = row.get("scene_key", "")
            physical = row.get("physical_track_id", "")
            try:
                oid = int(row.get("object_uid", "").split("::")[-1])
            except Exception:
                continue
            if not scene or not physical:
                continue
            track_objects[(physical, scene)].append((oid, float(row.get("area_m2") or 0.0)))
    selected: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    state_counts: dict[tuple[str, str], int] = defaultdict(int)
    with state_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            interval = row.get("interval", "")
            state = row.get("state", "")
            physical = row.get("physical_track_id", "")
            if interval not in SCENE_PAIR or state not in STATE_LABELS:
                continue
            state_counts[(interval, state)] += 1
            source_scene, target_scene = SCENE_PAIR[interval]
            scenes = (
                [target_scene]
                if state in {"STABLE", "ADDITION"}
                else [source_scene]
                if state == "DEMOLITION"
                else [source_scene, target_scene]
            )
            for sc in scenes:
                for oid, area in track_objects.get((physical, sc), []):
                    if area >= core.AREA_FILTER:
                        selected[(sc, oid)].add((interval, state))
    progress(
        f"state selection: {len(selected):,} date-object/state entries from {len(state_counts)} (interval,state) groups",
        progress_file,
    )
    return selected


def rasterize_state_fast(
    profile: dict,
    object_code: dict,
    object_area: dict,
    stripe_rows: int,
    progress_file: str,
    force: bool,
) -> None:
    selected = load_state_selection(object_code, object_area, progress_file)
    width, height = profile["width"], profile["height"]
    grid_left = profile["transform"].c
    grid_right = profile["transform"].c + profile["transform"].a * width
    n_stripes = (height + stripe_rows - 1) // stripe_rows

    datasets: dict = {}
    for interval in ("2018_2019", "2019_2020"):
        d = OUT_DIR / interval
        d.mkdir(parents=True, exist_ok=True)
        datasets[interval] = {}
        for state in STATE_LABELS:
            datasets[interval][state] = core.make_dataset(d / f"{state}_ge100m2.tif", profile, "uint8")
        # The COMBINED three-state raster is authored by
        # tools/xian_three_state_combine.py, which also detects the genuinely
        # overlapping pixels (class 5) and refuses an all-zero product.
    try:
        for year, (scene, _, filename) in core.SCENES.items():
            data = load_epoch_geometries(year, object_code, object_area, progress_file)
            geoms, codes, order, ymax_sorted = data["geoms"], data["codes"], data["order"], data["ymax_sorted"]
            xmin, xmax = data["xmin"], data["xmax"]
            started = time.time()
            cursor = 0
            written = 0
            for stripe_index, row_off, stripe_height in stripe_row_ranges(height, stripe_rows):
                win = Window(0, row_off, width, stripe_height)
                top, bottom = window_y_extent(win, profile["transform"])
                while cursor < len(order) and ymax_sorted[cursor] > top:
                    cursor += 1
                groups: dict = defaultdict(list)
                j = cursor
                while j < len(order):
                    yi = order[j]
                    if ymax_sorted[j] <= bottom:
                        break
                    if xmax[yi] >= grid_left and xmin[yi] <= grid_right:
                        labels = selected.get((scene, int(codes[yi])))
                        if labels:
                            for interval, state in labels:
                                groups[(interval, state)].append((geoms[yi], 1))
                    j += 1
                local = window_transform(win, profile["transform"])
                for (interval, state), shapes in groups.items():
                    burn = rasterize(
                        shapes,
                        out_shape=(stripe_height, width),
                        transform=local,
                        fill=0,
                        dtype="uint8",
                    )
                    datasets[interval][state].write(burn, 1, window=win)
                    written += len(shapes)
                if stripe_index == 1 or stripe_index == n_stripes or stripe_index % 20 == 0:
                    elapsed = time.time() - started
                    eta = (elapsed / stripe_index) * (n_stripes - stripe_index) / 60.0
                    progress(
                        f"state masks {year}: stripe {stripe_index}/{n_stripes} | burned {written:,} | eta {eta:.1f} min",
                        progress_file,
                    )
            del data, geoms, codes, order, ymax_sorted

        for interval, layers in datasets.items():
            summary = {}
            for _, row_off, stripe_height in stripe_row_ranges(height, stripe_rows):
                win = Window(0, row_off, width, stripe_height)
                for name in STATE_LABELS:
                    arr = layers[name].read(1, window=win)
                    summary[name] = summary.get(name, 0) + int(np.count_nonzero(arr))
            progress(f"{interval}: state masks finished; pixel counts {summary}", progress_file)
            if all(v == 0 for v in summary.values()):
                progress(f"{interval}: WARNING all state masks are empty", progress_file)
    finally:
        for layers in datasets.values():
            for ds in layers.values():
                ds.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="", help="comma list of epochs for the boundary/ID pass")
    ap.add_argument("--state", action="store_true", help="run the interval state-mask pass")
    ap.add_argument("--stripe-rows", type=int, default=512)
    ap.add_argument("--progress-file", default="RASTER_PROGRESS.md")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    core.STRIPE_ROWS = args.stripe_rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile, grid = core.grid_profile()
    (OUT_DIR / "GRID_METADATA.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")
    object_code, object_area, code_to_physical = core.load_identity_maps()

    if args.epochs:
        for year in [e.strip() for e in args.epochs.split(",") if e.strip()]:
            if year not in core.SCENES:
                raise SystemExit(f"unknown epoch {year}")
            rasterize_epoch_fast(
                year, profile, object_code, object_area, args.stripe_rows, args.progress_file, args.force
            )
        write_lookup(code_to_physical, object_code, object_area, args.progress_file)
    if args.state:
        rasterize_state_fast(
            profile, object_code, object_area, args.stripe_rows, args.progress_file, args.force
        )
    progress(f"process finished: epochs='{args.epochs}' state={args.state}", args.progress_file)


if __name__ == "__main__":
    main()
