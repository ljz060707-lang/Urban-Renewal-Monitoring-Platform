"""Single-pass rasterizer for the Xi'an current-best candidate delivery.

Reads each epoch's geometry ONCE and, in the same stripe loop, produces:

  * building_boundary_<epoch>_unfiltered.tif + _ge100m2.tif
  * building_id_<epoch>_unfiltered.tif       + _ge100m2.tif
  * 2018_2019 / 2019_2020 per-state masks (STABLE, ADDITION, DEMOLITION, UNKNOWN)

Burn semantics are identical to the approved row-stripe implementation:
same objects, same numeric labels, same rasterize(..., all_touched=False) call,
same full-city EPSG:4326 grid, same "first shape wins on overlap" ordering.

Label source is the authority chain given in the task:
  F:\\xian\\西安_当前最佳候选交付_20260917\\08_cloud_v2_candidate\\
      FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv   (object -> physical_track_id)
      STATE_V2_FROM_SCRATCH.csv                 (physical_track_id -> interval state)

Scientific scope: CANDIDATE product export only.  Polygons are never warped, no
state is invented, and UNKNOWN is never folded into STABLE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import importlib.util
import sys

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

OUT_DIR = Path(os.environ.get("XIAN_RASTER_OUT_DIR", str(core.DELIVERY / "01_raster_products")))
V2_DIR = Path(os.environ.get("XIAN_V2_DIR", str(core.DELIVERY / "08_cloud_v2_candidate")))
DEFAULT_TRACKING = V2_DIR / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv"
DEFAULT_STATE = V2_DIR / "STATE_V2_FROM_SCRATCH.csv"

SCENE_PAIR = {"2018_2019": ("xa180429", "xa190101"), "2019_2020": ("xa190101", "xa201231")}
STATE_LABELS = ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT")
# V2 writes the residual class as plain "UNKNOWN"; the delivered raster class is
# still called UNKNOWN_CONFLICT and is still kept separate from STABLE.
V2_STATE_MAP = {
    "STABLE": "STABLE",
    "ADDITION": "ADDITION",
    "DEMOLITION": "DEMOLITION",
    "UNKNOWN": "UNKNOWN_CONFLICT",
    "UNKNOWN_CONFLICT": "UNKNOWN_CONFLICT",
}
EXPECTED_SCENE_ROWS = {"xa180429": 195_140, "xa190101": 182_193, "xa201231": 281_417}


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
    """Existence is never enough: a killed writer leaves a valid, all-zero raster."""
    paths = epoch_outputs(year)
    if not all(p.exists() and p.stat().st_size > 0 for p in paths):
        return False, "missing output"
    height, width = profile["height"], profile["width"]
    for p in paths:
        try:
            with rasterio.open(p) as src:
                if src.width != width or src.height != height or src.crs is None:
                    return False, f"grid mismatch in {p.name}"
                tail = src.read(1, window=Window(0, max(0, height - stripe_rows), width, min(stripe_rows, height)))
                if np.count_nonzero(tail) == 0:
                    return False, f"{p.name}: no data in final stripe"
                mid = src.read(1, window=Window(0, height // 2, width, min(stripe_rows, height // 2)))
                if np.count_nonzero(mid) == 0:
                    return False, f"{p.name}: no data mid-grid"
        except Exception as exc:
            return False, f"{p.name} unreadable ({type(exc).__name__})"
    return True, "verified"


def load_labels(tracking_csv: Path, state_csv: Path, progress_file: str):
    """Return (object_code, object_area, code_to_physical, selected)."""
    if not tracking_csv.exists():
        raise FileNotFoundError(f"missing tracking table: {tracking_csv}")
    if not state_csv.exists():
        raise FileNotFoundError(f"missing state table: {state_csv}")

    scene_counts: dict[str, int] = defaultdict(int)
    physical_to_code: dict[str, int] = {}
    code_to_physical: dict[int, str] = {}
    object_code: dict[tuple[str, int], int] = {}
    object_area: dict[tuple[str, int], float] = {}
    track_objects: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    next_code = 1
    with tracking_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scene = row.get("scene_key", "")
            physical = row.get("physical_track_id", "")
            if not scene or not physical:
                continue
            try:
                oid = int(row.get("object_uid", "").split("::")[-1])
            except Exception:
                continue
            try:
                area = float(row.get("area_m2") or 0.0)
            except Exception:
                area = 0.0
            if physical not in physical_to_code:
                physical_to_code[physical] = next_code
                code_to_physical[next_code] = physical
                next_code += 1
            code = physical_to_code[physical]
            object_code[(scene, oid)] = code
            object_area[(scene, oid)] = area
            track_objects[(physical, scene)].append((oid, area))
            scene_counts[scene] += 1
    for scene, expected in EXPECTED_SCENE_ROWS.items():
        got = scene_counts.get(scene, 0)
        if got != expected:
            raise SystemExit(f"HARD STOP: {scene} label rows {got:,} != expected {expected:,}")

    selected: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    state_rows: dict[tuple[str, str], int] = defaultdict(int)
    unmapped = 0
    with state_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            interval = row.get("interval", "")
            raw_state = row.get("state", "")
            physical = row.get("physical_track_id", "")
            if interval not in SCENE_PAIR:
                continue
            state = V2_STATE_MAP.get(raw_state)
            if state is None:
                unmapped += 1
                continue
            state_rows[(interval, state)] += 1
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
        f"labels: {len(object_code):,} date-objects, {len(code_to_physical):,} physical tracks, "
        f"selected {len(selected):,} date-object/state entries, unmapped-state {unmapped}",
        progress_file,
    )
    progress(f"state rows by (interval,state): {dict(sorted(state_rows.items()))}", progress_file)
    return object_code, object_area, code_to_physical, selected


def load_epoch(year: str, object_code: dict, object_area: dict, progress_file: str) -> dict:
    scene, _, filename = core.SCENES[year]
    gpkg = core.geom_path(filename)
    if not gpkg.exists():
        raise FileNotFoundError(f"geometry not ready: {gpkg}")
    total = int(pyogrio.read_info(gpkg)["features"])
    expected = EXPECTED_SCENE_ROWS[scene]
    if total != expected:
        raise SystemExit(f"HARD STOP: {year} geometry count {total:,} != expected {expected:,} ({gpkg})")

    t0 = time.time()
    df = pyogrio.read_dataframe(gpkg, columns=["object_id", "area_m2", "geometry"], use_arrow=False)
    read_s = time.time() - t0
    tf = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    object_ids, codes, ge100, geoms = [], [], [], []
    for obj, area, geom in zip(df["object_id"], df["area_m2"], df.geometry):
        if geom is None or geom.is_empty:
            continue
        try:
            oid = int(obj)
        except Exception:
            continue
        code = int(object_code.get((scene, oid), 0))
        if code == 0:
            continue
        area_value = float(area or 0.0)
        object_ids.append(oid)
        codes.append(code)
        ge100.append(area_value >= core.AREA_FILTER and object_area.get((scene, oid), area_value) >= core.AREA_FILTER)
        geoms.append(geom)
    del df
    n = len(geoms)
    t1 = time.time()
    # shapely_transform returns a NEW geometry: the result must be written back,
    # and bounds may only be taken afterwards (metres vs degrees mismatch would
    # otherwise silently select zero geometries).
    for i in range(n):
        geoms[i] = shapely_transform(tf.transform, geoms[i])
    proj_s = time.time() - t1
    arr = np.empty(n, dtype=object)
    arr[:] = geoms
    bnd = shapely.bounds(arr)
    del arr
    ymax = bnd[:, 3]
    xmin = bnd[:, 0]
    xmax = bnd[:, 2]
    order = np.lexsort((np.arange(n), -ymax))
    progress(
        f"{year}: LOADED {total:,} | labelled {n:,} | ge100 {int(np.sum(ge100)):,} | gpkg-read {read_s:.1f}s "
        f"| project {proj_s:.1f}s | wgs84_y[{ymax.min():.6f},{ymax.max():.6f}] | source={gpkg}",
        progress_file,
    )
    return {
        "scene": scene,
        "object_ids": np.asarray(object_ids, dtype=np.int64),
        "geoms": geoms,
        "codes": np.asarray(codes, dtype=np.uint32),
        "ge100": np.asarray(ge100, dtype=bool),
        "order": order,
        "ymax_sorted": ymax[order],
        "xmin": xmin,
        "xmax": xmax,
        "total": total,
    }


def window_y_extent(win: Window, transform) -> tuple[float, float]:
    top = transform.f + transform.e * float(win.row_off)
    bottom = transform.f + transform.e * float(win.row_off + win.height)
    return (top, bottom) if top >= bottom else (bottom, top)


def open_target(path: Path, profile: dict, dtype: str, nodata: int = 0):
    p = dict(profile)
    p.update({"dtype": dtype, "nodata": nodata})
    return rasterio.open(path, "w+", **p)


def state_paths(interval: str, state: str) -> tuple[Path, Path]:
    final = OUT_DIR / interval / f"{state}_ge100m2.tif"
    return final, incomplete(final)


def run(args) -> None:
    core.STRIPE_ROWS = args.stripe_rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile, grid = core.grid_profile()
    (OUT_DIR / "GRID_METADATA.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")
    object_code, object_area, code_to_physical, selected = load_labels(
        Path(args.tracking), Path(args.state), args.progress_file
    )

    width, height = profile["width"], profile["height"]
    grid_left = profile["transform"].c
    grid_right = profile["transform"].c + profile["transform"].a * width
    stripe_rows = args.stripe_rows
    n_stripes = (height + stripe_rows - 1) // stripe_rows

    pending = [y for y in args.epochs if not outputs_are_complete(y, profile, stripe_rows)[0]]
    for y in args.epochs:
        if y not in pending:
            progress(f"{y}: existing rasters PASS the completeness gate, skipped", args.progress_file)
    if not pending:
        progress("all requested epochs already complete", args.progress_file)
        if args.finalize:
            finalize_state(args, profile, stripe_rows)
            write_lookup(code_to_physical, object_code, object_area, args.progress_file)
        return

    # State targets live at .incomplete paths while epochs accumulate.  They are
    # opened "r+" when a previous epoch already created them, so each epoch only
    # fills its own geo-referenced window and never truncates another's work.
    state_ds: dict = {}
    if not args.no_state:
        for interval in ("2018_2019", "2019_2020"):
            (OUT_DIR / interval).mkdir(parents=True, exist_ok=True)
            state_ds[interval] = {}
            for s in STATE_LABELS:
                _final, tmp = state_paths(interval, s)
                if tmp.exists():
                    # A prior interrupted handoff may leave a zero-byte or
                    # truncated/invalid placeholder.  Recreate only when it
                    # cannot be opened as the current grid; preserve a valid
                    # non-empty partial layer so the handoff remains resumable.
                    valid_partial = False
                    if tmp.stat().st_size > 0:
                        try:
                            with rasterio.open(tmp) as check:
                                valid_partial = (
                                    check.width == profile["width"]
                                    and check.height == profile["height"]
                                    and check.count == 1
                                    and check.crs == profile["crs"]
                                    and check.transform == profile["transform"]
                                    and check.dtypes[0] == "uint8"
                                )
                        except Exception:
                            valid_partial = False
                    if not valid_partial:
                        tmp.unlink()
                        progress(f"removed invalid incomplete state placeholder: {tmp.name}", args.progress_file)
                        state_ds[interval][s] = open_target(tmp, profile, "uint8")
                    else:
                        state_ds[interval][s] = rasterio.open(tmp, "r+")
                else:
                    state_ds[interval][s] = open_target(tmp, profile, "uint8")

    try:
        for year in pending:
            data = load_epoch(year, object_code, object_area, args.progress_file)
            object_ids, geoms, codes, ge100 = data["object_ids"], data["geoms"], data["codes"], data["ge100"]
            order, ymax_sorted = data["order"], data["ymax_sorted"]
            xmin, xmax, scene = data["xmin"], data["xmax"], data["scene"]
            targets = {
                "mask": incomplete(OUT_DIR / f"building_boundary_{year}_unfiltered.tif"),
                "ids": incomplete(OUT_DIR / f"building_id_{year}_unfiltered.tif"),
                "mask100": incomplete(OUT_DIR / f"building_boundary_{year}_ge100m2.tif"),
                "ids100": incomplete(OUT_DIR / f"building_id_{year}_ge100m2.tif"),
            }
            for p in targets.values():
                if p.exists():
                    p.unlink()
            mask = open_target(targets["mask"], profile, "uint8")
            ids = open_target(targets["ids"], profile, "uint32")
            mask100 = open_target(targets["mask100"], profile, "uint8")
            ids100 = open_target(targets["ids100"], profile, "uint32")
            started = time.time()
            cursor = 0
            burned = burned100 = burned_state = 0
            progress(
                f"{year}: SINGLE-PASS over {n_stripes} stripes of {stripe_rows} rows "
                f"(boundary + ID{' + state masks' if state_ds else ''})",
                args.progress_file,
            )
            try:
                for stripe_index, row_off in enumerate(range(0, height, stripe_rows), start=1):
                    stripe_height = min(stripe_rows, height - row_off)
                    win = Window(0, row_off, width, stripe_height)
                    top, bottom = window_y_extent(win, profile["transform"])
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
                    if hits:
                        local = window_transform(win, profile["transform"])
                        shapes = [(geoms[i], int(codes[i])) for i in hits]
                        ids_arr = rasterize(
                            shapes, out_shape=(stripe_height, width), transform=local, fill=0, dtype="uint32"
                        )
                        ids.write(ids_arr, 1, window=win)
                        mask.write((ids_arr > 0).astype("uint8"), 1, window=win)
                        burned += len(shapes)
                        shapes100 = [(geoms[i], int(codes[i])) for i in hits if ge100[i]]
                        if shapes100:
                            ids100_arr = rasterize(
                                shapes100, out_shape=(stripe_height, width), transform=local, fill=0, dtype="uint32"
                            )
                            ids100.write(ids100_arr, 1, window=win)
                            mask100.write((ids100_arr > 0).astype("uint8"), 1, window=win)
                            burned100 += len(shapes100)
                        if state_ds:
                            groups: dict = defaultdict(list)
                            for i in hits:
                                labels = selected.get((scene, int(object_ids[i])))
                                if not labels:
                                    continue
                                for interval, state in labels:
                                    groups[(interval, state)].append((geoms[i], 1))
                            for (interval, state), shp in groups.items():
                                burn = rasterize(
                                    shp, out_shape=(stripe_height, width), transform=local, fill=0, dtype="uint8"
                                )
                                # Multiple epoch passes contribute to the same
                                # interval/state layer (notably UNKNOWN from
                                # both source and target dates).  Preserve the
                                # prior pass by OR-merging the new burn.
                                ds = state_ds[interval][state]
                                existing = ds.read(1, window=win)
                                existing[burn > 0] = 1
                                ds.write(existing, 1, window=win)
                                burned_state += len(shp)
                    if stripe_index == 1 or stripe_index == n_stripes or stripe_index % 20 == 0:
                        el = time.time() - started
                        eta = (el / stripe_index) * (n_stripes - stripe_index) / 60.0
                        progress(
                            f"{year}: stripe {stripe_index}/{n_stripes} | hits {len(hits):,} | burned {burned:,} "
                            f"| ge100 {burned100:,} | state {burned_state:,} | elapsed {el/60.0:.1f} min | eta {eta:.1f} min",
                            args.progress_file,
                        )
            finally:
                for ds in (mask, ids, mask100, ids100):
                    ds.close()
            for p in epoch_outputs(year):
                os.replace(incomplete(p), p)
            progress(
                f"{year}: PROMOTED 4 rasters in {(time.time()-started)/60.0:.1f} min "
                f"(burned {burned:,} instances, {burned100:,} at >= {core.AREA_FILTER:g} m2, state {burned_state:,})",
                args.progress_file,
            )
            del data, object_ids, geoms, codes, ge100, order, ymax_sorted
    finally:
        if state_ds:
            for layers in state_ds.values():
                for ds in layers.values():
                    ds.close()

    if args.finalize:
        finalize_state(args, profile, stripe_rows)
    write_lookup(code_to_physical, object_code, object_area, args.progress_file)
    progress("single-pass rasterization complete", args.progress_file)


def finalize_state(args, profile: dict, stripe_rows: int) -> None:
    """Promote accumulated .incomplete state masks to final names after the last epoch."""
    if args.no_state:
        return
    with rasterio.open(OUT_DIR / "2018_2019" / f"{STATE_LABELS[0]}_ge100m2.tif.incomplete") as ref:
        height, width = ref.height, ref.width
    for interval in ("2018_2019", "2019_2020"):
        summary = {}
        for s in STATE_LABELS:
            final, tmp = state_paths(interval, s)
            if not tmp.exists():
                progress(f"state finalize: MISSING {tmp}", args.progress_file)
                continue
            with rasterio.open(tmp) as src:
                for row_off in range(0, height, stripe_rows):
                    win = Window(0, row_off, width, min(stripe_rows, height - row_off))
                    arr = src.read(1, window=win)
                    summary[s] = summary.get(s, 0) + int(np.count_nonzero(arr))
            os.replace(tmp, final)
        progress(f"{interval}: state masks PROMOTED; pixel counts {summary}", args.progress_file)
        if all(v == 0 for v in summary.values()):
            progress(f"{interval}: WARNING all state masks are empty", args.progress_file)


def _lock(fh) -> None:
    try:
        import msvcrt
    except ImportError:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return

    for _ in range(2400):
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.25)


def _unlock(fh) -> None:
    try:
        import msvcrt
    except ImportError:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return

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
    progress(f"BUILDING_ID_LOOKUP.csv written ({len(code_to_physical):,} raster ids -> physical_track_id)", progress_file)


def main() -> None:
    ap = argparse.ArgumentParser()
    # A single-valued --epoch is used deliberately: PowerShell's native-argument
    # binding turns a bare "--epochs 2018" into the character array '2','0','1','8'.
    ap.add_argument("--epoch", default="", help="one epoch: 2018, 2019 or 2020")
    ap.add_argument("--stripe-rows", type=int, default=512)
    ap.add_argument("--progress-file", default="RASTER_PROGRESS.md")
    ap.add_argument("--tracking", default=str(DEFAULT_TRACKING))
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--no-state", action="store_true", help="skip the interval state masks")
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="after this epoch, promote the accumulated state masks to their final names",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.epochs = [args.epoch] if args.epoch else []
    if not args.epochs:
        raise SystemExit("--epoch is required (2018, 2019 or 2020)")
    for y in args.epochs:
        if y not in core.SCENES:
            raise SystemExit(f"unknown epoch {y!r}")
    if args.force:
        for y in args.epochs:
            for p in epoch_outputs(y):
                if p.exists():
                    p.unlink()
    run(args)


if __name__ == "__main__":
    main()
