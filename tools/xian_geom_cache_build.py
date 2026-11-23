"""Build a compact on-disk geometry cache for cases, figures and ranking.

Why: the GPKG files are unordered and carry no R-tree, so every consumer either
re-reads 3.9 GB or rebuilds 658,750 shapely objects in RAM.  This builder reads
each epoch once with the low-level pyogrio API, parses the WKB polygons straight
into contiguous NumPy arrays, and appends fixed-size records to small shards.

The case/figure renderers then only need a bounded window of records.

Output layout (per epoch):
    <out>/<epoch>/shard_0000.npy     structured records, all fixed width
    <out>/<epoch>/index.json         shard sizes + format description
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "work" / "python-packages"))
spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)

import numpy as np  # noqa: E402
import pyogrio.raw as praw  # noqa: E402

RECORD = np.dtype(
    [
        ("object_id", "<i8"),
        ("area_m2", "<f8"),
        ("n_rings", "<i4"),
        ("n_points", "<i4"),
        ("ring_start", "<i8"),  # index into ring offsets (n_rings+1 entries)
        ("coord_start", "<i8"),  # index into the xy array
        ("cx", "<f8"),
        ("cy", "<f8"),
        ("minx", "<f8"),
        ("miny", "<f8"),
        ("maxx", "<f8"),
        ("maxy", "<f8"),
    ]
)

RING_RECORD = np.dtype([("start", "<i8"), ("count", "<i4"), ("pad", "<i4")])


def parse_epoch(year: str, out_root: Path, block: int = 20_000) -> dict:
    scene, _, filename = core.SCENES[year]
    gpkg = core.geom_path(filename)
    meta = praw.read(gpkg, columns=["object_id", "area_m2"], read_geometry=True)
    wkb = meta["geometry"]
    oids = np.asarray(meta["object_id"], dtype=np.int64)
    areas = np.asarray(meta["area_m2"], dtype=np.float64)
    n_total = len(oids)
    print(f"{year}: WKB blobs {n_total:,} from {gpkg}", flush=True)

    d = out_root / year
    d.mkdir(parents=True, exist_ok=True)
    records, ring_off, coords = [], [], []
    shard_idx, shard_records, shard_rings, shard_coords = 0, [], [], []
    coord_total, ring_total = 0, 0
    manifest = []
    n_poly, n_multipoly, n_skipped = 0, 0, 0

    def flush_shard():
        nonlocal shard_idx, shard_records, shard_rings, shard_coords, coord_total, ring_total
        if not shard_records:
            return
        base = d / f"shard_{shard_idx:04d}"
        np.save(base.with_suffix(".rec.npy"), np.asarray(shard_records, dtype=RECORD))
        np.save(base.with_suffix(".ring.npy"), np.asarray(shard_rings, dtype=RING_RECORD))
        np.save(base.with_suffix(".xy.npy"), np.asarray(shard_coords, dtype="<f8").reshape(-1, 2))
        manifest.append(
            {
                "shard": shard_idx,
                "records": len(shard_records),
                "rings": len(shard_rings),
                "points": len(shard_coords),
            }
        )
        print(
            f"  shard {shard_idx:04d}: {len(shard_records):,} polys, {len(shard_coords):,} points",
            flush=True,
        )
        shard_idx += 1
        shard_records, shard_rings, shard_coords = [], [], []
        coord_total, ring_total = 0, 0

    for i in range(n_total):
        raw = wkb[i]
        if raw is None:
            n_skipped += 1
            continue
        b = bytes(raw)
        try:
            endian = "<" if b[0] == 1 else ">"
            gtype = struct.unpack_from(endian + "I", b, 1)[0]
            base_type = gtype & 0xFF
            pos = 5
            polys = []
            if base_type == 3:  # Polygon
                polys = [b]
                n_poly += 1
            elif base_type == 6:  # MultiPolygon
                n_multipoly += 1
                (npolys,) = struct.unpack_from(endian + "I", b, pos)
                pos += 4
                for _ in range(npolys):
                    sub_endian = "<" if b[pos] == 1 else ">"
                    pos += 5  # endian + type
                    (nrings,) = struct.unpack_from(sub_endian + "I", b, pos)
                    pos += 4
                    start = pos
                    for _r in range(nrings):
                        (npts,) = struct.unpack_from(sub_endian + "I", b, pos)
                        pos += 4 + npts * 16
                    polys.append(b[start:pos])
            else:
                n_skipped += 1
                continue

            first_ring = len(shard_rings)
            first_coord = len(shard_coords)
            minx = miny = math.inf
            maxx = maxy = -math.inf
            sumx = sumy = 0.0
            npts_total = 0
            for pb in polys:
                pe = "<" if pb[0] == 1 else ">"
                p = 5
                (nrings,) = struct.unpack_from(pe + "I", pb, p)
                p += 4
                for _r in range(nrings):
                    (npts,) = struct.unpack_from(pe + "I", pb, p)
                    p += 4
                    if npts == 0:
                        shard_rings.append((len(shard_coords), 0, 0))
                        continue
                    xy = np.frombuffer(pb, dtype=pe + "f8", count=npts * 2, offset=p).reshape(-1, 2)
                    p += npts * 16
                    shard_rings.append((len(shard_coords), npts, 0))
                    shard_coords.append(xy)
                    npts_total += npts
                    minx = min(minx, float(xy[:, 0].min()))
                    maxx = max(maxx, float(xy[:, 0].max()))
                    miny = min(miny, float(xy[:, 1].min()))
                    maxy = max(maxy, float(xy[:, 1].max()))
                    sumx += float(xy[:, 0].sum())
                    sumy += float(xy[:, 1].sum())
            shard_rings.append((len(shard_coords), 0, 0))  # ring terminator
            if npts_total == 0 or not np.isfinite(minx):
                n_skipped += 1
                continue
            shard_records.append(
                (
                    int(oids[i]),
                    float(areas[i]),
                    len(polys),
                    npts_total,
                    first_ring,
                    first_coord,
                    sumx / npts_total,
                    sumy / npts_total,
                    minx,
                    miny,
                    maxx,
                    maxy,
                )
            )
        except Exception:
            n_skipped += 1
            continue
        if len(shard_records) >= block:
            flush_shard()
    flush_shard()

    index = {
        "epoch": year,
        "scene_key": scene,
        "source_gpkg": str(gpkg),
        "features_in_source": int(n_total),
        "polygons_parsed": int(n_poly),
        "multipolygons_parsed": int(n_multipoly),
        "skipped": int(n_skipped),
        "shards": manifest,
        "record_dtype": RECORD.descr,
        "ring_dtype": RING_RECORD.descr,
        "crs": "EPSG:32649",
        "note": "per-polygon rows: one record per WKB polygon; ring/xy arrays are shard-local",
    }
    (d / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{year}: cached {len(manifest)} shards, parsed {n_poly} polygons + {n_multipoly} multipolygons, "
        f"skipped {n_skipped}",
        flush=True,
    )
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"F:\xian\西安_当前最佳候选交付_20260917\07_geometry_cache")
    ap.add_argument("--epochs", default="2018,2019,2020")
    ap.add_argument("--block", type=int, default=20_000)
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for year in [e.strip() for e in args.epochs.split(",") if e.strip()]:
        summary[year] = parse_epoch(year, root, args.block)
    (root / "CACHE_SUMMARY.json").write_text(
        json.dumps(
            {
                "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
                "purpose": "compact geometry cache for case sheets, ranking and publication figures",
                "epochs": {k: {"polygons": v.get("polygons_parsed"), "shards": len(v.get("shards", []))} for k, v in summary.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("done")


if __name__ == "__main__":
    main()
