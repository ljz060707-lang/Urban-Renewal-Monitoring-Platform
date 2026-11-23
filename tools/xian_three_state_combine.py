"""Combine the per-state interval rasters into the delivered three-state product.

Encoding written (the user's suggested scheme, extended with one honest extra
class so that genuinely overlapping claims are never silently resolved):

    0   background
    1   STABLE
    2   ADDITION
    3   DEMOLITION
    4   UNKNOWN_CONFLICT (residual, kept separate, never folded into STABLE)
    5   STATE_OVERLAP (one pixel claimed by two different states)
    255 NODATA / outside the valid interval footprint

Why class 5 exists
------------------
UNKNOWN_CONFLICT is built from *both* epochs of an interval while STABLE and
ADDITION are built from the target epoch only.  A pixel can therefore be inside
a track that is resolved as STABLE and inside an unresolved track at the same
time.  Silently letting one class overwrite the other would hide real
ambiguity, so overlaps are counted, reported, and tagged as their own class.

Two orderings are computed for the record:
  A. definite-over-unknown  (STABLE > ADDITION > DEMOLITION > UNKNOWN)
  B. unknown-over-definite  (UNKNOWN > STABLE > ADDITION > DEMOLITION)
Class 5 is identical under both, and the per-class pixel counts of the
overlapping regions are reported so the effect of the ordering is auditable.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "work" / "python-packages"))
spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
spec.loader.exec_module(core)

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

OUT_DIR = core.DELIVERY / "01_raster_products"
STATE_CODES = {"STABLE": 1, "ADDITION": 2, "DEMOLITION": 3, "UNKNOWN_CONFLICT": 4}
OVERLAP = 5
NODATA = 255


def main() -> None:
    summary: dict = {}
    detail_rows = []
    for interval in ("2018_2019", "2019_2020"):
        d = OUT_DIR / interval
        paths = {s: d / f"{s}_ge100m2.tif" for s in STATE_CODES}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise SystemExit(f"{interval}: missing state rasters {missing}")
        stacks = {s: rasterio.open(p) for s, p in paths.items()}
        ref = stacks["STABLE"]
        profile = ref.profile.copy()
        profile.update({"dtype": "uint8", "nodata": NODATA, "count": 1})
        combined_path = d / "THREE_STATE_COMBINED_ge100m2.tif"
        tmp_path = combined_path.with_name(combined_path.name + ".incomplete")
        if tmp_path.exists():
            tmp_path.unlink()
        counts = {v: 0 for v in list(STATE_CODES.values()) + [OVERLAP, NODATA]}
        overlap_breakdown: dict[str, int] = {}
        try:
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for row_off in range(0, ref.height, 512):
                    h = min(512, ref.height - row_off)
                    win = Window(0, row_off, ref.width, h)
                    masks = {}
                    for s, ds in stacks.items():
                        masks[s] = ds.read(1, window=win) > 0
                    present = np.zeros(masks["STABLE"].shape, dtype=np.uint8)
                    for s in STATE_CODES:
                        present += masks[s].astype(np.uint8)
                    nodata = present == 0

                    arr = np.zeros(present.shape, dtype="uint8")
                    # definite-over-unknown
                    for s in ("DEMOLITION", "ADDITION", "STABLE"):
                        arr[masks[s]] = STATE_CODES[s]
                    arr[masks["UNKNOWN_CONFLICT"]] = STATE_CODES["UNKNOWN_CONFLICT"]
                    arr[nodata] = NODATA
                    # multi-claim detection, independent of ordering
                    multi = present >= 2
                    arr[multi] = OVERLAP
                    dst.write(arr, 1, window=win)

                    vals, cnts = np.unique(arr, return_counts=True)
                    for v, c in zip(vals.tolist(), cnts.tolist()):
                        counts[int(v)] = counts.get(int(v), 0) + int(c)
                    for pair in (
                        ("STABLE", "UNKNOWN_CONFLICT"),
                        ("ADDITION", "UNKNOWN_CONFLICT"),
                        ("DEMOLITION", "UNKNOWN_CONFLICT"),
                        ("STABLE", "ADDITION"),
                        ("STABLE", "DEMOLITION"),
                        ("ADDITION", "DEMOLITION"),
                        ("STABLE", "ADDITION", "UNKNOWN_CONFLICT"),
                    ):
                        m = np.ones(present.shape, dtype=bool)
                        for s in pair:
                            m &= masks[s]
                        if m.any():
                            key = "+".join(pair)
                            overlap_breakdown[key] = overlap_breakdown.get(key, 0) + int(np.count_nonzero(m))
        finally:
            for ds in stacks.values():
                ds.close()
        import os

        os.replace(tmp_path, combined_path)

        with rasterio.open(combined_path) as src:
            pixel_area_m2 = abs(src.transform.a * src.transform.e)
            transform = list(src.transform)[:6]
            crs = str(src.crs)
            width, height = src.width, src.height
        valid = width * height - counts.get(NODATA, 0)
        summary[interval] = {
            "combined_path": str(combined_path),
            "encoding": {
                "0": "background",
                "1": "STABLE",
                "2": "ADDITION",
                "3": "DEMOLITION",
                "4": "UNKNOWN_CONFLICT",
                "5": "STATE_OVERLAP",
                "255": "NODATA/outside valid interval footprint",
            },
            "pixel_area_m2": pixel_area_m2,
            "width": width,
            "height": height,
            "crs": crs,
            "transform": transform,
            "pixel_counts": {str(k): v for k, v in sorted(counts.items())},
            "valid_footprint_pixels": valid,
            "share_of_valid_footprint_pct": {
                str(k): (v / valid * 100.0 if valid else 0.0) for k, v in sorted(counts.items()) if k != NODATA
            },
            "area_km2_by_class": {
                str(k): v * pixel_area_m2 / 1e6 for k, v in sorted(counts.items()) if k != NODATA
            },
            "overlap_breakdown_pixels": dict(sorted(overlap_breakdown.items(), key=lambda kv: -kv[1])),
            "overlap_share_of_valid_pct": (counts.get(OVERLAP, 0) / valid * 100.0) if valid else 0.0,
            "ordering_used": "definite-over-unknown",
            "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
        }
        for k, v in summary[interval]["pixel_counts"].items():
            detail_rows.append({"interval": interval, "class_code": k, "pixels": v})
        print(
            f"{interval}: valid={valid:,} px  stable={counts.get(1,0):,} addition={counts.get(2,0):,} "
            f"demolition={counts.get(3,0):,} unknown={counts.get(4,0):,} overlap={counts.get(OVERLAP,0):,} "
            f"nodata={counts.get(NODATA,0):,}"
        )

    (OUT_DIR / "THREE_STATE_ENCODING_AND_QA.json").write_text(
        json.dumps(
            {
                "status": "CURRENT_BEST_CANDIDATE_DELIVERY",
                "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
                "authority_note": (
                    "class areas below are RASTER pixel areas reported for product QA only; the "
                    "scientific area authority is the vector statistics in 02_area_statistics"
                ),
                "intervals": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (OUT_DIR / "THREE_STATE_PIXEL_COUNTS.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["interval", "class_code", "pixels"])
        w.writeheader()
        w.writerows(detail_rows)
    print("wrote THREE_STATE_ENCODING_AND_QA.json and THREE_STATE_PIXEL_COUNTS.csv")


if __name__ == "__main__":
    main()
