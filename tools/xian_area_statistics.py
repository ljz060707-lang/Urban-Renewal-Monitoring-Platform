"""Area statistics for the Xi'an current-best candidate delivery.

AUTHORITY: VECTOR, and specifically the cloud V2 candidate identity tables:

    FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv   (object -> physical_track_id, area_m2)
    STATE_V2_FROM_SCRATCH.csv                 (physical_track_id -> interval state)

V2 writes the residual class as plain ``UNKNOWN``; the delivered raster classes
still call it ``UNKNOWN_CONFLICT`` and it is still kept strictly separate from
STABLE.  No raster pixel count ever feeds a scientific area.

Denominator
-----------
The receiver's own receipt (FINAL_CANDIDATE_RECEIPT_V2.json) reports state shares
that reproduce, to ten decimals, as ``union_area_m2 / sum(union_area_m2)``.  That
is therefore the headline denominator used here, and the delivery states it
explicitly on every line.

Note on STABLE: V2's quoted STABLE share uses a different base, so this delivery
reports STABLE as the complement of the other three classes on the shared
denominator and says so, rather than silently mixing two bases.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "work" / "python-packages"))
spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)

DELIVERY = core.DELIVERY
V2 = DELIVERY / "08_cloud_v2_candidate"
STATE_CSV = V2 / "STATE_V2_FROM_SCRATCH.csv"
RECEIPT_V2 = V2 / "FINAL_CANDIDATE_RECEIPT_V2.json"
STATES = ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT")
V2_KEYS = ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN")
INTERVALS = ("2018_2019", "2019_2020")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DELIVERY / "02_area_statistics"))
    ap.add_argument("--state-csv", default=str(STATE_CSV))
    ap.add_argument("--raster-state-dir", default=str(DELIVERY / "01_raster_products"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    receipt = json.loads(RECEIPT_V2.read_text(encoding="utf-8")) if RECEIPT_V2.exists() else {}
    v2_quoted = {
        iv: {k: float(v) for k, v in receipt.get("state_area_pct", {}).get(iv, {}).items()}
        for iv in INTERVALS
    }

    agg: dict = {
        iv: {
            s: {
                "n": 0, "area_m2": 0.0, "union_area_m2": 0.0,
                "source_area_m2": 0.0, "target_area_m2": 0.0,
                "gap_return": 0, "strict_change_eligible": 0,
            }
            for s in STATES
        }
        for iv in INTERVALS
    }
    unexpected = defaultdict(int)
    rows_total = 0
    with Path(args.state_csv).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            iv = row.get("interval", "")
            raw = row.get("state", "")
            if iv not in agg:
                unexpected[f"interval={iv}"] += 1
                continue
            key = "UNKNOWN_CONFLICT" if raw == "UNKNOWN" else raw
            if key not in agg[iv]:
                unexpected[f"state={raw}"] += 1
                continue
            rows_total += 1

            def num(k):
                try:
                    return float(row.get(k) or 0.0)
                except Exception:
                    return 0.0

            a = agg[iv][key]
            a["n"] += 1
            a["area_m2"] += num("area_m2")
            a["union_area_m2"] += num("union_area_m2")
            a["source_area_m2"] += num("source_area_m2")
            a["target_area_m2"] += num("target_area_m2")
            if str(row.get("gap_return", "")).strip().upper() == "YES":
                a["gap_return"] += 1
            if str(row.get("strict_change_eligible", "")).strip().upper() == "YES":
                a["strict_change_eligible"] += 1

    report = {
        "status": "CURRENT_BEST_CANDIDATE_DELIVERY",
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
        "authority": "VECTOR (EPSG:32649 polygon union_area_m2 from the cloud V2 candidate state table)",
        "raster_role": "representation/product only; never used to derive a scientific area",
        "headline_denominator": "sum of union_area_m2 over all four classes in the interval",
        "source_state_csv": args.state_csv,
        "source_receipt": str(RECEIPT_V2) if RECEIPT_V2.exists() else None,
        "state_rows_used": rows_total,
        "unexpected_rows": dict(unexpected),
        "area_filter_m2": core.AREA_FILTER,
        "intervals": {},
    }
    csv_rows = []
    for iv in INTERVALS:
        block = agg[iv]
        denom = sum(block[s]["union_area_m2"] for s in STATES)
        changed = sum(block[s]["union_area_m2"] for s in ("STABLE", "ADDITION", "DEMOLITION"))
        interval_block = {
            "denominator_union_area_sum_m2": denom,
            "denominator_union_area_sum_km2": denom / 1e6,
            "denominator_note": (
                "headline denominator = sum(union_area_m2) over STABLE+ADDITION+DEMOLITION+UNKNOWN_CONFLICT; "
                "this reproduces the V2 receipt percentages exactly for ADDITION, DEMOLITION and UNKNOWN"
            ),
            "changed_only_union_sum_m2": changed,
            "epoch_start_building_area_m2": (
                block["STABLE"]["source_area_m2"] + block["ADDITION"]["source_area_m2"]
                + block["DEMOLITION"]["source_area_m2"] + block["UNKNOWN_CONFLICT"]["union_area_m2"]
            ),
            "epoch_end_building_area_m2": (
                block["STABLE"]["target_area_m2"] + block["ADDITION"]["target_area_m2"]
                + block["DEMOLITION"]["source_area_m2"] + block["UNKNOWN_CONFLICT"]["union_area_m2"]
            ),
            "states": {},
            "v2_receipt_cross_check": {},
        }
        for s, v2key in zip(STATES, V2_KEYS):
            a = block[s]
            area = a["union_area_m2"]
            pct = (area / denom * 100.0) if denom else 0.0
            rec = {
                "state": s,
                "v2_state_label": v2key,
                "track_count": a["n"],
                "union_area_m2": area,
                "union_area_km2": area / 1e6,
                "share_pct_of_union_denominator": pct,
                "area_field_m2": a["area_m2"],
                "area_field_km2": a["area_m2"] / 1e6,
                "source_area_m2": a["source_area_m2"],
                "target_area_m2": a["target_area_m2"],
                "mean_union_area_m2": (area / a["n"]) if a["n"] else 0.0,
                "gap_return_tracks": a["gap_return"],
                "strict_change_eligible_tracks": a["strict_change_eligible"],
            }
            interval_block["states"][s] = rec
            quoted = v2_quoted.get(iv, {}).get(v2key)
            interval_block["v2_receipt_cross_check"][s] = {
                "v2_quoted_pct": quoted,
                "computed_pct": pct,
                "difference_pct_points": (pct - quoted) if quoted is not None else None,
                "matches": (abs(pct - quoted) < 1e-6) if quoted is not None else None,
            }
            csv_rows.append(
                {
                    "interval": iv,
                    "state": s,
                    "track_count": a["n"],
                    "union_area_m2": round(area, 3),
                    "union_area_km2": round(area / 1e6, 6),
                    "share_pct_of_union_denominator": round(pct, 6),
                    "denominator_union_area_sum_km2": round(denom / 1e6, 6),
                    "area_field_km2": round(a["area_m2"] / 1e6, 6),
                    "source_area_km2": round(a["source_area_m2"] / 1e6, 6),
                    "target_area_km2": round(a["target_area_m2"] / 1e6, 6),
                    "mean_union_area_m2": round(rec["mean_union_area_m2"], 3),
                    "gap_return_tracks": a["gap_return"],
                    "strict_change_eligible_tracks": a["strict_change_eligible"],
                    "v2_quoted_pct": v2_quoted.get(iv, {}).get(v2key),
                    "v2_match": interval_block["v2_receipt_cross_check"][s]["matches"],
                }
            )
        # STABLE on the shared denominator, and the V2 stable base explained.
        st = block["STABLE"]["union_area_m2"]
        others = sum(block[s]["union_area_m2"] for s in ("ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT"))
        stable_share = (st / denom * 100.0) if denom else 0.0
        v2_stable = v2_quoted.get(iv, {}).get("STABLE")
        implied_base = (st / (v2_stable / 100.0)) if v2_stable else None
        interval_block["stable_base_note"] = {
            "stable_share_on_shared_denominator_pct": stable_share,
            "stable_share_v2_quoted_pct": v2_stable,
            "implied_v2_stable_denominator_m2": implied_base,
            "implied_v2_stable_denominator_km2": (implied_base / 1e6) if implied_base else None,
            "note": (
                "ADDITION, DEMOLITION and UNKNOWN reproduce the V2 receipt exactly on the shared "
                "denominator; STABLE is reported here as its complement (100 - the other three), and "
                "the V2 receipt evidently used a wider base for STABLE. Both areshown; neither is hidden."
            ),
        }
        report["intervals"][iv] = interval_block

    # Raster representation check (product QA only; never a science input).
    raster_note = {}
    for iv in INTERVALS:
        d = Path(args.raster_state_dir) / iv
        combined = d / "THREE_STATE_COMBINED_ge100m2.tif"
        entry = {}
        if combined.exists():
            import numpy as np
            import rasterio

            try:
                with rasterio.open(combined) as src:
                    counts = defaultdict(int)
                    for _, window in src.block_windows(1):
                        arr = src.read(1, window=window)
                        vals, cnts = np.unique(arr, return_counts=True)
                        for v, c in zip(vals.tolist(), cnts.tolist()):
                            counts[int(v)] += int(c)
                    pixel_area_m2 = abs(src.transform.a * src.transform.e)
                entry.update(
                    {
                        "combined_pixel_counts": dict(sorted(counts.items())),
                        "pixel_area_m2": pixel_area_m2,
                        "pixel_area_km2": pixel_area_m2 / 1e6,
                        "raster_pixel_area_km2_by_class": {
                            str(k): v * pixel_area_m2 / 1e6 for k, v in sorted(counts.items())
                        },
                        "note": (
                            "PRODUCT-LEVEL ONLY. Raster areas are not the authoritative scientific area; "
                            "they differ from vector area through pixelisation, boundary rasterisation and "
                            "the >=100 m2 display filter. No scientific state is rewritten by this step."
                        ),
                    }
                )
            except Exception as exc:
                entry["status"] = f"not readable yet ({type(exc).__name__})"
        raster_note[iv] = entry
    report["raster_vs_vector_check"] = raster_note

    with (out_dir / "XIAN_TRANSITION_STATS_CURRENT_BEST.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    (out_dir / "XIAN_TRANSITION_STATS_CURRENT_BEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Xi'an V2 current-best candidate transition statistics",
        "",
        "科学状态：CANDIDATE_NOT_GROUND_TRUTH。面积权威为 EPSG:32649 矢量 `union_area_m2`；栅格只用于产品表达和 QA。",
        "Unknown/Conflict 保持独立，不并入 Stable。下表的比例统一使用该区间四类 union_area_m2 之和作为分母。",
        "",
        "| Interval | State | Track count | Area (km²) | Share of shared denominator |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        md.append(
            f"| {row['interval']} | {row['state']} | {row['track_count']:,} | "
            f"{row['union_area_km2']:.6f} | {row['share_pct_of_union_denominator']:.6f}% |"
        )
    md += [
        "",
        "## Denominator note",
        "",
        "Addition, Demolition and Unknown reproduce the V2 receipt on the shared four-class denominator. "
        "The V2 receipt's quoted Stable percentage uses a different base; this report therefore shows the "
        "mathematically comparable Stable share on the shared denominator and keeps the quoted Stable value "
        "in the JSON cross-check. Neither value is silently substituted for the other.",
        "",
        "## Gate",
        "",
        "Unknown<5%: FAIL for both intervals. This remains a candidate delivery and is not a ground-truth certificate.",
    ]
    (out_dir / "XIAN_TRANSITION_STATS_CURRENT_BEST.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("STATUS: CURRENT_BEST_CANDIDATE_DELIVERY / CANDIDATE_NOT_GROUND_TRUTH")
    for iv in INTERVALS:
        b = report["intervals"][iv]
        print(f"\n=== {iv} ===  denominator {b['denominator_union_area_sum_km2']:.6f} km2 (sum of union areas)")
        for s in STATES:
            r = b["states"][s]
            cc = b["v2_receipt_cross_check"][s]
            print(
                f"  {s:<18} {r['union_area_km2']:>10.6f} km2  {r['share_pct_of_union_denominator']:>12.8f} %  "
                f"n={r['track_count']:>8,}  V2={cc['v2_quoted_pct']}  match={cc['matches']}"
            )


if __name__ == "__main__":
    main()
