"""Small, interpretable threshold calibration for the held-edge recovery rule."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", type=Path, required=True)
    ap.add_argument("--formal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    review = pd.read_csv(args.repair / "REPAIRED_AFFECTED_VISUAL_REVIEW.csv", encoding="utf-8-sig", low_memory=False)
    review = review[review["recovery_decision"].isin(["SAFE_RECOVERY", "PARALLAX_SAFE_RECOVERY", "INSUFFICIENT_EVIDENCE"])].copy()
    review["label"] = review["recovery_decision"].ne("INSUFFICIENT_EVIDENCE")
    numeric = ["score", "buffered_iou", "area_ratio", "neighborhood_consistency", "alignment_aware_displacement_m"]
    for c in numeric:
        review[c] = pd.to_numeric(review[c], errors="coerce")

    # A simple conjunction is intentionally used: it is auditable and can be
    # applied to all held rows without training a black-box classifier.
    rows = []
    n_pos = int(review["label"].sum())
    n_neg = int((~review["label"]).sum())
    for score, iou, area, neigh, disp in itertools.product(
        [0.50, 0.55, 0.58, 0.60], [0.30, 0.40, 0.45, 0.50, 0.55], [0.35, 0.45, 0.50, 0.55], [0.40, 0.50, 0.55, 0.60], [30.0, 50.0]
    ):
        pred = (
            (review["score"] >= score)
            & (review["buffered_iou"] >= iou)
            & (review["area_ratio"] >= area)
            & (review["neighborhood_consistency"] >= neigh)
            & (review["alignment_aware_displacement_m"] <= disp)
        )
        tp = int((pred & review["label"]).sum())
        fp = int((pred & ~review["label"]).sum())
        rows.append({
            "rule": "REGULAR_CONJUNCTION",
            "score_min": score, "buffered_iou_min": iou, "area_ratio_min": area,
            "neighborhood_min": neigh, "displacement_max": disp,
            "tp": tp, "fp": fp, "positive_recall_pct": 100 * tp / n_pos,
            "false_positive_rate_pct": 100 * fp / n_neg,
        })
    out = pd.DataFrame(rows).sort_values(["false_positive_rate_pct", "positive_recall_pct"], ascending=[True, False])

    # A parallax branch is only considered when the displacement is outside
    # the regular envelope but bounded; it still requires all other metrics.
    par_rows = []
    par_review = review[review["recovery_decision"].eq("PARALLAX_SAFE_RECOVERY") | review["recovery_decision"].eq("INSUFFICIENT_EVIDENCE")].copy()
    par_review["par_label"] = par_review["recovery_decision"].eq("PARALLAX_SAFE_RECOVERY")
    par_pos = int(par_review["par_label"].sum())
    par_neg = int((~par_review["par_label"]).sum())
    for score, iou, area, neigh, dmin, dmax in itertools.product([0.55, 0.60, 0.65], [0.40, 0.50, 0.55], [0.45, 0.55, 0.60], [0.50, 0.60, 0.70], [45.0, 50.0], [75.0, 90.0]):
        pred = (
            (par_review["score"] >= score)
            & (par_review["buffered_iou"] >= iou)
            & (par_review["area_ratio"] >= area)
            & (par_review["neighborhood_consistency"] >= neigh)
            & (par_review["alignment_aware_displacement_m"] > dmin)
            & (par_review["alignment_aware_displacement_m"] <= dmax)
        )
        tp = int((pred & par_review["par_label"]).sum())
        fp = int((pred & ~par_review["par_label"]).sum())
        par_rows.append({
            "rule": "PARALLAX_CONJUNCTION", "score_min": score, "buffered_iou_min": iou,
            "area_ratio_min": area, "neighborhood_min": neigh, "displacement_min": dmin,
            "displacement_max": dmax, "tp": tp, "fp": fp,
            "positive_recall_pct": 100 * tp / par_pos if par_pos else 0,
            "false_positive_rate_pct": 100 * fp / par_neg if par_neg else 0,
        })
    par = pd.DataFrame(par_rows).sort_values(["false_positive_rate_pct", "positive_recall_pct"], ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    out.head(100).to_csv(stem.with_name(stem.name + "_regular.csv"), index=False, encoding="utf-8-sig")
    par.head(100).to_csv(stem.with_name(stem.name + "_parallax.csv"), index=False, encoding="utf-8-sig")
    print(out.head(10).to_string(index=False))
    print(par.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
