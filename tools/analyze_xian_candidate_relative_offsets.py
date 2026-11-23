#!/usr/bin/env python3
"""Read-only relative-shift diagnostic using repeated external polygon IDs.

The external layer's absolute accuracy is unknown. Repeated source FIDs can
only provide a differential reference across actual image dates; this tool
never applies the resulting vectors to footprints or products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import fiona
from shapely.geometry import shape


DEFAULT_RUN = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction\B04-TQA10_externalcandidate_02")
DEFAULT_OUTPUT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction\external_relative_offsets_B04_02")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mad(values: list[float]) -> float | None:
    if not values:
        return None
    center = median(values)
    return median([abs(value - center) for value in values])


def analyze(run_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite relative-offset output: {output_dir}")
    receipt = json.loads((run_dir / "EVIDENCE_RECEIPT.json").read_text(encoding="utf-8"))
    source = receipt.get("external_footprint_candidate_source")
    if not source or source.get("ground_truth") is not False:
        raise ValueError("Run does not contain an explicitly non-ground external candidate source")
    source_path = Path(source["source_path"])
    current_hashes = {}
    for name, expected in source["source_component_sha256"].items():
        path = source_path.with_name(name)
        current_hashes[name] = _sha256(path)
        if current_hashes[name] != expected:
            raise ValueError(f"External source component changed since reconstruction: {path}")

    observations = {item["uid"]: item for item in receipt["observations"]}
    roofs = {}
    with fiona.open(run_dir / "building_lineage.gpkg", layer="roof_candidates") as layer:
        for feature in layer:
            roofs[str(feature["properties"]["uid"])] = shape(feature["geometry"])

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with fiona.open(run_dir / "building_lineage.gpkg", layer="footprint_candidates") as layer:
        for feature in layer:
            props = dict(feature["properties"])
            uid = str(props["uid"])
            if uid not in observations or uid not in roofs or props.get("status") != "UNVERIFIED_CANDIDATE":
                continue
            obs = observations[uid]
            track_id = obs.get("candidate_identity")
            candidate_id = str(props["candidate_id"])
            if not track_id or not candidate_id or props.get("source_date") is not None:
                continue
            external = shape(feature["geometry"])
            roof = roofs[uid]
            grouped[(str(track_id), candidate_id, str(props["actual_date"]))].append({
                "uid": uid,
                "epoch": props["epoch"],
                "actual_date": props["actual_date"],
                "intersection_over_union": float(props["match_iou"]),
                "external_minus_roof_centroid_dx_m": float(external.centroid.x - roof.centroid.x),
                "external_minus_roof_centroid_dy_m": float(external.centroid.y - roof.centroid.y),
            })

    selected = []
    for (track_id, candidate_id, actual_date), rows in sorted(grouped.items()):
        best = max(rows, key=lambda item: (item["intersection_over_union"], item["uid"]))
        selected.append({"physical_track_candidate": track_id, "external_fid": candidate_id,
                         "actual_date": actual_date, "alternative_observations_same_date": len(rows) - 1,
                         **best})

    by_track_fid: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        by_track_fid[(row["physical_track_candidate"], row["external_fid"])].append(row)
    repeated = []
    changes_by_track_date_pair: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for (track_id, fid), rows in sorted(by_track_fid.items()):
        rows.sort(key=lambda item: item["actual_date"])
        if len(rows) < 2:
            continue
        repeated.append({"physical_track_candidate": track_id, "external_fid": fid,
                         "observations": rows})
        for earlier, later in zip(rows, rows[1:]):
            # For a static external polygon, the negative offset change is the
            # apparent roof displacement relative to that polygon.
            dx = earlier["external_minus_roof_centroid_dx_m"] - later["external_minus_roof_centroid_dx_m"]
            dy = earlier["external_minus_roof_centroid_dy_m"] - later["external_minus_roof_centroid_dy_m"]
            changes_by_track_date_pair[(earlier["actual_date"], later["actual_date"], track_id)].append({
                "external_fid": fid,
                "apparent_roof_shift_relative_to_static_polygon_dx_m": dx,
                "apparent_roof_shift_relative_to_static_polygon_dy_m": dy,
                "earlier_iou": earlier["intersection_over_union"],
                "later_iou": later["intersection_over_union"],
            })

    changes_by_date_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (earlier, later, track_id), rows in sorted(changes_by_track_date_pair.items()):
        dxs = [row["apparent_roof_shift_relative_to_static_polygon_dx_m"] for row in rows]
        dys = [row["apparent_roof_shift_relative_to_static_polygon_dy_m"] for row in rows]
        changes_by_date_pair[(earlier, later)].append({
            "physical_track_candidate": track_id,
            "candidate_fids": sorted({row["external_fid"] for row in rows}),
            "track_shift_dx_m": median(dxs), "track_shift_dy_m": median(dys),
            "supporting_fid_links": len(rows),
            "fid_link_details": rows,
        })

    pair_summaries = []
    for (earlier, later), rows in sorted(changes_by_date_pair.items()):
        dxs = [row["track_shift_dx_m"] for row in rows]
        dys = [row["track_shift_dy_m"] for row in rows]
        pair_summaries.append({
            "earlier_actual_date": earlier, "later_actual_date": later,
            "distinct_track_candidates": len(rows),
            "apparent_shift_median_dx_m": median(dxs), "apparent_shift_median_dy_m": median(dys),
            "apparent_shift_mad_dx_m": _mad(dxs), "apparent_shift_mad_dy_m": _mad(dys),
            "pair_details": rows,
        })

    output = {
        "schema_version": "xian_r5.relative_candidate_shift_diagnostic.v2",
        "source_run": str(run_dir), "source_layer": source_path.as_posix(),
        "source_component_sha256_revalidated": current_hashes,
        "source_capture_date": source.get("source_capture_date"),
        "source_accuracy_m": source.get("source_accuracy_m"),
        "ground_truth": False,
        "matching_basis": "same_external_FID_within_same_R4_candidate_track",
        "selected_track_fid_date_observations": selected,
        "repeated_track_fids": repeated,
        "adjacent_actual_date_pair_summaries": pair_summaries,
        "interpretation": "DIFFERENTIAL_DIAGNOSTIC_ONLY_NOT_GROUND_CORRECTION",
        "registration_or_footprint_translation_applied": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "RELATIVE_OFFSET_DIAGNOSTIC.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.output_dir)
    print(json.dumps({
        "repeated_track_fids": len(result["repeated_track_fids"]),
        "actual_date_pairs": [{key: row[key] for key in ("earlier_actual_date", "later_actual_date", "distinct_track_candidates", "apparent_shift_median_dx_m", "apparent_shift_median_dy_m", "apparent_shift_mad_dx_m", "apparent_shift_mad_dy_m")} for row in result["adjacent_actual_date_pair_summaries"]],
        "interpretation": result["interpretation"],
    }, ensure_ascii=False))
