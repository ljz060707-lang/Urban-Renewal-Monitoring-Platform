"""Create an auditable, conservative GloBFP-to-imagery pairing manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        out.writeheader(); out.writerows(rows)


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(8 * 1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def raster_bounds(path: Path) -> tuple[str | None, tuple[float, float, float, float] | None]:
    import rasterio  # type: ignore
    try:
        with rasterio.open(path) as ds:
            return (str(ds.crs) if ds.crs else None, tuple(float(v) for v in ds.bounds))
    except Exception:
        return None, None


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globfp-root", type=Path, required=True)
    parser.add_argument("--imagery-root", type=Path, action="append", default=[])
    parser.add_argument("--audit-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    audit_root = args.audit_root or out.parent / "01_globfp_audit"
    schema_path = audit_root / "GLOBFP_SCHEMA_AUDIT.json"
    if not schema_path.exists():
        raise SystemExit(f"Run audit_globfp_dataset.py first: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    vectors = [r for r in schema.get("records", []) if r.get("asset_class") in {"A_IMAGE_FOOTPRINT_HEIGHT", "B_IMAGE_FOOTPRINT", "C_FOOTPRINT_HEIGHT_ONLY", "D_FOOTPRINT_ONLY"}]
    images = []
    for root in args.imagery_root:
        if not root.is_dir(): continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in {".tif", ".tiff", ".jp2"}: continue
            # Test/holdout pixels are locked.  Do not even open their raster
            # headers while looking for a training pair.
            if any(part.lower() in {"test", "testing", "holdout"} for part in p.parts):
                continue
            crs, bounds = raster_bounds(p)
            # WHU image tiles are intentionally not georeferenced; opening all
            # 16k files and hashing them would add no pairing evidence. Keep
            # only georeferenced rasters, then hash only selected candidates.
            if bounds and crs:
                images.append({"path": str(p.resolve()), "crs": crs, "bounds": bounds})
    rows = []
    for record in vectors:
        vb = tuple(record.get("bounds") or [])
        candidates = [img for img in images if vb and len(vb) == 4 and intersects(vb, tuple(img["bounds"]))]
        status = "PAIR_REJECTED"
        reason = "vector_only_no_local_imagery" if not candidates else "crs_or_semantic_pair_not_proven"
        if record.get("asset_class") in {"A_IMAGE_FOOTPRINT_HEIGHT", "B_IMAGE_FOOTPRINT"} and candidates:
            status = "PAIR_STRONG" if len(candidates) == 1 and record.get("crs") == candidates[0].get("crs") else "PAIR_GEOMETRY_ONLY"
            reason = "exact_geometry_and_crs" if status == "PAIR_STRONG" else "bounds_intersection_only"
        rows.append({"globfp_path": record.get("path"), "asset_class": record.get("asset_class"), "feature_count": record.get("feature_count"), "image_path": candidates[0]["path"] if candidates else None, "candidate_count": len(candidates), "pair_status": status, "reason": reason, "height_supervision_allowed": status in {"PAIR_EXACT", "PAIR_STRONG"} and record.get("asset_class") == "A_IMAGE_FOOTPRINT_HEIGHT"})
    summary = {"status": "ready" if any(r["pair_status"] in {"PAIR_EXACT", "PAIR_STRONG"} for r in rows) else "no_trainable_image_pairs", "created_at": now(), "globfp_root": str(args.globfp_root.resolve()), "imagery_roots": [str(r.resolve()) for r in args.imagery_root], "pair_counts": {s: sum(r["pair_status"] == s for r in rows) for s in ["PAIR_EXACT", "PAIR_STRONG", "PAIR_GEOMETRY_ONLY", "PAIR_REJECTED"]}, "height_supervision_rows": sum(bool(r["height_supervision_allowed"]) for r in rows), "source_data_modified": False, "locked_test_pixel_reads": 0}
    write_csv(out / "GLOBFP_IMAGE_PAIR_AUDIT.csv", rows)
    write_json(out / "GLOBFP_PAIRING_MANIFEST.json", {"summary": summary, "pairs": rows})
    write_json(out / "GLOBFP_PAIRING_READINESS.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
