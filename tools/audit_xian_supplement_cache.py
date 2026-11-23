"""Register Xi'an supplemental Google Earth cache evidence.

The directory supplied by the user is not a normal JPEG/TIFF tile dump.  It
contains Google Earth Historical quadtree/cache packets and ``.datever``
records.  This tool treats those files as an independent supplemental
observation source, decodes the packed acquisition dates, and writes a small,
portable evidence registry for the cloud pipeline.

No source file is modified.  The registry deliberately keeps metadata-only
support separate from pixel observations: a cache packet can support date,
coverage, same-acquisition deduplication, and identity/observability review,
but it cannot be used as a geometry owner when no image tile is present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKED_DATE_SENTINEL_MAX = 545
LEAF_KEY_RE = re.compile(r"^0[0-3]{19}$")
QUAD_PACKET_RE = re.compile(r"^q-(0[0-3]+)\.dat(?:\.raw)?$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packed_date(value: int) -> str | None:
    """Decode Google's bit-packed acquisition date.

    The first field in a ``.datever`` record is ``(year << 9) | (month << 5)
    | day``.  Values at or below 545 are protocol sentinels, not dates.
    """

    if value <= PACKED_DATE_SENTINEL_MAX:
        return None
    year = value >> 9
    month = (value >> 5) & 0x0F
    day = value & 0x1F
    try:
        return date_type(year, month, day).isoformat()
    except ValueError:
        return None


def tile_key_from_name(name: str) -> str | None:
    path = Path(name)
    if path.suffix.lower() in {".raw", ".datever"}:
        stem = path.stem
        return stem if LEAF_KEY_RE.fullmatch(stem) else None
    match = QUAD_PACKET_RE.fullmatch(path.name)
    return match.group(1) if match else None


def iter_datever(path: Path) -> Iterable[dict[str, Any]]:
    data = path.read_bytes()
    usable = len(data) - (len(data) % 16)
    for offset in range(0, usable, 16):
        packed, version, layer, reserved = struct.unpack_from("<IIII", data, offset)
        yield {
            "record_offset": offset,
            "packed_date": packed,
            "actual_date": packed_date(packed),
            "version": version,
            "layer": layer,
            "reserved": reserved,
        }


def inventory_digest(files: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix().lower()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        stat = path.stat()
        digest.update(rel)
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(root: Path, output_root: Path) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    date_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    tile_date_rows: list[dict[str, Any]] = []
    folder_summaries: list[dict[str, Any]] = []
    duplicate_index: dict[tuple[str, str], set[str]] = defaultdict(set)

    folders = sorted(path for path in root.iterdir() if path.is_dir())
    for folder in folders:
        files = [path for path in folder.iterdir() if path.is_file()]
        extension_counts = Counter(path.suffix.lower() for path in files)
        datever_files = [path for path in files if path.suffix.lower() == ".datever"]
        pixel_files = [path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}]
        actual_counts: Counter[str] = Counter()
        tile_keys: set[str] = set()
        record_count = 0
        for path in datever_files:
            tile_key = tile_key_from_name(path.name)
            if tile_key is None:
                continue
            tile_keys.add(tile_key)
            for record in iter_datever(path):
                actual = record["actual_date"]
                record_count += 1
                if actual is None:
                    continue
                actual_counts[actual] += 1
                duplicate_index[(tile_key, actual)].add(folder.name)
                tile_date_rows.append(
                    {
                        "source_asset_id": f"xian_supplement::{folder.name}",
                        "folder_label": folder.name,
                        "tile_key": tile_key,
                        "actual_acquisition_date": actual,
                        "packed_date": record["packed_date"],
                        "version": record["version"],
                        "layer": record["layer"],
                        "datever_file": str(path),
                        "record_offset": record["record_offset"],
                        "pixel_payload_present": bool(pixel_files),
                    }
                )

        file_digest = inventory_digest(files, root)
        payload_status = "PIXEL_TILES_PRESENT" if pixel_files else "METADATA_ONLY_NO_PIXEL_TILES"
        observation_rows.append(
            {
                "observation_id": f"xian_supplement::{folder.name}",
                "source_asset_id": f"xian_supplement::{folder.name}",
                "source_ref": str(folder),
                "folder_label_not_acquisition_date": folder.name,
                "observation_kind": "SUPPLEMENTAL_GOOGLE_EARTH_CACHE",
                "pixel_payload_present": bool(pixel_files),
                "quality_status": payload_status,
                "file_count": len(files),
                "pixel_file_count": len(pixel_files),
                "tile_key_count": len(tile_keys),
                "datever_record_count": record_count,
                "actual_dates": ";".join(sorted(actual_counts)),
                "inventory_sha256": file_digest,
                "raw_inputs_modified": False,
            }
        )
        folder_summaries.append(
            {
                "folder": folder.name,
                "source_asset_id": f"xian_supplement::{folder.name}",
                "payload_status": payload_status,
                "file_count": len(files),
                "extension_counts": dict(sorted(extension_counts.items())),
                "pixel_file_count": len(pixel_files),
                "tile_key_count": len(tile_keys),
                "datever_record_count": record_count,
                "actual_date_record_counts": dict(sorted(actual_counts.items())),
                "actual_dates": sorted(actual_counts),
                "inventory_sha256": file_digest,
            }
        )

    duplicate_rows = [
        {
            "tile_key": tile_key,
            "actual_acquisition_date": actual,
            "folder_count": len(folders_for_key),
            "folders": ";".join(sorted(folders_for_key)),
            "duplicate_acquisition": len(folders_for_key) > 1,
        }
        for (tile_key, actual), folders_for_key in sorted(duplicate_index.items())
    ]
    for row in tile_date_rows:
        folders_for_key = duplicate_index[(row["tile_key"], row["actual_acquisition_date"])]
        row["same_acquisition_duplicate_folder_count"] = len(folders_for_key)
        row["same_acquisition_duplicate"] = len(folders_for_key) > 1

    payload_statuses = {row["payload_status"] for row in folder_summaries}
    overall_payload = "PIXEL_TILES_PRESENT" if "PIXEL_TILES_PRESENT" in payload_statuses else "METADATA_ONLY_NO_PIXEL_TILES"
    manifest = {
        "schema_version": "urrs.xian.supplement.observation_registry.v2",
        "created_at_utc": utc_now(),
        "root": str(root),
        "role": [
            "actual_date_support",
            "same_acquisition_source_collapse",
            "same_date_deduplication_evidence",
            "alignment_parallax_support",
            "identity_tracking_support",
            "observability_support",
        ],
        "formal_epoch_policy": "supplemental observations are never a fourth formal epoch",
        "overall_payload_status": overall_payload,
        "geometry_owner": "MAIN_TIF_B0_ONLY",
        "pixel_support_available": overall_payload == "PIXEL_TILES_PRESENT",
        "date_decode": "Google Earth packed date (year << 9) | (month << 5) | day; <=545 sentinels ignored",
        "duplicate_policy": "same tile_key + actual acquisition date is one acquisition; retain folder provenance and do not collapse distinct actual dates",
        "raw_inputs_modified": False,
        "folders": folder_summaries,
        "observation_count": len(observation_rows),
        "tile_date_evidence_count": len(tile_date_rows),
        "same_acquisition_duplicate_count": sum(1 for row in duplicate_rows if row["duplicate_acquisition"]),
        "source_sha256_scope": "inventory_sha256 is path-plus-size inventory digest; pixel content SHA is not asserted when pixel tiles are absent",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "supplement_observation_registry.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_root / "supplement_observations.csv", observation_rows)
    write_csv(output_root / "supplement_tile_date_evidence.csv", tile_date_rows)
    write_csv(output_root / "supplement_same_acquisition_duplicates.csv", duplicate_rows)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = audit(args.root, args.output_root)
    print(json.dumps({
        "status": "PASS_SUPPLEMENT_REGISTERED",
        "payload_status": manifest["overall_payload_status"],
        "observation_count": manifest["observation_count"],
        "tile_date_evidence_count": manifest["tile_date_evidence_count"],
        "same_acquisition_duplicate_count": manifest["same_acquisition_duplicate_count"],
        "output_root": str(args.output_root.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
