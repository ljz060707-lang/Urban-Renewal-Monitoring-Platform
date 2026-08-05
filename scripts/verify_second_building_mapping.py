from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify SECOND building class mapping without reading Test images."
    )
    parser.add_argument(
        "--building-class-id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/change_public_v2_multisource_windows.yaml"),
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("outputs/v2_readiness/second_mapping_review"),
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path(
            r"D:\Data Set\SECOND\mapping\SECOND_MAPPING_VERIFIED.json"
        ),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    review_dir = args.review_dir.resolve()
    mapping_output = args.mapping_output

    stats_path = review_dir / "SECOND_CLASS_STATS.csv"
    audit_path = review_dir / "SECOND_MAPPING_REVIEW_AUDIT.json"
    sheet_path = review_dir / f"class_{args.building_class_id}_review.jpg"

    # ---------------------------------------------------------
    # Evidence checks
    # ---------------------------------------------------------
    required = [
        config_path,
        stats_path,
        audit_path,
        sheet_path,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required verification evidence is missing:\n"
            + "\n".join(missing)
        )

    # ---------------------------------------------------------
    # Verify selected class really exists in audited labels
    # ---------------------------------------------------------
    observed_ids: set[int] = set()

    with stats_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or "class_id" not in reader.fieldnames:
            raise RuntimeError(
                f"class_id column missing from {stats_path}"
            )

        for row in reader:
            try:
                observed_ids.add(int(row["class_id"]))
            except (TypeError, ValueError):
                continue

    if args.building_class_id not in observed_ids:
        raise ValueError(
            f"Class {args.building_class_id} is not present in "
            f"audited semantic IDs: {sorted(observed_ids)}"
        )

    if args.building_class_id == 255:
        raise ValueError("255 is an ignore/void class and cannot be building.")

    # ---------------------------------------------------------
    # Verify mapping review was Test-safe
    # ---------------------------------------------------------
    audit = json.loads(
        audit_path.read_text(encoding="utf-8")
    )

    test_reads = audit.get("test_image_reads")

    if test_reads != 0:
        raise RuntimeError(
            f"Mapping review is not Test-safe: test_image_reads={test_reads}"
        )

    # ---------------------------------------------------------
    # Write verified mapping
    # ---------------------------------------------------------
    mapping_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = {
        "dataset": "SECOND",
        "mapping_verified": True,
        "verified_by_user": True,
        "building_class_id": args.building_class_id,
        "semantic_ids_observed": sorted(observed_ids),
        "ignore_index": 255,
        "verification_scope": "building class only",
        "other_numeric_semantic_names_inferred": False,
        "test_image_reads": 0,
        "evidence": {
            "review_audit": str(audit_path),
            "review_audit_sha256": sha256_file(audit_path),
            "class_stats": str(stats_path),
            "class_stats_sha256": sha256_file(stats_path),
            "building_review_sheet": str(sheet_path),
            "building_review_sheet_sha256": sha256_file(sheet_path),
        },
        "verified_at": datetime.now().astimezone().isoformat(),
    }

    mapping_output.write_text(
        json.dumps(
            mapping,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Backup config before changing it
    # ---------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_path = config_path.with_name(
        f"{config_path.stem}.before_second_mapping_{stamp}{config_path.suffix}"
    )

    shutil.copy2(
        config_path,
        backup_path,
    )

    # ---------------------------------------------------------
    # Patch canonical V2 YAML location:
    #
    # datasets:
    #   second:
    # ---------------------------------------------------------
    cfg = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if cfg is None:
        cfg = {}

    datasets = cfg.setdefault("datasets", {})

    if not isinstance(datasets, dict):
        raise RuntimeError("config['datasets'] is not a mapping.")

    second = datasets.setdefault("second", {})

    if not isinstance(second, dict):
        raise RuntimeError("config['datasets']['second'] is not a mapping.")

    second["mapping_verified"] = True
    second["building_class_id"] = int(args.building_class_id)
    second["class_mapping_file"] = str(mapping_output).replace("\\", "/")

    # Keep SECOND Test protected.
    second["allow_test"] = False

    # If an old duplicate top-level `second:` section exists,
    # keep it synchronized rather than leaving contradictory settings.
    if isinstance(cfg.get("second"), dict):
        cfg["second"]["mapping_verified"] = True
        cfg["second"]["building_class_id"] = int(args.building_class_id)
        cfg["second"]["class_mapping_file"] = str(mapping_output).replace(
            "\\", "/"
        )

    config_path.write_text(
        yaml.safe_dump(
            cfg,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("SECOND BUILDING MAPPING VERIFIED")
    print("=" * 60)
    print(f"building_class_id : {args.building_class_id}")
    print(f"observed IDs      : {sorted(observed_ids)}")
    print(f"mapping file      : {mapping_output}")
    print(f"config            : {config_path}")
    print(f"config backup     : {backup_path}")
    print("SECOND Test reads : 0")
    print("allow_test        : false")
    print("=" * 60)


if __name__ == "__main__":
    main()
