from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


FORMAL_TRAINING_STATES = {
    "FORMAL_TRAINING_ALLOWED",
    "FORMAL_TRAINING_BLOCKED_MISSING_IMAGES",
    "FORMAL_TRAINING_BLOCKED_LABEL_MAPPING",
    "FORMAL_TRAINING_BLOCKED_SPLIT",
    "FORMAL_TRAINING_BLOCKED_LICENSE",
    "FORMAL_TRAINING_BLOCKED_OTHER",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_status(gate: dict, package: dict, split: dict) -> str:
    if not gate.get("pairing_pass") or int(package.get("unpaired_label_count", 0)) > 0:
        return "FORMAL_TRAINING_BLOCKED_MISSING_IMAGES"
    if not gate.get("license_verified") or str(package.get("license", "")).startswith("not_declared"):
        return "FORMAL_TRAINING_BLOCKED_LICENSE"
    if not package.get("rgb_band_semantics_verified"):
        return "FORMAL_TRAINING_BLOCKED_LABEL_MAPPING"
    if not gate.get("split_leakage_pass") or not split.get("test_manifest_locked"):
        return "FORMAL_TRAINING_BLOCKED_SPLIT"
    if not gate.get("formal_ucs_training_allowed"):
        return "FORMAL_TRAINING_BLOCKED_OTHER"
    return "FORMAL_TRAINING_ALLOWED"


def _blocking_statuses(gate: dict, package: dict, split: dict) -> list[str]:
    statuses = []
    if not gate.get("pairing_pass") or int(package.get("unpaired_label_count", 0)) > 0:
        statuses.append("FORMAL_TRAINING_BLOCKED_MISSING_IMAGES")
    if not gate.get("license_verified") or str(package.get("license", "")).startswith("not_declared"):
        statuses.append("FORMAL_TRAINING_BLOCKED_LICENSE")
    if not package.get("rgb_band_semantics_verified"):
        statuses.append("FORMAL_TRAINING_BLOCKED_LABEL_MAPPING")
    if not gate.get("split_leakage_pass") or not split.get("test_manifest_locked"):
        statuses.append("FORMAL_TRAINING_BLOCKED_SPLIT")
    if not gate.get("formal_ucs_training_allowed") and not statuses:
        statuses.append("FORMAL_TRAINING_BLOCKED_OTHER")
    return statuses


def build_formal_training_readiness(audit_dir: Path, output_dir: Path | None = None) -> dict:
    """Build a read-only readiness receipt from an existing audit directory."""
    audit_dir = Path(audit_dir).resolve()
    gate = _load(audit_dir / "ONBOARDING_GATE.json")
    discovery = _load(audit_dir / "DATA_DISCOVERY.json")
    package = _load(audit_dir / "UCS_PACKAGE_AUDIT.json")
    relationship = _load(audit_dir / "UCS_RECORD_RELATIONSHIP.json")
    split = _load(audit_dir / "UCS_SPLIT_AUDIT.json")
    cwld = _load(audit_dir / "CWLD_REUSE_AUDIT.json")

    status = _primary_status(gate, package, split)
    blocking_statuses = _blocking_statuses(gate, package, split)
    if status not in FORMAL_TRAINING_STATES:
        raise ValueError(f"invalid formal training status: {status}")
    output_dir = Path(output_dir) if output_dir else audit_dir.parent / f"construction_formal_readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = int(split.get("counts", {}).get("train", 0))
    validation_count = int(split.get("counts", {}).get("validation", 0))
    test_count = int(split.get("counts", {}).get("test", 0))
    readiness = {
        "status": status,
        "allowed_status_values": sorted(FORMAL_TRAINING_STATES),
        "blocking_statuses": blocking_statuses,
        "formal_training_allowed": bool(gate.get("formal_ucs_training_allowed")),
        "formal_validation_allowed": False,
        "formal_validation_status": "BLOCKED_BY_FORMAL_TRAINING_GATE",
        "audit_dir": str(audit_dir),
        "ucs_resolved_root": discovery.get("resolved_ucs_root", package.get("resolved_root")),
        "cwld_resolved_root": discovery.get("resolved_cwld_root"),
        "data_record_version": package.get("data_version"),
        "record_relationship": relationship,
        "real_images": {
            "status": "partial" if int(package.get("unpaired_label_count", 0)) else "complete",
            "unique_tiff_images": int(package.get("imagery_count", 0)),
            "unique_pixel_labels": int(package.get("label_count", 0)),
            "paired_unique_records": int(package.get("paired_count", 0)),
            "unpaired_label_count": int(package.get("unpaired_label_count", 0)),
            "unpaired_image_count": int(package.get("unpaired_image_count", 0)),
        },
        "pixel_labels": {
            "available": bool(package.get("label_count", 0)),
            "encoding": package.get("label_encoding"),
            "class_names": package.get("class_names", []),
            "uncertain_encoding": "not separately documented in audited package; do not infer uncertain as background",
        },
        "image_label_pairing": {
            "pass": bool(package.get("pairing_pass")),
            "matching_unique_count": int(package.get("paired_count", 0)),
            "unmatched_label_count": int(package.get("unpaired_label_count", 0)),
            "unmatched_image_count": int(package.get("unpaired_image_count", 0)),
        },
        "scene_identity": {
            "city": "not documented as a formal training split field",
            "aoi": "not documented as a formal training split field",
            "source_scene": "GeoTIFF metadata and local package records available; RGB semantics not verified",
        },
        "split": {
            "train": train_count,
            "validation": validation_count,
            "locked_test": test_count,
            "split_mode": split.get("split_mode"),
            "hash_overlap_count": split.get("hash_overlap_count"),
            "group_overlap_count": split.get("group_overlap_count"),
            "test_manifest_locked": bool(split.get("test_manifest_locked")),
            "test_pixel_reads": split.get("test_pixel_reads", 0),
        },
        "neighbor_tiles_and_augmentation": {
            "status": "not_formally_audited_for_augmentation_grouping",
            "required_rule": "same source scene, neighboring tiles, and augmentation descendants must remain in one split",
        },
        "license": {
            "verified": bool(gate.get("license_verified")),
            "status": relationship.get("license_status", package.get("license")),
            "article_license_not_assumed_for_dataset": True,
        },
        "rgb_band_semantics": {
            "verified": bool(package.get("rgb_band_semantics_verified")),
            "status": "undocumented_in_local_package" if not package.get("rgb_band_semantics_verified") else "verified",
        },
        "cwld_reuse": {
            "status": cwld.get("reuse_status"),
            "prohibited_interpretation": cwld.get("prohibited_interpretation"),
        },
        "test_pixel_reads": gate.get("test_pixel_reads", {"UCS": 0, "CWLD": 0, "WHU": 0, "SECOND": 0}),
        "original_data_unchanged": bool(gate.get("original_data_unchanged", True)),
        "current_urrs_release_unchanged": bool(gate.get("current_urrs_release_unchanged", True)),
        "blocking_reasons": gate.get("blocking_reasons", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    (output_dir / "FORMAL_TRAINING_READINESS.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")

    status_lines = [
        "# UCS formal training readiness",
        "",
        f"- Status: `{status}`",
        f"- UCS resolved root: `{readiness['ucs_resolved_root']}`",
        f"- Data record: `{readiness['data_record_version']}`",
        f"- Real TIFF images: `{readiness['real_images']['unique_tiff_images']}`",
        f"- Pixel labels: `{readiness['real_images']['unique_pixel_labels']}`",
        f"- Matched image/label records: `{readiness['real_images']['paired_unique_records']}`",
        f"- Unpaired labels: `{readiness['real_images']['unpaired_label_count']}`",
        f"- Train / validation / locked test: `{train_count} / {validation_count} / {test_count}`",
        f"- License verified: `{readiness['license']['verified']}`",
        f"- RGB band semantics verified: `{readiness['rgb_band_semantics']['verified']}`",
        f"- Formal validation allowed: `{readiness['formal_validation_allowed']}`",
        f"- Test pixel reads: `{json.dumps(readiness['test_pixel_reads'], ensure_ascii=False)}`",
        "",
        "## Blocking reasons",
        "",
    ] + [f"- {reason}" for reason in readiness["blocking_reasons"]] + [
        "",
        "No formal training, locked-test reads, or validation claim is authorized by this receipt.",
    ]
    (output_dir / "UCS_DATASET_STATUS.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    missing_rows = [
        {
            "asset_id": "ucs_image_label_pairing",
            "category": "images",
            "status": "missing_or_unmapped",
            "observed": f"{package.get('label_count', 0)} labels vs {package.get('imagery_count', 0)} unique TIFF images; {package.get('unpaired_label_count', 0)} labels unmatched",
            "required_before_formal_training": "yes",
            "source": str(audit_dir / "UCS_PACKAGE_AUDIT.json"),
            "action": "obtain or verify the missing image records and their exact label mapping; do not fabricate matches",
        },
        {
            "asset_id": "ucs_dataset_license",
            "category": "license",
            "status": "unknown_requires_review",
            "observed": "license is not declared in audited Zenodo API metadata",
            "required_before_formal_training": "yes",
            "source": str(audit_dir / "LICENSE_AND_PROVENANCE.md"),
            "action": "record an authoritative dataset license/provenance statement",
        },
        {
            "asset_id": "ucs_rgb_band_semantics",
            "category": "label_mapping",
            "status": "undocumented",
            "observed": "local package does not verify RGB band semantics",
            "required_before_formal_training": "yes",
            "source": str(audit_dir / "UCS_PACKAGE_AUDIT.json"),
            "action": "add authoritative RGB band/channel semantics to the local manifest",
        },
        {
            "asset_id": "human_seed_temporal_labels",
            "category": "temporal_labels",
            "status": "pending_human_seed_labels",
            "observed": "no formal human temporal seed labels are available yet",
            "required_before_formal_training": "no_for_ucs_pixel_model_but_required_for_tfr_mti",
            "source": "URRS transition preview inventory",
            "action": "use the independent Seed Temporal Labeling console",
        },
    ]
    with (output_dir / "REQUIRED_MISSING_ASSETS.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(missing_rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(missing_rows)

    next_actions = {
        "status": "blocked",
        "formal_training_status": status,
        "formal_training_allowed": False,
        "formal_validation_allowed": False,
        "resume_required": True,
        "allowed_now": [
            {
                "action": "seed_temporal_labeling",
                "status": "ALLOWED_NOW",
                "purpose": "collect model-independent human seed labels from the 96 valid transition previews",
                "model_prediction_display": False,
                "formal_ground_truth": False,
            },
            {
                "action": "resolve_ucs_pairing",
                "status": "REQUIRED_BEFORE_FORMAL_TRAINING",
                "purpose": "resolve the 214 unpaired labels without modifying raw data",
            },
            {
                "action": "verify_license_and_rgb_semantics",
                "status": "REQUIRED_BEFORE_FORMAL_TRAINING",
                "purpose": "complete authoritative provenance and channel semantics",
            },
        ],
        "blocked_until": blocking_statuses,
        "forbidden_now": [
            "formal UCS training",
            "UCS+CWLD joint fine-tuning",
            "locked Test pixel reads",
            "formal TFR/MTI evaluation",
            "using human seed labels as pixel ground truth",
        ],
        "test_pixel_reads": readiness["test_pixel_reads"],
        "current_urrs_release_unchanged": readiness["current_urrs_release_unchanged"],
        "created_at": readiness["created_at"],
    }
    (output_dir / "NEXT_ALLOWED_ACTIONS.json").write_text(json.dumps(next_actions, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**readiness, "output_dir": str(output_dir.resolve()), "next_actions": next_actions}
