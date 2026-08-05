"""Metadata-first dataset onboarding; no Test pixels and no invented label semantics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .adapters import BeijingWasteAdapter, GenericMTIAdapter, GenericTFRAdapter
from ..safety.test_guard import TestAccessGuard, default_policies


ADAPTERS = {
    "beijing_waste": BeijingWasteAdapter,
    "generic_tfr": GenericTFRAdapter,
    "generic_mti": GenericMTIAdapter,
}


def _load(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def audit_dataset(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = _load(config_path)
    adapter_name = config.get("adapter")
    if adapter_name not in ADAPTERS:
        raise ValueError(f"Unknown onboarding adapter: {adapter_name}")
    adapter = ADAPTERS[adapter_name](config)
    mapping_path = config.get("label_mapping_file")
    mapping = _load(mapping_path) if mapping_path and Path(mapping_path).is_file() else {}
    issues = adapter.validate_configuration()
    mapping_issues = adapter.validate_label_mapping(mapping)
    samples = []
    if not issues:
        samples = list(adapter.discover_metadata())
    duplicates = len({sample.sample_id for sample in samples}) != len(samples)
    missing_labels = sum(sample.label is None for sample in samples)
    guard = TestAccessGuard(default_policies(), phase="validation")
    report = {
        "dataset_id": config.get("dataset_id"), "adapter": adapter_name,
        "status": "ready" if not issues and not mapping_issues and samples and not duplicates and missing_labels == 0 else "blocked",
        "metadata_only": True, "image_pixel_reads": 0, "test_image_pixel_reads": guard.read_count,
        "sample_count": len(samples), "missing_label_count": missing_labels, "duplicate_sample_ids": duplicates,
        "configuration_issues": issues, "mapping_issues": mapping_issues,
        "next_steps": ["verify label mapping", "prepare deterministic train/validation manifest"] if mapping_issues else ["dataset-prepare"],
    }
    destination = Path(output_dir or "outputs/dataset_onboarding") / str(config.get("dataset_id") or "unconfigured")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "DATASET_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_dataset(config_path: str | Path, output_dir: str | Path | None = None) -> Path:
    report = audit_dataset(config_path, output_dir)
    if report["status"] != "ready":
        raise ValueError("Dataset audit is blocked; fix configuration and verified label mapping before prepare")
    config = _load(config_path)
    adapter = ADAPTERS[config["adapter"]](config)
    rows = list(adapter.discover_metadata())
    destination = Path(output_dir or "outputs/dataset_onboarding") / str(config["dataset_id"])
    manifest = destination / "manifest.csv"
    with manifest.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "image", "label", "date", "split"])
        writer.writeheader()
        for sample in rows:
            writer.writerow({"sample_id": sample.sample_id, "image": sample.image, "label": sample.label, "date": sample.date or "", "split": sample.split})
    return manifest
