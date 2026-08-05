"""Dataset onboarding protocol for single-date, paired and temporal sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CanonicalSample:
    sample_id: str
    image: Path | None
    date: date | None
    label: Path | None
    native_label_schema: str | None
    canonical_targets: tuple[str, ...]
    crs: str | None
    transform: tuple[float, ...] | None
    quality_metadata: dict[str, Any] = field(default_factory=dict)
    t1: Path | None = None
    t2: Path | None = None
    sequence: tuple[tuple[date, Path], ...] = ()
    split: str = "unassigned"


class BaseDatasetAdapter(ABC):
    adapter_name = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        root = config.get("dataset_root")
        self.root = Path(root) if root else None

    def validate_configuration(self) -> list[str]:
        issues = []
        if self.root is None:
            issues.append("dataset_root is not configured")
        elif not self.root.is_dir():
            issues.append(f"dataset_root does not exist: {self.root}")
        if not self.config.get("image_pattern"):
            issues.append("image_pattern is not configured")
        if not self.config.get("label_pattern"):
            issues.append("label_pattern is not configured")
        return issues

    @abstractmethod
    def discover_metadata(self) -> Iterable[CanonicalSample]:
        """Discover declared files without opening image pixels."""

    def validate_label_mapping(self, mapping: dict[str, Any]) -> list[str]:
        allowed = {"background_or_other", "vacant_land", "construction_site", "building", "unknown"}
        issues = []
        if not mapping.get("verified", False):
            issues.append("label mapping is not verified")
        for row in mapping.get("mappings", []):
            if row.get("canonical_class") not in allowed:
                issues.append(f"invalid canonical class: {row.get('canonical_class')}")
        return issues
