from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from .base import BaseDatasetAdapter, CanonicalSample


class PatternDatasetAdapter(BaseDatasetAdapter):
    """Conservative metadata adapter. Native class meaning always comes from YAML."""

    def discover_metadata(self) -> Iterable[CanonicalSample]:
        issues = self.validate_configuration()
        if issues:
            raise ValueError("; ".join(issues))
        assert self.root is not None
        images = sorted(self.root.glob(str(self.config["image_pattern"])))
        labels = {path.stem: path for path in self.root.glob(str(self.config["label_pattern"]))}
        date_format = self.config.get("date_parser")
        for image in images:
            parsed = None
            if date_format:
                parsed = datetime.strptime(image.stem, date_format).date()
            yield CanonicalSample(
                sample_id=image.stem, image=image, date=parsed, label=labels.get(image.stem),
                native_label_schema=str(self.config.get("native_label_schema") or ""),
                canonical_targets=tuple(self.config.get("canonical_targets", ())),
                crs=None, transform=None, quality_metadata={}, split="unassigned",
            )
