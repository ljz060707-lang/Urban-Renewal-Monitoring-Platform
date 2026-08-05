"""Immutable data contracts for the URRS modular rebuild.

These contracts intentionally carry paths and audit metadata rather than open
file handles or mutable model tensors.  A downstream module can therefore
consume a sealed result without gaining a way to rewrite its producer's
outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SceneRecord:
    date: str
    path: str
    filename: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    bands: int
    dtype: tuple[str, ...]
    crs: str | None
    transform: tuple[float, ...]
    bounds: tuple[float, ...]
    resolution: tuple[float, ...]
    nodata: Any = None
    duplicate_observation: bool = False
    excluded_from_temporal_analysis: bool = False
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class AlignedSceneReference:
    date: str
    source_path: str
    source_sha256: str
    grid_signature: tuple[Any, ...]
    overlap_status: str
    resampling_required: bool
    source_data_modified: bool = False


@dataclass(frozen=True)
class Full20PairResult:
    pair_name: str
    date_from: str
    date_to: str
    output_root: str
    manifest_path: str
    checkpoint_sha256: str
    config_sha256: str
    thresholds: Mapping[str, float]
    tile_size: int
    overlap: int
    vectorization_mmu_m2: float
    test_images_read: int = 0


@dataclass(frozen=True)
class BuildingGeometryResult:
    date: str
    output_root: str
    probability_path: str
    mask_path: str
    boundary_path: str
    vector_path: str
    manifest_path: str
    checkpoint_sha256: str
    threshold: float
    tile_size: int
    overlap: int
    test_reads: Mapping[str, int] = field(default_factory=lambda: {"WHU": 0, "SECOND": 0})


@dataclass(frozen=True)
class BaselineChangeResult:
    pair_name: str
    output_root: str
    full20_input: str
    rsbuilding_inputs: tuple[str, ...]
    valid_overlap_mask: str
    output_files: tuple[str, ...]
    metric_table: str
    scientific_status: str = "candidate_not_ground_truth"


@dataclass(frozen=True)
class BuildingTrackResult:
    track_table: str
    transition_table: str
    object_count: int
    matching_rule: str
    scientific_status: str = "geometry_only_baseline_tracking"


@dataclass(frozen=True)
class ConstructionElementResult:
    date: str
    output_root: str
    source_path: str
    probability_outputs: tuple[str, ...]
    candidate_vector: str
    candidate_mask: str
    high_confidence_mask: str
    low_confidence_mask: str
    uncertainty_mask: str
    model_sha256: str
    rgb_mapping: tuple[int, int, int]
    human_review_required: bool = True
    test_pixel_reads: int = 0


@dataclass(frozen=True)
class ConstructionTemporalResult:
    output_root: str
    dates: tuple[str, ...]
    adjacent_pairs: tuple[str, ...]
    cumulative_pairs: tuple[str, ...]
    integration: str
    scientific_status: str
    reliable_lifecycle: bool
    temporal_flicker_rate: float | None


@dataclass(frozen=True)
class CrossReferenceResult:
    output_root: str
    baseline_seal_manifest: str
    construction_manifest: str
    metrics_table: str
    output_files: tuple[str, ...]
    upstream_modified: bool = False


def contract_to_dict(value: Any) -> dict[str, Any]:
    """Serialize one of the frozen contracts without exposing mutability."""

    if not hasattr(value, "__dataclass_fields__"):
        raise TypeError(f"Expected a dataclass contract, got {type(value)!r}")
    return asdict(value)
