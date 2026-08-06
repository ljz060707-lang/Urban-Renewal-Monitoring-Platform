"""Detached Boundary Objectizer contracts and deterministic utilities.

This module deliberately operates on arrays and explicit registries.  It never
discovers a Test split and it never changes the binary Building mask: learned
boundary evidence can only change instance IDs inside pixels already accepted
by B0.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .boundary import boundary_target, bootstrap_ci
from .object_metrics import match_objects, size_stratified_recall


@dataclass(frozen=True)
class BoundaryObjectizationParams:
    boundary_threshold: float = 0.50
    interior_seed_threshold: float = 0.70
    minimum_seed_area: int = 16
    minimum_object_area: int = 10
    watershed_compactness: float = 0.0
    minimum_separation_distance: int = 2

    def validate(self) -> None:
        if not 0.0 < self.boundary_threshold < 1.0:
            raise ValueError("boundary_threshold must be in (0, 1)")
        if not 0.0 < self.interior_seed_threshold <= 1.0:
            raise ValueError("interior_seed_threshold must be in (0, 1]")
        if self.minimum_seed_area < 1 or self.minimum_object_area < 1:
            raise ValueError("minimum areas must be positive")
        if self.watershed_compactness < 0:
            raise ValueError("watershed_compactness must be non-negative")
        if self.minimum_separation_distance < 0:
            raise ValueError("minimum_separation_distance must be non-negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connected_components(mask: np.ndarray, *, connectivity: int = 8) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {value.shape}")
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, _ = ndimage.label(value, structure=structure)
    return labels.astype(np.int32, copy=False)


def _remove_small_labels(labels: np.ndarray, minimum_area: int) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int32).copy()
    if minimum_area <= 1:
        return result
    counts = np.bincount(result.reshape(-1))
    for label, count in enumerate(counts):
        if label and count < minimum_area:
            result[result == label] = 0
    return result


def objectization_preserves_building_area(labels: np.ndarray, building_mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(building_mask, dtype=bool)
    object_mask = np.asarray(labels) > 0
    if mask.shape != object_mask.shape:
        raise ValueError("object labels and Building mask shapes differ")
    disagreement = int(np.logical_xor(mask, object_mask).sum())
    return {
        "building_mask_disagreement": disagreement,
        "building_area_before": int(mask.sum()),
        "building_area_after": int(object_mask.sum()),
        "preserved": disagreement == 0 and int(mask.sum()) == int(object_mask.sum()),
    }


def connected_component_objectization(building_mask: np.ndarray) -> np.ndarray:
    """O0: deterministic connected components of the fixed B0 mask."""
    return connected_components(np.asarray(building_mask, dtype=bool))


def morphology_boundary_objectization(building_mask: np.ndarray, *, width: int = 2) -> np.ndarray:
    """O1: morphology-derived seeds with strict B0 area preservation."""
    mask = np.asarray(building_mask, dtype=bool)
    boundary = boundary_target(mask.astype(np.uint8), width=width, method="symmetric") > 0
    # The boundary is evidence only.  Connected components remain the
    # deterministic fallback when it does not yield separable interiors.
    interior = mask & ~boundary
    labels = connected_components(interior)
    fallback = connected_component_objectization(mask)
    for label in np.unique(fallback):
        if label == 0:
            continue
        component = fallback == label
        if not np.any(labels[component] > 0):
            labels[component] = int(label)
    # Assign boundary pixels to their nearest existing object, never to
    # background, so the binary coverage remains exactly the B0 coverage.
    missing = mask & (labels == 0)
    if np.any(missing) and np.any(labels > 0):
        _, nearest = ndimage.distance_transform_edt(labels == 0, return_distances=True, return_indices=True)
        # distance_transform indices are (row, col) arrays.
        labels[missing] = labels[nearest[0][missing], nearest[1][missing]]
    labels[~mask] = 0
    return labels.astype(np.int32, copy=False)


def learned_boundary_watershed(
    building_mask: np.ndarray,
    boundary_probability: np.ndarray,
    params: BoundaryObjectizationParams | None = None,
) -> np.ndarray:
    """O2: watershed using detached boundary probability inside B0 pixels.

    The watershed is constrained to ``building_mask`` and all unassigned B0
    pixels are deterministically assigned to the nearest object.  Thus O2 can
    split objects, but cannot create or delete Building pixels.
    """
    params = params or BoundaryObjectizationParams()
    params.validate()
    mask = np.asarray(building_mask, dtype=bool)
    probability = np.asarray(boundary_probability, dtype=np.float32)
    if mask.shape != probability.shape:
        raise ValueError("boundary probability and Building mask shapes differ")
    if not np.isfinite(probability).all():
        raise ValueError("boundary probability contains NaN/Inf")
    probability = np.clip(probability, 0.0, 1.0)
    base_labels = connected_component_objectization(mask)
    seed = mask & (probability < float(params.boundary_threshold))
    if params.minimum_separation_distance:
        seed = ndimage.binary_erosion(seed, iterations=int(params.minimum_separation_distance), border_value=0)
    markers = connected_components(seed)
    markers = _remove_small_labels(markers, int(params.minimum_seed_area))
    # A component without a viable interior seed must remain one object.
    for base_label in np.unique(base_labels):
        if base_label == 0:
            continue
        component = base_labels == base_label
        if not np.any(markers[component] > 0):
            markers[component] = int(markers.max()) + 1
    # Elevation favours low boundary probability and distance from the edge.
    distance = ndimage.distance_transform_edt(mask)
    elevation = probability + (1.0 / (1.0 + distance)) * 0.05
    labels = watershed(elevation, markers=markers, mask=mask, compactness=float(params.watershed_compactness))
    labels = _remove_small_labels(labels, int(params.minimum_object_area))
    missing = mask & (labels == 0)
    if np.any(missing):
        source = labels > 0
        if not np.any(source):
            labels = base_labels.copy()
        else:
            _, nearest = ndimage.distance_transform_edt(~source, return_distances=True, return_indices=True)
            labels[missing] = labels[nearest[0][missing], nearest[1][missing]]
    labels[~mask] = 0
    return labels.astype(np.int32, copy=False)


def oracle_boundary_objectization(building_mask: np.ndarray, ground_truth_mask: np.ndarray, params: BoundaryObjectizationParams | None = None) -> np.ndarray:
    """O3 diagnostic upper bound; never valid for model selection."""
    gt_boundary = boundary_target(np.asarray(ground_truth_mask, dtype=np.uint8), width=2, method="symmetric")
    return learned_boundary_watershed(building_mask, gt_boundary, params)


def objectization_metrics(
    predicted_labels: np.ndarray,
    ground_truth_mask: np.ndarray,
    *,
    building_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_labels) > 0
    target = np.asarray(ground_truth_mask) > 0
    result: dict[str, Any] = {}
    for threshold in (0.3, 0.5, 0.7):
        result[f"object_{threshold:.1f}"] = match_objects(predicted, target, threshold=threshold)
    result["size_stratified_recall"] = size_stratified_recall(predicted, target, iou_threshold=0.5)
    # The contract is against the fixed B0 pixel mask, never against the
    # objectized prediction itself.  ``building_mask`` is optional for
    # backwards-compatible callers; new evaluation code must pass B0's mask.
    fixed_mask = predicted if building_mask is None else np.asarray(building_mask) > 0
    result["area_contract"] = objectization_preserves_building_area(predicted_labels, fixed_mask)
    return result


def bounded_parameter_grid(values: dict[str, Sequence[Any]], *, maximum: int = 64) -> list[BoundaryObjectizationParams]:
    keys = ["boundary_threshold", "interior_seed_threshold", "minimum_seed_area", "minimum_object_area", "watershed_compactness", "minimum_separation_distance"]
    unknown = set(values) - set(keys)
    if unknown:
        raise ValueError(f"unknown objectization parameters: {sorted(unknown)}")
    grid: list[BoundaryObjectizationParams] = [BoundaryObjectizationParams()]
    for key in keys:
        options = list(values.get(key, [getattr(grid[0], key)]))
        next_grid: list[BoundaryObjectizationParams] = []
        for item in grid:
            for option in options:
                payload = {name: getattr(item, name) for name in keys}
                payload[key] = option
                candidate = BoundaryObjectizationParams(**payload)
                candidate.validate()
                next_grid.append(candidate)
        grid = next_grid
        if len(grid) > int(maximum):
            raise ValueError(f"objectization parameter grid exceeds maximum {maximum}")
    return grid


def select_objectization_parameters(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select only from Calibration rows; O3 is explicitly ineligible."""
    materialized = list(rows)
    explicit_splits = {str(row.get("split")) for row in materialized if row.get("split") is not None}
    if explicit_splits and explicit_splits != {"boundary_calibration"}:
        raise ValueError(f"objectization selection requires boundary_calibration rows, got {sorted(explicit_splits)}")
    candidates = [row for row in materialized if str(row.get("method")) in {"O0_CONNECTED_COMPONENTS", "O1_MORPHOLOGICAL_BOUNDARY", "O2_LEARNED_BOUNDARY_WATERSHED"}]
    if not candidates:
        raise ValueError("no selectable O0/O1/O2 calibration rows")
    candidates.sort(key=lambda row: (-float(row.get("object_f1@0.5", -1.0)), float(row.get("split_rate", 1e9)), float(row.get("merge_rate", 1e9)), str(row.get("method"))))
    selected = dict(candidates[0])
    selected["selection_split"] = "train_calibration_only"
    selected["oracle_selectable"] = False
    selected["selected_method"] = selected.get("method")
    return selected


def bootstrap_object_differences(rows: Iterable[dict[str, Any]], *, n_bootstrap: int = 1000, seed: int = 20260805) -> dict[str, Any]:
    """Bootstrap per-image O2-O0/O1 differences using images as units."""
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        method = str(row.get("method"))
        grouped.setdefault(sample_id, {})[method] = {key: float(row.get(key, 0.0)) for key in ("object_f1@0.5", "split_rate", "merge_rate", "absolute_count_error", "relative_area_error")}
    sample_ids = sorted(grouped)
    result: dict[str, Any] = {"n_images": len(sample_ids), "n_bootstrap": int(n_bootstrap), "seed": int(seed), "comparisons": {}}
    for reference in ("O0_CONNECTED_COMPONENTS", "O1_MORPHOLOGICAL_BOUNDARY"):
        deltas: dict[str, list[float]] = {key: [] for key in ("object_f1@0.5", "split_rate", "merge_rate", "absolute_count_error", "relative_area_error")}
        for sample_id in sample_ids:
            methods = grouped[sample_id]
            if "O2_LEARNED_BOUNDARY_WATERSHED" not in methods or reference not in methods:
                continue
            for key in deltas:
                deltas[key].append(methods["O2_LEARNED_BOUNDARY_WATERSHED"][key] - methods[reference][key])
        comparison: dict[str, Any] = {"sample_count": len(next(iter(deltas.values()), []))}
        for key, values in deltas.items():
            comparison[key] = bootstrap_ci(values, seed=int(seed), n_bootstrap=int(n_bootstrap))
        result["comparisons"][f"O2_minus_{reference.split('_')[0]}"] = comparison
    return result


__all__ = [
    "BoundaryObjectizationParams",
    "connected_component_objectization",
    "morphology_boundary_objectization",
    "learned_boundary_watershed",
    "oracle_boundary_objectization",
    "objectization_metrics",
    "bounded_parameter_grid",
    "select_objectization_parameters",
    "bootstrap_object_differences",
    "objectization_preserves_building_area",
    "sha256_file",
]
