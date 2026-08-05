from __future__ import annotations


def distance_feature(geometry, reference_geometries) -> float:
    candidates = list(reference_geometries)
    if not candidates:
        return float("nan")
    return float(min(geometry.distance(item) for item in candidates))


def urban_context(geometry, *, roads=(), urban_boundary=(), built_up=()):
    return {
        "roads": distance_feature(geometry, roads),
        "urban_boundary": distance_feature(geometry, urban_boundary),
        "existing_built_up": distance_feature(geometry, built_up),
    }
