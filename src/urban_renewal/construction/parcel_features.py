from __future__ import annotations

import math

import numpy as np

PARCEL_FEATURE_SCHEMA = (
    "exposed_ratio", "dustproof_net_ratio", "building_ratio", "facility_ratio",
    "debris_ratio", "uncertainty_ratio", "site_area", "perimeter", "compactness",
    "elongation", "internal_holes", "texture_heterogeneity", "distance_to_roads",
    "distance_to_urban_boundary", "distance_to_existing_built_up", "building_overlap",
    "prior_building_coverage", "subsequent_building_coverage", "full20_change_evidence",
    "sentinel_seasonal_evidence",
)


def calculate_parcel_features(geometry, evidence: dict[str, float], *, texture_values=None, distances=None, context=None):
    distances = distances or {}
    context = context or {}
    area = float(geometry.area)
    perimeter = float(geometry.length)
    minx, miny, maxx, maxy = geometry.bounds
    sides = sorted((maxx - minx, maxy - miny))
    result = {
        "exposed_ratio": float(evidence.get("exposed_ground_evidence", 0.0)),
        "dustproof_net_ratio": float(evidence.get("dustproof_net_evidence", 0.0)),
        "building_ratio": float(evidence.get("building_evidence", 0.0)),
        "facility_ratio": float(evidence.get("engineering_facility_evidence", 0.0)),
        "debris_ratio": float(evidence.get("material_debris_evidence", 0.0)),
        "uncertainty_ratio": float(evidence.get("uncertain_evidence", 0.0)),
        "site_area": area,
        "perimeter": perimeter,
        "compactness": float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-9)),
        "elongation": float(sides[1] / max(sides[0], 1e-9)),
        "internal_holes": int(sum(len(poly.interiors) for poly in getattr(geometry, "geoms", [geometry]))),
        "texture_heterogeneity": float(np.std(texture_values)) if texture_values is not None and len(texture_values) else 0.0,
        "distance_to_roads": float(distances.get("roads", math.nan)),
        "distance_to_urban_boundary": float(distances.get("urban_boundary", math.nan)),
        "distance_to_existing_built_up": float(distances.get("existing_built_up", math.nan)),
        "building_overlap": float(context.get("building_overlap", 0.0)),
        "prior_building_coverage": float(context.get("prior_building_coverage", 0.0)),
        "subsequent_building_coverage": float(context.get("subsequent_building_coverage", 0.0)),
        "full20_change_evidence": float(context.get("full20_change_evidence", 0.0)),
        "sentinel_seasonal_evidence": float(context.get("sentinel_seasonal_evidence", math.nan)),
    }
    return result
