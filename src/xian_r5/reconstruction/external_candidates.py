"""Load external polygons as explicitly unverified footprint candidates."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import fiona
from pyproj import CRS, Transformer
from shapely.affinity import translate
from shapely.geometry import shape
from shapely.ops import transform, unary_union

from .core import ExternalFootprintCandidate, Observation


def _source_component_hashes(source_path: Path) -> dict[str, str]:
    if source_path.suffix.lower() == ".shp":
        allowed = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".xml",
                   ".sbn", ".sbx", ".qix"}
        components = [
            component for component in source_path.parent.glob(f"{source_path.stem}.*")
            if component.is_file() and component.suffix.lower() in allowed
        ]
    else:
        # Single-file vector formats (notably GeoPackage) must also have
        # content lineage. Include SQLite sidecars when present because GDAL
        # can read committed data from a WAL during a concurrent write.
        components = [source_path]
        if source_path.suffix.lower() in {".gpkg", ".sqlite", ".db"}:
            components.extend(
                Path(f"{source_path}{suffix}")
                for suffix in ("-wal", "-shm", "-journal")
                if Path(f"{source_path}{suffix}").is_file()
            )

    hashes = {}
    for component in sorted(components):
        digest = hashlib.sha256()
        with component.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[component.name] = digest.hexdigest()
    return hashes


def attach_overlapping_external_candidates(
    observations: Iterable[Observation], source_path: Path, *, layer: str | None = None,
    max_candidate_distance_m: float = 20.0,
    source_epoch_label: str | None = None,
    source_geometry_semantics: str | None = None,
) -> dict:
    """Attach overlap or nearby polygons as unverified geometry candidates.

    Disjoint candidates are retained only within the explicit metric search
    radius. Centroid-alignment IoU is a diagnostic on a translated copy of the
    roof; no shift is applied to observations and no candidate becomes ground.
    User-reported epoch and geometry semantics are provenance labels only.
    """
    records = list(observations)
    if not records:
        raise ValueError("At least one observation is required")
    if (not math.isfinite(max_candidate_distance_m)
            or max_candidate_distance_m < 0):
        raise ValueError("Candidate distance radius must be finite and nonnegative")
    if source_epoch_label is not None:
        if not isinstance(source_epoch_label, str) or not source_epoch_label.strip():
            raise ValueError("Source epoch label must be a non-empty string when provided")
        source_epoch_label = source_epoch_label.strip()
    allowed_geometry_semantics = {
        "ROOF_BOUNDARY_FROM_IMAGE",
        "GROUND_FOOTPRINT_FROM_ORTHO_OR_SURVEY",
        "UNKNOWN",
    }
    if source_geometry_semantics is not None:
        if source_geometry_semantics not in allowed_geometry_semantics:
            raise ValueError(
                "Unsupported source geometry semantics: "
                f"{source_geometry_semantics!r}")
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_component_sha256 = _source_component_hashes(source_path)
    observation_union = unary_union([item.roof for item in records])
    attached: dict[str, list[ExternalFootprintCandidate]] = {item.uid: [] for item in records}
    invalid_features = 0
    source_feature_count = None
    source_bounds = None
    source_bounds_status = "UNAVAILABLE"
    source_schema = None

    with fiona.open(source_path, layer=layer) as source:
        source_crs_input = source.crs_wkt or source.crs
        if not source_crs_input:
            raise ValueError(f"External footprint layer has no CRS: {source_path}")
        source_feature_count = len(source)
        try:
            source_bounds = list(source.bounds) if source.bounds else None
            source_bounds_status = "AVAILABLE" if source_bounds else "EMPTY"
        except fiona.errors.DriverError:
            # A broken/unavailable layer extent does not invalidate the source
            # CRS or its spatial index; the local bbox filter below is still
            # bounded by the observation geometries and metric search radius.
            source_bounds = None
            source_bounds_status = "UNAVAILABLE_DRIVER_ERROR"
        source_schema = {
            "geometry": source.schema.get("geometry"),
            "properties": dict(source.schema.get("properties", {})),
        }
        source_crs = CRS.from_user_input(source_crs_input)
        source_crs_label = source_crs.to_string()
        to_source = Transformer.from_crs("EPSG:32649", source_crs, always_xy=True)
        to_ground = Transformer.from_crs(source_crs, "EPSG:32649", always_xy=True)
        query_geometry = transform(
            to_source.transform, observation_union.buffer(max_candidate_distance_m))
        query_bbox = query_geometry.bounds
        scanned = 0
        for feature in source.filter(bbox=query_bbox):
            if not feature.get("geometry"):
                invalid_features += 1
                continue
            try:
                raw_geometry = shape(feature["geometry"])
                candidate_geometry = transform(to_ground.transform, raw_geometry)
            except Exception:
                invalid_features += 1
                continue
            if candidate_geometry.is_empty or not candidate_geometry.is_valid:
                invalid_features += 1
                continue
            scanned += 1
            fid = str(feature["id"])
            raw_properties = dict(feature["properties"])
            source_properties = {
                str(key): (value if value is None or isinstance(value, (str, int, float, bool)) else str(value))
                for key, value in raw_properties.items()
            }
            intersection_metrics_by_uid = []
            for observation in records:
                intersection_area = observation.roof.intersection(candidate_geometry).area
                polygon_distance = observation.roof.distance(candidate_geometry)
                if (intersection_area <= 0.01
                        and polygon_distance > max_candidate_distance_m):
                    continue
                roof_area = observation.roof.area
                candidate_area = candidate_geometry.area
                union_area = observation.roof.union(candidate_geometry).area
                iou = intersection_area / union_area if union_area > 0 else 0.0
                roof_centroid = observation.roof.centroid
                candidate_centroid = candidate_geometry.centroid
                centroid_dx = float(candidate_centroid.x - roof_centroid.x)
                centroid_dy = float(candidate_centroid.y - roof_centroid.y)
                shifted_roof = translate(
                    observation.roof, xoff=centroid_dx, yoff=centroid_dy)
                shifted_union_area = shifted_roof.union(candidate_geometry).area
                centroid_aligned_iou = (
                    shifted_roof.intersection(candidate_geometry).area / shifted_union_area
                    if shifted_union_area > 0 else 0.0
                )
                association_method = (
                    "POSITIVE_AREA_OVERLAP_ONLY" if intersection_area > 0.01
                    else "NEARBY_WITHIN_SEARCH_RADIUS")
                intersection_metrics_by_uid.append((
                    observation, association_method, float(iou), float(intersection_area),
                    float(intersection_area / roof_area) if roof_area > 0 else 0.0,
                    float(intersection_area / candidate_area) if candidate_area > 0 else 0.0,
                    centroid_dx, centroid_dy, math.hypot(centroid_dx, centroid_dy),
                    float(polygon_distance),
                    (None if intersection_area > 0.01 else float(max_candidate_distance_m)),
                    float(centroid_aligned_iou), float(centroid_aligned_iou - iou),
                ))
            for (observation, association_method, iou, intersection_area,
                 roof_overlap_fraction, candidate_overlap_fraction, centroid_dx,
                 centroid_dy, centroid_distance, polygon_distance, search_radius,
                 centroid_aligned_iou, centroid_aligned_iou_gain) in intersection_metrics_by_uid:
                attached[observation.uid].append(ExternalFootprintCandidate(
                    geometry=candidate_geometry,
                    candidate_id=fid,
                    source_ref=f"{source_path}#fid={fid}",
                    source_crs=source_crs_label,
                    source_capture_date=None,
                    source_epoch_label=source_epoch_label,
                    source_geometry_semantics=source_geometry_semantics,
                    source_geometry_semantics_provenance=(
                        "USER_REPORTED_SEMANTICS_NOT_INDEPENDENTLY_VERIFIED"
                        if source_geometry_semantics is not None else "NOT_PROVIDED"),
                    source_accuracy_m=None,
                    association_method=association_method,
                    match_iou=iou,
                    intersection_area_m2=intersection_area,
                    roof_overlap_fraction=roof_overlap_fraction,
                    candidate_overlap_fraction=candidate_overlap_fraction,
                    candidate_centroid_dx_from_roof_m=centroid_dx,
                    candidate_centroid_dy_from_roof_m=centroid_dy,
                    centroid_distance_m=centroid_distance,
                    source_properties=source_properties,
                    candidate_polygon_distance_m=polygon_distance,
                    candidate_search_radius_m=search_radius,
                    centroid_aligned_roof_iou=centroid_aligned_iou,
                    centroid_aligned_iou_gain=centroid_aligned_iou_gain,
                ))

    source_component_sha256_after = _source_component_hashes(source_path)
    if source_component_sha256_after != source_component_sha256:
        raise RuntimeError(f"External footprint source changed while being read: {source_path}")

    for observation in records:
        observation.footprint_candidates.extend(
            sorted(attached[observation.uid], key=lambda item: (-float(item.match_iou or 0.0), item.candidate_id))
        )

    matched = sum(bool(values) for values in attached.values())
    association_count = sum(len(values) for values in attached.values())
    return {
        "schema_version": "xian_r5.external_footprint_candidates.v5",
        "source_path": str(source_path),
        "source_layer": layer,
        "source_crs": source_crs_label,
        "source_feature_count": source_feature_count,
        "source_bounds": source_bounds,
        "source_bounds_status": source_bounds_status,
        "source_schema": source_schema,
        "source_capture_date": None,
        "source_epoch_label": source_epoch_label,
        "source_epoch_label_provenance": (
            "USER_REPORTED_EPOCH_LABEL_NOT_EXACT_ACQUISITION_DATE"
            if source_epoch_label is not None else "NOT_PROVIDED"),
        "source_geometry_semantics": source_geometry_semantics,
        "source_geometry_semantics_provenance": (
            "USER_REPORTED_SEMANTICS_NOT_INDEPENDENTLY_VERIFIED"
            if source_geometry_semantics is not None else "NOT_PROVIDED"),
        "source_accuracy_m": None,
        "source_component_sha256": source_component_sha256,
        "source_component_sha256_after": source_component_sha256_after,
        "source_hash_recheck": "PASS",
        "ground_truth": False,
        "association_policy": "POSITIVE_AREA_OVERLAP_OR_NEARBY_WITHIN_DISTANCE",
        "max_candidate_distance_m": float(max_candidate_distance_m),
        "metric_semantics": {
            "intersection_area_m2": "Positive-area geometry intersection in EPSG:32649",
            "roof_overlap_fraction": "intersection area divided by roof candidate area",
            "candidate_overlap_fraction": "intersection area divided by external candidate area",
            "candidate_centroid_dx_from_roof_m": "external candidate centroid easting minus roof centroid easting",
            "candidate_centroid_dy_from_roof_m": "external candidate centroid northing minus roof centroid northing",
            "centroid_distance_m": "Euclidean distance between centroids; diagnostic only, not a correction",
            "candidate_polygon_distance_m": "Minimum roof-to-candidate polygon separation in EPSG:32649",
            "candidate_search_radius_m": "Maximum polygon separation for a disjoint nearby candidate; null for positive-area overlap",
            "centroid_aligned_roof_iou": "IoU after translating a copy of the roof by candidate-centroid minus roof-centroid; diagnostic only, not applied",
            "centroid_aligned_iou_gain": "Centroid-aligned roof IoU minus original overlap IoU; diagnostic only",
        },
        "features_examined_in_local_bbox": scanned,
        "invalid_features_skipped": invalid_features,
        "observations_total": len(records),
        "observations_with_candidates": matched,
        "candidate_associations": association_count,
    }
