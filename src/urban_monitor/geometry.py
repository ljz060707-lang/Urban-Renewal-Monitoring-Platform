from __future__ import annotations

from collections.abc import Iterable
from math import atan2, cos, pi, sin

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry


def polygon_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.is_empty or b.is_empty or not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return float(inter / union) if union > 0 else 0.0


def intersection_over_area(a: BaseGeometry, b: BaseGeometry, *, denominator: str = "min") -> float:
    """Intersection divided by min/max/first/second polygon area.

    This is more tolerant than IoU for redevelopment lineage, where a large new
    footprint may cover several smaller former buildings.
    """
    if a.is_empty or b.is_empty or not a.intersects(b):
        return 0.0
    inter = float(a.intersection(b).area)
    if denominator == "first":
        denom = float(a.area)
    elif denominator == "second":
        denom = float(b.area)
    elif denominator == "max":
        denom = max(float(a.area), float(b.area))
    else:
        denom = min(float(a.area), float(b.area))
    return inter / denom if denom > 0 else 0.0


def buffered_iou(a: BaseGeometry, b: BaseGeometry, buffer_m: float) -> float:
    if buffer_m <= 0:
        return polygon_iou(a, b)
    return polygon_iou(a.buffer(buffer_m), b.buffer(buffer_m))


def bbox_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    return polygon_iou(box(*a.bounds), box(*b.bounds))


def centroid_distance(a: BaseGeometry, b: BaseGeometry) -> float:
    return float(a.centroid.distance(b.centroid))


def area_similarity(a: BaseGeometry, b: BaseGeometry) -> float:
    aa, bb = float(a.area), float(b.area)
    if aa <= 0 or bb <= 0:
        return 0.0
    return min(aa, bb) / max(aa, bb)


def greedy_polygon_nms(
    geometries: Iterable[BaseGeometry], scores: Iterable[float], iou_threshold: float = 0.5
) -> list[int]:
    geoms = list(geometries)
    score_arr = np.asarray(list(scores), dtype=float)
    order = list(np.argsort(-score_arr))
    keep: list[int] = []
    while order:
        idx = int(order.pop(0))
        keep.append(idx)
        order = [j for j in order if polygon_iou(geoms[idx], geoms[j]) < iou_threshold]
    return keep


def clean_polygon(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty:
        return geometry
    fixed = geometry.buffer(0)
    if isinstance(fixed, (Polygon, MultiPolygon)):
        return fixed
    polygonal = [part for part in getattr(fixed, "geoms", []) if isinstance(part, Polygon)]
    return MultiPolygon(polygonal) if polygonal else fixed.convex_hull


def _main_axis_angle(poly: Polygon) -> float:
    """Return the dominant angle (radians) of the minimum rotated rectangle."""
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]
    if len(coords) < 2:
        return 0.0
    edges = []
    for p, q in zip(coords, coords[1:] + coords[:1]):
        dx, dy = q[0] - p[0], q[1] - p[1]
        edges.append((dx * dx + dy * dy, atan2(dy, dx)))
    _, angle = max(edges, key=lambda x: x[0])
    while angle >= pi / 2:
        angle -= pi / 2
    while angle < -pi / 2:
        angle += pi / 2
    return angle


def _line_intersection(p1: tuple[float, float], axis1: str, p2: tuple[float, float], axis2: str):
    if axis1 == axis2:
        return None
    if axis1 == "h":
        return (p2[0], p1[1])
    return (p1[0], p2[1])


def _orthogonalize_axis_aligned(poly: Polygon) -> Polygon | None:
    """Conservatively snap polygon edges to horizontal/vertical directions.

    The polygon is expected to be rotated so its dominant building axis is near
    horizontal. Each original edge is represented by a horizontal or vertical
    line through its midpoint, and adjacent line intersections form the new ring.
    """
    coords = list(poly.exterior.coords)[:-1]
    if len(coords) < 4:
        return None
    lines: list[tuple[tuple[float, float], str]] = []
    for p, q in zip(coords, coords[1:] + coords[:1]):
        dx, dy = q[0] - p[0], q[1] - p[1]
        if abs(dx) + abs(dy) < 1e-9:
            continue
        axis = "h" if abs(dx) >= abs(dy) else "v"
        lines.append((((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), axis))
    if len(lines) < 4:
        return None

    # Collapse consecutive lines with the same orientation; retain the area-
    # weighted mean location by simply averaging their midpoints.
    collapsed: list[tuple[tuple[float, float], str]] = []
    for midpoint, axis in lines:
        if collapsed and collapsed[-1][1] == axis:
            old_mid, _ = collapsed[-1]
            collapsed[-1] = (((old_mid[0] + midpoint[0]) / 2, (old_mid[1] + midpoint[1]) / 2), axis)
        else:
            collapsed.append((midpoint, axis))
    if len(collapsed) >= 2 and collapsed[0][1] == collapsed[-1][1]:
        a, axis = collapsed[0]
        b, _ = collapsed[-1]
        collapsed[0] = (((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), axis)
        collapsed.pop()
    if len(collapsed) < 4:
        return None

    out: list[tuple[float, float]] = []
    for i in range(len(collapsed)):
        prev_mid, prev_axis = collapsed[i - 1]
        curr_mid, curr_axis = collapsed[i]
        intersection = _line_intersection(prev_mid, prev_axis, curr_mid, curr_axis)
        if intersection is None:
            return None
        out.append(intersection)
    candidate = clean_polygon(Polygon(out))
    if candidate.is_empty or not candidate.is_valid or candidate.area <= 0:
        return None
    return candidate if isinstance(candidate, Polygon) else None


def regularize_building_polygon(
    geometry: BaseGeometry,
    *,
    simplify_tolerance: float = 0.35,
    orthogonalize: bool = True,
    rectangularity_threshold: float = 0.82,
    min_area: float = 4.0,
    max_area_change_ratio: float = 1.75,
    max_centroid_shift: float | None = 2.0,
    max_hausdorff_distance: float | None = 4.0,
) -> BaseGeometry:
    """Clean, simplify and conservatively orthogonalize a building polygon.

    Highly rectangular objects are replaced by their minimum rotated rectangle.
    More complex footprints are angle-snapped only when the resulting geometry is
    valid and does not change area excessively. Otherwise the simplified polygon
    is retained. This keeps QGIS editing manageable without forcing every building
    into a rectangle.
    """
    original = clean_polygon(geometry)
    if original.is_empty or original.area < min_area:
        return original
    if isinstance(original, MultiPolygon):
        parts = [
            regularize_building_polygon(
                part,
                simplify_tolerance=simplify_tolerance,
                orthogonalize=orthogonalize,
                rectangularity_threshold=rectangularity_threshold,
                min_area=min_area,
                max_area_change_ratio=max_area_change_ratio,
                max_centroid_shift=max_centroid_shift,
                max_hausdorff_distance=max_hausdorff_distance,
            )
            for part in original.geoms
        ]
        return MultiPolygon([part for part in parts if isinstance(part, Polygon) and not part.is_empty])
    simplified = clean_polygon(original.simplify(simplify_tolerance, preserve_topology=True))
    if not isinstance(simplified, Polygon) or simplified.area <= 0 or not orthogonalize:
        return simplified

    def acceptable(candidate: BaseGeometry) -> bool:
        if candidate.is_empty or not candidate.is_valid:
            return False
        ratio = float(candidate.area / max(simplified.area, 1e-9))
        if not (1.0 / max(max_area_change_ratio, 1.0) <= ratio <= max(max_area_change_ratio, 1.0)):
            return False
        if max_centroid_shift is not None and candidate.centroid.distance(simplified.centroid) > max_centroid_shift:
            return False
        if max_hausdorff_distance is not None and candidate.hausdorff_distance(simplified) > max_hausdorff_distance:
            return False
        return True

    if simplified.interiors:
        return simplified

    rect = simplified.minimum_rotated_rectangle
    rect_area = float(rect.area)
    rectangularity = float(simplified.area / rect_area) if rect_area > 0 else 0.0
    if rectangularity >= rectangularity_threshold and acceptable(rect):
        return clean_polygon(rect)

    angle = _main_axis_angle(simplified)
    origin = simplified.centroid
    rotated = affinity.rotate(simplified, -angle * 180 / pi, origin=origin)
    snapped = _orthogonalize_axis_aligned(rotated)
    if snapped is None:
        return simplified
    snapped = affinity.rotate(snapped, angle * 180 / pi, origin=origin)
    snapped = clean_polygon(snapped)
    if not acceptable(snapped):
        return simplified
    return snapped
