"""Unshifted source-component/legacy-object overlay, not identity acceptance.

All relations are direct positive-area geometry evidence. Historical objects
cannot set displacement, replace roof observations, or certify clean roofs.
"""
from dataclasses import dataclass
import math
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


@dataclass(frozen=True)
class InstanceEvidencePolicy:
    diagnostic_covered_fraction: float = .80
    geometric_coincidence_iou: float = .95
    positive_area_epsilon_m2: float = 1e-6

    def __post_init__(self):
        if (any(not math.isfinite(v) for v in self.__dict__.values())
                or not 0 < self.diagnostic_covered_fraction <= 1
                or not 0 < self.geometric_coincidence_iou <= 1
                or self.positive_area_epsilon_m2 <= 0):
            raise ValueError('Explicit finite diagnostic policy required')


def component_evidence(observed, auxiliary, regions, *, policy=InstanceEvidencePolicy()):
    """Return topology hypotheses and untouched metric source provenance."""
    def check(items):
        for key, g in items.items():
            if (not isinstance(key, str) or not isinstance(g, (Polygon, MultiPolygon))
                    or g.is_empty or not g.is_valid or not math.isfinite(g.area)):
                raise ValueError('Valid keyed metric polygons required; no silent repair')
    check(observed); check(auxiliary); check(regions)
    for i, a in enumerate(regions.values()):
        for b in list(regions.values())[i+1:]:
            if a.intersection(b).area > policy.positive_area_epsilon_m2:
                raise ValueError('Source support overlap unresolved')
    if not regions: raise ValueError('Explicit support required')
    keys = sorted(auxiliary); geometries = np.array([auxiliary[k] for k in keys], dtype=object)
    tree = shapely.STRtree(geometries); rows = []
    for component_key, source in observed.items():
        relations = []; clipped = []; substantial_keys = []
        for ix in sorted(tree.query(source, predicate='intersects')):
            key = keys[int(ix)]; other = auxiliary[key]; intersection = source.intersection(other)
            area = intersection.area
            if area <= policy.positive_area_epsilon_m2: continue
            sf = area/source.area; af = area/other.area
            relations.append({'historical_object_id': key, 'intersection_area_m2': area,
                'source_covered_fraction': sf, 'historical_covered_fraction': af,
                'direct_unshifted_iou': area/(source.area+other.area-area)})
            clipped.append(intersection)
            if af >= policy.diagnostic_covered_fraction: substantial_keys.append(key)
        union_area = unary_union(clipped).area if clipped else 0.
        coverage = union_area/source.area
        substantial_geoms = [auxiliary[k] for k in substantial_keys]
        overlap = 0.
        if substantial_geoms:
            overlap = sum(g.area for g in substantial_geoms)-unary_union(substantial_geoms).area
        if not relations: status='NO_DIRECT_HISTORICAL_GEOMETRY_OVERLAP'
        elif overlap > policy.positive_area_epsilon_m2: status='HISTORICAL_GEOMETRY_OVERLAP_CONFLICT'
        elif len(substantial_keys)>=2 and coverage >= policy.diagnostic_covered_fraction:
            status='POSSIBLE_MERGED_SOURCE_COMPONENT_NOT_VERIFIED'
        elif len(relations)==1 and relations[0]['direct_unshifted_iou']>=policy.geometric_coincidence_iou:
            status='SINGLE_GEOMETRIC_COINCIDENCE_NOT_IDENTITY'
        elif coverage < policy.diagnostic_covered_fraction:
            status='UNEXPLAINED_SOURCE_AREA_NOT_CONFIRMED_CONTAMINATION'
        else: status='PARTIAL_OR_AMBIGUOUS_GEOMETRIC_ASSOCIATION'
        rows.append({'component_key':component_key,'source_area_m2':source.area,
            'source_regions':[k for k,g in regions.items() if source.intersection(g).area>policy.positive_area_epsilon_m2],
            'historical_union_covered_fraction':coverage,'historical_union_covered_area_m2':union_area,
            'historical_candidate_count':len(relations),'substantially_contained_object_ids':substantial_keys,
            'substantial_historical_overlap_area_m2':max(0.,overlap),'relations':relations,'status':status,
            'source_geometry_modified':False,'verified_clean_roof':False,'physical_identity_verified':False})
    return {'records':rows,'policy':policy.__dict__,
        'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
        'source_components':len(observed),'historical_objects':len(auxiliary),
        'new_displacements':0,'new_clean_roofs':0,'formal_accepts':0,'physical_ids':0}
