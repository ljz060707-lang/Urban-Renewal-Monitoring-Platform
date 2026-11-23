"""Derived union of edge-adjacent supports sharing one frozen experimental field.

This does not identify camera acquisition units or estimate a new displacement.
Unknown regions and original SourceRegions remain unchanged and traceable.
"""
import hashlib
import json
from shapely.ops import unary_union
from .source_partition import SourceRegion


def coalesce_equal_field_regions(regions):
    regions = list(regions)
    if not regions or any(not isinstance(r, SourceRegion) for r in regions):
        raise ValueError('Explicit frozen source regions required')
    if len({r.region_id for r in regions}) != len(regions): raise ValueError('Duplicate region')
    for i, a in enumerate(regions):
        for b in regions[i+1:]:
            if a.support.intersection(b.support).area > 1e-6: raise ValueError('Overlapping source supports')
    used = set(); groups = []; output = []
    for i, first in enumerate(regions):
        if i in used: continue
        if first.correction_m is None:
            used.add(i); output.append(first); continue
        if first.evidence_sha256 is None: raise ValueError('Numeric field requires frozen evidence')
        members = [i]; used.add(i); queue = [i]
        while queue:
            current = regions[queue.pop()]
            for j, other in enumerate(regions):
                if j in used or other.correction_m is None: continue
                if other.actual_date != first.actual_date or other.correction_m != first.correction_m: continue
                if current.support.boundary.intersection(other.support.boundary).length <= 1e-6: continue
                if other.evidence_sha256 is None: raise ValueError('Numeric field requires frozen evidence')
                used.add(j); members.append(j); queue.append(j)
        originals = [regions[j] for j in members]
        support = unary_union([r.support for r in originals])
        member_records = [{'region_id': r.region_id, 'support_wkb_sha256': hashlib.sha256(r.support.wkb).hexdigest(),
            'evidence_sha256': r.evidence_sha256, 'actual_date': r.actual_date, 'correction_m': list(r.correction_m)} for r in originals]
        digest = hashlib.sha256(json.dumps(member_records, sort_keys=True).encode()).hexdigest()
        name = 'equal_field_' + '_'.join(sorted(r.region_id.removeprefix('roi_') for r in originals))
        output.append(SourceRegion(name, first.actual_date, support, first.correction_m, digest))
        groups.append({'derived_region_id': name, 'members': member_records,
            'support_area_m2': support.area, 'support_wkb_sha256': hashlib.sha256(support.wkb).hexdigest(),
            'correction_m': list(first.correction_m), 'actual_date': first.actual_date,
            'camera_acquisition_unit_verified': False, 'new_displacement_estimated': False,
            'formal_alignment_accepted': False})
    if not unary_union([r.support for r in output]).equals(unary_union([r.support for r in regions])):
        raise RuntimeError('Source support changed during field grouping')
    return output, groups
