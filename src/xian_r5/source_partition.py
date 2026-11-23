"""Source-specific candidate transport, never a registration approval gate.

Regions are explicit spatial supports, not inferred camera acquisitions.
No interpolation, shared global shift, synthetic GT or silent seam repair.
"""
from dataclasses import dataclass
from datetime import date
import math
import re
from shapely.geometry import Polygon, MultiPolygon
from .label_transfer import bounded_transport


def _support_ok(geometry):
    return (isinstance(geometry,(Polygon,MultiPolygon)) and
            not geometry.is_empty and geometry.is_valid and
            math.isfinite(geometry.area) and geometry.area>0)


@dataclass(frozen=True)
class SourceRegion:
    region_id: str
    actual_date: str
    support: object
    correction_m: tuple | None = None
    evidence_sha256: str | None = None
    crs: str = 'EPSG:32649'

    def __post_init__(self):
        if not isinstance(self.region_id,str) or not self.region_id.strip():
            raise ValueError('Explicit source-region ID required')
        if not isinstance(self.actual_date,str):
            raise ValueError('Actual source date required')
        date.fromisoformat(self.actual_date)
        if self.crs!='EPSG:32649' or not _support_ok(self.support):
            raise ValueError('Valid nonempty metric polygon support required')
        if self.correction_m is not None:
            if (not isinstance(self.correction_m,tuple) or len(self.correction_m)!=2
                    or any(isinstance(x,bool) or not isinstance(x,(int,float))
                           or not math.isfinite(x) for x in self.correction_m)):
                raise ValueError('Candidate vector must contain two finite metres')
            if (not isinstance(self.evidence_sha256,str) or
                    re.fullmatch(r'[0-9a-fA-F]{64}',self.evidence_sha256) is None):
                raise ValueError('Candidate vector needs a bound evidence hash')


def source_partition_transport(objects, regions, target_support):
    """Preserve object identity and stop at ambiguous or incomplete supports.

    Caller supplies already metric, immutable polygons and candidate vectors.
    An evidence hash binds provenance, not correctness: all outputs remain
    candidates. No object spanning source regions is independently shifted in
    pieces. Unresolved originals remain with the caller and are never replaced
    by a guessed zero transform. Source footprints are not known-background
    validity. Formal acceptance remains the existing upstream review contract.
    """
    regions=list(regions)
    if not _support_ok(target_support):
        raise ValueError('Valid target support required')
    if any(not isinstance(r,SourceRegion) for r in regions):
        raise ValueError('SourceRegion values required')
    ids=[r.region_id for r in regions]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate source-region ID')
    for i,left in enumerate(regions):
        for right in regions[i+1:]:
            if left.support.intersection(right.support).area>1e-6:
                raise ValueError('Overlapping source regions need explicit resolution')
    records={};candidates={}
    for key,g in objects.items():
        if not _support_ok(g):
            raise ValueError('Invalid observed object geometry')
        hits=[r for r in regions if g.intersection(r.support).area>1e-6]
        record={'source_regions':[r.region_id for r in hits],
                'source_dates':sorted({r.actual_date for r in hits}),
                'original_area_m2':g.area,'formal_corrected':False}
        if not hits:
            record['status']='UNRESOLVED_OUTSIDE_SOURCE_SUPPORT'
        elif len(hits)>1:
            record['status']='UNRESOLVED_SOURCE_SEAM'
        elif not hits[0].support.covers(g):
            record['status']='UNRESOLVED_PARTIAL_SOURCE_OBJECT'
        elif hits[0].correction_m is None:
            record['status']='UNRESOLVED_NO_VECTOR'
        else:
            region=hits[0]
            moved,_=bounded_transport({key:g},region.correction_m,
                                       region.support,target_support)
            candidate=moved[key]
            record.update(candidate_area_m2=candidate.area,
                          correction_m=list(region.correction_m),
                          evidence_sha256=region.evidence_sha256)
            if candidate.is_empty:
                record['status']='UNRESOLVED_OUTSIDE_TARGET'
            else:
                candidates[key]=candidate
                record['status']=('CANDIDATE_TARGET_EDGE_CLIPPED'
                    if abs(candidate.area-g.area)>1e-6 else 'CANDIDATE_TRANSLATED')
        records[key]=record
    return {'candidate_geometries':candidates,'objects':records,
            'formal_corrected_objects':0,'training_gt_eligible':False,
            'position_semantics':'RELATIVE_IMAGE_GRID_ONLY',
            'absolute_position_accuracy':'UNKNOWN',
            'roof_to_ground_correction':'NOT_ESTABLISHED',
            'source_coverage_is_label_validity':False}
