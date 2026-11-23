"""Frozen, geometry-only roof correspondences; not ground ties or physical IDs.

Shapes are centroid-normalized for identity descriptors only. Source geometry
is never modified. Displacement is computed only after mutual uniqueness is
decided, and is not fed back into candidate ranking or mask reconstruction.
"""
from dataclasses import dataclass
import math
import numpy as np
import shapely
from shapely.geometry import box,Polygon,MultiPolygon
from shapely.affinity import translate


@dataclass(frozen=True)
class RoofMatchPolicy:
    maximum_axis_distance_m:float=30.
    minimum_area_m2:float=100.
    minimum_area_similarity:float=.75
    minimum_centered_iou:float=.70
    minimum_bidirectional_gap:float=.10

    def __post_init__(self):
        values=tuple(self.__dict__.values())
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Finite policy required')
        if self.maximum_axis_distance_m<=0 or self.minimum_area_m2<=0:raise ValueError('Positive spatial/area scope required')
        if not all(0<v<=1 for v in (self.minimum_area_similarity,self.minimum_centered_iou,self.minimum_bidirectional_gap)):
            raise ValueError('Similarity/gap policy outside (0,1]')


def roof_reference_candidates(observed,references,regions,*,policy=RoofMatchPolicy()):
    """Metric geometry census and conservative mutual-shape candidate pairs.

    Caller supplies EPSG:32649 polygons. Regions are frozen nonoverlapping
    supports, not inferred camera acquisitions. Results remain hypotheses;
    no local transform, review, uncertainty or raster is generated here.
    """
    if not regions:raise ValueError('Explicit source support required')
    region_items=list(regions.items())
    for index,(key,g) in enumerate(region_items):
        if not isinstance(key,str) or not g.is_valid or g.is_empty:raise ValueError('Invalid region')
        for _,other in region_items[index+1:]:
            if g.intersection(other).area>1e-6:raise ValueError('Overlapping regions need resolution')
    def prepare(items):
        selected={};records={}
        for key,g in items.items():
            if not isinstance(key,str) or not isinstance(g,(Polygon,MultiPolygon)) or not g.is_valid or g.is_empty:
                raise ValueError('Valid keyed polygon observations required')
            hits=[rid for rid,support in region_items if g.intersection(support).area>1e-6]
            status=('OUTSIDE_SCOPE' if not hits else 'SOURCE_SEAM_OR_PARTIAL' if len(hits)!=1 or not regions[hits[0]].covers(g)
                    else 'BELOW_PARENT_AREA_FILTER' if g.area<policy.minimum_area_m2 else 'STRUCTURALLY_ELIGIBLE')
            records[key]={'regions':hits,'status':status,'area_m2':g.area}
            if status=='STRUCTURALLY_ELIGIBLE':
                center=g.centroid
                selected[key]={'geometry':g,'center':center,'region':hits[0],
                               'descriptor':translate(g,xoff=-center.x,yoff=-center.y)}
        return selected,records
    source,source_records=prepare(observed);target,target_records=prepare(references)
    target_keys=sorted(target)
    centers=np.array([target[k]['center'] for k in target_keys],dtype=object)
    tree=shapely.STRtree(centers);pairs=[]
    for key in sorted(source):
        src=source[key];p=src['center'];d=policy.maximum_axis_distance_m
        for idx in tree.query(box(p.x-d,p.y-d,p.x+d,p.y+d)):
            fid=target_keys[int(idx)];ref=target[fid]
            if src['region']!=ref['region']:continue
            a,b=src['geometry'].area,ref['geometry'].area;similarity=min(a,b)/max(a,b)
            if similarity<policy.minimum_area_similarity:continue
            left,right=src['descriptor'],ref['descriptor']
            intersection=left.intersection(right).area;union=a+b-intersection
            iou=intersection/union if union else 0.
            score=.8*iou+.2*similarity
            pairs.append({'component_key':key,'source_fid':fid,'region_id':src['region'],
                'centered_shape_iou':iou,'area_similarity':similarity,'shape_score':score})
    by_source={};by_target={}
    for pair in pairs:
        by_source.setdefault(pair['component_key'],[]).append(pair)
        by_target.setdefault(pair['source_fid'],[]).append(pair)
    for groups in (by_source,by_target):
        for rows in groups.values():rows.sort(key=lambda p:(-p['shape_score'],p['component_key'],p['source_fid']))
    unique=[];decisions=[]
    for key in sorted(source):
        options=by_source.get(key,[])
        if not options:
            decisions.append({'component_key':key,'status':'NO_AREA_COMPATIBLE_REFERENCE_CANDIDATE'});continue
        best=options[0];reverse=by_target[best['source_fid']]
        source_gap=best['shape_score']-options[1]['shape_score'] if len(options)>1 else None
        target_gap=best['shape_score']-reverse[1]['shape_score'] if len(reverse)>1 else None
        reasons=[]
        if best['centered_shape_iou']<policy.minimum_centered_iou:reasons.append('CENTERED_SHAPE_MISMATCH')
        if reverse[0]['component_key']!=key:reasons.append('NOT_MUTUAL_BEST')
        if source_gap is not None and source_gap<policy.minimum_bidirectional_gap:reasons.append('SOURCE_COMPETITOR_AMBIGUITY')
        if target_gap is not None and target_gap<policy.minimum_bidirectional_gap:reasons.append('REFERENCE_COMPETITOR_AMBIGUITY')
        record={**best,'source_gap':source_gap,'reference_gap':target_gap,
            'source_candidate_count':len(options),'reference_candidate_count':len(reverse),
            'status':'GEOMETRIC_CORRESPONDENCE_HYPOTHESIS' if not reasons else 'UNRESOLVED_GEOMETRIC_CORRESPONDENCE',
            'reasons':reasons,'physical_identity_verified':False,'is_ground_tie':False}
        if not reasons:
            a=source[key]['center'];b=target[best['source_fid']]['center']
            record['roof_to_reference_candidate_m']=[b.x-a.x,b.y-a.y]
            unique.append(record)
        decisions.append(record)
    return {'hypotheses':unique,'decisions':decisions,'source_structure':source_records,
        'reference_structure':target_records,'candidate_pair_count':len(pairs),
        'source_eligible_count':len(source),'reference_eligible_count':len(target),
        'policy':policy.__dict__,'physical_ids_created':0,'ground_ties_created':0,
        'formal_alignment_accepts':0,'geometry_modified':False,'reference_to_Sentinel_binding':'NOT_ESTABLISHED',
        'relative_registration_uncertainty_m':None,'absolute_position_accuracy':'UNKNOWN'}
