"""Direct-overlap reference groups, not physical identities or registration.

Own each reference by strongest direct source overlap within the frozen cell
and axis-distance scope. Group union descriptors never move source geometry.
No combinations or shifts are searched to make the shape statistic pass.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPolygon,box
from shapely.ops import unary_union
from shapely.affinity import translate
from .roof_reference_candidates import RoofMatchPolicy


def roof_group_correspondence(observed,references,regions,*,policy=RoofMatchPolicy()):
    if not regions:raise ValueError('Explicit immutable source support required')
    for collection in (observed,references,regions):
        if any(not isinstance(k,str) or not isinstance(g,(Polygon,MultiPolygon)) or not g.is_valid or g.is_empty for k,g in collection.items()):
            raise ValueError('Valid keyed metric polygons required')
    values=list(regions.values())
    if any(a.intersection(b).area>1e-6 for i,a in enumerate(values) for b in values[i+1:]):raise ValueError('Support overlap unresolved')
    def eligible(items):
        result={}
        for key,g in items.items():
            owner=[rid for rid,r in regions.items() if r.intersection(g).area>1e-6]
            if len(owner)==1 and regions[owner[0]].covers(g) and g.area>=policy.minimum_area_m2:
                result[key]=(g,owner[0])
        return result
    sources=eligible(observed);refs=eligible(references);keys=sorted(sources)
    tree=shapely.STRtree(np.array([sources[k][0] for k in keys],dtype=object))
    owned={k:[] for k in keys};assignments=[]
    for fid,(r,region) in refs.items():
        options=[]
        for index in tree.query(r,predicate='intersects'):
            key=keys[int(index)];s,scope=sources[key]
            if scope!=region:continue
            if abs(s.centroid.x-r.centroid.x)>policy.maximum_axis_distance_m or abs(s.centroid.y-r.centroid.y)>policy.maximum_axis_distance_m:continue
            intersection=s.intersection(r).area
            if intersection<=1e-6:continue
            options.append((intersection/r.area,key,intersection))
        options.sort(key=lambda row:(-row[0],row[1]))
        if not options:continue
        gap=options[0][0]-options[1][0] if len(options)>1 else None
        if gap is not None and gap<policy.minimum_bidirectional_gap:
            assignments.append({'source_fid':fid,'status':'DIRECT_OVERLAP_OWNERSHIP_AMBIGUOUS','options':options});continue
        best=options[0];owned[best[1]].append(fid)
        assignments.append({'source_fid':fid,'source_candidate_id':best[1],
            'direct_reference_overlap_fraction':best[0],'ownership_gap':gap,
            'status':'DIRECT_GEOMETRIC_OWNERSHIP_NOT_IDENTITY'})
    records=[]
    for key,(s,region) in sources.items():
        fids=sorted(owned[key]);reasons=[]
        if not fids:
            records.append({'source_candidate_id':key,'reference_fids':[],'status':'NO_DIRECT_REFERENCE_GROUP'});continue
        parts=[refs[fid][0] for fid in fids];group=unary_union(parts)
        overlap=sum(g.area for g in parts)-group.area
        area_similarity=min(s.area,group.area)/max(s.area,group.area)
        a=translate(s,xoff=-s.centroid.x,yoff=-s.centroid.y)
        b=translate(group,xoff=-group.centroid.x,yoff=-group.centroid.y)
        intersection=a.intersection(b).area;centered_iou=intersection/(a.area+b.area-intersection)
        if overlap>1e-6:reasons.append('REFERENCE_GROUP_OVERLAP_CONFLICT')
        if area_similarity<policy.minimum_area_similarity:reasons.append('GROUP_AREA_MISMATCH')
        if centered_iou<policy.minimum_centered_iou:reasons.append('GROUP_CENTERED_SHAPE_MISMATCH')
        records.append({'source_candidate_id':key,'region_id':region,'reference_fids':fids,
            'relation':'1:N' if len(fids)>1 else '1:1','source_area_m2':s.area,'reference_union_area_m2':group.area,
            'reference_overlap_area_m2':max(0.,overlap),'area_similarity':area_similarity,'centered_shape_iou':centered_iou,
            'status':'GROUPED_GEOMETRIC_CORRESPONDENCE_HYPOTHESIS' if not reasons else 'UNRESOLVED_GROUP_GEOMETRY',
            'reasons':reasons,'physical_identity_verified':False,'stable_independent_pairs_count':0,
            'registration_vector_generated':False,'source_geometry_modified':False})
    return {'records':records,'reference_assignments':assignments,'policy':policy.__dict__,
        'hypotheses':sum(r['status']=='GROUPED_GEOMETRIC_CORRESPONDENCE_HYPOTHESIS' for r in records),
        'no_shift_or_reference_subset_optimization':True,'stable_independent_pairs_verified':0,
        'physical_ids_created':0,'formal_accepts':0,'reference_to_Sentinel_binding':'NOT_ESTABLISHED'}
