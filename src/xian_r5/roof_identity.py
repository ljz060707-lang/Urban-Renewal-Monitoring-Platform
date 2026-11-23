"""Non-mutating, multi-feature identity hypotheses, never physical-ID creation."""
import math
from collections import Counter
from shapely import STRtree
from shapely.affinity import translate
from shapely.geometry import box

STATUSES = ('CONFIRMED', 'PLAUSIBLE', 'AMBIGUOUS', 'REJECTED', 'UNRESOLVED')


def orientation(g):
    points = list(g.minimum_rotated_rectangle.exterior.coords)
    segments = [(math.hypot(b[0]-a[0], b[1]-a[1]),
                 math.atan2(b[1]-a[1], b[0]-a[0])) for a,b in zip(points,points[1:])]
    return max(segments)[1] % math.pi


def candidates(observed, references, search_m=30.):
    """BBox search is only a scope gate; ranking is not nearest-centroid.

    Centered/simplified descriptors are diagnostic copies, not output geometry.
    Missing roads, precise dates and identity evidence prevent CONFIRMED.
    """
    if not math.isfinite(search_m) or search_m <= 0:
        raise ValueError('Positive bounded search required')
    keys = sorted(references)
    shapes = [references[k] for k in keys]
    if not shapes:
        return [], {k:'UNRESOLVED' for k in observed}
    tree = STRtree(shapes)
    obs_shapes=list(observed.values())
    obs_tree=STRtree(obs_shapes)
    descriptors = {}
    def describe(key,g):
        if key not in descriptors:
            center = g.centroid
            local_tree,local_shapes=(obs_tree,obs_shapes) if key[0]=='OBS' else (tree,shapes)
            nearby = local_tree.query(box(center.x-60,center.y-60,center.x+60,center.y+60))
            quadrants = [0,0,0,0]
            for j in nearby:
                q=local_shapes[int(j)].centroid
                if q.distance(center) > 1e-6:
                    quadrants[(2 if q.x>=center.x else 0)+(1 if q.y>=center.y else 0)] += 1
            descriptors[key]=(translate(g,xoff=-center.x,yoff=-center.y).simplify(.5,preserve_topology=True),
                              orientation(g),quadrants)
        return descriptors[key]
    rows=[]
    for oid,g in sorted(observed.items()):
        if not g.is_valid or g.is_empty or g.area<10:
            continue
        left,angle,left_top=describe(('OBS',oid),g)
        bounds=g.bounds
        for index in tree.query(box(bounds[0]-search_m,bounds[1]-search_m,bounds[2]+search_m,bounds[3]+search_m)):
            rid=keys[int(index)]; r=references[rid]
            if not r.is_valid or r.is_empty or r.area<=0:
                continue
            ratio=min(g.area,r.area)/max(g.area,r.area)
            if ratio<.35:
                continue
            right,rangle,right_top=describe(('REF',rid),r)
            inter=left.intersection(right).area
            union=left.area+right.area-inter
            iou=inter/union if union else 0.
            if iou<.35:
                continue
            angle_diff=abs(angle-rangle); angle_diff=min(angle_diff,math.pi-angle_diff)
            boundary=left.boundary.hausdorff_distance(right.boundary)
            topology=sum(abs(a-b) for a,b in zip(left_top,right_top))
            score=.55*iou+.25*ratio+.1*(1-angle_diff/(math.pi/2))+.1/(1+boundary)
            rows.append({'roof_obs_id':oid,'reference_id':rid,'shape_iou':iou,'area_similarity':ratio,
                         'orientation_difference_rad':angle_diff,'centered_boundary_hausdorff_m':boundary,
                         'neighborhood_quadrant_l1':topology,'diagnostic_score':score,
                         'road_relation':'UNKNOWN','temporal_continuity':'NOT_VERIFIED',
                         'evidence_level':'MULTIFEATURE_GEOMETRY_HYPOTHESIS'})
    by_obs={}; by_ref={}
    for row in rows:
        by_obs.setdefault(row['roof_obs_id'],[]).append(row)
        by_ref.setdefault(row['reference_id'],[]).append(row)
    statuses={}
    for oid in observed:
        options=sorted(by_obs.get(oid,[]),key=lambda r:(-r['diagnostic_score'],r['reference_id']))
        if not options:
            statuses[oid]='UNRESOLVED'; continue
        best=options[0]
        reverse=sorted(by_ref[best['reference_id']],key=lambda r:-r['diagnostic_score'])
        gap=best['diagnostic_score']-options[1]['diagnostic_score'] if len(options)>1 else 1.
        reverse_gap=reverse[0]['diagnostic_score']-reverse[1]['diagnostic_score'] if len(reverse)>1 else 1.
        status=('AMBIGUOUS' if gap<.10 or reverse_gap<.10 or reverse[0]['roof_obs_id']!=oid
                else 'PLAUSIBLE' if best['shape_iou']>=.70 and best['area_similarity']>=.75
                and best['neighborhood_quadrant_l1']<=4 else 'UNRESOLVED')
        statuses[oid]=status
        for row in options:
            row['match_status']=status if row is best else 'AMBIGUOUS'
            row['is_best_diagnostic']=row is best
            row['GT_USED_FOR_GEOMETRY']=False
    return rows,statuses


def validate_table(rows, observed_ids, gt_ids, object_ids, physical_ids):
    """Fail closed on dangling IDs, illegal promotion, missing denominator."""
    seen=set()
    for row in rows:
        if row['roof_obs_id'] not in observed_ids or row['match_status'] not in STATUSES:
            raise ValueError('Invalid observation/status')
        for field,allowed in [('roof_gt_id',gt_ids),('building_object_id',object_ids),('physical_id',physical_ids)]:
            if row.get(field) and row[field] not in allowed:
                raise ValueError('Dangling '+field)
        if row['match_status']=='CONFIRMED' and not row.get('explicit_identity_evidence'):
            raise ValueError('Geometry alone cannot confirm identity')
        if row.get('GT_USED_FOR_GEOMETRY'):
            raise ValueError('Observed geometry is immutable in this task')
        seen.add(row['roof_obs_id'])
    if seen != set(observed_ids):
        raise ValueError('Incomplete observation denominator')
    return Counter(row['match_status'] for row in rows)
