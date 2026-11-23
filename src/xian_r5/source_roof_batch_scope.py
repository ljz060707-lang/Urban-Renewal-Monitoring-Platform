"""Complete-ROI source-only SAM proposal worklist, no GT-based selection."""
import hashlib
from shapely.geometry import Polygon,MultiPolygon


def source_roof_batch_scope(objects,regions):
    if len(regions)!=16:raise ValueError('Exactly the frozen sixteen source units required')
    if any(not isinstance(g,(Polygon,MultiPolygon)) or not g.is_valid or g.is_empty for g in regions.values()):
        raise ValueError('Valid frozen supports required')
    values=list(regions.values())
    if any(a.intersection(b).area>1e-6 for i,a in enumerate(values) for b in values[i+1:]):raise ValueError('Source supports overlap')
    selected=[];excluded=[]
    for key,g in sorted(objects.items()):
        if not isinstance(key,str) or not isinstance(g,(Polygon,MultiPolygon)) or not g.is_valid or g.is_empty:
            raise ValueError('Valid immutable source components required')
        hits=[rid for rid,support in regions.items() if support.intersection(g).area>1e-6]
        reason=('OUTSIDE_SOURCE_SCOPE' if not hits else 'SOURCE_SEAM_OR_PARTIAL' if len(hits)!=1 or not regions[hits[0]].covers(g)
            else 'BELOW_SOURCE_PARENT_AREA' if g.area<100 else None)
        if reason:excluded.append({'component_key':key,'regions':hits,'reason':reason});continue
        selected.append({'component_key':key,'region_id':hits[0],'source_area_m2':g.area,
            'source_geometry_wkb_sha256':hashlib.sha256(g.wkb).hexdigest(),
            'prompt_basis':'WEAK_RSBUILDING_COMPONENT_NOT_VERIFIED_ROOF_SURFACE',
            'source_model_mask_is_ground_truth':False,'physical_id':None})
    counts={rid:sum(r['region_id']==rid for r in selected) for rid in regions}
    if any(n==0 for n in counts.values()):raise ValueError('Requested sixteen units but an eligible source unit is missing; no expansion')
    return {'targets':selected,'excluded_components':excluded,'per_unit_targets':counts,
        'selection':'ALL_STRUCTURALLY_ELIGIBLE_SOURCE_COMPONENTS_NO_TOPK_NO_GT_NO_NCC',
        'source_geometries_changed':0,'registration_ties_generated':0,'formal_roof_admissions':0}
