"""Prepare factual RGB crops and supplementary topology diagnostics."""
import hashlib
import math
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds,reproject,Resampling
from affine import Affine
from PIL import Image
from xian_identity_minimal import OUT,PREV,GE,write,read,graph_rows


def main():
    revised='--revision' in sys.argv
    cases=read(OUT/('PANEL_CASES_FIXED_02.json' if revised else 'PANEL_CASES.json'))['cases']
    assets=OUT/('panel_assets_fixed_02' if revised else 'panel_assets');assets.mkdir(exist_ok=False)
    provenance=[]
    with rasterio.open(GE) as src:
        for i,c in enumerate(cases):
            x0,y0,x1,y1=c['bounds'];res=.25
            width=math.ceil((x1-x0)/res);height=math.ceil((y1-y0)/res)
            dst_aff=Affine(res,0,x0,0,-res,y1)
            bounds=transform_bounds(32649,src.crs,*c['bounds'],densify_pts=21);inv=~src.transform
            pts=[inv*(bounds[0],bounds[1]),inv*(bounds[2],bounds[3])]
            c0=math.floor(min(p[0] for p in pts))-3;r0=math.floor(min(p[1] for p in pts))-3
            c1=math.ceil(max(p[0] for p in pts))+3;r1=math.ceil(max(p[1] for p in pts))+3
            win=Window(c0,r0,c1-c0,r1-r0);raw=src.read([1,2,3],window=win);aff=src.window_transform(win)
            image=np.zeros((3,height,width),dtype=np.uint8)
            for band in range(3):reproject(raw[band],image[band],src_transform=aff,src_crs=src.crs,
                dst_transform=dst_aff,dst_crs='EPSG:32649',resampling=Resampling.bilinear)
            Image.fromarray(np.moveaxis(raw,0,-1)).save(assets/f'case_{i+1}_GE_NATIVE_RGB.png')
            Image.fromarray(np.moveaxis(image,0,-1)).save(assets/f'case_{i+1}_GE_DISPLAY.png')
            provenance.append({'case':i+1,'observation_id':c['observation_id'],'source_path':str(GE),
                'native_window':[c0,r0,c1-c0,r1-r0],'native_affine':list(aff)[:6],
                'native_array_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                'display_bounds':[x0,y1-height*res,x0+width*res,y1],
                'display_resolution_m':res,'display_reprojection_only':True,'displacement_applied':False,
                'GE_actual_date':'2018-11-01'})
    write('PANEL_RGB_PROVENANCE_FIXED_02.json' if revised else 'PANEL_RGB_PROVENANCE.json',{'records':provenance,'ground_accuracy':'UNKNOWN'})
    if revised:return
    observed=graph_rows('roof_instance'); objects=graph_rows('building_object_reference')
    all_records=read(OUT/'PLAUSIBLE_41_ADJUDICATION.json')['records']+read(OUT/'AMBIGUOUS_RECOVERY_DIAGNOSTIC.json')['records']
    traced={r['observation_id']:r['building_object_id'] for r in all_records if r['link_level']=='TRACEABLE_SINGLE_DATE_LINK'}
    old=read(PREV/'IDENTITY_FEATURE_EVIDENCE.json')['existing_object_edges'];lookup={(r['roof_obs_id'],r['reference_id']):r for r in old}
    output=[]
    for r in all_records:
        oid=r['observation_id'];uid=r['building_object_id'];g=observed[oid][1];h=objects[uid][1]
        keys=[other for other in traced if other!=oid and observed[other][1].centroid.distance(g.centroid)<100]
        left=sorted(keys,key=lambda k:math.atan2(observed[k][1].centroid.y-g.centroid.y,observed[k][1].centroid.x-g.centroid.x))
        right=sorted(keys,key=lambda k:math.atan2(objects[traced[k]][1].centroid.y-h.centroid.y,objects[traced[k]][1].centroid.x-h.centroid.x))
        cyclic=bool(left) and any(left==right[j:]+right[:j] for j in range(len(right)))
        output.append({'observation_id':oid,'building_object_id':uid,'baseline_multifeature_geometry':lookup.get((oid,uid)),
            'observation_area_m2':g.area,'object_area_m2':h.area,'geometry_hausdorff_m_unshifted':g.hausdorff_distance(h),
            'neighbor_order_observations':left,'neighbor_order_object_associations':right,
            'neighbor_order_status':'INSUFFICIENT_TRACED_NEIGHBORS' if len(keys)<2 else 'ORDER_CONSISTENT' if cyclic else 'ORDER_DIFFERS',
            'neighborhood_diagnostic_not_independent_validation':True,'GT_used_for_new_source_adjudication':False,
            'GT_used_for_prior_candidate_generation':True,'GT_used_for_display':True})
    write('GEOMETRY_AND_NEIGHBOR_ORDER_DIAGNOSTIC.json',{'records':output,'geometry_modified':False})


if __name__=='__main__':main()
