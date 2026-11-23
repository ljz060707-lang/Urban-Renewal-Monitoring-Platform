"""Close strip-fragment ambiguity, legacy edge links, and bounded native evidence."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import cv2,fiona,numpy as np,rasterio
from pyproj import Transformer
from shapely.geometry import shape,box
from shapely.ops import transform
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.native_fragment_evidence import native_fragment_evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment');prior=parent/'SINGLE2019_INSTANCE_EVIDENCE_20261012_13'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh native-fragment audit required')
    old_plan=json.loads((prior/'FROZEN_INSTANCE_EVIDENCE_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'INSTANCE_EVIDENCE_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(old_plan['input_hashes']);protected.update(old_plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'INSTANCE_EVIDENCE_RECEIPT.json')]=sha(prior/'INSTANCE_EVIDENCE_RECEIPT.json')
    recipe=ROOT/'tools/infer_shenzhen_sparse_geometry.py';protected[str(recipe)]=sha(recipe)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen evidence drift: '+path)
    code=[ROOT/'src/xian_r5/native_fragment_evidence.py',ROOT/'tests/test_xian_native_fragment_evidence.py',Path(__file__).resolve()]
    plan={'run':out.name,'scope':'CURRENT_256x256_SOURCE_ROOF_EVIDENCE_ONLY',
        'source_geometry_and_vectors_modified':False,'formal_review_or_clean_roof_acceptance':False,
        'native_serialization_tolerance_px':.001,'diagnostic_strip_row_multiple':1024,
        'strip_multiple_does_not_prove_exact_historical_run_config':True,
        'known_writer_behavior':'stream strip-owned polygons into separate object IDs; global dissolve skipped',
        'known_writer_source':str(recipe),'known_writer_is_not_exact_historical_runtime_receipt':True,
        'native_render_case_selection':'3 largest fully in ROI, single-unresolved-cell merge hypotheses, distinct cells; 1 largest single-coincidence control in a different unresolved cell',
        'native_visual_output_role':'DIAGNOSTIC_NOT_HUMAN_REVIEW_NOT_PRECISION_OR_GROUND_TIES',
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}}
    out.mkdir();save_json(out/'FROZEN_NATIVE_FRAGMENT_PLAN.json',plan)
    bundles=json.loads((prior/'COMPONENT_EVIDENCE_BUNDLES.json').read_text(encoding='utf-8'))['records']
    with fiona.open(prior/'BOUNDED_HISTORICAL_OBJECTS.gpkg') as s:
        old={str(f.properties['object_id']):shape(f.geometry) for f in s}
    ge_path=Path('F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif')
    rows=[];ge=rasterio.open(ge_path);inv=~ge.transform
    to_native_crs=Transformer.from_crs(32649,ge.crs,always_xy=True)
    def native(x,y,z=None):
        x,y=to_native_crs.transform(x,y)
        return inv.a*np.asarray(x)+inv.b*np.asarray(y)+inv.c,inv.d*np.asarray(x)+inv.e*np.asarray(y)+inv.f
    for b in bundles:
        if b['status']!='POSSIBLE_MERGED_SOURCE_COMPONENT_NOT_VERIFIED':continue
        ids=b['substantially_contained_object_ids']
        result=native_fragment_evidence({key:transform(native,old[key]) for key in ids})
        covered_ids={key for group in result['fragment_groups'] for key in group['historical_object_ids']}
        complete=len(result['fragment_groups'])==1 and covered_ids==set(ids)
        rows.append({'component_key':b['component_key'],'source_regions':b['source_regions'],
            'touches_unresolved_cells':b['touches_unresolved_cells'],'source_area_m2':b['source_area_m2'],
            'historical_union_covered_fraction':b['historical_union_covered_fraction'],
            'all_historical_parts_share_native_strip_edge_set':complete,'native_partition_evidence':result,
            'corrected_diagnostic':'COEXTENSIVE_NATIVE_STRIP_FRAGMENT_SET_NOT_MULTIPLE_BUILDING_EVIDENCE' if complete
                else 'MERGE_HYPOTHESIS_NOT_EXPLAINED_BY_ONE_NATIVE_STRIP_FRAGMENT_SET',
            'new_clean_roof_instances':0,'physical_ids_created':0,'source_geometry_modified':False})
    save_json(out/'NATIVE_FRAGMENT_RELATIONS.json',{'records':rows,'native_grid':{
        'source':str(ge_path),'crs':ge.crs.to_string(),'shape':list(ge.shape),'affine':list(ge.transform)[:6]},
        'rounding_role':'DISPOSABLE_NATIVE_GRID_SERIALIZATION_VIEW_NOT_LOCALIZATION_BOUND'})
    # Link only structurally valid rows actually observed in the partial edge
    # CSV. A complete object UID join does not prove complete temporal edges.
    edges=json.loads((prior/'BOUNDED_HISTORICAL_EDGES.json').read_text(encoding='utf-8'))
    links={}
    for ix,row in enumerate(edges['selected_rows']):
        for uid in {row['source_object_id'],row['target_object_id']}:
            if uid.startswith('xa190101::'):links.setdefault(uid,[]).append(ix)
    save_json(out/'LEGACY_EDGE_BINDING_SIDECAR.json',{'source':str(prior/'BOUNDED_HISTORICAL_EDGES.json'),
        'source_sha256':sha(prior/'BOUNDED_HISTORICAL_EDGES.json'),'indices_zero_based':True,
        'valid_observed_edge_indices_by_object_uid':links,'edge_completeness':'UNKNOWN_PARTIAL_CSV_LAST_ROW_WIDTH_MISMATCH',
        'legacy_edge_indices_empty_in_stage13_bundle':'UNPOPULATED_PLACEHOLDER_NOT_OBSERVED_ZERO; use this sidecar',
        'historical_accepted_for_identity_is_formal_human_review':False})
    roi=box(308460,3792190,311020,3794750)
    candidates=sorted([b for b in bundles if b['touches_unresolved_cells'] and len(b['source_regions'])==1
        and b['status']=='POSSIBLE_MERGED_SOURCE_COMPONENT_NOT_VERIFIED' and roi.covers(shape(b['source_geometry']))],
        key=lambda b:(-b['source_area_m2'],b['component_key']))
    selected=[];used_cells=set()
    for b in candidates:
        if b['source_regions'][0] in used_cells:continue
        selected.append(b);used_cells.add(b['source_regions'][0])
        if len(selected)==3:break
    controls=sorted([b for b in bundles if b['touches_unresolved_cells'] and len(b['source_regions'])==1
        and b['source_regions'][0] not in used_cells and b['status']=='SINGLE_GEOMETRIC_COINCIDENCE_NOT_IDENTITY'
        and roi.covers(shape(b['source_geometry']))],key=lambda b:(-b['source_area_m2'],b['component_key']))
    if controls:selected.append(controls[0])
    save_json(out/'FROZEN_NATIVE_VIEW_CASES.json',{'selected_component_keys':[b['component_key'] for b in selected],
        'selection_before_native_image_reads':True,'uses_alignment_scores_or_corrected_labels':False})
    views=[]
    for b in selected:
        component=shape(b['source_geometry']);window_shape=transform(to_native_crs.transform,component.buffer(20))
        left,bottom,right,top=window_shape.bounds
        corners=[inv*(x,y) for x,y in ((left,bottom),(right,top))]
        col0=max(0,math.floor(min(p[0] for p in corners)));row0=max(0,math.floor(min(p[1] for p in corners)))
        col1=min(ge.width,math.ceil(max(p[0] for p in corners)));row1=min(ge.height,math.ceil(max(p[1] for p in corners)))
        window=Window(col0,row0,col1-col0,row1-row0)
        scale=min(1.,1280/max(window.height,window.width));h=max(1,round(window.height*scale));w=max(1,round(window.width*scale))
        rgb=np.moveaxis(ge.read([1,2,3],window=window,out_shape=(3,h,w)),0,-1)
        left_panel=rgb.copy();right_panel=rgb.copy()
        def draw(canvas,g,color):
            pixel=transform(native,g)
            for part in shapely_parts(pixel):
                for ring in [part.exterior,*part.interiors]:
                    xy=np.asarray(ring.coords);xy[:,0]=(xy[:,0]-col0)*w/window.width;xy[:,1]=(xy[:,1]-row0)*h/window.height
                    cv2.polylines(canvas,[np.rint(xy).astype(np.int32)],True,color,2)
        draw(left_panel,component,(255,60,60))
        for rel in b['historical_evidence']:draw(right_panel,old[rel['historical_object_id']],(50,255,255))
        title=np.zeros((64,w*3,3),np.uint8)
        for k,text0 in enumerate(['RAW GE / no displacement','SOURCE mask boundary','HISTORICAL IDs / not independent GT']):
            cv2.putText(title,text0,(k*w+8,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(title,b['component_key']+' '+b['source_regions'][0],(8,52),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        panel=np.concatenate([title,np.concatenate([rgb,left_panel,right_panel],axis=1)],axis=0)
        path=out/(b['component_key']+'_NATIVE_SOURCE_DIAGNOSTIC.png')
        if not cv2.imwrite(str(path),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR)):raise RuntimeError('PNG rendering failed')
        views.append({'component_key':b['component_key'],'path':str(path),'source_window':[col0,row0,window.width,window.height],
            'native_read_out_shape':[h,w],'native_window_pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
            'source_date':'2018-11-01','experiment_year':2019,'displacement_applied':False,'model_visual_result':'NOT_YET_INSPECTED'})
    ge.close();save_json(out/'NATIVE_VIEW_PROVENANCE.json',{'records':views,'original_geometry_modified':False})
    total=sum(r['all_historical_parts_share_native_strip_edge_set'] for r in rows)
    unresolved=sum(r['touches_unresolved_cells'] and r['all_historical_parts_share_native_strip_edge_set'] for r in rows)
    summary={'run':out.name,'all_roi_initial_merge_hypotheses':len(rows),'native_strip_fragment_sets':total,
        'unresolved_initial_merge_hypotheses':sum(r['touches_unresolved_cells'] for r in rows),
        'unresolved_native_strip_fragment_sets':unresolved,
        'remaining_unexplained_merge_hypotheses_in_unresolved':sum(r['touches_unresolved_cells'] and not r['all_historical_parts_share_native_strip_edge_set'] for r in rows),
        'native_diagnostic_views':len(views),'legacy_edges_linked_uid_count':len(links),
        'legacy_edge_completeness':'UNKNOWN_PARTIAL_SOURCE','new_geometry_or_vectors':0,
        'new_label_pixels':0,'formal_gt_pixels':0,'physical_ids':0}
    save_json(out/'NATIVE_FRAGMENT_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen artifact changed: '+path)
    save_json(out/'NATIVE_FRAGMENT_RECEIPT.json',{'run':out.name,'new_code_sha256':plan['new_code_sha256'],
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'summary':summary,
        'whole_window_governance_completed':False})
    print(json.dumps(summary))


def shapely_parts(g):
    return [g] if g.geom_type=='Polygon' else list(g.geoms)


if __name__=='__main__':main()
