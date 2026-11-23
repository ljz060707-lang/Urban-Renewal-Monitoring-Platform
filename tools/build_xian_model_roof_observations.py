"""Bind raw-model roof surfaces; never rasterize them as whole-roof labels."""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2,fiona,numpy as np,rasterio
from affine import Affine
from shapely.geometry import shape,box,mapping
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from run_xian_single2019_experiment import write
from xian_r5.roof_observation import source_roof_observation
from xian_r5.fraction_shadow import _areas_and_counts
from xian_r5.label_transfer import Grid
from xian_r5.alignment.source_template_alignment import _sample_target


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if root.parent!=parent.resolve() or (root/'MODEL_OBSERVATION_RECEIPT.json').exists():raise ValueError('Append-only roof-packet run required')
    plan=json.loads((root/'FROZEN_ROOF_OBSERVATION_PLAN.json').read_text(encoding='utf-8'))
    initial=json.loads((root/'RAW_SOURCE_PACKET_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(root/n):h for n,h in initial['output_sha256'].items()})
    protected[str(root/'RAW_SOURCE_PACKET_RECEIPT.json')]=sha(root/'RAW_SOURCE_PACKET_RECEIPT.json')
    annotation_path=root/'MODEL_RAW_ROOF_SURFACE_ANNOTATIONS.json';protected[str(annotation_path)]=sha(annotation_path)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen evidence drift: '+path)
    code=[ROOT/'src/xian_r5/roof_observation.py',ROOT/'tests/test_xian_roof_observation.py',Path(__file__).resolve()]
    bake_plan={'coordinate_conversion_only':True,'GT_or_target_positions_may_not_set_geometry':True,
        'sampling_only_not_full_instance_or_fraction_labels':True,'no_candidate_reestimation':True,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}}
    save_json(root/'FROZEN_OBSERVATION_BINDING_PLAN.json',bake_plan)
    packets={p['source_parent_component']:p for p in json.loads((root/'RAW_SOURCE_PACKETS.json').read_text(encoding='utf-8'))['records']}
    annotations=json.loads(annotation_path.read_text(encoding='utf-8'))
    if annotations['uses_target_or_reference_position'] or annotations['complete_instance']:raise ValueError('Observation role breach')
    prior=parent/'SINGLE2019_INSTANCE_EVIDENCE_20261012_13'
    bundles={r['component_key']:r for r in json.loads((prior/'COMPONENT_EVIDENCE_BUNDLES.json').read_text(encoding='utf-8'))['records']}
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649');scope=box(308460,3792190,311020,3794750)
    observed={};records=[];unbound=[]
    for annotation0 in annotations['records']:
        key=annotation0['source_parent_component'];packet=packets[key]
        if sha(packet['image_path'])!=packet['image_sha256']:raise ValueError('Raw image drift')
        annotation=dict(annotation0,method=annotations['method'],coordinate_convention=annotations['coordinate_convention'],
            complete_instance=False,uses_target_or_reference_position=False)
        try:record,geometry=source_roof_observation(packet,annotation,shape(bundles[key]['source_geometry']),scope)
        except ValueError as exc:
            unbound.append({'observation_id':annotation['observation_id'],'reason':str(exc),'source_annotation_retained':True,
                'source_parent_component':key,'formal_disposition':None});continue
        if record['observation_id'] in observed:raise ValueError('Duplicate observation ID')
        observed[record['observation_id']]=geometry
        record['historical_uid_context']=[r['legacy_object_uid'] for r in bundles[key]['historical_evidence']]
        record['historical_uid_context_is_verified_roof_identity']=False
        records.append(record)
    save_json(root/'SOURCE_ROOF_OBSERVATIONS.json',{'records':records,'unbound_model_observations':unbound,
        'role':'MODEL_ROOF_SURFACES_NOT_COMPLETE_ROOF_INSTANCES','source_raster_positions_modified':False})
    if not observed:raise ValueError('No model observation bound; do not generate empty diagnostics')
    schema={'geometry':'Polygon','properties':{'observation_id':'str','parent_component':'str','area_m2':'float',
        'complete_instance':'int','physical_id':'str','method':'str','formal_gt':'int'}}
    with fiona.open(root/'SOURCE_ROOF_SURFACES.gpkg','w',driver='GPKG',layer='model_surface_candidates',crs='EPSG:32649',schema=schema) as s:
        for record in records:s.write({'geometry':mapping(observed[record['observation_id']]),'properties':{
            'observation_id':record['observation_id'],'parent_component':record['source_parent_component'],
            'area_m2':record['area_m2'],'complete_instance':0,'physical_id':None,'method':record['method'],'formal_gt':0}})
    cells=np.array([grid.cell(c,r) for r in range(256) for c in range(256)],dtype=object)
    areas,counts,conflicts=_areas_and_counts(observed.values(),cells)
    coverage=(areas/100).reshape(256,256).astype(np.float32)
    if conflicts.any():raise RuntimeError('Roof-surface observation overlap conflict')
    write(root,'SOURCE_ROOF_SURFACE_SUPPORT_DIAGNOSTIC.tif',coverage,grid,['ANNOTATED_SURFACE_FRACTION_NOT_OCCUPANCY'],
        extra={'ZERO_SEMANTICS':'UNANNOTATED_NOT_BACKGROUND','TRAINING_LABEL_USE':'PROHIBITED_PARTIAL_ROOF_SURFACE'})
    evidence_mask=coverage>=.8
    write(root,'SOURCE_ROOF_SURFACE_SAMPLE_SCOPE.tif',evidence_mask.astype(np.uint8),grid,['CANDIDATE_SURFACE_SAMPLE_SCOPE_NOT_LABEL'],
        extra={'ZERO_SEMANTICS':'UNANNOTATED_OR_MIXED_NOT_BACKGROUND','TRAINING_LABEL_USE':'PROHIBITED'})
    field=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    # Count fixed FIT/CHECK populations only; no target RGB read and no score,
    # peak search, displacement estimation, or old candidate modification.
    yy,xx=np.indices((256,256));fit=(yy//16+xx//16)%2==0;population=[]
    for cell in field:
        rr,cc=(int(v[1:]) for v in cell['region_id'].split('_')[1:]);unit=np.zeros((256,256),bool)
        unit[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
        numbers={}
        for role,fold in [('FIT',fit),('CHECK',~fit)]:
            common=evidence_mask&unit&fold;target_scope=unit&fold
            for dx in range(-3,4):
                for dy in range(-3,4):common&=_sample_target(target_scope[None],dx,dy)[0]
            numbers[role]=int(common.sum())
        population.append({'region_id':cell['region_id'],'candidate_surface_pixels':int((evidence_mask&unit).sum()),
            'fixed_fit_pixels':numbers['FIT'],'fixed_check_pixels':numbers['CHECK'],
            'minimum_fit_pixels':100,'minimum_check_pixels':100,
            'has_required_population':numbers['FIT']>=100 and numbers['CHECK']>=100,
            'alignment_fit_run':False,'reason':'COUNT_PREFLIGHT_ONLY_NO_DISPLACEMENT_OR_PRECISION_EVIDENCE'})
    save_json(root/'SURFACE_POPULATION_PREFLIGHT.json',{'records':population,
        'source_surface_threshold':.8,'search_radius_px':3,'fold_block_px':16,
        'target_RGB_or_GT_positions_read':False,'new_numeric_candidates':0})
    for key,packet in packets.items():
        group=[r for r in records if r['source_parent_component']==key]
        if not group:continue
        image=cv2.imread(packet['image_path']);h,w=image.shape[:2]
        for r in group:
            xy=np.rint(r['native_vertices_xy']).astype(np.int32)
            cv2.polylines(image,[xy],True,(80,255,80),2)
            x,y=xy.min(axis=0);cv2.putText(image,r['observation_id'].rsplit('_',1)[1],(int(x),max(12,int(y)-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(80,255,80),1)
        if not cv2.imwrite(str(root/(key+'_MODEL_SURFACE_OVERLAY.png')),image):raise RuntimeError('Overlay failed')
    expected=sum(g.area for g in observed.values());delivered=float(coverage.astype(np.float64).sum()*100)
    if abs(expected-delivered)>.01:raise RuntimeError('Surface coverage area differs after raster serialization')
    with fiona.open(root/'SOURCE_ROOF_SURFACES.gpkg') as s:
        if len(s)!=len(records) or s.crs.to_epsg()!=32649:raise RuntimeError('Observation layer readback failed')
        for f in s:
            if not shape(f.geometry).equals_exact(observed[f.properties['observation_id']],0):raise RuntimeError('Source observation geometry changed on disk')
    summary={'run':root.name,'source_packets':len(packets),'model_surface_annotations':len(annotations['records']),
        'bound_surface_observations':len(records),'unbound_surface_observations':len(unbound),
        'total_candidate_surface_area_m2':expected,'float32_area_error_m2':abs(expected-delivered),
        'surface_pixels_at_80_percent':int(evidence_mask.sum()),
        'cells_with_100_fit_and_100_check_surface_pixels':sum(r['has_required_population'] for r in population),
        'complete_observed_roof_instances':0,'verified_clean_roof_instances':0,'new_numeric_displacements':0,
        'original_masks_or_candidates_replaced':0,'new_training_label_pixels':0,'formal_gt_eligible':0,
        'reviewer_signatures_created':0,'absolute_position_accuracy':'UNKNOWN'}
    save_json(root/'MODEL_SURFACE_OBSERVATION_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen evidence changed: '+path)
    save_json(root/'MODEL_OBSERVATION_RECEIPT.json',{'summary':summary,'new_code_sha256':bake_plan['new_code_sha256'],
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in root.iterdir() if f.is_file()},'whole_window_governance_completed':False})
    print(json.dumps(summary))


if __name__=='__main__':main()
