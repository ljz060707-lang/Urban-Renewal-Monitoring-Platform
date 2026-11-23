"""Append-only post-run checks, including actual downstream deterministic replay."""
import sys, json, argparse
from pathlib import Path
from datetime import datetime
import fiona
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from affine import Affine
from xian_r5.integration import sha256,verify_artifacts,read_json,run_r5_state,build_gt10m_shadow
from xian_r5.label_transfer import Grid,load_rasters

def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();o=args.output.resolve()
    r=read_json(o/'FINAL_RECEIPT.json');m=read_json(o/'S4_12_END_TO_END_RUN_MANIFEST.json');checks=[]
    def check(n,v):
        assert v,n
        checks.append({'test':n,'PASS':True})
    check('receipt_SHA',sha256(o/'FINAL_RECEIPT.json')==(o/'FINAL_RECEIPT.sha256').read_text().strip())
    for name,meta in r['artifacts'].items():check('artifact_SHA '+name,sha256(o/name)==meta['sha256'])
    check('frozen_inputs_unchanged',verify_artifacts(m['inputs'])==r['verified_inputs_sha256'])
    obs=read_json(o/'FRESH_OBSERVATIONS_RAW.json');rows=obs['records'];byid={x['observation_id']:x for x in rows};invalid=obs['invalid_native_geometries_unmodified']
    pred=read_json(o/'ROOF_CANDIDATES_NATIVE.json')['ROOF_INSTANCE_CANDIDATES']
    check('runtime_original_IDs_preserved',{x['candidate_id'] for x in pred}=={x['observation_id'] for x in rows+invalid})
    inference=read_json(o/'S4_12_RGB_ONLY_INFERENCE.json')
    check('pixel_area_conserved_before_CRS_conversion',sum(x['pixel_area'] for x in pred)==inference['raw_mask_positive_pixels'])
    gp=o/'S4_12_INSTANCE_ALIGNMENT_CANDIDATES.gpkg'
    with fiona.open(gp,layer='roof_alignment_candidates') as src:
        check('candidate_GPKG_crs',src.crs.to_epsg()==32649)
        check('candidate_GPKG_count',len(src)==len(rows)+len(invalid))
        for f in src:
            p=dict(f.properties);key=p['observation_id']
            check('no_fabricated_transform_or_identity '+key,all(p[n] is None for n in ['source_object_id','physical_id','total_dx_m','total_dy_m','uncertainty_m','region_dx_m','region_dy_m','residual_dx_m','residual_dy_m']))
            if key in byid:check('GPKG_native_geometry_SHA '+key,sha256_bytes(shape(f.geometry).wkb)==byid[key]['geometry_sha256'])
    with fiona.open(gp,layer='registered_roofs_qualified') as src:check('no_registered_roofs',len(src)==0)
    state=read_json(o/'PRESENCE_AND_STATE_SHADOW.json')
    check('Presence_U_only_and_no_transition',not state['transitions'] and all(x['presence']=='U' and x['physical_id'] is None for x in state['presence']))
    # Actual repeat of downstream calls from immutable newly generated observations.
    result,_=run_r5_state(rows,{'ACCEPTED_LOCAL_REGIONS':0,'APPLIED_SHADOW_LOCAL_REGIONS':0},[m['inputs']['grid_template']['path']])
    check('Presence_deterministic_replay',result.presence==state['presence'] and result.transitions==state['transitions'])
    tiles=read_json(m['inputs']['source_registry']['path'])['tiles'];packet=m['packet'];support=unary_union([shape(t['support_geometry']) for t in tiles]).intersection(shape(packet['exact_support']))
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    f,e,s=build_gt10m_shadow(rows,[packet],support,grid,result)
    a=load_rasters([o/'S4_12_GT10M_FRACTION_SHADOW.tif',o/'S4_12_GT10M_QUALITY_SHADOW.tif',o/'GT10M_VALID_UNKNOWN.tif',o/'GT10M_TRAINING_WEIGHT.tif',o/'GT10M_STATE_IGNORE.tif',o/'GT10M_SUPERVISION_FRACTIONS.tif'])
    check('fixed_grid_unchanged',a['grid']==grid)
    check('actual_fraction_replay_bitwise_after_float32_export',np.array_equal(a['arrays'][0][0],f['raw_fraction'].astype('float32')))
    check('actual_quality_replay_bitwise',np.array_equal(a['arrays'][1],f['quality']))
    check('registered_fraction_all_NoData',np.all(a['arrays'][0][1]==-9999))
    check('valid_unknown_weight_ignore_are_unpromoted',np.all(a['arrays'][2][0]==0) and np.all(a['arrays'][2][1]==1) and np.all(a['arrays'][3]==0) and np.all(a['arrays'][4]==255))
    check('state_channels_not_interpreted_as_absence',np.all(a['arrays'][5]==0) and np.all(a['arrays'][2][0]==0))
    check('raw_zeros_not_background',np.all(a['arrays'][1][8]==1))
    check('source_support_before_transform_area_conservation',max(f['area_conservation_error_m2'].values())<=.001)
    hand=read_json(o/'S4_12_SOURCE_IDENTITY_HANDOFF.json')
    check('historical53_preserved_not_inherited',hand['historical_single_date_source_links_revalidated']==53 and all(not x['building_object_ids'] and x['physical_id'] is None for x in hand['fresh_records']))
    check('torch_real_DataLoader_pass',read_json(o/'DATALOADER_READBACK.json')['nonzero_training_weight']==0)
    check('target_alignment_remains_unvalidated',r['INSTANCE_ALIGNMENT_VALIDATED'] is False and r['QUALIFIED_10M_PIXEL_COUNT']==0 and r['READY_FOR_CITYWIDE'] is False)
    report={'status':'PASS','checks':checks,'checks_passed':len(checks),'new_cross_module_regression_tests':m['regression'],
            'actual_downstream_replay':{'Presence':'EXECUTED_EXACT_MATCH','fraction':'EXECUTED_EXACT_FLOAT32_MATCH','quality':'EXECUTED_EXACT_MATCH'},
            'additional_network_inference':'NOT_RUN_NOT_NEEDED_FOR_DOWNSTREAM_REPLAY','additional_training':'NOT_RUN',
            'manual_GT_geometry_read':False,'physical_position_validation':'NOT_RUN','no_new_scientific_metrics':True}
    save(o/'POST_RUN_CROSS_MODULE_VERIFICATION.json',report)
    source=Path(__file__).resolve()
    save(o/'SUPPLEMENTAL_RECEIPT.json',{'completed_at':datetime.now().astimezone().isoformat(),'append_only':True,'parent_FINAL_RECEIPT_SHA256':sha256(o/'FINAL_RECEIPT.json'),
       'verifier_source_path':str(source),'verifier_source_sha256':sha256(source),'checks_passed':len(checks),'artifacts':{'POST_RUN_CROSS_MODULE_VERIFICATION.json':{'sha256':sha256(o/'POST_RUN_CROSS_MODULE_VERIFICATION.json')}}})
    with (o/'SUPPLEMENTAL_RECEIPT.sha256').open('x',encoding='ascii') as out:out.write(sha256(o/'SUPPLEMENTAL_RECEIPT.json'))
    print(json.dumps({'status':'PASS','checks':len(checks),'regressions':m['regression']['tests_run'],'downstream_replay':'EXACT','qualified_pixels':0}),flush=True)
def sha256_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()
if __name__=='__main__':main()
