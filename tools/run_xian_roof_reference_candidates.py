"""Bounded geometry-only roof correspondence inventory under frozen rules."""
from pathlib import Path
import argparse,csv,json,sys
import fiona
from pyproj import Transformer
from shapely.geometry import shape,Polygon
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.roof_reference_candidates import RoofMatchPolicy,roof_reference_candidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    prior=parent/'SINGLE2019_SUPPORT_REPAIR_20261012_07'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh hypothesis run required')
    receipt=json.loads((prior/'SUPPORT_REPAIR_IMPLEMENTATION_RECEIPT.json').read_text(encoding='utf-8'))
    prior_plan=json.loads((prior/'FROZEN_SUPPORT_REPAIR_PLAN.json').read_text(encoding='utf-8'))
    protected=dict(prior_plan['protected_sha256']);protected.update(receipt['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'SUPPORT_REPAIR_IMPLEMENTATION_RECEIPT.json')]='281d4087e421469665d59b987e6f0a68910ce8eb5accfe585fd79c65765b114d'
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected source drift: '+path)
    inventory_path=parent/'GT10M_VALIDATION_PREFLIGHT_20261012_05/REFERENCE_EVIDENCE_CENSUS.json'
    inventory=json.loads(inventory_path.read_text(encoding='utf-8'))
    fids={r['source_fid'] for r in inventory['records']}
    source=Path('F:/xian/西安建筑轮廓数据/西安.shp')
    cache=parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
    cells_path=parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json'
    field=json.loads(cells_path.read_text(encoding='utf-8'))
    regions={c['region_id']:shape(c['support_geometry']) for c in field['cells']}
    policy=RoofMatchPolicy()
    plan={'run':out.name,'scope':'SAME_256x256_TRAINROI','crs':'EPSG:32649','experiment_year':2019,
        'source_observed_date':'2018-11-01','reference_year':'USER_CONFIRMED_2019',
        'policy':policy.__dict__,'ranking':'0.8 centered-shape IoU + 0.2 area similarity; no offset/outcome score',
        'hypothesis_displacement_calculated_only_after_mutual_unique_selection':True,
        'source_geometry_may_not_snap_scale_rotate_or_merge':True,
        'manual_reference_may_not_be_modified':True,'ground_or_pixel_alignment_vectors_used_in_matching':False,
        'reviewer_or_ground_tie_coordinates_generated':False,'physical_identity_verified':False,
        'reference_to_Sentinel_binding':'NOT_ESTABLISHED','hypothesis_is_not_clean_roof_verification':True,
        'independent_validation_claim':False,'all_loaded_reference_fids_logged_as_development_exposure':True,
        'no_common_shift_fit_or_application_this_run':True,'protected_sha256':protected,
        'reference_source':str(source),'source_components':str(cache),'formal_state':receipt['formal_state']}
    out.mkdir(parents=True);save_json(out/'FROZEN_ROOF_MATCH_PLAN.json',plan)
    observed={}
    with fiona.open(cache) as s:
        if s.crs.to_epsg()!=32649:raise ValueError('Source frame drift')
        for f in s:observed[str(f.properties['component_key'])]=shape(f.geometry)
    to_ll=Transformer.from_crs(32649,4326,always_xy=True)
    to_metric=Transformer.from_crs(4326,32649,always_xy=True)
    corners=[(308460,3792190),(311020,3792190),(311020,3794750),(308460,3794750)]
    ring=[to_ll.transform(a[0]+(b[0]-a[0])*n/64,a[1]+(b[1]-a[1])*n/64)
          for a,b in zip(corners,corners[1:]+corners[:1]) for n in range(64)]
    references={}
    with fiona.open(source) as s:
        if s.crs.to_epsg()!=4326:raise ValueError('Reference frame drift')
        for f in s.filter(bbox=Polygon(ring).bounds):
            fid=str(f.id)
            if fid in fids:references[fid]=transform(to_metric.transform,shape(f.geometry))
    if set(references)!=fids:raise ValueError('Frozen reference IDs changed')
    print('Frozen roof descriptors; comparing same-region shapes without correction feedback.',flush=True)
    result=roof_reference_candidates(observed,references,regions,policy=policy)
    save_json(out/'ROOF_CORRESPONDENCE_HYPOTHESES.json',result)
    keys=['component_key','source_fid','region_id','centered_shape_iou','area_similarity','shape_score',
        'source_gap','reference_gap','source_candidate_count','reference_candidate_count','roof_to_reference_candidate_m',
        'status','physical_identity_verified','is_ground_tie']
    with (out/'ROOF_CORRESPONDENCE_HYPOTHESES.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in result['hypotheses']:
            writer.writerow({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in keys})
    save_json(out/'REFERENCE_DEVELOPMENT_EXPOSURE.json',{'role':'GEOMETRY_DESCRIPTOR_DEVELOPMENT_NOT_REVIEW',
        'reference_file_sha256':sha(source),'reference_fids_loaded':sorted(references),
        'source_component_keys_loaded':sorted(observed),'selected_reference_fids':sorted(r['source_fid'] for r in result['hypotheses']),
        'external_unseen_validation_may_not_be_claimed_for_loaded_geometries':True,
        'prior_usage_records_or_roles_rewritten':False})
    summary={'run':out.name,'input_components':len(observed),'input_reference_roofs':len(references),
        'source_structurally_eligible':result['source_eligible_count'],
        'reference_structurally_eligible':result['reference_eligible_count'],
        'area_compatible_candidate_pairs':result['candidate_pair_count'],
        'mutual_unique_shape_hypotheses':len(result['hypotheses']),
        'per_cell_hypothesis_counts':{rid:sum(r['region_id']==rid for r in result['hypotheses']) for rid in regions},
        'clean_observed_roof_instances_verified':0,'stable_physical_pairs_verified':0,
        'corrected_label_pixels_added':0,'no_candidate_application':True,'formal_state':receipt['formal_state']}
    save_json(out/'ROOF_MATCH_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Protected source changed: '+path)
    print(json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
