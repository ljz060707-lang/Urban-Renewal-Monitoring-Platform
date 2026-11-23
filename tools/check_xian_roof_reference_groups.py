"""Append direct group geometry diagnostics without modifying source retention."""
from pathlib import Path
import argparse,json,sys
import fiona
from shapely.geometry import shape,Polygon
from shapely.ops import transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.roof_group_correspondence import roof_group_correspondence
from xian_r5.roof_reference_candidates import RoofMatchPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if root.parent!=parent.resolve() or (root/'ROOF_REFERENCE_GROUP_RECEIPT.json').exists():raise ValueError('Append-only retained-source run required')
    plan0=json.loads((root/'FROZEN_RETENTION_AND_SHAPE_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((root/'SOURCE_ROOF_RETENTION_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan0['protected_sha256']);protected.update(plan0['new_code_sha256'])
    protected.update({str(root/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(root/'SOURCE_ROOF_RETENTION_RECEIPT.json')]=sha(root/'SOURCE_ROOF_RETENTION_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected source changed: '+path)
    code=[ROOT/'src/xian_r5/roof_group_correspondence.py',ROOT/'tests/test_xian_roof_group_correspondence.py',Path(__file__).resolve()]
    save_json(root/'FROZEN_ROOF_REFERENCE_GROUP_PLAN.json',{'operation':'DIRECT_REFERENCE_OWNERSHIP_THEN_CENTROID_NORMALIZED_GROUP_SHAPE',
        'no_reference_subset_or_shift_search':True,'one_to_one_gate_changed':False,
        'no_group_members_counted_as_independent_stable_pairs':True,'source_masks_may_not_change':True,
        'policy':RoofMatchPolicy().__dict__,'development_exposed_roi':True,'external_validation':False,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}})
    retention=json.loads((root/'SOURCE_ROOF_RETENTION.json').read_text(encoding='utf-8'))['records']
    ids={r['retained_candidate_id'] for r in retention if r['shape_diagnostic_eligible']}
    with fiona.open(root/'RETAINED_SOURCE_ROOF_CANDIDATES.gpkg') as s:
        source={f.properties['candidate_id']:shape(f.geometry) for f in s if f.properties['candidate_id'] in ids}
    regions={c['region_id']:shape(c['support_geometry']) for c in json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']}
    fids=set(json.loads((root/'REFERENCE_DEVELOPMENT_EXPOSURE.json').read_text(encoding='utf-8'))['source_fids_loaded'])
    ll=Transformer.from_crs(32649,4326,always_xy=True);metric=Transformer.from_crs(4326,32649,always_xy=True)
    corners=[(308460,3792190),(311020,3792190),(311020,3794750),(308460,3794750)]
    ring=[ll.transform(a[0]+(b[0]-a[0])*n/64,a[1]+(b[1]-a[1])*n/64)
        for a,b in zip(corners,corners[1:]+corners[:1]) for n in range(64)]
    refs={}
    with fiona.open('F:/xian/西安建筑轮廓数据/西安.shp') as s:
        for f in s.filter(bbox=Polygon(ring).bounds):
            if str(f.id) in fids:refs[str(f.id)]=transform(metric.transform,shape(f.geometry))
    if set(refs)!=fids:raise ValueError('Reference ID inventory drift')
    result=roof_group_correspondence(source,refs,regions)
    save_json(root/'ROOF_REFERENCE_GROUP_DIAGNOSTIC.json',result)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed during group diagnostic: '+path)
    save_json(root/'ROOF_REFERENCE_GROUP_RECEIPT.json',{'new_code_sha256':{str(f):sha(f) for f in code},
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in root.iterdir() if f.is_file()},
        'group_hypotheses':result['hypotheses'],'stable_independent_pairs':0,'new_registration_vectors':0,
        'formal_gt':0,'whole_window_governance_completed':False})
    print(json.dumps({'records':result['records'],'group_hypotheses':result['hypotheses'],'source_masks_changed':0,'stable_independent_pairs':0}))


if __name__=='__main__':main()
