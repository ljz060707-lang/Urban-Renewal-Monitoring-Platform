"""Recompute all frozen 267 pair vectors; internal per-source uniformity check."""
from pathlib import Path
import argparse,json,sys
import fiona
from shapely.geometry import shape,Polygon
from shapely.ops import transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json,write_new
from xian_r5.roof_pair_translation import roof_pair_translation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh diagnostic run required')
    prior=parent/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21'
    plan=json.loads((prior/'FROZEN_MANUAL_REFERENCE_BINDING.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'MANUAL_REFERENCE_BINDING_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    for n in ('MANUAL_REFERENCE_BINDING_RECEIPT.json','REFERENCE_BINDING_ANNEX_RECEIPT.json',
              'REFERENCE_BINDING_ENGINEERING_QA.json','SOURCE_SEMANTIC_TRIAGE_CHECKPOINT.json'):
        protected[str(prior/n)]=sha(prior/n)
    pairpath=parent/'SINGLE2019_ROOF_EVIDENCE_20261012_08/ROOF_CORRESPONDENCE_HYPOTHESES.json'
    protected[str(pairpath)]=sha(pairpath)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen input drift: '+path)
    field=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    pairs=json.loads(pairpath.read_text(encoding='utf-8'))['hypotheses']
    if len(pairs)!=267 or len(field)!=16:raise ValueError('Frozen full-ROI cohort drift')
    code=[Path(__file__).resolve(),ROOT/'src/xian_r5/roof_pair_translation.py',ROOT/'tests/test_xian_roof_pair_translation.py']
    out.mkdir();save_json(out/'FROZEN_PAIR_TRANSLATION_PLAN.json',{
        'scope':'CURRENT_256x256_ALL_16_UNITS_NOT_CITYWIDE','frozen_pair_census':267,
        'matching_not_rerun':True,'source_GEOMETRY_or_reference_not_modified':True,
        'new_case_or_outlier_selection':False,'minimum_per_fold':6,'minimum_spatial_groups':3,
        'calibration_estimator':'COMPONENTWISE_MEDIAN_CANDIDATE_ONLY',
        'maximum_centroid_consistency_residual_m':2.,
        'consistency_threshold_is_NOT_localization_bound_or_V2_acceptance':True,
        'fold_assignment':'SOURCE_centroid_4x4_checker','coverage_groups':'SOURCE_centroid_3x3',
        'all_reference_objects_development_exposed':True,'formal_human_review':False,
        'candidate_application':False,'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}})
    sources={}
    with fiona.open(parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
        for f in s:sources[str(f.properties['component_key'])]=shape(f.geometry)
    gt=Path('F:/xian/西安建筑轮廓数据/西安.shp');refs={};needed={r['source_fid'] for r in pairs}
    ll=Transformer.from_crs(32649,4326,always_xy=True);metric=Transformer.from_crs(4326,32649,always_xy=True)
    corners=[(308460,3792190),(311020,3792190),(311020,3794750),(308460,3794750)]
    ring=[ll.transform(a[0]+(b[0]-a[0])*n/64,a[1]+(b[1]-a[1])*n/64)
          for a,b in zip(corners,corners[1:]+corners[:1]) for n in range(64)]
    with fiona.open(gt) as s:
        for f in s.filter(bbox=Polygon(ring).bounds):
            if str(f.id) in needed:refs[str(f.id)]=transform(metric.transform,shape(f.geometry))
    if set(refs)!=needed:raise ValueError('Frozen reference IDs missing')
    results=[]
    for cell in field:
        rid=cell['region_id'];support=shape(cell['support_geometry']);rows=[]
        for r in pairs:
            if r['region_id']!=rid:continue
            source=sources[r['component_key']];ref=refs[r['source_fid']]
            if not support.covers(source) or not support.covers(ref):raise ValueError('Pair geometry outside exact support')
            rows.append({'component_key':r['component_key'],'source_fid':r['source_fid'],
                'source_centroid_xy':[source.centroid.x,source.centroid.y],
                'reference_centroid_xy':[ref.centroid.x,ref.centroid.y],
                'frozen_pair_vector_m':r['roof_to_reference_candidate_m']})
        result=roof_pair_translation(rows,support.bounds);results.append({'region_id':rid,**result})
        print(rid,result['calibration_n'],result['internal_check_n'],result['diagnostic_candidate_vector_m'],
              result['internal_check_max_centroid_residual_m'],result['reasons'],flush=True)
    summary={'units_evaluated':16,'frozen_pairs_recomputed':sum(len(r['records']) for r in results),
        'units_with_estimable_candidate':sum(r['diagnostic_candidate_vector_m'] is not None for r in results),
        'units_diagnostically_uniform':sum(r['uniform_translation_diagnostically_consistent'] for r in results),
        'new_training_admissions':0,'new_displacement_applications':0,'formal_review_accept_reject':0,
        'recommended_experimental_pixels_unchanged':12903,'whole_window_governance_completed':False}
    save_json(out/'FULLROI_PAIR_TRANSLATION_DIAGNOSTIC.json',{'summary':summary,'records':results})
    write_new(out/'PAIR_TRANSLATION_REPORT.md','# 当前窗口分来源屋顶平移一致性检查\n\n'
        '267个既有形态对应假设全部保留，没有按残差删样本，也未重新挑GT或SAM掩膜。'
        '机器从原始源组件和人工GT复算每对质心向量，再按源位置冻结的空间棋盘分为校准和内部检查。'
        '至少各6对及各3个空间组才估计公共平移；中位数仅估计候选，检查统计量是最大残差。\n\n'
        '这不是稳定地面tie合同，不生成localization bounds，不将质心残差当作V2 uncertainty；'
        '形态假设、内部交叉检查和人工GT角色均不构成源屋顶纯度或物理身份认证。'
        '原人工review合同及旧位移保持不变。本轮所有结果只作定位问题的诊断，不应用新位移。\n\n'
        f'16个单元中{summary["units_with_estimable_candidate"]}个可估计候选，'
        f'{summary["units_diagnostically_uniform"]}个满足本次均匀性诊断；详细失败原因保存在JSON。\n')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    save_json(out/'PAIR_TRANSLATION_RECEIPT.json',{'summary':summary,'new_code_sha256':{str(f):sha(f) for f in code},
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})


if __name__=='__main__':main()
