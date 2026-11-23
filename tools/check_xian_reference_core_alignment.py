"""Fixed corroborated-source cohort replay; no new displacement application."""
from pathlib import Path
import argparse,hashlib,io,json,sys,unittest
import numpy as np,rasterio
from affine import Affine
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json,write_new
from xian_r5.label_transfer import Grid,export_raster
from xian_r5.reference_core_cohort import reference_core_cohort
from xian_r5.alignment.point_roof_alignment import point_roof_check,reconcile_roof_descriptors
from xian_r5.alignment.source_template_alignment import source_template_check


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh diagnostic replay required')
    prior=parent/'SINGLE2019_REFERENCE_SCREENED_WINDOW_20261012_23'
    plan=json.loads((prior/'FROZEN_WINDOW_REFERENCE_SCREEN_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'CURRENT_WINDOW_DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    for n in ('CURRENT_WINDOW_DELIVERY_RECEIPT.json','INDEPENDENT_DELIVERY_READBACK.json'):
        protected[str(prior/n)]=sha(prior/n)
    old=parent/'SINGLE2019_POINT_RGB_20261012_11/POINT_RGB_REPLAY.json'
    protected[str(old)]=sha(old)
    inputs={'raw':parent/'SINGLE2019_20261012_06/2019_ROOF_FRACTION_RAW.tif',
        'reference':parent/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21/2019_MANUAL_REFERENCE_ROOF_FRACTION_10m.tif',
        'manual_quality':parent/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21/2019_MANUAL_REFERENCE_QUALITY.tif',
        'source_rgb':parent/'SINGLE2019_SUPPORT_REPAIR_20261012_07/2019_BOUNDED_GE_RGB_CACHE.tif',
        'target_rgb':Path('F:/xian/10m底图/Xian_2019_TrueColor.tif'),
        'field':parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json'}
    for path in inputs.values():protected[str(path)]=sha(path)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input drift: '+path)
    code=[Path(__file__).resolve(),ROOT/'src/xian_r5/reference_core_cohort.py',ROOT/'tests/test_xian_reference_core_cohort.py']
    frozen={'scope':'FIXED_256x256_ALL_16_UNITS','source_cohort':'RAW>=0.8 AND UNTRANSLATED_MANUAL_REFERENCE>=0.8 NO_OVERLAP_NO_ROI_TRUNCATION',
        'manual_reference_position_information_used':True,'manual_reference_or_source_geometry_modified':False,
        'ten_m_model_predictions_used':False,'target_feature_mask_uses_manual_annotation':False,
        'search_radius_px':3,'minimum_fit_and_check_distinct_pixels':100,'fold_block_size_px':16,
        'minimum_fit_peak_gap':.02,'minimum_check_gain':.01,'minimum_positive_check_blocks':3,
        'minimum_distinct_check_block_samples':20,'materially_negative_check_blocks_allowed':0,
        'minimum_improving_rgb_bands':2,'materially_negative_rgb_bands_allowed':0,
        'reconcile_rule':'IF_DERIVATIVE_FIT_AND_CHECK_COMPUTABLE_REQUIRE_SAME_DIRECTION',
        'estimator_or_descriptor_not_selected_by_coverage':True,
        'nominal_year':2019,'source_actual_date':'2018-11-01','external_independent_accuracy':False,
        'no_new_labels_or_displacement_applications_this_run':True,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}}
    out.mkdir();save_json(out/'FROZEN_REFERENCE_CORE_REPLAY_PLAN.json',frozen)
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    def read(p):
        with rasterio.open(p) as s:
            if Grid(s.transform,s.width,s.height,str(s.crs))!=grid:raise ValueError('Source/reference crop grid drift')
            return s.read()
    raw=read(inputs['raw'])[0];ref=read(inputs['reference'])[0];rq=read(inputs['manual_quality'])
    source=np.moveaxis(read(inputs['source_rgb']),0,-1)
    source_valid=np.isfinite(source).all(axis=2)&np.any(source>0,axis=2)
    cohort=reference_core_cohort(raw,ref,rq[1].astype(bool),rq[2].astype(bool),source_valid)
    mask=cohort['mask'];save_json(out/'SOURCE_REFERENCE_CORE_COHORT.json',{
        k:v for k,v in cohort.items() if k!='mask'})
    export_raster(out/'SOURCE_REFERENCE_CORE_COHORT.tif',mask[None].astype(np.uint8),grid,
                  ['FIXED_UNSHIFTED_SOURCE_REFERENCE_CORROBORATED_COHORT'],root=out)
    with rasterio.open(inputs['target_rgb']) as s:
        window=Window(1516,1765,256,256);target=np.moveaxis(s.read([1,2,3],window=window),0,-1)
        target_valid=(s.read_masks([1,2,3],window=window)>0).all(axis=0)
        if s.window_transform(window)!=grid.affine:raise ValueError('Target grid drift')
    fields=json.loads(inputs['field'].read_text(encoding='utf-8'))['cells']
    if len(fields)!=16:raise ValueError('Frozen scope mismatch')
    legacy={r['region_id']:r for r in json.loads(old.read_text(encoding='utf-8'))['records']}
    records=[];candidate_field=np.full((2,256,256),-9999.,np.float32)
    changed=source.copy();changed[~mask]=np.nan
    for cell in fields:
        rid=cell['region_id'];rr,cc=(int(x[1:]) for x in rid.split('_')[1:])
        scope=np.zeros((256,256),bool);scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
        point=point_roof_check(source,target,mask,target_valid,scope)
        derivative=source_template_check(source,target,mask,target_valid,scope)
        agreement=reconcile_roof_descriptors(point,derivative)
        # Perturb every excluded source value: no hidden tree/ground data may affect the result.
        if point_roof_check(changed,target,mask,target_valid,scope)!=point:
            raise ValueError('Excluded-source point evidence leak: '+rid)
        if source_template_check(changed,target,mask,target_valid,scope)!=derivative:
            raise ValueError('Excluded-source derivative evidence leak: '+rid)
        vector=None
        if point['best'] is not None:
            vector=[10*point['best']['correction_dx_px'],-10*point['best']['correction_dy_px']]
        supported=agreement['experimental_recommendation']
        if supported:candidate_field[:,scope]=np.array(vector,np.float32)[:,None]
        row={'region_id':rid,'source_cohort_pixels':int((mask&scope).sum()),
            'point_diagnosis':point,'derivative_diagnosis':derivative,'descriptor_reconciliation':agreement,
            'diagnostic_candidate_vector_m':vector,'experimental_supported_vector_m':vector if supported else None,
            'old_point_supported_vector_m':legacy[rid]['recommended_experiment_vector_m'],
            'manual_reference_position_used_for_source_population':True,
            'source_roof_purity_certified':False,'formal_alignment_accepted':False,'candidate_applied':False}
        records.append(row)
        print(rid,point['fit_fixed_population_n'],point['check']['n'],vector,supported,agreement['reasons'],flush=True)
    export_raster(out/'REFERENCE_CORE_CANDIDATE_FIELD_NOT_APPLIED.tif',candidate_field,grid,
                  ['DIAGNOSTIC_DX_M','DIAGNOSTIC_DY_M'],root=out,nodata=-9999.)
    old_supported={rid for rid,r in legacy.items() if r['recommended_experiment_vector_m'] is not None}
    new_supported={r['region_id'] for r in records if r['experimental_supported_vector_m'] is not None}
    summary={'source_units_evaluated':16,'source_prior_pixels':cohort['source_prior_pixels'],
        'reference_corroborated_source_pixels':int(mask.sum()),'experimental_supported_units':len(new_supported),
        'newly_supported_units_vs_legacy':sorted(new_supported-old_supported),
        'legacy_units_not_supported_by_new_cohort':sorted(old_supported-new_supported),
        'excluded_source_perturbation_verified_units':16,'new_displacement_applications':0,
        'new_training_labels':0,'formal_alignment_acceptances':0,'external_unseen_validation':False,
        'absolute_position_accuracy':'UNKNOWN','whole_window_governance_completed':False}
    save_json(out/'REFERENCE_CORE_ALIGNMENT_REPLAY.json',{'summary':summary,'records':records})
    sys.path.insert(0,str(ROOT/'tests'));names=sorted(p.stem for p in (ROOT/'tests').glob('test_xian_*.py')
        if p.stem!='test_xian_additive_post_inference')
    stream=io.StringIO();tr=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    write_new(out/'REGRESSION_TEST_LOG.txt',stream.getvalue())
    if not tr.wasSuccessful():raise ValueError('Regression failed')
    write_new(out/'REFERENCE_CORE_REPLAY_REPORT.md',f'''# 人工参考参与的固定源取样检查

当前窗口16个来源支持域均实际复核。原始源屋顶占比≥0.8且人工屋顶GT未平移占比≥0.8，排除GT重叠及窗口截断风险，
得到{int(mask.sum())}个参考一致源取样像元。它们只是位置参考参与筛选的固定取样，不是已独立认证的原生屋顶，也不生成地面tie。

复用旧点RGB与受支持方向差分检查：±30m搜索，100个不同FIT及CHECK像元，峰差0.02，增益0.01，至少3个20样本正增益空间块，
没有明显负增益块；RGB至少2通道改善且无明显负通道。所有候选使用同一源取样，不按GT移位动态挑样本。
目标特征支持域仍仅由目标影像有效性、冻结来源范围与空间fold决定；GT位置确实用于源取样，必须明确披露，不称为无参考信息的检查。

{len(new_supported)}个来源单元通过本次内部实验门槛。相对旧点RGB分支新增{sorted(new_supported-old_supported)}，
旧支持但新取样不支持{sorted(old_supported-new_supported)}。这不是选择覆盖更大的分支，更不是自动ACCEPT。
每个来源支持域还实际将全部排除的源RGB改为NaN复算；两种描述结果均保持一致，证明没有隐含特征污染。

本轮未应用新候选、未生成新标签、未覆盖此前10,096像元推荐交付。完整{tr.testsRun}项相关unittest通过；依赖pytest的旧模块未运行。
未知绝对精度、已知2018-11-01源日期与名义2019实验角色均保留。人工GT已有开发暴露，不能声称外部独立检验。
codebase-design使参考参与源取样的规则和真实信息来源集中在一个独立模块中，旧模块、旧合同保持不变。
''')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input changed: '+path)
    save_json(out/'REFERENCE_CORE_REPLAY_RECEIPT.json',{'summary':summary,'tests_run':tr.testsRun,
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'new_code_sha256':{str(f):sha(f) for f in code},
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
