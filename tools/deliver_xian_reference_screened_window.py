"""Whole current-window reference-screened experiment delivery, never formal GT."""
from pathlib import Path
import argparse,io,json,shutil,sys,unittest,zipfile
import numpy as np
import rasterio
from affine import Affine
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json,write_new
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.reference_label_filter import reference_label_filter
from xian_r5.single_year_experiment import experiment_batch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh window delivery required')
    prior=parent/'SINGLE2019_ROOF_PAIR_CHECK_20261012_22'
    plan=json.loads((prior/'FROZEN_PAIR_TRANSLATION_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'PAIR_TRANSLATION_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'PAIR_TRANSLATION_RECEIPT.json')]=sha(prior/'PAIR_TRANSLATION_RECEIPT.json')
    baseline=parent/'SINGLE2019_COHERENT_TRANSFER_20261012_12'
    manual=parent/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21'
    base_input=parent/'SINGLE2019_20261012_06/2019_S1S2_INPUT_WINDOW.tif'
    copied={'2019_ROOF_FRACTION_CANDIDATE.tif':baseline/'2019_ROOF_FRACTION.tif',
        '2019_SOURCE_QUALITY.tif':baseline/'2019_QUALITY.tif',
        '2019_FROZEN_CANDIDATE_FIELD.tif':baseline/'2019_FROZEN_CANDIDATE_FIELD.tif',
        '2019_SOURCE_COMPONENT_COUNT.tif':baseline/'2019_SOURCE_COMPONENT_COUNT.tif',
        '2019_SOURCE_DISPOSITION.tif':baseline/'2019_TRANSFER_DISPOSITION.tif',
        '2019_S1S2_INPUT_WINDOW.tif':base_input,
        '2019_MANUAL_REFERENCE_ROOF_FRACTION.tif':manual/'2019_MANUAL_REFERENCE_ROOF_FRACTION_10m.tif',
        'SOURCE_COMPONENT_LINEAGE.json':baseline/'SOURCE_COMPONENT_LINEAGE.json',
        'DERIVED_SUPPORT_LINEAGE.json':baseline/'DERIVED_SUPPORT_LINEAGE.json',
        'FROZEN_SOURCE_CELL_CANDIDATES.json':parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json',
        'USER_CONFIRMED_MANUAL_REFERENCE_BINDING.json':manual/'FROZEN_MANUAL_REFERENCE_BINDING.json',
        'FULLROI_PAIR_TRANSLATION_DIAGNOSTIC.json':prior/'FULLROI_PAIR_TRANSLATION_DIAGNOSTIC.json'}
    for f in copied.values():protected[str(f)]=sha(f)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen input drift: '+path)
    code=[Path(__file__).resolve(),ROOT/'src/xian_r5/reference_label_filter.py',
          ROOT/'tests/test_xian_reference_label_filter.py']
    out.mkdir();save_json(out/'FROZEN_WINDOW_REFERENCE_SCREEN_PLAN.json',{
        'scope':'ENTIRE_CURRENT_256x256_GRID_16_UNITS_NO_CITYWIDE',
        'new_geometric_calibration_fits_or_applications':False,
        'source_fraction_not_replaced_by_manual_reference':True,'threshold':.43,'comparison':'>',
        'filter':'ONLY_REMOVE inherited samples: require same reference threshold class, no manual polygon overlap, no ROI-truncated manual roof',
        'background_is_assumption':True,'positive_only_companion_required':True,
        'known_source_actual_date':'2018-11-01','experiment_year':2019,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}})
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    def read(p):
        with rasterio.open(p) as s:
            if Grid(s.transform,s.width,s.height,str(s.crs))!=grid:raise ValueError('Window grid drift')
            return s.read()
    label=read(baseline/'2019_EXPERIMENT_LABEL.tif')[0]
    weight=read(baseline/'2019_EXPERIMENT_WEIGHT.tif')[0]
    fraction=read(copied['2019_ROOF_FRACTION_CANDIDATE.tif'])[0]
    reference=read(copied['2019_MANUAL_REFERENCE_ROOF_FRACTION.tif'])[0]
    rq=read(manual/'2019_MANUAL_REFERENCE_QUALITY.tif')
    result=reference_label_filter(label,weight,fraction,reference,rq[1].astype(bool),rq[2].astype(bool))
    if result['counts']['inherited_experiment_selected']!=12903:raise ValueError('Inherited selection changed')
    for name,src in copied.items():
        shutil.copy2(src,out/name)
        if sha(out/name)!=protected[str(src)]:raise ValueError('Copied artifact differs')
    write(out,'2019_RECOMMENDED_PSEUDO_LABEL.tif',result['label'],grid,['EXPERIMENT_PSEUDO_LABEL'],nodata=255,
          extra={'MANUAL_REFERENCE_FILTER':'THRESHOLD_AGREEMENT_NO_OVERLAP_NO_TRUNCATED_ROOF'})
    write(out,'2019_RECOMMENDED_PSEUDO_WEIGHT.tif',result['weight'],grid,['EXPERIMENT_SELECTION_NOT_CONFIDENCE'])
    write(out,'2019_POSITIVE_ONLY_LABEL.tif',result['positive_only_label'],grid,['REFERENCE_CORROBORATED_POSITIVE_ONLY'],nodata=255)
    write(out,'2019_POSITIVE_ONLY_WEIGHT.tif',result['positive_only_weight'],grid,['POSITIVE_ONLY_SELECTION'])
    write(out,'2019_REFERENCE_SCREEN_QUALITY.tif',result['quality'],grid,result['quality_bands'])
    write(out,'2019_REFERENCE_EXCLUSION_FLAGS.tif',result['exclusion_flags'],grid,['BIT1_DISAGREE_BIT2_OVERLAP_BIT4_TRUNCATED'])
    inputs=read(base_input)
    b=experiment_batch(inputs,result['label'],result['weight'])
    p=experiment_batch(inputs,result['positive_only_label'],result['positive_only_weight'])
    if b['optimization_steps'] or p['optimization_steps']:raise ValueError('Training unauthorized')
    if int((b['target']==-100).sum())!=result['counts']['ignore']:raise ValueError('Ignore dataloader mismatch')
    # No selected samples may escape the inherited four supported units.
    scope=np.zeros((256,256),bool)
    for rr,cc in ((1,2),(2,1),(3,0),(3,1)):scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
    if np.any(result['weight'][~scope]):raise ValueError('Unresolved region gained samples')
    per_cell=[]
    for rr in range(4):
        for cc in range(4):
            sl=np.s_[rr*64:(rr+1)*64,cc*64:(cc+1)*64]
            per_cell.append({'region_id':f'roi_r{rr}_c{cc}',
                'experiment_samples':int(result['weight'][sl].sum()),
                'reference_positive_samples':int(result['positive_only_weight'][sl].sum()),
                'unresolved_or_ignored_pixels':int((result['label'][sl]==255).sum()),
                'formal_alignment_accepted':False,'absolute_position_accuracy':'UNKNOWN'})
    # Run the actual related test suite, not a caller-supplied green count.
    sys.path.insert(0,str(ROOT/'tests'))
    names=sorted(p.stem for p in (ROOT/'tests').glob('test_xian_*.py') if p.stem!='test_xian_additive_post_inference')
    stream=io.StringIO();testresult=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    write_new(out/'ENGINEERING_TEST_LOG.txt',stream.getvalue())
    if not testresult.wasSuccessful():raise ValueError('Related regression tests failed')
    summary={**result['counts'],'all_window_pixels_accounted':result['counts']['recommended_experiment_selected']+result['counts']['ignore']==65536,
        'source_units':16,'source_units_with_experiment_samples':sum(r['experiment_samples']>0 for r in per_cell),
        'source_units_without_experiment_samples':sum(r['experiment_samples']==0 for r in per_cell),
        'new_vector_applications':0,'new_physical_IDs_or_ground_footprints':0,'optimization_steps':0,
        'tests_run':testresult.testsRun,'tests_failed':len(testresult.failures),'tests_errors':len(testresult.errors),
        'pytest_only_module':'test_xian_additive_post_inference NOT_RUN_DEPENDENCY_ABSENT',
        'current_window_artifact_chain_closed':True,'all_window_alignment_verified':False,
        'whole_window_governance_completed':False,'absolute_position_accuracy':'UNKNOWN'}
    save_json(out/'WINDOW_REFERENCE_SCREEN_SUMMARY.json',{'summary':summary,'per_source_unit':per_cell})
    save_json(out/'RECOMMENDED_EXPERIMENT_DATASET.json',{
        'scope':'CURRENT_256x256_ONLY','inputs':'2019_S1S2_INPUT_WINDOW.tif',
        'fraction':'2019_ROOF_FRACTION_CANDIDATE.tif','label':'2019_RECOMMENDED_PSEUDO_LABEL.tif',
        'weight':'2019_RECOMMENDED_PSEUDO_WEIGHT.tif','source_quality':'2019_SOURCE_QUALITY.tif',
        'reference_quality':'2019_REFERENCE_SCREEN_QUALITY.tif',
        'positive_only_label':'2019_POSITIVE_ONLY_LABEL.tif','positive_only_weight':'2019_POSITIVE_ONLY_WEIGHT.tif',
        'threshold':.43,'comparison':'>','ignore_class':255,'loss_ignore_index':-100,
        'grid':{'crs':grid.crs,'shape':[256,256],'affine':list(grid.affine)[:6],
                'gdal_geotransform':[grid.affine.c,grid.affine.a,grid.affine.b,grid.affine.f,grid.affine.d,grid.affine.e]},
        'source_actual_date':'2018-11-01','experiment_year':2019,'year_policy':'USER_AUTHORIZED_NOMINAL_YEAR',
        'weight_semantics':'EXPERIMENT_SAMPLE_SELECTION_NOT_CONFIDENCE','normalization':'NOT_RUN',
        'training_use':'PSEUDO_LABEL_EXPERIMENT_ONLY_NOT_FORMAL_GT',
        'negative_class':'ASSUMED_BELOW_THRESHOLD_NOT_VERIFIED_ABSENCE',
        'formal_alignment_accepted':False,'external_unseen_accuracy_verified':False,
        'complete_background_reference':'UNKNOWN','absolute_position_accuracy':'UNKNOWN',
        'minimum_warning':'Positive-only companion has no negative examples; cannot treat as balanced complete training set'})
    counts=result['counts']
    write_new(out/'CURRENT_WINDOW_DELIVERY_REPORT.md',f'''# 当前256×256窗口：人工参考筛选后的伪标签交付

按用户最新范围要求，仅治理当前窗口，不扩整市。这里“交付链闭合”不代表所有位置已经可靠校正。

整个65,536像元均已登记：推荐实验选入{counts['recommended_experiment_selected']:,}，ignore为{counts['ignore']:,}。
其中{counts['reference_corroborated_positive']:,}为与2019人工屋顶GT阈值类别一致的正例，
{counts['assumed_background']:,}为模型阈值下背景假设，不是已验证无建筑。
仅正例版本不使用背景假设，但不是具备正负样本的完整训练集。

旧版12,903个实验像元中，人工GT类别冲突{counts['reference_disagreement_exclusions']:,}个；
标注重叠风险{counts['reference_overlap_exclusions']:,}个；标注屋顶在窗口截断风险{counts['reference_truncation_exclusions']:,}个。
原因允许重叠，合并剔除{counts['union_reference_exclusions']:,}个。筛选只删除，不增加未经支持像元，不将GT直接替换源fraction。

仍沿用旧的四个通过内部图像检查的实验单元及其南移10m候选；其余12个单元全部ignore。
267对冻结形态假设的公共平移一致性诊断未产生可应用的新向量。旧源几何、GT、位移场和全部旧交付不改。
源影像真实2018-11-01仍保留；名义2019按用户授权用于单年实验。0.43严格大于规则不改，不做阈值搜索。

包含源面积分数、候选场、原质量、人工参考fraction、新质量、排除原因、对象来源、六波段输入及读取清单。
标签255转loss ignore_index=-100；权重仅表示实验选入。标签与输入同EPSG:32649、同10m原点及256×256网格。

实际运行{testresult.testsRun}项相关unittest，无失败或错误。一个依赖pytest的旧模块未运行，未算通过。
没有训练或加载最终模型权重，没有新物理ID、正式人工review或地面足迹；正式GT准入仍0。

本次采用codebase-design，把参考筛选、原因登记和双版本标签统一在一个可测试模块里，避免不同导出器各自解释筛选规则。
人工参考已有开发暴露，不能把过滤后与GT一致率当作独立验证精度；位置误差上界和绝对位置精度仍未知。
当前窗口尚有未可靠校正区域，不能宣布空间治理全部完成。后续仅针对这些当前窗口区域处理，不扩展范围。
''')
    # Freeze all currently written outputs into the archive manifest; archive never contains itself.
    save_json(out/'PACKAGE_MANIFEST.json',{'files_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},
        'package_role':'REFERENCE_SCREENED_SINGLE2019_EXPERIMENT_WITH_EXPLICIT_UNKNOWN_REGIONS'})
    archive=out/'CURRENT_WINDOW_REFERENCE_SCREENED_EXPERIMENT.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f!=archive:z.write(f,f.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failure')
        if z.read('RECOMMENDED_EXPERIMENT_DATASET.json')!=(out/'RECOMMENDED_EXPERIMENT_DATASET.json').read_bytes():
            raise ValueError('Packaged reader manifest differs')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input changed: '+path)
    save_json(out/'CURRENT_WINDOW_DELIVERY_RECEIPT.json',{'summary':summary,
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'new_code_sha256':{str(f):sha(f) for f in code},
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
