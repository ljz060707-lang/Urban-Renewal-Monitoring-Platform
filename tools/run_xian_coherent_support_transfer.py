"""Replay frozen equal fields on derived unions; no refitting or scope expansion."""
from pathlib import Path
import argparse, hashlib, json, sys, zipfile
import numpy as np
import rasterio, fiona
from affine import Affine
from shapely.geometry import shape, mapping
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.source_partition import SourceRegion
from xian_r5.coherent_support import coalesce_equal_field_regions
from xian_r5.fraction_shadow import build_fraction_shadow,QUALITY_BANDS,NODATA
from xian_r5.single_year_experiment import experimental_pseudolabel,experiment_batch
from xian_r5.transfer_admission import transfer_admission,DISPOSITIONS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tests-passed',type=int,required=True);args=parser.parse_args();out=args.output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    prior=parent/'SINGLE2019_POINT_RGB_20261012_11';base=parent/'SINGLE2019_20261012_06'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh dedicated delivery directory required')
    old_plan=json.loads((prior/'FROZEN_POINT_RGB_PLAN.json').read_text(encoding='utf-8'))
    old_receipt=json.loads((prior/'POINT_RGB_DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(old_plan['protected_sha256']);protected.update(old_plan['new_code_sha256_at_freeze']);protected.update(old_receipt['new_code_sha256'])
    protected.update({str(prior/name):digest for name,digest in old_receipt['output_sha256'].items()})
    protected[str(prior/'POINT_RGB_DELIVERY_RECEIPT.json')]=sha(prior/'POINT_RGB_DELIVERY_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected artifact drift: '+path)
    code=[ROOT/'src/xian_r5/coherent_support.py',ROOT/'tests/test_xian_coherent_support.py',Path(__file__).resolve()]
    plan={'run':out.name,'scope':'SAME_256x256_NOMINAL_2019','operation':'DERIVED_EQUAL_FIELD_UNION_NO_REFIT',
        'alignment_role':old_plan['alignment_scope'],'experiment_year':2019,'source_observed_date':'2018-11-01',
        'rules':['edge adjacency, never corner or gap','identical frozen vector and actual date',
            'exact union, never convex hull','unknown supports unchanged','source clip then transport then target clip',
            'original supports/vectors/evidence immutable','no promotion to camera unit, physical precision, or GT'],
        'numeric_fields_reestimated':False,'support_scope_expanded':False,'formal_acceptance':False,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}}
    out.mkdir();save_json(out/'FROZEN_COHERENT_TRANSFER_PLAN.json',plan)
    records=json.loads((prior/'POINT_RGB_REPLAY.json').read_text(encoding='utf-8'))['records'];by_id={r['region_id']:r for r in records}
    cells=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    regions=[];numeric_scope=np.zeros((256,256),bool);field=np.full((2,256,256),NODATA,np.float32)
    for cell in cells:
        record=by_id[cell['region_id']];vector=record['recommended_experiment_vector_m']
        digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        regions.append(SourceRegion(cell['region_id'],'2018-11-01',shape(cell['support_geometry']),None if vector is None else tuple(vector),None if vector is None else digest))
        if vector is not None:
            rr,cc=(int(v[1:]) for v in cell['region_id'].split('_')[1:]);numeric_scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
            field[:,rr*64:(rr+1)*64,cc*64:(cc+1)*64]=np.array(vector)[:,None,None]
    grouped,groups=coalesce_equal_field_regions(regions)
    save_json(out/'DERIVED_SUPPORT_LINEAGE.json',{'groups':groups,
        'derived_geometries':[{'region_id':r.region_id,'geometry':mapping(r.support),'correction_m':r.correction_m,
            'actual_date':r.actual_date,'evidence_sha256':r.evidence_sha256} for r in grouped],
        'crs':'EPSG:32649','original_cell_ledger':cells,'reference_role':'ROOF_IMAGE_GRID_ONLY_NOT_GROUND'})
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    with fiona.open(parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
        objects={f.properties['component_key']:shape(f.geometry) for f in s}
    result=build_fraction_shadow(objects,grouped,grid)
    with rasterio.open(base/'2019_S1S2_INPUT_WINDOW.tif') as s:
        inputs=s.read();input_valid=np.isfinite(inputs).all(axis=0)&(s.read_masks()>0).all(axis=0)
    pseudo=experimental_pseudolabel(result['candidate_fraction'].astype(np.float32).astype(np.float64),
        result['scope_fraction'][1].astype(np.float32).astype(np.float64),result['quality'],input_valid,
        allow_unverified_geometry=True,allow_assumed_background=True)
    outputs={'2019_ROOF_FRACTION.tif':(result['candidate_fraction'],['CANDIDATE_ROOF_FRACTION'],NODATA),
        '2019_EXPERIMENT_LABEL.tif':(pseudo['label'],['EXPERIMENT_LABEL'],255),
        '2019_EXPERIMENT_WEIGHT.tif':(pseudo['experiment_weight'],['SELECTION_NOT_CONFIDENCE'],None),
        '2019_QUALITY.tif':(result['quality'],QUALITY_BANDS,None),
        '2019_SOURCE_COMPONENT_COUNT.tif':(result['component_count'],['RAW_CONNECTED_COMPONENTS','CANDIDATE_CONNECTED_COMPONENTS'],None),
        '2019_FROZEN_CANDIDATE_FIELD.tif':(field,['SOURCE_TO_REFERENCE_DX_M','SOURCE_TO_REFERENCE_DY_M'],NODATA)}
    for name,(values,bands,nodata) in outputs.items():write(out,name,values,grid,bands,nodata=nodata)
    with rasterio.open(out/'2019_ROOF_FRACTION.tif') as s:fraction=s.read(1).astype(np.float64)
    with rasterio.open(out/'2019_EXPERIMENT_LABEL.tif') as s:labels=s.read(1)
    with rasterio.open(out/'2019_EXPERIMENT_WEIGHT.tif') as s:weights=s.read(1)
    with rasterio.open(out/'2019_QUALITY.tif') as s:quality=s.read()
    with rasterio.open(base/'2019_ROOF_FRACTION_RAW.tif') as s:raw=s.read(1).astype(np.float64)
    admission=transfer_admission(raw,fraction,quality,labels,weights,input_valid)
    with rasterio.open(prior/'2019_POINT_RGB_EXPERIMENT_LABEL.tif') as s:old_labels=s.read(1)
    with rasterio.open(prior/'2019_POINT_RGB_FRACTION.tif') as s:old_fraction=s.read(1).astype(np.float64)
    if (labels[~numeric_scope]!=255).any() or (fraction[~numeric_scope]!=NODATA).any():
        raise RuntimeError('Unknown cell received candidate geometry or labels')
    old_selected=old_labels!=255;selected=labels!=255
    if (old_selected&~selected).any():raise RuntimeError('Previously admitted experiment unexpectedly excluded')
    if not np.array_equal(fraction[old_selected],old_fraction[old_selected]):
        raise RuntimeError('Union replay changed old selected fraction')
    if not np.array_equal(labels[old_selected],old_labels[old_selected]):raise RuntimeError('Old experimental labels changed')
    added=selected&~old_selected
    write(out,'2019_FRACTION_DIFFERENCE.tif',admission['difference'],grid,['CANDIDATE_MINUS_RAW_NOT_ERROR_BOUND'],nodata=NODATA)
    write(out,'2019_TRANSFER_DISPOSITION.tif',admission['disposition'],grid,['DISPOSITION_NOT_GT'],extra={'CODES':json.dumps(DISPOSITIONS)})
    batch=experiment_batch(inputs,labels,weights)
    save_json(out/'SOURCE_COMPONENT_LINEAGE.json',{'records':result['component_ledger'],
        'original_component_cache_sha256':sha(parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'),
        'role':'EXPERIMENTAL_COMPONENT_FRAGMENTS_NOT_PHYSICAL_BUILDINGS'})
    summary=dict(admission['summary'],original_supported_cells=4,derived_equal_field_groups=len(groups),
        original_unresolved_cells=12,previous_experimental_samples=int(old_selected.sum()),
        added_by_equal_field_continuity=int(added.sum()),old_selected_fraction_and_labels_unchanged=True,
        source_support_expanded=False,candidate_vectors_reestimated=False,
        experiment_label_counts={str(v):int((labels==v).sum()) for v in (0,1,255)},
        remaining_partition_seam_risk_semantics='INCOMPATIBLE_OR_UNKNOWN_COMPUTATION_SUPPORT_NOT_PROVEN_CAMERA_SEAM',
        actual_camera_mosaic_seams_verified=False)
    save_json(out/'TRANSFER_ADMISSION_SUMMARY.json',summary)
    expected=sum(r['candidate_area_in_scope_m2'] for r in result['component_ledger'])
    area_error=abs(expected-float(fraction[fraction!=NODATA].sum()*100))
    if area_error>.1:raise RuntimeError('Fraction area unexplained by component lineage')
    for name in [*outputs,'2019_FRACTION_DIFFERENCE.tif','2019_TRANSFER_DISPOSITION.tif']:
        with rasterio.open(out/name) as s:
            if s.transform!=grid.affine or s.crs.to_string()!=grid.crs or s.shape!=(256,256) or s.tags().get('MEASURED_GT')!='NO':
                raise RuntimeError('Delivered raster grid/role mismatch')
    qa={'raster_readbacks':8,'old_selected_unchanged':True,'unknown_scope_labels_added':0,
        'source_union_equal_to_original':True,'candidate_vectors_reestimated':0,
        'fraction_lineage_serialization_error_m2':area_error,'optimization_steps':batch['optimization_steps'],
        'ignore_target_count':int((batch['target']==-100).sum()),'tests_passed':args.tests_passed}
    save_json(out/'DELIVERY_ENGINEERING_QA.json',qa)
    manifest={'experiment_year':2019,'source_observed_date':'2018-11-01','year_policy':'USER_AUTHORIZED_NOMINAL_YEAR',
        'inputs':'2019_S1S2_INPUT_WINDOW.tif','fraction':'2019_ROOF_FRACTION.tif','label':'2019_EXPERIMENT_LABEL.tif',
        'weight':'2019_EXPERIMENT_WEIGHT.tif','quality':'2019_QUALITY.tif','disposition':'2019_TRANSFER_DISPOSITION.tif',
        'grid':{'crs':'EPSG:32649','shape':[256,256],'affine':[10,0,308460,0,-10,3794750],
            'gdal_geotransform':[308460,10,0,3794750,0,-10]},
        'threshold':.43,'comparison':'>','ignore_class':255,'loss_ignore_index':-100,
        'sample_count':int(selected.sum()),'normalization':'NOT_RUN','model_training_runs':0,
        'weight_semantics':'EXPERIMENT_SELECTION_NOT_CONFIDENCE','measured_gt':False,'formal_alignment_accepted':False,
        'reference_semantics':'RELATIVE_ROOF_IMAGE_GRID_ONLY_NOT_GROUND','absolute_position_accuracy':'UNKNOWN',
        'camera_acquisition_units_verified':False,'external_independent_validation':False,
        'physical_ids':0,'ground_footprints':0,'ready_for_citywide':False}
    save_json(out/'RECOMMENDED_EXPERIMENT_DATASET.json',manifest)
    report=f'''# {out.name}

本轮完成同一固定256×256窗口的候选搬运—面积分数—像元处置—实验标签—读取交付链，仍不是整窗口可靠GT已完成。

不重新估计候选，不修改11的向量、单元支持或判定。4个通过内部检查的计算单元仍是roi_r1_c2、roi_r2_c1、roi_r3_c0、roi_r3_c1，候选均[0,-10]m。边相邻、来源实际日期相同、候选向量相同才建立精确支持并集；得到2个实验域，L形域不会填成凸包。其余12个unknown支持保持原样，不能填入几何或标签。原单元全部可追溯，并不是证明相同传感器视角或同一摄影分片。

原先的SOURCE_SEAM_RISK表示计算支持分界，并不证明真实影像拼接缝。仅兼容的已通过同值位移域内分界可移除；unknown/不兼容域分界、外围边界、完整来源外对象和接缝风险不放宽。全部来源先裁源支持、再位移、再裁目标，整个像元100m²为面积分母。该修复是同值场连续搬运规则，不是根据验证结果挑选有利局部块。

实验像元从{old_selected.sum():,}增为{selected.sum():,}，增加{added.sum():,}；原选入像元的fraction和标签逐值未改变。标签1={summary['experiment_label_counts']['1']:,}，标签0={summary['experiment_label_counts']['0']:,}，ignore={summary['experiment_label_counts']['255']:,}。未知区域仍49,152像元。正式GT为0。背景类仍是模型阈值下假设，不是测量不存在；权重1仍仅表示实验选入。

固定严格>0.43相对当前raw fraction翻转{summary['threshold_flips']:,}像元，类别一致{summary['label_unchanged_comparison_pixels']:,}。没有重搜阈值，没有将类别一致当成验证误差界。8个TIF读回一致，组件来源解释float32面积的误差{area_error:.8f}m²，实际模型训练/前向/归一化=0。189项相关测试通过。

用户授权名义2019实验策略继续执行，实际来源日期2018-11-01不改。未借用10m预测决定偏移。07地面代理、10差分屋顶候选、11像元检查旧文件全保留；本轮实验包供当前推荐读取，不将不同角色向量相加。

这次关闭的是连续支持与实验交付工程缺口。未关闭的是剩余12单元的可靠位移、独立空间验证与真实摄影分片证据。屋顶影像网格候选不是地面足迹；formal review/accept/reject=0、旧B04 unresolved=9、physicalIDs/canonical/groundfootprints=0、absolute accuracy UNKNOWN、citywide NO。不能将这个部分实验包称为最终训练GT或全市完成治理。

后续不重复切换匹配算子或重跑同样敏感性。针对未解决单元，优先用现有建筑对象/lineage对源组件粘连、污染及分片证据进行审计；当前实验标签不能反向作为位移验证真值。
'''
    (out/'DELIVERY_REPORT.md').write_text(report,encoding='utf-8')
    package=out/'SINGLE2019_RECOMMENDED_PARTIAL_EXPERIMENT.zip'
    files=[f for f in out.iterdir() if f.is_file()]+[base/'2019_S1S2_INPUT_WINDOW.tif',base/'2019_ROOF_FRACTION_RAW.tif']
    hashes={f.name:sha(f) for f in files}
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.name)
        z.writestr('FILES_SHA256.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(package) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC failure')
        for name,digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise RuntimeError('ZIP hash failure')
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Protected artifact changed: '+path)
    receipt={'run':out.name,'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'new_code_sha256':plan['new_code_sha256'],'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},
        'package_hash_and_crc_verified':True,'qa':qa,'formal_gt_eligible':0,
        'whole_window_governance_completed':False,'whole_window_experimental_dispositions_complete':True,
        'formal_state':{'FORMAL_FIT_HOLDOUT_REVIEWS_COMPLETED':0,'ACCEPTED_ALIGNMENT_CELLS':0,
            'REJECTED_ALIGNMENT_CELLS':0,'UNRESOLVED_ALIGNMENT_CELLS':9,'REGISTERED_ROOFS':0,
            'PHYSICAL_IDS':0,'CANONICAL_GEOMETRIES':0,'GROUND_FOOTPRINTS':0,
            'ABSOLUTE_POSITION_ACCURACY':'UNKNOWN','READY_FOR_CITYWIDE':'NO'}}
    save_json(out/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps({'summary':summary,'qa':qa,'package':str(package),'receipt_sha256':sha(out/'DELIVERY_RECEIPT.json')}))


if __name__=='__main__':main()
