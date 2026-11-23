"""Deliver all-pixel dispositions, fraction effects, lineage, and a safe bundle."""
from pathlib import Path
import argparse, hashlib, json, sys, zipfile
import numpy as np
import rasterio, fiona
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import shape
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from run_xian_source_partition_alignment import sha, save_json
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.source_partition import SourceRegion
from xian_r5.fraction_shadow import build_fraction_shadow, NODATA
from xian_r5.alignment.point_roof_alignment import point_roof_check
from xian_r5.transfer_admission import transfer_admission, DISPOSITIONS
from xian_r5.single_year_experiment import experiment_batch


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--tests-passed', type=int, required=True); args = p.parse_args(); root = args.run.resolve()
    parent = Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if root.parent != parent.resolve() or (root / 'POINT_RGB_DELIVERY_RECEIPT.json').exists():
        raise ValueError('Append-only dedicated run required')
    plan = json.loads((root / 'FROZEN_POINT_RGB_PLAN.json').read_text(encoding='utf-8'))
    replay = json.loads((root / 'POINT_RGB_REPLAY.json').read_text(encoding='utf-8'))
    if not replay['cross_descriptor_supported_cells']: raise ValueError('No supported experiment; do not generate empty packages')
    protected = dict(plan['protected_sha256']); protected.update(plan['new_code_sha256_at_freeze'])
    for path, digest in protected.items():
        if sha(path) != digest: raise RuntimeError('Protected input/code drift: ' + path)
    base = parent / 'SINGLE2019_20261012_06'; grid = Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    names = ['2019_POINT_RGB_FRACTION.tif', '2019_POINT_RGB_EXPERIMENT_LABEL.tif',
             '2019_POINT_RGB_EXPERIMENT_WEIGHT.tif', '2019_POINT_RGB_QUALITY.tif']
    read = []
    for name in names:
        with rasterio.open(root / name) as s:
            if (s.transform != grid.affine or s.crs.to_string() != grid.crs or s.shape != (256,256)
                    or s.tags().get('MEASURED_GT') != 'NO'): raise RuntimeError('Output grid/role mismatch')
            read.append(s.read())
    fraction, labels, weights, quality = read[0][0], read[1][0], read[2][0], read[3]
    with rasterio.open(base / '2019_ROOF_FRACTION_RAW.tif') as s: raw = s.read(1)
    with rasterio.open(base / '2019_S1S2_INPUT_WINDOW.tif') as s:
        inputs = s.read(); input_valid = np.isfinite(inputs).all(axis=0) & (s.read_masks() > 0).all(axis=0)
        if s.transform != grid.affine or s.crs.to_string() != grid.crs: raise RuntimeError('Input grid drift')
    admitted = transfer_admission(raw.astype(np.float64), fraction.astype(np.float64), quality, labels, weights, input_valid)
    if admitted['summary']['experimental_selected'] != replay['experimental_samples']: raise RuntimeError('Selection count drift')
    batch = experiment_batch(inputs, labels, weights)
    if batch['optimization_steps'] or not np.array_equal(batch['target'] == -100, labels == 255):
        raise RuntimeError('Ignore interface disagreement')
    cells = json.loads((parent / 'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    by_id = {r['region_id']: r for r in replay['records']}; regions = []
    selected_scope = np.zeros((256,256), bool)
    for cell in cells:
        r = by_id[cell['region_id']]; v = r['recommended_experiment_vector_m']
        regions.append(SourceRegion(cell['region_id'], '2018-11-01', shape(cell['support_geometry']),
            None if v is None else tuple(v), None if v is None else hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()))
        if v is not None:
            rr,cc=(int(z[1:]) for z in cell['region_id'].split('_')[1:]); selected_scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
    if (labels[~selected_scope] != 255).any(): raise RuntimeError('Unresolved scope gained training samples')
    with fiona.open(parent / 'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
        objects = {f.properties['component_key']: shape(f.geometry) for f in s}
    rebuilt = build_fraction_shadow(objects, regions, grid)
    if not np.array_equal(rebuilt['candidate_fraction'].astype(np.float32), fraction):
        raise RuntimeError('Fraction deterministic geometry replay differs')
    if not np.array_equal(rebuilt['quality'], quality): raise RuntimeError('Quality replay differs')
    # Lineage component areas must explain delivered float32 area, including
    # clipped fragments. They are not independent physical building counts.
    expected = sum(r['candidate_area_in_scope_m2'] for r in rebuilt['component_ledger'])
    delivered = float(fraction[fraction != NODATA].astype(np.float64).sum() * 100)
    if abs(expected - delivered) > .1: raise RuntimeError('Delivered roof area unexplained by lineage')
    write(root, '2019_FRACTION_DIFFERENCE.tif', admitted['difference'], grid,
        ['CANDIDATE_MINUS_RAW_NOT_ERROR_ENVELOPE'], nodata=NODATA)
    write(root, '2019_TRANSFER_DISPOSITION.tif', admitted['disposition'], grid,
        ['EXCLUSIVE_EXPERIMENT_DISPOSITION_NOT_GT'], extra={'DISPOSITION_CODES': json.dumps(DISPOSITIONS)})
    write(root, '2019_SOURCE_COMPONENT_COUNT.tif', rebuilt['component_count'], grid,
        ['RAW_CONNECTED_COMPONENT_COUNT', 'CANDIDATE_CONNECTED_COMPONENT_COUNT'],
        extra={'COUNT_SEMANTICS': 'NATIVE_MASK_COMPONENTS_NOT_PHYSICAL_BUILDINGS'})
    save_json(root / 'PIXEL_ADMISSION_AND_FRACTION_EFFECTS.json', admitted['summary'])
    # Actual excluded-pixel perturbation test across every cell. A diagnostic
    # offset must not depend on hidden source RGB outside its fixed roof mask.
    with rasterio.open(parent / 'SINGLE2019_SUPPORT_REPAIR_20261012_07/2019_BOUNDED_GE_RGB_CACHE.tif') as s:
        source = np.moveaxis(s.read(),0,-1)
    with rasterio.open('F:/xian/10m底图/Xian_2019_TrueColor.tif') as s:
        window=Window(1516,1765,256,256); target=np.moveaxis(s.read([1,2,3],window=window),0,-1)
        valid=(s.read_masks([1,2,3],window=window)>0).all(axis=0)
    roof=(raw>=.8)&np.any(source>0,axis=2); changed=source.copy(); changed[~roof]=np.nan
    for r in replay['records']:
        rr,cc=(int(z[1:]) for z in r['region_id'].split('_')[1:]); scope=np.zeros((256,256),bool)
        scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
        if point_roof_check(changed,target,roof,valid,scope)!=r['diagnosis']:
            raise RuntimeError('Excluded source values influence diagnostic: '+r['region_id'])
    qa={'raster_readbacks':4,'deterministic_geometry_fraction_replay_identical':True,
        'deterministic_quality_replay_identical':True,'excluded_source_perturbation_cells_unchanged':16,
        'excluded_source_pixels_modified':int((~roof).sum()),'lineage_candidate_area_m2':expected,
        'delivered_float32_area_m2':delivered,'float32_area_serialization_error_m2':abs(expected-delivered),
        'same_candidate_population_all_offsets':True,'ignore_target_count':int((batch['target']==-100).sum()),
        'optimization_steps':0,'measured_gt_pixels':0,'tests_passed':args.tests_passed}
    save_json(root / 'POINT_RGB_ENGINEERING_QA.json',qa)
    report = f'''# {root.name}

2019 单期候选转移继续推进，整窗口正式校正尚未完成。

本轮使用固定来源屋顶像元的原始 RGB，不使用目标手绘注释掩膜或 10m 模型预测。三个通道分别中心化计算空间相关，不能靠平均屋顶颜色获得虚假空间证据。原搜索范围±30m、100个不同FIT像元、100个不同CHECK像元、20个不同像元的检查块、峰分差0.02及至少3个正增益/无明显负增益检查块不放宽。新增两个通道改善且没有通道明显退化的门槛。对可计算的差分描述方向冲突直接排除，不按覆盖率选择描述器。

16个计算单元均完成数值诊断。6个单元通过单独RGB内部检查，4个方向冲突使部分候选不能推荐；最终4个单元为屋顶相对影像网格实验候选：roi_r1_c2、roi_r2_c1、roi_r3_c0、roi_r3_c1，均南移10m。不能外推给其余12个单元。这里的单元只是640m计算支持，不是经过认证的摄影角度分片。

实验选入{admitted['summary']['experimental_selected']:,} / 65,536个像元：屋顶阈值标签1为6,450，阈值下模型假设标签0为5,732，ignore为53,354。权重1仅表示实验选入，不是置信度或测量精度。正式GT仍为0。所有像元获得独占处置码，未解决不是背景。

同一选入像元上的raw fraction与candidate fraction在固定严格>0.43规则下翻转{admitted['summary']['threshold_flips']:,}个；保持类别{admitted['summary']['label_unchanged_comparison_pixels']:,}个。比较的是当前源掩膜重新聚合的raw，不是已追回旧训练的原始阈值前面积分数，也不是误差包络或独立精度。没有重新优化0.43阈值。

输出面积分数、差值、10层质量、处置、来源连接组件计数、实验标签/权重；分母始终整个100m²像元。先裁源支持，再位移，再裁目标；接缝、边界和冲突不进入实验标签。全部来源片段可追溯，不把连接组件当真实建筑身份。独立重建与交付fraction/quality一致，float32面积序列化误差{abs(expected-delivered):.8f}m²。原始影像被排除来源像元改为NaN后16个单元诊断均不变。

10的roi_r2_c2差分候选仍保留原文件，但本轮RGB局部检查不一致，因此不并入本轮推荐；这是另一描述器的失败证据，不是正式REJECT。07的地面代理roi_r1_c0与本轮屋顶对齐估计对象不同，不相加、不合并成统一地面纠偏。

输入六波段已读回，ignore转为损失ignore_index=-100检查通过，实际归一化、模型前向和训练均未执行，权重未修改。来源实际2018-11-01保留；按用户授权作为2019实验归组，不制造单日2019日期或真实变化状态。

全ROI已用于多轮方法开发。内部空间检查不是未见过的独立验证；相关性、峰分差、10m像元尺寸都不能生成定位uncertainty。屋顶到参考图像的相对位置候选不是地面足迹，人工屋顶GT与Sentinel地面框架的独立绑定尚未建立。正式review/accept/reject=0，旧B04 unresolved=9；ground footprints/physical IDs/canonical=0，absolute accuracy UNKNOWN，citywide NO。

下一步停止通过反复切换匹配算子增加覆盖：应针对剩余单元审计实例污染、合并和支持分片，复用历史对象与原始影像。不得选择有利检查块后重新宣称独立验证，也不得将现有12,182实验像元升级为正式训练GT。
'''
    report_path=root/'POINT_RGB_DELIVERY_REPORT.md'
    if report_path.exists(): raise ValueError('Report already exists')
    report_path.write_text(report,encoding='utf-8')
    manifest={'branch':'ROOF_POINT_RGB_CROSS_DESCRIPTOR_EXPERIMENT', 'experiment_year':2019,
        'source_observed_date':'2018-11-01','alignment_scope':plan['alignment_scope'],
        'inputs':'2019_S1S2_INPUT_WINDOW.tif','label':names[1],'weight':names[2],'fraction':names[0],'quality':names[3],
        'experimental_samples':replay['experimental_samples'], 'source_cells':[r['region_id'] for r in replay['records'] if r['recommended_experiment_vector_m'] is not None],
        'nominal_year_policy':'USER_AUTHORIZED_NOMINAL_2019','threshold':.43,'threshold_comparison':'>',
        'ignore_class':255,'loss_ignore_index':-100,'weight_semantics':'SELECTION_NOT_CONFIDENCE',
        'normalization':'NOT_RUN','model_training_runs':0,'measured_gt_eligible':0,
        'absolute_position_accuracy':'UNKNOWN','ground_footprints':0,'ready_for_citywide':False,
        'do_not_merge_old_candidate_vectors':True,'independent_validation':False}
    save_json(root/'CURRENT_RECOMMENDED_DATASET.json',manifest)
    package=root/'SINGLE2019_POINT_RGB_EXPERIMENT.zip'
    files=[f for f in root.iterdir() if f.is_file()]+[base/'2019_S1S2_INPUT_WINDOW.tif',base/'2019_ROOF_FRACTION_RAW.tif']
    checksums={f.name:sha(f) for f in files}
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.name)
        z.writestr('FILES_SHA256.json',json.dumps(checksums,indent=2))
    with zipfile.ZipFile(package) as z:
        if z.testzip() is not None:raise RuntimeError('Package CRC failure')
        for name,digest in checksums.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise RuntimeError('Package hash failure')
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen artifact changed: '+path)
    new_code={str(f.resolve()):sha(f) for f in (Path(__file__),ROOT/'src/xian_r5/transfer_admission.py',ROOT/'tests/test_xian_transfer_admission.py')}
    receipt={'run':root.name,'package_crc_and_hash_verified':True,'tests_passed':args.tests_passed,
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'new_code_sha256':new_code,'output_sha256':{f.name:sha(f) for f in root.iterdir() if f.is_file()},
        'whole_window_formal_governance_completed':False,'manifest':manifest,'qa':qa,
        'formal_state':{'FORMAL_FIT_HOLDOUT_REVIEWS_COMPLETED':0,'ACCEPTED_ALIGNMENT_CELLS':0,
            'REJECTED_ALIGNMENT_CELLS':0,'UNRESOLVED_ALIGNMENT_CELLS':9,'REGISTERED_ROOFS':0,
            'PHYSICAL_IDS':0,'CANONICAL_GEOMETRIES':0,'GROUND_FOOTPRINTS':0,
            'ABSOLUTE_POSITION_ACCURACY':'UNKNOWN','READY_FOR_CITYWIDE':'NO'}}
    save_json(root/'POINT_RGB_DELIVERY_RECEIPT.json',receipt)
    print(json.dumps({'package':str(package),'summary':admitted['summary'],'qa':qa,
        'receipt_sha256':sha(root/'POINT_RGB_DELIVERY_RECEIPT.json')}))


if __name__=='__main__':main()
