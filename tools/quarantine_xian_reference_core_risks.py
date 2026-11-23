"""Remove known-negative image-check blocks; keep source/target risk frames apart."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,zipfile
import numpy as np,rasterio
from affine import Affine
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json,write_new
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.single_year_experiment import experiment_batch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh append-only risk quarantine required')
    base=parent/'SINGLE2019_REFERENCE_SCREENED_WINDOW_20261012_23'
    diagnostic=parent/'SINGLE2019_REFERENCE_CORE_REPLAY_20261012_24_02'
    plan=json.loads((diagnostic/'FROZEN_REFERENCE_CORE_REPLAY_PLAN.json').read_text(encoding='utf-8'))
    prior=json.loads((diagnostic/'REFERENCE_CORE_REPLAY_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(diagnostic/n):h for n,h in prior['output_sha256'].items()})
    protected[str(diagnostic/'REFERENCE_CORE_REPLAY_RECEIPT.json')]=sha(diagnostic/'REFERENCE_CORE_REPLAY_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected evidence drift: '+path)
    out.mkdir();save_json(out/'FROZEN_RISK_QUARANTINE_PLAN.json',{
        'scope':'CURRENT_256x256_FULL_WINDOW','policy':'ONLY_REMOVE_INHERITED_EXPERIMENT_SAMPLES',
        'reason':'MATERIAL_NEGATIVE_POINT_RGB_CHECK_BLOCK_ON_MANUAL_CORROBORATED_COHORT',
        'minimum_distinct_samples':20,'negative_gain_threshold':-.01,'source_block_size_px':16,
        'mapping':'SOURCE risk to TARGET via already frozen legacy vector only; no new vector application',
        'unknown_vector_blocks_stay_SOURCE_FRAME_ONLY':True,'formal_accept_or_reject':False,
        'protected_sha256':protected,'new_code_sha256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())}})
    records=json.loads((diagnostic/'REFERENCE_CORE_ALIGNMENT_REPLAY.json').read_text(encoding='utf-8'))['records']
    src_risk=np.zeros((256,256),np.uint8);target_risk=src_risk.copy();lineage=[]
    for r in records:
        for b in r['point_diagnosis']['check_blocks']:
            if b['n']<20 or b['gain'] is None or b['gain']>=-.01:continue
            sr,sc=b['row']*16,b['col']*16
            src_risk[sr:sr+16,sc:sc+16]=1
            vector=r['old_point_supported_vector_m'];mapped=None
            if vector is not None:
                if vector!=r['diagnostic_candidate_vector_m']:raise ValueError('New and frozen vector disagree; cannot map block silently')
                dx=vector[0]/10;dy=-vector[1]/10
                if dx!=int(dx) or dy!=int(dy):raise ValueError('Exact pixel-grid frozen vector required')
                tr,tc=sr+int(dy),sc+int(dx)
                if not 0<=tr<=240 or not 0<=tc<=240:raise ValueError('Risk support would leave ROI')
                target_risk[tr:tr+16,tc:tc+16]=1;mapped=[tc,tr,16,16]
            lineage.append({'region_id':r['region_id'],'source_pixel_window':[sc,sr,16,16],
                'target_pixel_window':mapped,'mapping_vector_m':vector,'distinct_source_samples':b['n'],
                'check_gain':b['gain'],'risk_only_not_formal_reject':True})
    if len(lineage)!=7 or sum(x['target_pixel_window'] is not None for x in lineage)!=2:
        raise ValueError('Known material-negative block census drift')
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    def read(name):
        with rasterio.open(base/name) as s:
            if Grid(s.transform,s.width,s.height,str(s.crs))!=grid:raise ValueError('Baseline grid drift')
            return s.read()
    labels=read('2019_RECOMMENDED_PSEUDO_LABEL.tif')[0];weights=read('2019_RECOMMENDED_PSEUDO_WEIGHT.tif')[0]
    old_selection=weights>0;excluded=old_selection&target_risk.astype(bool)
    labels[excluded]=255;weights[excluded]=0
    positives=np.full(labels.shape,255,np.uint8);positives[labels==1]=1;positive_weight=(labels==1).astype(np.float32)
    q=read('2019_REFERENCE_SCREEN_QUALITY.tif');q[4:7,excluded]=0
    changed_names={'2019_RECOMMENDED_PSEUDO_LABEL.tif','2019_RECOMMENDED_PSEUDO_WEIGHT.tif',
        '2019_POSITIVE_ONLY_LABEL.tif','2019_POSITIVE_ONLY_WEIGHT.tif','2019_REFERENCE_SCREEN_QUALITY.tif'}
    for f in base.iterdir():
        if f.suffix.lower()=='.tif' and f.name not in changed_names:
            shutil.copy2(f,out/f.name)
            if sha(out/f.name)!=sha(f):raise ValueError('Copied source data drift')
    for name in ('SOURCE_COMPONENT_LINEAGE.json','DERIVED_SUPPORT_LINEAGE.json','FROZEN_SOURCE_CELL_CANDIDATES.json'):
        shutil.copy2(base/name,out/name)
    write(out,'2019_RECOMMENDED_PSEUDO_LABEL.tif',labels,grid,['EXPERIMENT_PSEUDO_LABEL'],nodata=255)
    write(out,'2019_RECOMMENDED_PSEUDO_WEIGHT.tif',weights,grid,['EXPERIMENT_SELECTION_NOT_CONFIDENCE'])
    write(out,'2019_POSITIVE_ONLY_LABEL.tif',positives,grid,['REFERENCE_CORROBORATED_POSITIVE_ONLY'],nodata=255)
    write(out,'2019_POSITIVE_ONLY_WEIGHT.tif',positive_weight,grid,['POSITIVE_ONLY_SELECTION'])
    with rasterio.open(base/'2019_REFERENCE_SCREEN_QUALITY.tif') as s:bands=list(s.descriptions)
    write(out,'2019_REFERENCE_SCREEN_QUALITY.tif',q,grid,bands)
    write(out,'2019_SOURCE_ALIGNMENT_RISK_DIAGNOSTIC.tif',src_risk,grid,['SOURCE_FRAME_NEGATIVE_CHECK_BLOCK'])
    write(out,'2019_TARGET_ALIGNMENT_RISK_QUARANTINE.tif',target_risk,grid,['TARGET_FRAME_FROZEN_VECTOR_RISK_QUARANTINE'])
    batch=experiment_batch(read('2019_S1S2_INPUT_WINDOW.tif'),labels,weights)
    if (weights[~old_selection]>0).any() or weights[target_risk.astype(bool)].any():raise ValueError('Selection leaked into unsupported scope')
    counts={'full_window_denominator':65536,'previous_experimental_pixels':int(old_selection.sum()),
        'newly_quarantined_pixels':int(excluded.sum()),'recommended_experimental_pixels':int((weights>0).sum()),
        'reference_corroborated_positives':int((labels==1).sum()),'assumed_background':int((labels==0).sum()),
        'ignore_pixels':int((labels==255).sum()),'source_negative_blocks':7,'target_mapped_risk_blocks':2,
        'new_selection_outside_previous_scope':0,'new_displacement_applications':0,'formal_GT':0,
        'formal_review_accept_reject':0,'tests_from_prior_actual_run':prior['tests_run'],
        'model_training_steps':batch['optimization_steps'],'whole_window_governance_completed':False}
    if counts['recommended_experimental_pixels']+counts['ignore_pixels']!=65536:raise ValueError('Window accounting differs')
    save_json(out/'RISK_QUARANTINE_SUMMARY.json',counts);save_json(out/'RISK_BLOCK_SOURCE_TARGET_LINEAGE.json',{'records':lineage})
    config=json.loads((base/'RECOMMENDED_EXPERIMENT_DATASET.json').read_text(encoding='utf-8'))
    config['source_alignment_risk']='2019_SOURCE_ALIGNMENT_RISK_DIAGNOSTIC.tif'
    config['target_alignment_risk']='2019_TARGET_ALIGNMENT_RISK_QUARANTINE.tif'
    config['sample_count']=counts['recommended_experimental_pixels']
    config['risk_notice']='Manual-reference source core replay passed no new unit; known-negative mapped blocks excluded; retained labels remain experimental, not independently verified geometry'
    save_json(out/'RECOMMENDED_EXPERIMENT_DATASET.json',config)
    write_new(out/'CURRENT_WINDOW_RISK_REPORT.md',f'''# 当前窗口：局部负增益风险隔离

新的人工参考源取样检查从20,373个原屋顶核心候选缩小到9,849个参考一致像元。
16个单元均未通过原检查门槛，未产生可应用的新位移；这是新信息，不是旧分支的正式人工REJECT。
原推荐roi_r1_c2在源16×16块(row4,col11)，23个CHECK样本增益-0.07010；
roi_r2_c1在(row9,col4)，24样本增益-0.16246。不能忽略局部反证继续无风险推荐。

7个源坐标负增益块全部记录；只有2个有原冻结向量且新诊断方向一致，
按既有南移10m将风险范围映射到目标网格。其余来源向量未推荐，不编造目标风险位置。
这仅搬运风险支持范围，不应用新校正、不改任何原屋顶或候选向量。

原10,096个实验选样新增隔离{counts['newly_quarantined_pixels']}个，现{counts['recommended_experimental_pixels']}个；
正例{counts['reference_corroborated_positives']}个，假设背景{counts['assumed_background']}个，ignore {counts['ignore_pixels']}个。
源坐标风险图与目标坐标隔离图分开，不误把源位置当作已校正位置。
原23、失败24、成功24_02均保留；读取接口255→-100不改，没有训练，原B04所有正式计数不变。

此前23交付仍可复现，但出现新风险信息后应使用此25隔离版本进行实验。
保留像元不因此升级为独立精度合格GT；不能把挑选后与人工参考一致率作为验证。
空间治理尚未完成，后续仅继续当前256×256窗口的局部真实屋顶证据与位置验证，不扩整市。
''')
    save_json(out/'PACKAGE_MANIFEST.json',{'files_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    archive=out/'CURRENT_WINDOW_RISK_QUARANTINED_EXPERIMENT.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f!=archive:z.write(f,f.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive corruption')
        m=json.loads(z.read('PACKAGE_MANIFEST.json'))['files_sha256']
        for n,d in m.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=d:raise ValueError('Package artifact hash mismatch')
    for f in out.glob('*.tif'):
        with rasterio.open(f) as s:
            if Grid(s.transform,s.width,s.height,str(s.crs))!=grid:raise ValueError('Delivered raster grid mismatch')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected evidence changed: '+path)
    save_json(out/'RISK_QUARANTINE_RECEIPT.json',{'summary':counts,'protected_artifacts_checked':len(protected),
        'protected_artifacts_changed':0,'new_code_sha256':{str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
