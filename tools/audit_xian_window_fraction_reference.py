"""All-window fixed-block annotation-fraction audit, without sample selection."""
from pathlib import Path
import argparse,json,sys
import numpy as np,rasterio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json,write_new
from xian_r5.label_transfer import Grid,export_raster


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh full-window audit required')
    prior=parent/'SINGLE2019_NEGATIVE_BLOCK_NATIVE_20261012_26'
    plan=json.loads((prior/'FROZEN_NATIVE_RISK_EVIDENCE_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'NATIVE_NEGATIVE_BLOCK_PACKET_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'NATIVE_NEGATIVE_BLOCK_PACKET_RECEIPT.json')]=sha(prior/'NATIVE_NEGATIVE_BLOCK_PACKET_RECEIPT.json')
    base=parent/'SINGLE2019_RISK_QUARANTINED_WINDOW_20261012_25'
    paths={'raw':parent/'SINGLE2019_20261012_06/2019_ROOF_FRACTION_RAW.tif',
        'candidate':base/'2019_ROOF_FRACTION_CANDIDATE.tif','reference':base/'2019_MANUAL_REFERENCE_ROOF_FRACTION.tif',
        'source_quality':base/'2019_SOURCE_QUALITY.tif',
        'manual_quality':parent/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21/2019_MANUAL_REFERENCE_QUALITY.tif'}
    for path in paths.values():protected[str(path)]=sha(path)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input drift: '+path)
    out.mkdir();save_json(out/'FROZEN_FRACTION_BLOCK_AUDIT_PLAN.json',{
        'scope':'ALL_65536_PIXELS_ALL_256_FIXED_16x16_BLOCKS','comparison_before_label_agreement_selection':True,
        'comparison':'SAME_TARGET_PIXEL_AND_FULL_100m2_DENOMINATOR',
        'population':'SOURCE_QUALITY_CODE_1 AND NO_MANUAL_OVERLAP_OR_TRUNCATION; positive reference subset separate',
        'unknown_vector_or_support_not_zero_fraction':True,'negative_annotations_not_certified':True,
        'mean_fraction_difference_is_NOT_V2_uncertainty_or_acceptance_statistic':True,
        'no_new_vectors_or_labels_or_threshold_search':True,'roundoff_tolerance_fraction':1e-7,
        'protected_sha256':protected,'new_code_sha256':{str(Path(__file__).resolve()):sha(__file__)}})
    data={};grid=None
    for name,path in paths.items():
        with rasterio.open(path) as s:
            current=Grid(s.transform,s.width,s.height,str(s.crs))
            if grid is not None and current!=grid:raise ValueError('Audit grid drift')
            grid=current;data[name]=s.read()
    raw,can,ref=[data[k][0].astype(np.float64) for k in ('raw','candidate','reference')]
    q=data['source_quality'];rq=data['manual_quality']
    valid=(q[0]==1)&~rq[1].astype(bool)&~rq[2].astype(bool)
    for a in (raw,can,ref):
        if not np.isfinite(a[valid]).all() or np.any((a[valid]<0)|(a[valid]>1)):
            raise ValueError('Available reference comparison contains unknown fraction')
    positive=valid&(ref>0);delta=np.abs(can-ref)-np.abs(raw-ref)
    changes=np.full((256,256),-9999.,np.float64);changes[positive]=delta[positive]
    flags=np.zeros((256,256),np.uint8)
    flags[positive&(delta<-1e-7)]=1;flags[positive&(np.abs(delta)<=1e-7)]=2;flags[positive&(delta>1e-7)]=3
    records=[]
    for rr in range(16):
        for cc in range(16):
            block=np.zeros((256,256),bool);block[rr*16:(rr+1)*16,cc*16:(cc+1)*16]=True
            groups={}
            for role,m in [('annotation_union_all_available',valid&block),('positive_annotation_only',positive&block)]:
                n=int(m.sum());change=None if n==0 else float(delta[m].mean())
                groups[role]={'pixels':n,'raw_annotation_fraction_MAE':None if n==0 else float(np.abs(raw[m]-ref[m]).mean()),
                    'candidate_annotation_fraction_MAE':None if n==0 else float(np.abs(can[m]-ref[m]).mean()),
                    'candidate_minus_raw_annotation_MAE':change,
                    'status':'UNCOMPARED' if n==0 else 'IMPROVED' if change<-1e-7 else 'WORSENED' if change>1e-7 else 'UNCHANGED'}
            records.append({'block_row':rr,'block_col':cc,'region_id':f'roi_r{rr//4}_c{cc//4}',**groups})
    summary={'window_pixels':65536,'fixed_blocks_audited':len(records),'source_units':16,
        'all_available_comparison_pixels':int(valid.sum()),'positive_annotation_comparison_pixels':int(positive.sum()),
        'positive_annotation_block_counts':{status:sum(r['positive_annotation_only']['status']==status for r in records)
            for status in ('IMPROVED','WORSENED','UNCHANGED','UNCOMPARED')},
        'positive_annotation_pixel_counts':{name:int((flags==code).sum()) for name,code in
            [('improved',1),('unchanged',2),('worsened',3),('uncompared',0)]},
        'new_labels_or_vector_applications':0,'recommended_experiment_samples_unchanged':9866,
        'absolute_position_accuracy':'UNKNOWN','whole_window_governance_completed':False}
    if sum(summary['positive_annotation_pixel_counts'].values())!=65536:raise ValueError('Pixel accounting incomplete')
    export_raster(out/'FRACTION_ANNOTATION_DIFFERENCE_CHANGE.tif',changes[None],grid,
        ['CANDIDATE_MINUS_RAW_ABSOLUTE_DIFFERENCE_TO_ANNOTATED_UNION'],root=out,nodata=-9999.)
    export_raster(out/'FRACTION_ANNOTATION_COMPARISON_DISPOSITION.tif',flags[None],grid,
        ['0_UNCOMPARED_1_IMPROVED_2_UNCHANGED_3_WORSENED_NOT_TRAINING_GATE'],root=out)
    save_json(out/'FULL_WINDOW_FRACTION_BLOCK_AUDIT.json',{'summary':summary,'records':records})
    write_new(out/'FULL_WINDOW_FRACTION_AUDIT_REPORT.md',f'''# 当前窗口全范围面积分数参考检查

65,536个目标像元、16个来源支持及256个固定16×16块全部登记；不是只挑两个反例或选容易改善的屋顶。
先用原源支持质量与人工标注重叠/窗口截断风险决定比较域，不使用过滤后的标签类别来选比较像元。
完整100m²像元为分母；在同一目标像元上比较原源fraction与冻结南移10m候选fraction到人工屋顶标注并集的差异。

可比较{int(valid.sum())}个像元；其中{int(positive.sum())}个含人工屋顶正面积标注。
人工未标注区域的完整背景可靠性未知，所以另外报告仅正面积参考域，不能把全部未标注像元解释成已验证背景。
正面积参考域的块结果：{summary['positive_annotation_block_counts']}。
这些是开发集参考差异，不是V2 uncertainty、外部精度或自动训练准入；没有产生新平移或新标签。

16个来源支持不是已认证的摄影分片。未知向量区域记UNCOMPARED，不填零或将其当背景。
像元与块结果均保留，不能用整个窗口的平均改善掩盖局部变差。
此前26的原生影像检查显示两处均有不同尺度屋顶、可见立面、树影和复杂体/参考分件差异；无法由视觉独立归因于唯一摄影角度或求准确新位移。
已复算的RGB负增益和同目标网格fraction差异变差共同支持继续保留已隔离风险区；不能仅因看见屋顶就重新启用样本。
推荐实验版本仍25的9,866像元，未重新训练或重新搜索0.43阈值，未修改任一原始或历史产物。
''')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input changed: '+path)
    save_json(out/'FULL_WINDOW_FRACTION_AUDIT_RECEIPT.json',{'summary':summary,
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'new_code_sha256':{str(Path(__file__).resolve()):sha(__file__)},
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
