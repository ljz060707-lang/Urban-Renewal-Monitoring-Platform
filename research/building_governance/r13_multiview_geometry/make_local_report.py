from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent;D=R/'incoming';receipt=json.loads((D/'TRAINING_RECEIPT.json').read_text());full=json.loads((D/'FULL_POLYGON_EVALUATION.json').read_text());curves=list(csv.DictReader((D/'FIT_CURVES.csv').open()));x=[int(r['iteration']) for r in curves]
fig,ax=plt.subplots(figsize=(7.2,4),constrained_layout=True)
for name,label in [('mean_reference_residual_m','TRAIN mean'),('p95_reference_residual_m','TRAIN P95')]:ax.plot(x,[float(r[name]) for r in curves],label=label)
ax.set(xlabel='Actual coordinate-descent iteration',ylabel='Map-reference residual (m)',title='R13 robust multiview fit: TRAIN only');ax.legend();fig.savefig(R/'REAL_ALS_TRAIN_CURVE.png',dpi=180);plt.close(fig)
methods=['RAW','DATE_FIELD','TEMPORAL_DATE_FIELD_MEAN','TEMPORAL_RAW_MEAN','SELECTED_FACTOR'];labels=['Raw','Date field','Date field + mean','Raw temporal mean','Selected factor'];records=full['records'];ious=[np.mean([r['scores'][m]['reference_IoU'] for r in records]) for m in methods];metrics=receipt['TUNE_reports'];fig,axs=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
axs[0].barh(labels,ious,color=['#888888','#5286b8','#3d9b86','#aa9c4c','#7556a5']);axs[0].set(xlim=(0,.6),xlabel='Mean full-polygon map-reference IoU',title='399 reused development observations')
for j,v in enumerate(ious):axs[0].text(v+.005,j,f'{v:.3f}',va='center')
axs[1].plot(labels,[metrics[m]['all']['mean_error_m'] for m in methods],'o-',label='Mean');axs[1].plot(labels,[metrics[m]['all']['P95_error_m'] for m in methods],'s-',label='P95');axs[1].tick_params(axis='x',rotation=35);axs[1].set(ylabel='Reference centroid error (m)',title='Provided map-associated groups');axs[1].legend();fig.savefig(R/'REFERENCE_CONDITIONED_COMPARISON.png',dpi=180);plt.close(fig)
rows=['# R13 多期位置试验（已完成，未晋升）','','这是基于已提供地图参考分组的条件试验，不代表自动同栋识别，更不是独立可见屋顶精度。用 TRAIN 9,504 个对象记录拟合，TUNE 399 个对象记录复用开发评价；未知或无对应建筑不据此判为背景。','','| 方法 | 平均位置误差 m | P95 m | 完整多边形平均 IoU |','|---|---:|---:|---:|']
for m,name,v in zip(methods,labels,ious):rows.append(f"| {name} | {metrics[m]['all']['mean_error_m']:.3f} | {metrics[m]['all']['P95_error_m']:.3f} | {v:.5f} |")
rows+=['','真实优化是 40 次稳健交替最小二乘迭代（Huber 3 m、日期位置场加一维残差因子）；深度学习反向更新为 0。参数、选择过程、每次真实迭代曲线和全部对象预测保存在 incoming。残差因子不能解释为物理楼高，也不是完整摄影测量重建。','','轮廓评价采用完整多边形平移而非裁窗，逐对象面积误差最大值 0。相对原始几何确有参考对齐收益，但选中模型未在所有参考来源和尾部指标上优于简单时序平均，因此保持未晋升。自动关联、原生屋顶/外立面质量和全市面积仍需另行通过。','','推理不读取参考坐标的毒化功能测试通过；分组本身仍由参考对应关系提供，不能据此消除 GT 辅助性质。原全市产品未修改。']
(R/'REPORT_CN.md').write_text('\n'.join(rows),encoding='utf-8');manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='LOCAL_ARTIFACT_SHA256.json'];(R/'LOCAL_ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('REPORT_AND_CURVES',len(manifest))
