# coding: utf-8
from pathlib import Path
import json,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent;I=R/'incoming/output'
def read(p,default=None):
 try:return json.loads(p.read_text(encoding='utf-8'))
 except (OSError,ValueError):return default
prep=read(I/'PREPARATION_RECEIPT.json',{});receipt=read(I/'training/TRAINING_RECEIPT.json',{});rgb=read(I/'rgb_identity/training/STATUS.json',{});position=read(I/'automatic_position/FULL_POLYGON_REFERENCE_EVALUATION.json',{});health=read(R/'SYNC_HEALTH.json',{})
lines=['# R14 原生西安候选、关联与自动位置试验',f'本地回传更新：{time.strftime("%Y-%m-%d %H:%M:%S")}','','持续执行中；没有新全市交付版。源数据、历史权重和原产品均保留。',f'已封存模型观察 {prep.get("nodes","待回传")} 个，窗口 {prep.get("windows","")} 个，有准入地图参考标签 {prep.get("labels","")} 个。GT 在候选图封存后才连接。无 GT 观察保留，不作监督负例。','','## 已完成的真实关联训练','','两组各 2,000 次 CPU AdamW 更新；A10 同时用于原生屋顶实例 CNN。TRAIN 内按 2 km 块划分、600 m 中心隔离并剔除参考身份交叉，阈值和检查点冻结后才评价复用 TUNE。','', '| 方法 | 部分标签 IDF1 | AssA | 已标注端点错边 | 轨迹碎片 |','|---|---:|---:|---:|---:|']
for name,v in receipt.get('TUNE_results',{}).items():lines.append(f"| {name} | {v['IDF1']:.5f} | {v['AssA']:.5f} | {v['wrong_reference_direct_edges']} | {v['identity_fragments']} |")
lines+=['','全部 2,523 个 TUNE 模型观察均初始化轨迹，其中 399 个有地图参考身份（15.8%）。表中指标只评价这 399 个对象的参考关联；未知对象间的错配及全场景物理身份精度尚不可由此确定。标签是几何准入的参考配对，不是独立确认的真实建筑时序身份。','',f'已完成两组晋升结果：{receipt.get("qualified_partial_label_research_arms",[])}。不因新增训练而替换原产品。','','## 自动关联后的位置复核','','轨迹由模型/几何规则产生，推理不再提供 GT 分组或参考坐标。位置场参数来自 TRAIN 参考监督；399 个复用开发对象上按完整多边形评价，未裁切损失面积。','', '| 固定方法 | 平均参考 IoU | 平均误差 m | P95 m |','|---|---:|---:|---:|']
summary=position.get('summary',{})
for name in ['RAW','GEOMETRY_ONLY__DATE_FIELD','GEOMETRY_ONLY__TEMPORAL_DATE_FIELD_MEAN','GEOMETRY_ONLY__SELECTED_FACTOR','GEOMETRY_AND_FROZEN_ROOF64__SELECTED_FACTOR']:
 if name in summary:
  v=summary[name]['all'];lines.append(f"| {name} | {v['mean_reference_IoU']:.5f} | {v['mean_reference_error_m']:.3f} | {v['P95_reference_error_m']:.3f} |")
lines+=['','自动关联的错误会传播到位置共识：部分方法平均值改善而尾部变差，旧语义特征组甚至整体恶化。所有分组结果和负对照保存在 incoming/output/automatic_position，未据 TUNE 结果事后挑选正式版本。','','## 当前接续训练','',f'真实屋顶 RGB 外观对比学习：{rgb.get("event","准备或等待回传")}。参照 SupCon 官方损失，使用 ImageNet ResNet18，训练 layer4 与 64 维投影；不同观测期为正对，只对已准入 FIT 参考身份监督。128 px 是表征用 ROI，不是提高影像真实分辨率。','', '论文：[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)；官方代码的提交号、许可、SHA 和实际参数见 incoming/code/official_supcontrast 及 incoming/output/rgb_identity/training。','',f'回传核验：{health.get("status","尚未完成")}，权重 {health.get("weight_files",0)} 个。']
lines+=['','## 后续实验汇总','','R14 已完成真实参数更新 14,000 次：基础集合关联 4,000；受限残差 4,000；RGB 对比学习 2,000；居中原生形状关联 4,000。DINOv2 特征及全局超图均为零项目参数更新，单列为表征/求解对照。','','|固定方法|部分标签 IDF1|已标注直接错边|轨迹碎片|','|---|---:|---:|---:|','|基础几何网络|0.954887|10|12|','|受限几何/语义残差|0.954887|10|12|','|原生 RGB 对比学习|0.944862|13|12|','|原生形状受限残差|0.954887|10|12|','|冻结 DINOv2|0.962406|6|13|','','DINOv2 的参考关联有局部提升，但碎片数增加，未通过完整准入。全局超图把碎片降到 11，但 IDF1 为 0.959900，仍未通过；其冲突口径改为轨迹内全部已知身份对，不能与直接错边列混比。所有结果来自重复使用的部分标签开发集，尚不证明全场景物理身份或真实新增/拆除精度。','', '当前 GPU 接续任务为 R15 真实原生屋顶 SAM 解码器，详见相邻目录的 PROGRESS_CN.md；本目录全部实验保持封存。']
new_status=read(I/'dino_pair_residual/STATUS.json',{});new_receipt=read(I/'dino_pair_residual/TRAINING_RECEIPT.json',{})
if new_status:
 lines+=['','## DINO 外观与原生形状的受限深度修正','','同一冻结几何网络起点，比较匹配的几何 MLP 与 DINO/形状 MLP；编码器和 dustbin 均冻结，修正幅度受限。每组计划 2,000 次真实 CPU AdamW 更新，四线程与屋顶 GPU 任务分工。',f'最近阶段：{new_status.get("arm")}，组内实际更新 {new_status.get("actual_new_optimizer_updates")}。未知候选与无 GT 查询直接梯度为零，已用真实候选批次检查。']
if new_receipt:
 lines+=['','新实验已完成实际参数更新 4,000 次；R14 系列累计 18,000 次。研究准入：'+str(new_receipt['qualified_partial_label_research_candidate'])+'。','', '|新方法|部分标签 IDF1|已标注直接错边|轨迹碎片|','|---|---:|---:|---:|']
 for name,m in new_receipt['TUNE_results'].items():lines.append(f"|{name}|{m['IDF1']:.6f}|{m['wrong_reference_direct_edges']}|{m['identity_fragments']}|")
(R/'PROGRESS_CN.md').write_text('\n'.join(lines),encoding='utf-8')
for name,path in [('SET_ASSOCIATION',I/'training/HISTORY.json'),('RGB_CONTRASTIVE',I/'rgb_identity/training/HISTORY.json'),('DINO_PAIR_RESIDUAL',I/'dino_pair_residual/HISTORY.json')]:
 rows=read(path,[])
 if not rows:continue
 fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
 for arm in sorted({r.get('arm','RGB_CONTRASTIVE') for r in rows}):
  rr=[r for r in rows if r.get('arm','RGB_CONTRASTIVE')==arm];ax.plot([r.get('actual_optimizer_updates',r.get('actual_new_optimizer_updates')) for r in rr],[r['loss'] for r in rr],label=arm)
 ax.set(xlabel='Actual AdamW optimizer updates',ylabel='Actual minibatch loss',title=name+' / weak map-reference supervision');ax.legend(fontsize=8);fig.savefig(R/(name+'_REAL_TRAIN_CURVE.png'),dpi=170);plt.close(fig)
print('LOCAL_REPORT_UPDATED')
