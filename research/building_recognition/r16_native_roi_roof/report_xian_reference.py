"""Summarize paired Xi'an map-reference comparisons on the original 24 development sites."""
import json
from pathlib import Path
import numpy as np

folder=Path(__file__).parent/'xian_map_reference'
def read(name):
    return json.loads((folder/name).read_text(encoding='utf-8'))
base=read('BASELINE_SUMMARY.json')['by_epoch']
position=read('POSITION_SUMMARY.json')
windows=read('POSITION_OBJECTS.json')
raw=read('BASELINE_OBJECTS.json')
field='GEOMETRY__DATE_FIELD'
rng=np.random.default_rng(20261111)
lines=['# 西安三期开发窗口的屋顶与位置对照','',
'24 处地点，每处 3 次观测。两次评估都从窗口内有影像支持的指定人工 GT 建筑出发，保留预测完全未命中的观测。没有人工标签的预测继续保留，未计为假阳性。','',
'## 已有外立面模型对屋顶候选的影响','',
'按原生格网像元中心栅格化参考多边形，每期有 995 条可计量观测。下表的面积覆盖是已标注建筑像元被候选覆盖的比例。','',
'|实际拍摄期|一对一匹配数，剔除前 → 后|GT 面积覆盖，剔除前 → 后|完全未覆盖的参考观测，剔除前 → 后|','|---|---:|---:|---:|']
dates=['2018-04-29','2018-11-01、11-08','2019-12 至 2020-06，多日拼接']
for (epoch,rec),date in zip(sorted(base.items()),dates):
    a,b=rec['metrics']['RAW'],rec['metrics']['AFTER_FACADE']
    lines.append(f"|{date}|{a['matches_at_IoU50']} → {b['matches_at_IoU50']}|{a['reference_pixel_coverage']:.2%} → {b['reference_pixel_coverage']:.2%}|{a['zero_overlap']} → {b['zero_overlap']}|")
lines+=['','以上数据衡量相对于人工地图的覆盖情况。窗口内没有西安逐像元的可见外立面真值，立面剔除准确率仍待验证。','',
'## 已有自动位移对轮廓和位置的影响','',
'将治理后的原生像元边界转成矢量，投影到 EPSG:32649，采用冻结的按拍摄日期估计的位移。各期 998 条参考观测与支持区有正面积交集，其中 3 条未覆盖任何原生像元中心。每个查询独立平移，位置模型没有给出位移的候选沿用原坐标。匹配采用 IoU≥0.50、一对一最大匹配。','',
'|实际拍摄期|未移 → 平移后匹配数|新增 / 丢失|所有参考建筑平均最佳 IoU|同一批持续匹配对象的平均质心偏差|','|---|---:|---:|---:|---:|']
uncertainty={}
for (epoch,methods),date in zip(sorted(position['epochs'].items()),dates):
    a,b=methods['RAW'],methods[field]
    lines.append(f"|{date}|{a['matched']} → {b['matched']}|{b['gained']} / {b['lost']}|{a['mean_best_IoU']:.3f} → {b['mean_best_IoU']:.3f}|{b['raw_paired_m']:.2f} → {b['final_paired_m']:.2f} m（{b['paired_identical_matches']} 个）|")
    site=[]
    for w in windows:
        if w['epoch']!=epoch:continue
        a0,b0=w['results']['RAW'],w['results'][field]
        site.append((w['site'].rsplit('_',1)[0],len(a0),sum((b['matched_query'] is not None)-(a['matched_query'] is not None) for a,b in zip(a0,b0))))
    arr=np.array([x[1:] for x in site],dtype=float)
    draws=np.zeros((4000,2))
    for st in sorted({x[0] for x in site}):
        idx=[i for i,x in enumerate(site) if x[0]==st]
        draws+=arr[rng.choice(idx,(4000,len(idx)),replace=True)].sum(axis=1)
    ci=np.quantile(draws[:,1]/draws[:,0],[.025,.975]).tolist()
    uncertainty[epoch]=dict(net_gain_pp=100*arr[:,1].sum()/arr[:,0].sum(),dev_site_bootstrap_95pct_pp=[100*x for x in ci],sites_improved=int((arr[:,1]>0).sum()),sites_worse=int((arr[:,1]<0).sum()),sites_tied=int((arr[:,1]==0).sum()))
lines+=['','按地点配对、在原抽样分组内重抽 4,000 次，各期净匹配率变化如下。区间描述这些开发窗口的敏感性，城市总体代表性和跨日期真实存在性另需验证。','']
for (epoch,u),date in zip(sorted(uncertainty.items()),dates):
    lo,hi=u['dev_site_bootstrap_95pct_pp']
    lines.append(f"- {date} 增加 {u['net_gain_pp']:.2f} 个百分点，重抽区间 [{lo:.2f}, {hi:.2f}]，改善 / 下降 / 持平地点数 {u['sites_improved']} / {u['sites_worse']} / {u['sites_tied']}。")
lines+=['','## 下轮训练要解决的误差','',
'在原始屋顶候选中，三期各有 644、628、421 条参考观测已有至少一半面积被覆盖，但与任一单个预测对象的 IoU 都低于 0.50。这里需要分别检查邻栋粘连、轮廓和位移。逐栋裁剪细化只能修正已有候选，对检测器完全没有提出的目标还需要单独评估候选生成。','',
'跨期位置平均的分期对照也已保存。将日期位移再作跨期平均，三期匹配数分别为 160、153、107；各期效果方向不一致。以上位移均来自已训练模型在开发窗口的冻结预测。R16 新权重的结果另行记录。','']
fraction=read('fraction_10m/MANIFEST.json')
lines+=['## 10 m 面积分数','',
'在冻结格网原点 (293300, 3812400) 上逐格计算矢量并集与像元的交集，分母为 100 m²。72 个窗口各保存一份六波段 TIF，依次为影像支持、人工 GT 并集、剔除立面后、剔除立面并平移后、前者与 GT 的交集、后者与 GT 的交集。支持区外编码为 -1。','',
'每期有 6,251 个格子与指定参考有正面积交集。在这些格子中，模型与人工地图之间的建筑面积分数平均绝对差如下。','']
for (epoch,f),date in zip(sorted(fraction['summary'].items()),dates):
    m=f['mean_abs_fraction_difference_on_positive_reference_cells']
    lines.append(f"- {date} 从 {m['after_facade']:.4f} 降至 {m['after_facade_and_shift']:.4f}。")
lines+=['','GT 遗漏建筑也会产生差值。开发窗口及其中的参考正面积格子均不能代表全市。所有 TIF 已在本地复核 SHA 并读回各波段积分，432 个波段与投影矢量并集或交集面积的最大读回差为 %.6f m²。'%fraction['max_area_readback_residual_m2'],'']
(folder/'SITE_SENSITIVITY.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8')
(folder/'西安三期治理实测.md').write_text('\n'.join(lines),encoding='utf-8')

