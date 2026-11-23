from pathlib import Path
import json
from datetime import datetime,timezone,timedelta
R=Path(__file__).parent
I=R/'incoming'
D=R/'observability'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
vis=read(D/'DIAGNOSTIC.json')
m=read(I/'data/READY.json')['summary']
st=read(I/'output/STATUS.json')
lines=['# 西安三期可判读性与独立屋顶候选训练','',
'## 三处对第三期统计影响最大的地点','',
'相同位置、相同图幅的三期原图与叠加图已回传。三处的末期实际拍摄日均为 2019-12-02。三期 24 个窗口全部参与统计。','',
'| 地点 | 可栅格化人工参考数 | 原候选完全未覆盖，2018-04 / 2018-11 / 末期 | 末期现场影像 |',
'|---|---:|---|---|']
notes={'DENSE_REFERENCE_00':'前两期为密集屋顶，末期运动场保留、周边大面积呈裸露或清理地表。个体建筑是否拆除尚未经核定。',
       'RANDOM_02':'末期大片灰白低对比区域，屋顶细节不清。','RANDOM_03':'末期大片灰白低对比区域，局部地物仍可见。'}
for site,t in notes.items():
    rr=sorted([r for r in vis['rows'] if r['site']==site],key=lambda r:r['epoch'])
    lost=[sum(v['raw_zero_overlap'] for v in r['rows']) for r in rr]
    lines.append(f"|{site}|{len(rr[0]['rows'])}|{' / '.join(map(str,lost))}|{t}|")
lines+=['','上述 3 处占末期全部 342 条零重叠参考观测的 291 条。参考年份、拍摄日期、可见性及模型性能都影响该统计。各期未标注预测仍保留，零重叠不生成背景或拆除标签。',
'','[三期同位置原图](observability/THREE_SITE_RGB.jpg)　[黄线人工参考、紫线原始候选](observability/THREE_SITE_OVERLAY.jpg)','',
'## 影像可判读性诊断','',
'每个 64×64 原生像元块记录亮度中位数、90–10 百分位跨度和相邻像元平均绝对差。明亮且变化小的块按三个固定条件标记（≥145、≤36、≤9）。这是本次查看开发窗口后的诊断规则，云、薄雾、空地与浅色屋顶都可能触发。','',
'| 实际拍摄期 | 零重叠参考观测 | 其中至少一半面积落在明亮低纹理块 |','|---|---:|---:|']
for (epoch,v),date in zip(sorted(vis['summary'].items()),['2018-04','2018-11','2019-12 至 2020-06']):
    lines.append(f"|{date}|{v['zero_raw_overlap']}|{v['zero_and_majority_bright_flat']}|")
lines+=['','72 份四波段诊断 TIF 的摘要、原生空间参考及本地文件哈希已回读。','',
'## 独立候选训练','',
'检测器读取整幅 1024×1024 影像的冻结 SAM 特征，在步长 8 的网格上预测屋顶内部点、亚像元偏移和四向框边距，并用已知屋顶面积分数提供辅助监督。候选生成无需原模型提出框；与原候选重叠的重复框在融合时处理。','',
'| 来源及用途 | 影像 | 可见且有标注的屋顶 | 原候选零像元覆盖 |','|---|---:|---:|---:|']
for role,loc in [('FIT','北京、哈尔滨训练'),('INNER','成都选模')]:
    v=m[role];lines.append(f"|{loc}|{v['frames']}|{v['annotated']:,}|{v['zero_raw_overlap']:,}|")
lines+=['','步长 8 产生的同格中心冲突在训练集有 103 对应观测，在成都有 3 条；这些建筑仍留在逐栋评价分母。未知和拥挤像元不作负监督。另有 370 / 31 条标注在训练 / 选模窗口内没有有效可见像元。','',
'设定为 2,000 次 AdamW 更新，批量 16，初始最大学习率 0.0006，100 步预热后余弦衰减。按 250、500、1,000、2,000 步记录不仅整体匹配，还记录原模型完全漏掉的建筑实际找回数。候选命中要求内部点落入标注屋顶且框 IoU≥0.50，一对一匹配。候选框与分割准确率分别评价。','']
if (I/'output/SOURCE_COMPLETE_MISS_PROFILE.json').exists():
    results=read(I/'output/SOURCE_COMPLETE_MISS_PROFILE.json')['results']
    v=results['FIT'];bins=v['by_min_native_box_side_px']
    lines.append(f"训练样本中，完全漏检屋顶的包围框短边中位数为 {v['median_min_side_px_completely_uncovered']:.0f} 像元，至少部分覆盖的为 {v['median_min_side_px_any_coverage']:.0f} 像元。短边为 9–16 像元时，零覆盖率为 {bins['8-16']['uncovered_rate']:.1%}；短边超过 64 像元时为 {bins['64-inf']['uncovered_rate']:.1%}。")
lines+=['','## 最近一次同步','',
    f"北京时间 {datetime.fromtimestamp(st['time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}，阶段 {st['event']}，已记录优化器更新 {st.get('actual_optimizer_updates',0)} 次。",
    '','达到源域冻结选模条件后，在济南 12 个已用开发窗口复核。冻结权重随后用于西安 72 个窗口的研究性独立搜索，输出新增候选掩膜与人工地图对应情况。全市产品另需通过部署前质量检查。','']
(R/'研究进展.md').write_text('\n'.join(lines),encoding='utf-8')

