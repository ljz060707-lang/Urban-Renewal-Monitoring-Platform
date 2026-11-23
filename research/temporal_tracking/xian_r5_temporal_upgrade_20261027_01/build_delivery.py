# -*- coding: utf-8 -*-
"""Evidence-linked research report, measured plots and product conservation QA."""
import json,csv,math,hashlib,html,sys
from pathlib import Path
import numpy as np,rasterio
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import shape
from shapely.ops import unary_union
from upgrade_application import ROOT,BASE,jsonl,inputs
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def products_path(run):
    pointer=run/'PRODUCTS_POINTER.json'
    return Path(read(pointer)['path']) if pointer.exists() else run/'PRODUCTS'
def source_plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f,axes=plt.subplots(1,2,figsize=(12,4),dpi=160)
    for seed in [20261027,20261028,20261029]:
        p=ROOT/'runs/T2_FULL_SPAN'/('seed_'+str(seed));a=read(p/'history.json');axes[0].plot([r['epoch']+1 for r in a],[r['tune']['macro_f1'] for r in a],label=str(seed))
        curve=read(p/'CALIBRATION.json')['same_edge_curve'];xx=[r['coverage'] for r in curve if r['false_same_rate'] is not None];yy=[r['false_same_rate'] for r in curve if r['false_same_rate'] is not None]
        axes[1].plot(xx,yy,label=str(seed))
    axes[0].set(xlabel='Complete FIT epochs',ylabel='TUNE macro F1',title='Internal source model selection')
    axes[1].set(xlabel='Accepted SAME / all known candidate pairs',ylabel='Empirical false SAME rate',title='CALIBRATION coverage-error curve')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    f.text(.5,.01,'GT proposals; overlapping query windows; historically exposed TRAIN AOIs. These are not XiAn accuracy or independent confidence bounds.',ha='center',fontsize=8)
    f.tight_layout(rect=[0,.05,1,1]);f.savefig(out/'SOURCE_TRAINING_AND_RISK.png');plt.close(f)
def source_views(out):
    c,packets,obs,tracks=inputs();extra=jsonl(ROOT/'runs/ACTIVE_EXPANDED/RECOVERED_OBSERVATIONS_VALIDATED.jsonl')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    canvas=Image.new('RGB',(1920,540),(18,24,32));draw=ImageDraw.Draw(canvas)
    draw.text((20,12),'Current RGB source evidence: cyan = original B0; orange = extra-context B0 candidate',(235,240,245),font=font)
    for j,p in enumerate(packets):
        im=Image.open(p['image']).convert('RGB');d=ImageDraw.Draw(im)
        for rows,color in [(obs,(30,215,240)),(extra,(255,150,40))]:
            for r in rows:
                if r['epoch_id']!=p['epoch_id']:continue
                g=shape(r['native_geometry']);parts=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
                for poly in parts:
                    if poly.geom_type=='Polygon':d.line(list(poly.exterior.coords),fill=color,width=2)
        im=im.resize((640,384),Image.Resampling.LANCZOS);canvas.paste(im,(j*640,84));draw.text((j*640+15,53),p['actual_date'],fill=(235,240,245),font=font)
        a=sum(r['epoch_id']==p['epoch_id'] for r in obs);b=sum(r['epoch_id']==p['epoch_id'] for r in extra)
        draw.text((j*640+15,480),f'{a} old observations + {b} new candidates; no independent roof GT',fill=(220,225,230),font=small)
    draw.text((18,515),'A same-model context version is not independent evidence. Facades/roads and incomplete roof boundaries remain possible.',fill=(215,220,225),font=small)
    canvas.save(out/'REAL_SOURCE_CANDIDATES.png')
def conservation(runs):
    qa=[]
    for run in runs:
        p=products_path(run)
        for summary in (p/'single_date').glob('*/PRODUCT_MANIFEST.json'):
            s=read(summary)['summary'];tif=summary.parent/'FRACTION_10M.tif'
            with rasterio.open(tif) as src:
                integral=float(src.read(1).sum(dtype=np.float64)*abs(src.transform.determinant));assert src.crs.is_projected
            error=abs(integral-s['within_source_support_union_m2']);assert error<.003
            qa.append(dict(run=run.name,product=str(tif),area_integral_m2=integral,vector_union_m2=s['within_source_support_union_m2'],difference_m2=error))
        for summary in (p/'pairs').glob('*/PAIR_SUMMARY.json'):
            s=read(summary);tif=summary.parent/'CHANGE_FRACTION_10M.tif'
            with rasterio.open(tif) as src:
                a=src.read();cell=abs(src.transform.determinant);assert a.shape[0]==3 and np.isfinite(a).all()
                for i,state in enumerate(['STABLE','ADDITION','DEMOLITION']):
                    area=next(r['within_pair_support_union_m2'] for r in s['area_by_class'] if r['state']==state);integral=float(a[i].sum(dtype=np.float64)*cell);error=abs(integral-area);assert error<.003
                    qa.append(dict(run=run.name,product=str(tif)+':'+state,area_integral_m2=integral,vector_union_m2=area,difference_m2=error))
            with rasterio.open(summary.parent/'CHANGE_WEIGHT_10M.tif') as src:assert np.all(src.read()==0)
    write(ROOT/'evidence/EXPORTED_PRODUCT_CONSERVATION_QA.json',dict(status='PASS',records=qa,max_error_m2=max(r['difference_m2'] for r in qa),formal_weights_all_zero=True))
    return qa
def lineage_areas(runs):
    rows=[]
    for run in runs:
        a=jsonl(run/'OLD_120_UNKNOWN_LINEAGE.jsonl');assert len(a)==120
        for e in ['t0','t1','t2']:
            selected=[r for r in a if r['epoch_id']==e];unknown=[r for r in selected if r['new_presence_candidate']=='U']
            rows.append(dict(run=run.name,epoch=e,old_U_nodes=len(selected),new_source_observed=sum(r['new_presence_candidate']==1 for r in selected),remaining_U=len(unknown),
                fixed_old_candidate_union_m2=unary_union([shape(r['old_geometry']) for r in selected]).area,
                unresolved_old_candidate_union_m2=unary_union([shape(r['old_geometry']) for r in unknown]).area,
                qualified_unknown_nodes=len(selected)))
    table(ROOT/'evidence/FIXED_OLD_UNKNOWN_AREA_LINEAGE.csv',rows)
def main():
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    names=['B2_BASELINE_ARGMAX_FIXED55','B2_BASELINE_ARGMAX_EXPANDED_VALIDATED','LEARNED_FULLSPAN_NATIVE_FIXED55','LEARNED_FULLSPAN_NATIVE_EXPANDED_VALIDATED','LEARNED_FULLSPAN_NATIVE_MANAGED_FIXED55','LEARNED_FULLSPAN_NATIVE_MANAGED_EXPANDED_VALIDATED']
    runs=[ROOT/'runs'/n for n in names];assert all((r/'COMPLETE.json').exists() for r in runs)
    source=read(ROOT/'evidence/SOURCE_COMPARISON_FULLSPAN.json');oldsource=read(ROOT/'evidence/SOURCE_COMPARISON.json');cross=read(ROOT/'evidence/CROSS_EPOCH_ONLY_SOURCE_METRICS.json')
    dataset=read(ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json');active=read(ROOT/'runs/ACTIVE_EXPANDED/COMPLETE.json');t1=read(ROOT/'runs/T1_BT/COMPLETE.json')
    training=[read(ROOT/'runs/T2_FULL_SPAN'/('seed_'+str(s))/'TRAINING_COMPLETE.json') for s in [20261027,20261028,20261029]]
    comparison=[]
    for run in runs:
        rec=read(run/'COMPLETE.json');p=products_path(run);pairs=[read(v) for v in sorted((p/'pairs').glob('*/PAIR_SUMMARY.json'))]
        row=dict(variant=run.name,observations=rec['observations'],tracks=rec['tracks'],accepted_edges=rec['accepted_edges'],candidate_present=rec['presence']['1'],candidate_U=rec['presence']['U'],
            fixed_old_U_observed=rec['old_U_now_observed'],fixed_old_U_remaining=rec['old_U_remaining'],fixed_old_U_to_absent=rec['old_U_to_absent'],
            stable_candidates=sum(x['candidate_state_counts']['STABLE'] for x in pairs),addition_candidates=sum(x['candidate_state_counts']['ADDITION'] for x in pairs),demolition_candidates=sum(x['candidate_state_counts']['DEMOLITION'] for x in pairs),
            formal_qualified_events=0,target_accuracy=None,products=str(p))
        comparison.append(row)
    table(out/'TARGET_FIXED_AND_EXPANDED_COMPARISON.csv',comparison);write(out/'TARGET_COMPARISON.json',comparison)
    qa=conservation(runs);lineage_areas(runs);source_plots(out);source_views(out)
    epochs=[dict(seed=r['seed'],epochs=r['epochs'],best_epoch=r['best_epoch'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],source_macro_f1=r['calibration']['macro_f1']) for r in training]
    write(out/'TRAINED_CHECKPOINTS.json',epochs)
    # These are sampled/overlapping query-window metrics; no end-to-end claims.
    mrows=[]
    for name,r in source.items():
        m=r['classification'];s=r['sequence'];mrows.append(dict(model=name,macro_f1=cross[name]['macro_f1'],macro_f1_including_self_control=m['macro_f1'],query_IDF1_strict_argmax=s['query_strict_argmax_IDF1'],
            stable_precision=s['event_precision'][0],addition_precision=s['event_precision'][1],demolition_precision=s['event_precision'][2],
            stable_recall=s['event_recall'][0],addition_recall=s['event_recall'][1],demolition_recall=s['event_recall'][2],
            ID_switches_common_point=s['common_half_probability_ID_switches'],fragments_common_point=s['query_window_fragments']))
        if name=='B2_BASELINE':
            for key in ['stable_precision','addition_precision','demolition_precision','stable_recall','addition_recall','demolition_recall']:mrows[-1][key]=None
    table(out/'SOURCE_ASSOCIATION_AND_EVENTS.csv',mrows)
    primary=next(r for r in comparison if r['variant']=='LEARNED_FULLSPAN_NATIVE_FIXED55');b2=next(r for r in comparison if r['variant']=='B2_BASELINE_ARGMAX_FIXED55')
    stage=dict(T0='COMPLETE_FIXED120_LINEAGE',T1='PARTIAL_BT_VECTOR_AUX_TRAINED; exact flow warp binding unresolved; SpaceNet4 unavailable locally',
        T2='COMPLETE_DEFINED_SOURCE_COVERAGE_FULL_EPOCHS_THREE_SEEDS',T3='PARTIAL_ACTIVE_SET_COMPLETE_FIXED_INPUT; target pseudo-label adaptation and outer-source expansion not performed',
        T4='PARTIAL_SOURCE_INTERNAL_CALIBRATION; alpha proxy is not cloud-free visibility; target absence NOT_ADMITTED',T5='COMPARISONS_AND_MAP_EXPORTS_COMPLETE; production upgrade not qualified')
    write(out/'STAGE_STATUS.json',stage)
    texts=[]
    texts.append('# 西安时序身份升级：训练、重检与同应用对照\n')
    texts.append('本轮完成了真实训练、历史时序权重复用、预定主动重检和同应用导出。**可靠应用升级尚未通过验收，正式监督权重仍为 0。** 全部结果限研究候选，未将目标域一致预测当作真值。\n')
    texts.append(f"日期按相邻实际观测处理：2018-04-29 → 2018-11-08 → 2020-06-14；不要求相隔一年。原 APPLICATION_V1 与其 55 条观察保持原样。\n")
    texts.append('## 实际完成\n')
    texts.append(f"- 新主网络为可训练 RGB/实例掩膜 CNN、几何分支与双向时序 GRU，共 168,415 参数。候选集合先按期汇聚，再跨期建模，不将候选序号冒充持久 ID。全序列整数优化同时使用所有日期对，保留空匹配出口，禁止一期多实例进入同轨。\n- SpaceNet7 原 TRAIN 共 42 个 AOI，生成 {dataset['sequences']:,} 条五期序列；FIT/TUNE/CALIBRATION 为 {dataset['roles']['FIT']:,}/{dataset['roles']['TUNE']:,}/{dataset['roles']['CALIBRATION']:,}。包含连续、长间隔、跳期和所有符合 alpha 条件的标签 ID 消失/返回；没有声称遍历每栋建筑。\n- 三种子实际完整训练 epoch 为 {' / '.join(str(r['epochs']) for r in training)}，按冻结平台规则停止；不是固定500/1500次短更新。模型、优化器、随机状态、数据及源码哈希均保存。\n- BANDON 取回1,689对官方 TRAIN 的 BT-flow/变化标签，训练只使用继承角色允许的1,635对：FIT/TUNE/CALIBRATION 样本23,154/921/889。BT向量辅助 TUNE EPE {t1['zero_vector_EPE']:.3f} → {t1['best_tune_EPE']:.3f} 像素，训练{t1['epochs']}轮；此结果不是身份精度。\n- 原120条U逐条保留。主动搜索登记240条上下文检查、104个唯一有效上下文，未支持60条、去重引用76条；新增131条当前期RGB模型候选。重复模型和像素不计作独立证据，未新增任何已准入缺失。\n- 恢复并运行 B2 22特征身份权重，以及V8R5三种子11通道退出/返回/占用模型。旧V8、V9.5的历史拒绝结论保留，未因文件名或时间较新而晋升。\n")
    texts.append('## 目标案例：固定旧55条与扩展186条分开比较\n')
    texts.append('|方案|观察|轨迹|关联边|旧120 U中现有观察|旧U仍U|Stable/Addition/Demolition候选|\n|---|---:|---:|---:|---:|---:|---|\n')
    for r in comparison:texts.append(f"|{r['variant']}|{r['observations']}|{r['tracks']}|{r['accepted_edges']}|{r['fixed_old_U_observed']}|{r['fixed_old_U_remaining']}|{r['stable_candidates']}/{r['addition_candidates']}/{r['demolition_candidates']}|\n")
    texts.append('\n这些“现有观察”计数包含观察字段与身份资格解耦的影响，不能宣传为已解决相同数量的真实建筑 Unknown。新增候选可能含立面、道路或重复分割；186也不是186栋已认证建筑。固定旧对象几何、日期和逐条去向见 `evidence/FIXED_OLD_UNKNOWN_AREA_LINEAGE.csv` 与各运行的 `OLD_120_UNKNOWN_LINEAGE.jsonl`。\n')
    texts.append(f"新主模型在固定55条上接纳 {primary['accepted_edges']} 条关联边；B2恢复基线为 {b2['accepted_edges']} 条。即使公共源分数较高，也不能据此宣布跨视角西安泛化成功。未以调低目标阈值、强填Stable或删除难例来改变这个结论。\n")
    texts.append('最终比较已统一到训练使用的像素X向右/Y向下框架。初版把米制北向上坐标直接代入新模型的运行属于输入合同错误，全部排除出选用结果；修复后重跑原生几何与V2候选位置两条分支，均保留在上表。V2分支只是研究几何候选，不代表绝对位置已认证。\n')
    texts.append('## 公开源评价与界限\n\n|模型|Macro F1|查询条件IDF1|Stable召回|Addition召回|Demolition召回|\n|---|---:|---:|---:|---:|---:|\n')
    fmt=lambda v:'N/A' if v is None else f'{v:.4f}'
    for r in mrows:texts.append(f"|{r['model']}|{r['macro_f1']:.4f}|{r['query_IDF1_strict_argmax']:.4f}|{fmt(r['stable_recall'])}|{fmt(r['addition_recall'])}|{fmt(r['demolition_recall'])}|\n")
    texts.append('\nB2没有本轮的Presence头，三态头指标为N/A；不把其几何SAME概率0.5代理造成的零事件当作可比性能。新模型的标签消失召回已从连续短窗诊断的0提高，但其标签消失精确率仅为 '+ ' / '.join(f"{r['demolition_precision']:.3f}" for r in mrows if r['model']!='B2_BASELINE')+'，不足以支持安全缺失准入。\n')
    texts.append('\n表中候选分类Macro F1已排除同图参考自对照，见 `evidence/CROSS_EPOCH_ONLY_SOURCE_METRICS.json`。训练/选模曲线及checkpoint原始指标保留冻结协议中的自对照，未事后改写选模结果；查询条件IDF1含初始化观察，须结合下面联合池追踪指标阅读。\n')
    assoc=read(ROOT/'evidence/ORACLE_PROPOSAL_ASSOCIATION_METRICS.json')
    texts.append('\n另对相同真值查询/候选联合池，按每期一对一带空出口匹配形成轨迹，再算IDF1及AssA关联项；未获得查询支持的提案保留为单点轨迹，不从分母删除。该共同选择器仅用于关联诊断，非完整场景检测评价。\n\n|模型|联合池IDF1|AssA|ID切换|身份碎片|\n|---|---:|---:|---:|---:|\n')
    for name,m in assoc.items():texts.append(f"|{name}|{m['IDF1']:.4f}|{m['AssA']:.4f}|{m['ID_switches']}|{m['identity_fragments']}|\n")
    texts.append('\n联合池共 '+str(assoc['B2_BASELINE']['nodes'])+' 个窗口内提案节点、'+str(assoc['B2_BASELINE']['windows'])+' 个窗口。新模型的联合关联结果未超过B2，不能用分类Macro F1提升代替追踪成功。AssA按[HOTA原论文](https://doi.org/10.1007/s11263-020-01375-2)关联定义计算；提供真值提案，相当于预先给定检测/定位，故完整HOTA仍不报告。\n')
    texts.append('\n上述评价使用官方ID和真值候选，按重叠五期查询窗口计数；是关联诊断，不是端到端IDF1、HOTA或SCOT。原TRAIN历史上已参与旧研究，这里的AOI组仅对本轮拟合隔离，不能冒充全新独立最终测试。alpha支持仅表示像素覆盖；Presence监督是标签记录存在，不是可见屋顶认证。源曲线中的二项Wilson上界忽略窗口相关性，仅为研发门槛，不能当目标误差保证。完整屋顶召回、目标三类精度、面积误差和独立目标准确率均未测得，保留null。\n')
    texts.append('三分类的实际语义是 SAME_ID / DIFFERENT_ID / NO_LABELLED_COUNTERPART；低置信或不可观测通过拒绝产生U。没有把第三类直接解释为真实拆除，也没有声称已有精确的视觉不可观测真值。\n')
    texts.append('## 保留的证据与缺口\n')
    texts.append('- 原3.7527227141平方米拆除候选及几何在 `evidence/OLD_DEMOLITION_CANDIDATE_RETAINED.json`，未按面积删除，亦未认证为真拆除。\n- BANDON BT坐标绑定对32个FIT父场景检查了锚定期、分量顺序和正负号。没有假定一个符号后直接生成身份真值；严格足点一致性门槛未通过，精确跨期warp监督仍未准入，故T1未记为全部完成。SpaceNet4仅在限定本地盘点中未发现，未声称全盘或网上不存在。\n- 未训练目标伪标签适配，也未新增独立分割模型；扩大B0上下文不等于独立分割器。不支持的60条上下文仅限当前固定输入，未等同真实不可观测，也未声称全市原图信息已经穷尽。无法判明的状态保留U。\n- 身份、当前期屋顶观察、alpha支持、相对位置、绝对位置、物理面积和正式资格分别输出。真实物理面积保持null；城墙形状筛查不生成新增/拆除。\n- 初始整段官方TRAIN辅助文件获取包含54对继承FINAL_TEST的辅助字节，已CRC解压后才按继承角色识别；这些文件随后隔离在 `data/QUARANTINE_INHERITED_FINAL_TEST`，未解码其RGB、未参与训练或指标。此范围问题与108个文件的去向完整披露在 `evidence/INHERITED_FINAL_TEST_QUARANTINE.json`。原SN7 Validation/Holdout未读取，原保护资产未改写，不宣称新增正式FINAL_TEST评价。\n')
    counts=read(ROOT/'evidence/INDEPENDENT_RELOAD_QA.json')
    texts.append('\n参数核算：总计 '+str(counts['total_parameters'])+'，其中T2主身份/时序/存在路径 '+str(counts['primary_T2_path_parameters'])+' 参数；BT辅助回归头 '+str(counts['auxiliary_BT_head_T2_unused'])+' 参数在T2不更新、也不用于部署追踪。T1到T2仅复用编码器。三种子独立CPU重载与已保存CUDA输出一致。源码修复包含候选排列不变性、多类argmax成本一致性、V8输入锚定掩膜双线性合同及面积不变的自相接触修复；初始失败/诊断尝试保留，不进入默认对照结果。\n')
    texts.append('## 工程验证与复跑\n')
    texts.append(f"54项现有应用与新增时序测试通过。{len(qa)}项面积积分核验通过，最大误差 {max(r['difference_m2'] for r in qa):.8f} 平方米；正式变化权重均为0。真实源图和学习/风险曲线见同目录PNG。\n\n`RUN_COMPARISON.cmd b2 --expanded` 与 `RUN_COMPARISON.cmd learned --family fullspan --expanded` 复用原 `run_xian_r5_integration.py` 入口的 `--temporal-upgrade` 分支；原 `--application` 路径保留。运行快照与本地数据路径绑定，未声称离线跨机器即插即用。\n\n继续研究的明确缺口是亚米跨视角的可验证对应监督和目标存在/缺失校准；当前产物证明训练与应用对照真实执行，没有证明可靠建筑变化识别已达成。\n")
    texts.append('\n来源与历史上下文：[BANDON官方实现](https://github.com/fitzpchao/BANDON)、[SpaceNet7官方页面](https://spacenet.ai/sn7-challenge/)。已读取本项目“升级 XIAN-R5 建筑身份识别”“提升西安R5屋顶分割精度”的实际对话，提取证据见 `evidence/PROJECT_CHAT_CONTEXT_VERIFIED.json`；旧任务结论未冒充本轮新结果。\n')
    import re
    report=re.sub(r'(^#{1,6}[^\n]*\n)',r'\1\n',''.join(texts),flags=re.MULTILINE)
    (ROOT/'FINAL_REPORT.md').write_text(report,encoding='utf8')
    body='<h1>XiAn temporal upgrade: research comparison</h1><p>Real training and rechecks executed. Production qualification remains NO.</p><p><a href="../FINAL_REPORT.md">完整中文报告</a></p>'
    body+='<img src="SOURCE_TRAINING_AND_RISK.png"><img src="REAL_SOURCE_CANDIDATES.png"><h2>Target comparisons</h2><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in ['variant','observations','tracks','accepted_edges','fixed_old_U_remaining'])+'</tr>'
    for r in comparison:body+='<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ['variant','observations','tracks','accepted_edges','fixed_old_U_remaining'])+'</tr>'
    body+='</table><h2>Products</h2>'
    for r in comparison:body+='<p>'+html.escape(r['variant'])+': <a href="'+html.escape(Path(r['products']).relative_to(ROOT).as_posix().replace('runs/','../runs/'),quote=True)+'/reports/INTERVAL_CHANGE_AREA_SUMMARY.csv">area table</a></p>'
    (out/'INDEX.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1400px;margin:36px auto;line-height:1.5;background:#121a23;color:#e8edf2}a{color:#8cd6ff}img{width:100%;margin:20px 0}table{border-collapse:collapse}td,th{border:1px solid #456;padding:8px;text-align:left}</style>'+body,encoding='utf8')
    write(out/'RESEARCH_ADMISSION.json',dict(engineering_executed=True,trained_primary=True,source_test_scope='historically exposed TRAIN internal groups and GT proposals',independent_target_accuracy=None,
        primary_weights_promoted=False,formal_supervision_admitted=False,production_ready=False,stages=stage))
if __name__=='__main__':main()
