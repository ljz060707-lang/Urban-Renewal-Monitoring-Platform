"""Offline DSIFN final-test diagnostic review; deliberately has no model dependency."""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image


RUN_ID = "20260725_232317"
THRESHOLD = 0.75
DECISIONS = [
    ("unreviewed", "未审核"), ("correct", "模型基本正确"), ("false_positive", "模型明显误检"),
    ("false_negative", "模型明显漏检"), ("both_fp_fn", "模型同时误检和漏检"),
    ("label_issue", "标签存在问题"), ("misregistration", "影像配准存在问题"), ("uncertain", "无法判断"),
]
FIELDS = {
    "manual_error_type": [("unselected", "未选择"), ("no_obvious_problem", "无明显问题"), ("true_model_false_positive", "真实模型误检"), ("true_model_false_negative", "真实模型漏检"), ("misregistration", "配准偏移"), ("shadow_or_illumination", "阴影或光照变化"), ("seasonal_or_vegetation", "季节或植被变化"), ("radiometric_difference", "亮度/色彩/辐射差异"), ("bare_land_or_construction", "裸地或施工地干扰"), ("water_edge", "水体或水体边缘干扰"), ("boundary_error", "建筑边界预测不准"), ("tiny_change", "变化区域过小"), ("large_complex_change", "变化区域过大或过于复杂"), ("label_error", "标签错误"), ("other", "其他难以归类问题"), ("uncertain", "无法判断")],
    "registration_quality": [("unselected", "未选择"), ("good", "配准良好"), ("minor_offset", "轻微偏移"), ("moderate_offset", "明显偏移"), ("serious_offset", "严重错位"), ("uncertain", "无法判断")],
    "label_quality": [("unselected", "未选择"), ("good", "标签良好"), ("minor_boundary_issue", "存在轻微边界问题"), ("missing_label", "存在明显漏标"), ("wrong_label", "存在明显错标"), ("unusable", "标签基本不可用"), ("uncertain", "无法判断")],
    "scene_type": [("unselected", "未选择"), ("dense_urban", "高密度城区"), ("low_density_urban", "低密度城区"), ("residential", "住宅区"), ("industrial", "工业区/大型厂房"), ("peri_urban", "城乡混合区"), ("vegetation_mixed", "植被较多区域"), ("bare_land_construction", "裸地/施工地"), ("water_edge", "水体边缘"), ("road_hard_surface", "道路或硬化地面"), ("other", "其他"), ("uncertain", "无法判断")],
    "severity": [("unselected", "未选择"), ("none", "无问题"), ("minor", "轻微"), ("moderate", "中等"), ("severe", "严重"), ("catastrophic", "灾难性失败")],
}


def _safe(sample_id: str) -> str:
    return sample_id.replace(":", "_").replace("/", "_")


def _number(row: pd.Series, key: str) -> float:
    return float(pd.to_numeric(row.get(key, 0), errors="coerce") if pd.notna(pd.to_numeric(row.get(key, 0), errors="coerce")) else 0)


def _category(row: pd.Series) -> str:
    f1, fp, fn = _number(row, "f1"), _number(row, "false_positive_pixels"), _number(row, "false_negative_pixels")
    if f1 == 0: return "catastrophic_zero"
    if fp and fn and f1 < .30: return "both_poor"
    if fp > fn * 1.5: return "false_positive_dominant"
    if fn > fp * 1.5: return "false_negative_dominant"
    if f1 < .50: return "weak_mixed"
    return "strong_control" if f1 >= .80 else "medium_control"


def _choose_validation(frame: pd.DataFrame, limit: int = 80) -> pd.DataFrame:
    """Deterministic stratified queue; each sample can occur only once."""
    pool, chosen = frame.copy(), []
    pool["diagnostic_category"] = pool.apply(_category, axis=1)
    def take(rows: pd.DataFrame, count: int) -> None:
        for index in rows.index:
            if index not in chosen and len(chosen) < limit:
                chosen.append(index)
                if sum(item in chosen for item in rows.index) >= count: break
    take(pool.sort_values("f1"), 25)
    take(pool[pool.diagnostic_category == "false_positive_dominant"].sort_values("f1"), 15)
    take(pool[pool.diagnostic_category == "false_negative_dominant"].sort_values("f1"), 10)
    take(pool.sort_values("uncertainty", ascending=False), 10)
    rng = random.Random(42)
    for subset, count in ((pool[pool.f1 >= .8], 10), (pool[(pool.f1 >= .5) & (pool.f1 < .8)], 10)):
        ids = list(subset.index); rng.shuffle(ids); take(pool.loc[ids], count)
    remainder = [index for index in pool.sort_values(["f1", "uncertainty"], ascending=[True, False]).index if index not in chosen]
    chosen.extend(remainder[: max(0, limit - len(chosen))])
    return pool.loc[chosen[:limit]].copy()


def _copy_image(source: str, destination: Path, *, mask: bool = False) -> bool:
    path = Path(str(source))
    if not path.is_file(): return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        image = image.convert("L" if mask else "RGB")
        image.thumbnail((1024, 1024))
        image.save(destination)
    return True


def _overlay(t2_path: Path, gt_path: Path, probability_path: Path, destination: Path) -> bool:
    if not all(path.is_file() for path in (t2_path, gt_path, probability_path)): return False
    with Image.open(t2_path) as image:
        base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(gt_path) as image:
        gt = np.asarray(image.convert("L").resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST)) > 0
    with Image.open(probability_path) as image:
        prediction = np.asarray(image.convert("L").resize((base.shape[1], base.shape[0]), Image.Resampling.NEAREST)) >= round(255 * THRESHOLD)
    colors = np.zeros_like(base); colors[prediction & gt] = (41, 196, 93); colors[prediction & ~gt] = (230, 57, 70); colors[~prediction & gt] = (52, 120, 246)
    visible = prediction | gt
    base[visible] = (base[visible].astype(np.float32) * .40 + colors[visible].astype(np.float32) * .60).astype(np.uint8)
    Image.fromarray(base).save(destination)
    return True


def _materialize_assets(rows: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    assets, counts, records = root / "assets", {key: 0 for key in ("t1", "t2", "gt", "prediction", "overlay")}, []
    for _, row in rows.iterrows():
        item, safe = row.to_dict(), _safe(str(row.sample_id))
        sources = {"t1": row.get("t1_path", ""), "t2": row.get("t2_path", ""), "gt": row.get("gt_binary_path", ""), "prediction": row.get("binary_probability_path", "")}
        suffixes = {"t1": "t1.jpg", "t2": "t2.jpg", "gt": "gt.png", "prediction": "prediction.png"}
        paths = {}
        for name, source in sources.items():
            target = assets / f"{safe}_{suffixes[name]}"; ok = _copy_image(source, target, mask=name in {"gt", "prediction"})
            item[f"{name}_asset"] = target.relative_to(root).as_posix() if ok else ""; paths[name] = target
            counts[name] += int(ok)
        overlay = assets / f"{safe}_overlay.jpg"; ok = _overlay(paths["t2"], paths["gt"], paths["prediction"], overlay)
        item["overlay_asset"] = overlay.relative_to(root).as_posix() if ok else ""; counts["overlay"] += int(ok)
        records.append(item)
    return pd.DataFrame(records), counts


def _html(records: list[dict], run_id: str) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    options = json.dumps({"manual_decision": DECISIONS, **FIELDS}, ensure_ascii=False)
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DSIFN Final Test 泛化诊断</title>
<style>:root{{--bg:#090c12;--panel:#121822;--line:#2d3748;--text:#eaf0f8;--muted:#9aa9bd;--test:#ff6577;--val:#ffb454}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,"Microsoft YaHei",sans-serif}}header{{position:sticky;top:0;z-index:5;background:#090c12f7;border-bottom:1px solid var(--line);padding:14px 22px}}.warning{{color:#ffd5da;font-weight:700}}.tools,.info,.fields{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}select,button{{background:#202938;border:1px solid #46566e;color:var(--text);border-radius:6px;padding:8px}}button{{cursor:pointer}}main{{max-width:1800px;margin:auto;padding:18px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.tag{{border-radius:5px;padding:4px 8px;font-weight:700}}.test{{background:#622330;color:#ffd9de}}.validation{{background:#624719;color:#ffe1a6}}.views{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:16px 0}}.view{{background:#05070b;border:1px solid #263247;border-radius:7px;padding:6px;text-align:center}}.view img{{width:100%;aspect-ratio:1;object-fit:contain;background:#000;cursor:zoom-in}}.fields{{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));align-items:end}}label{{display:grid;gap:5px;color:var(--muted)}}.progress{{margin-left:auto;color:var(--muted)}}.empty{{color:#ffb4bd}}dialog{{border:0;background:#000;padding:0;max-width:95vw}}dialog img{{max-width:94vw;max-height:92vh}}@media(max-width:1100px){{.views{{grid-template-columns:repeat(2,1fr)}}.fields{{grid-template-columns:1fr}}}}</style>
<header><div class="warning">⚠ Held-out Test 已完成最终评估：人工审核仅用于误差诊断。禁止将 Test 样本用于训练、阈值调整、checkpoint 选择或超参数调整。</div><div class="tools"><b>DSIFN 泛化诊断审核</b><span id="progress" class="progress"></span><select id="split"><option value="all">数据集：全部</option><option value="test">Test</option><option value="validation">Validation</option></select><select id="category"><option value="all">诊断类别：全部</option></select><select id="reviewed"><option value="all">审核状态：全部</option><option value="no">未审核</option><option value="yes">已审核</option></select><select id="decisionFilter"><option value="all">人工判断：全部</option></select><select id="sort"><option value="priority">排序：审核优先级</option><option value="f1asc">F1 从低到高</option><option value="f1desc">F1 从高到低</option><option value="precision">Precision 从低到高</option><option value="recall">Recall 从低到高</option><option value="uncertainty">Uncertainty 从高到低</option></select><button id="export">导出审核结果 CSV</button></div></header>
<main><section class="card"><div id="info" class="info"></div><div class="views" id="views"></div><div class="fields" id="fields"></div><div class="tools" style="margin-top:14px"><button id="prev">← 上一张</button><button id="next">下一张 →</button><button id="save">保存当前</button><button data-d="correct">标记为模型正确</button><button data-d="false_positive">标记为误检</button><button data-d="false_negative">标记为漏检</button><button data-d="misregistration">标记为配准问题</button><button data-d="label_issue">标记为标签问题</button></div></section></main><dialog id="zoom"><img></dialog>
<script>const DATA=__PAYLOAD__, OPTIONS=__OPTIONS__, KEY='dsifn-diagnostic-review:__RUN__:';let rows=[],pos=0;const $=s=>document.querySelector(s),state=id=>JSON.parse(localStorage.getItem(KEY+id)||'{}'),put=(id,v)=>localStorage.setItem(KEY+id,JSON.stringify(v));const cn=(field,value)=>{let x=(OPTIONS[field]||[]).find(a=>a[0]===value);return x?x[1]:value||''};
for(const v of [...new Set(DATA.map(x=>x.diagnostic_category))])$('#category').insertAdjacentHTML('beforeend',`<option value="${{v}}">${{v}}</option>`);for(const [v,l] of OPTIONS.manual_decision)$('#decisionFilter').insertAdjacentHTML('beforeend',`<option value="${{v}}">${{l}}</option>`);
function choose(){{let split=$('#split').value,cat=$('#category').value,reviewed=$('#reviewed').value,decision=$('#decisionFilter').value;rows=DATA.filter(x=>{{let s=state(x.sample_id);return(split==='all'||x.source_split===split)&&(cat==='all'||x.diagnostic_category===cat)&&(decision==='all'||s.manual_decision===decision)&&(reviewed==='all'||(reviewed==='yes'?!!s.reviewed:!s.reviewed))}});let sort=$('#sort').value;rows.sort((a,b)=>sort==='f1asc'?+a.f1-+b.f1:sort==='f1desc'?+b.f1-+a.f1:sort==='precision'?+a.precision-+b.precision:sort==='recall'?+a.recall-+b.recall:sort==='uncertainty'?+b.uncertainty-+a.uncertainty:(+b.review_priority||0)-(+a.review_priority||0));pos=Math.min(pos,Math.max(0,rows.length-1));render()}}
function select(field,value){{return `<label>${{field}}<select data-field="${{field}}">${{(OPTIONS[field]||[]).map(([v,l])=>`<option value="${{v}}" ${{v===value?'selected':''}}>${{l}}</option>`).join('')}</select></label>`}}
function render(){{let x=rows[pos];if(!x){{$('#info').innerHTML='<span class="empty">没有匹配样本</span>';$('#views').innerHTML='';return}}let saved=state(x.sample_id),all=DATA,reviewed=all.filter(r=>state(r.sample_id).reviewed),test=all.filter(r=>r.source_split==='test'),val=all.filter(r=>r.source_split==='validation');$('#progress').textContent=`${{pos+1}} / ${{rows.length}} · 总 ${{all.length}} · 已审核 ${{reviewed.length}} (${{Math.round(reviewed.length/all.length*100)}}%) · Test ${{test.filter(r=>state(r.sample_id).reviewed).length}}/${{test.length}} · Validation ${{val.filter(r=>state(r.sample_id).reviewed).length}}/${{val.length}}`;let tag=x.source_split==='test'?'test':'validation';$('#info').innerHTML=`<b>${{x.sample_id}}</b><span class="tag ${{tag}}">${{x.source_split==='test'?'TEST · 仅诊断 · 禁止训练':'VALIDATION · 诊断用途'}}</span><span>F1 ${{(+x.f1).toFixed(3)}} · IoU ${{(+x.iou).toFixed(3)}} · P ${{(+x.precision).toFixed(3)}} · R ${{(+x.recall).toFixed(3)}}</span><span>Uncertainty ${{(+x.uncertainty).toFixed(3)}}</span><span>${{x.diagnostic_category}}</span>`;let cards=[['T1',x.t1_asset],['T2',x.t2_asset],['GT',x.gt_asset],['Prediction',x.prediction_asset],['Overlay (TP绿 FP红 FN蓝)',x.overlay_asset]];$('#views').innerHTML=cards.map(([name,path])=>`<div class="view"><b>${{name}}</b>${{path?`<img src="${{path}}" alt="${{name}}">`:'<p class="empty">资产缺失</p>'}}</div>`).join('');$('#fields').innerHTML=['manual_decision','manual_error_type','registration_quality','label_quality','scene_type','severity'].map(f=>select(f,saved[f]||(f==='manual_decision'?'unreviewed':'unselected'))).join('');document.querySelectorAll('[data-field]').forEach(e=>e.onchange=save)}}
function save(){{let x=rows[pos];if(!x)return;let v={{...state(x.sample_id)}};document.querySelectorAll('[data-field]').forEach(e=>v[e.dataset.field]=e.value);v.reviewed=v.manual_decision&&v.manual_decision!=='unreviewed';v.review_time=new Date().toISOString();put(x.sample_id,v);render()}}function move(n){{save();pos=(pos+n+rows.length)%rows.length;render()}}$('#prev').onclick=()=>move(-1);$('#next').onclick=()=>move(1);$('#save').onclick=save;document.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>{{let s=$('[data-field="manual_decision"]');s.value=b.dataset.d;save()}});['split','category','reviewed','decisionFilter','sort'].forEach(id=>$('#'+id).onchange=choose);document.onkeydown=e=>{{if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);let d={{'1':'correct','2':'false_positive','3':'false_negative','4':'both_fp_fn','5':'label_issue','6':'misregistration','0':'uncertain'}}[e.key];if(d){{let s=$('[data-field="manual_decision"]');if(s){{s.value=d;save()}}}}}};$('#views').onclick=e=>{{if(e.target.tagName==='IMG'){{$('#zoom img').src=e.target.src;$('#zoom').showModal()}}}};$('#zoom').onclick=()=>$('#zoom').close();$('#export').onclick=()=>{{let keys=['sample_id','dataset_name','source_split','f1','iou','precision','recall','uncertainty','diagnostic_category','manual_decision','manual_decision_cn','manual_error_type','manual_error_type_cn','registration_quality','registration_quality_cn','label_quality','label_quality_cn','scene_type','scene_type_cn','severity','severity_cn','reviewed','review_time','training_eligible','threshold_selection_eligible','model_selection_eligible'];let q=v=>'"'+String(v??'').replaceAll('"','""')+'"',out=DATA.map(x=>{{let s=state(x.sample_id),r={{...x,...s}};for(let f of ['manual_decision','manual_error_type','registration_quality','label_quality','scene_type','severity'])r[f+'_cn']=cn(f,r[f]);r.training_eligible=false;r.threshold_selection_eligible=false;r.model_selection_eligible=false;return r}});let csv='\ufeff'+keys.join(',')+'\n'+out.map(r=>keys.map(k=>q(r[k])).join(',')).join('\n'),a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='DSIFN_MANUAL_REVIEW_DECISIONS___RUN__.csv';a.click()}};choose();</script></html>'''
    # The template originated as an f-string, so collapse its escaped braces
    # before injecting JSON (never transform JSON content itself).
    return page.replace("{{", "{").replace("}}", "}").replace("__PAYLOAD__", payload).replace("__OPTIONS__", options).replace("__RUN__", run_id)


def build_manual_diagnostic_review(run: str | Path, *, validation_limit: int = 80) -> dict:
    """Build assets/page from already persisted review artifacts only; never forwards a model."""
    run = Path(run).resolve(); root = run / "manual_diagnostic_review"; root.mkdir(parents=True, exist_ok=True)
    test_csv, validation_csv = run / "review" / "review_samples_test.csv", run / "review" / "review_samples.csv"
    if not test_csv.is_file() or not validation_csv.is_file():
        raise FileNotFoundError("Manual diagnostic review requires existing review_samples_test.csv and review_samples.csv; it never reruns inference.")
    test, validation = pd.read_csv(test_csv, dtype=str).fillna(""), pd.read_csv(validation_csv, dtype=str).fillna("")
    if len(test) != 48:
        raise ValueError(f"Expected the locked DSIFN final-test receipt to contain 48 samples, got {len(test)}")
    for frame, split in ((test, "test"), (validation, "validation")):
        numeric = ["f1", "iou", "precision", "recall", "uncertainty", "false_positive_pixels", "false_negative_pixels"]
        for key in numeric: frame[key] = pd.to_numeric(frame[key], errors="coerce").fillna(0.0)
        frame["source_split"] = split; frame["diagnostic_category"] = frame.apply(_category, axis=1)
        frame["training_eligible"] = False; frame["threshold_selection_eligible"] = False; frame["model_selection_eligible"] = False
        frame["manual_decision"] = "unreviewed"; frame["manual_error_type"] = "unselected"; frame["registration_quality"] = "unselected"; frame["label_quality"] = "unselected"; frame["scene_type"] = "unselected"; frame["severity"] = "unselected"
    validation = _choose_validation(validation, validation_limit)
    test["review_priority"] = np.arange(len(test), 0, -1); validation["review_priority"] = np.arange(len(validation), 0, -1)
    combined = pd.concat([test, validation], ignore_index=True)
    materialized, counts = _materialize_assets(combined, root)
    test_output, validation_output = materialized[materialized.source_split == "test"], materialized[materialized.source_split == "validation"]
    test_output.to_csv(root / "TEST_ALL48_MANUAL_DIAGNOSTIC.csv", index=False, encoding="utf-8-sig")
    validation_output.to_csv(root / "VALIDATION_MANUAL_REVIEW_QUEUE_80.csv", index=False, encoding="utf-8-sig")
    (root / "index.html").write_text(_html(materialized.to_dict(orient="records"), run.name), encoding="utf-8")
    report = {"run_id": run.name, "test_count": len(test_output), "validation_count": len(validation_output), "asset_counts": counts, "missing_assets": {key: len(materialized) - value for key, value in counts.items()}, "test_inference_performed": False, "source_files": [str(test_csv), str(validation_csv)]}
    (root / "MANUAL_DIAGNOSTIC_BUILD.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
