from __future__ import annotations

import html
import json
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image


DECISIONS = ["correct", "prediction_error", "label_error", "misregistration", "uncertain", "exclude", "needs_mask_edit"]


def _asset(source: str | Path, destination: Path, max_size: int = 640) -> str:
    source = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB") if source.suffix.casefold() not in {".png"} else image.copy()
        image.thumbnail((max_size, max_size))
        image.save(destination)
    return destination.as_posix()


def build_static_review(run: str | Path, samples_csv: str | Path | None = None) -> Path:
    run = Path(run).resolve(); review = run / "review"; assets = review / "assets"; review.mkdir(parents=True, exist_ok=True)
    samples_csv = Path(samples_csv) if samples_csv else review / "review_samples.csv"
    if not samples_csv.is_file():
        raise FileNotFoundError(f"Review samples not found: {samples_csv}")
    frame = pd.read_csv(samples_csv).fillna("")
    records = []
    for index, row in frame.iterrows():
        safe = str(row["sample_id"]).replace(":", "_").replace("/", "_")
        item = row.to_dict()
        for key, suffix in (("t1_path", "t1.jpg"), ("t2_path", "t2.jpg"), ("binary_probability_path", "binary.png"),
                            ("new_probability_path", "new.png"), ("demolition_probability_path", "demolition.png"), ("preview_path", "preview.jpg")):
            if item.get(key) and Path(str(item[key])).is_file():
                destination = assets / f"{safe}_{suffix}"
                _asset(item[key], destination)
                item[key] = destination.relative_to(review).as_posix()
        errors = []
        preview_name = Path(str(row.get("preview_path", ""))).name
        for folder in (run / "errors").iterdir() if (run / "errors").is_dir() else []:
            if folder.is_dir() and (folder / preview_name).exists(): errors.append(folder.name)
        item["errors"] = errors; records.append(item)
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(f"<option value='{value}'>{value}</option>" for value in DECISIONS)
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>变化教师人工审核</title><style>
:root{{--bg:#0b0f14;--panel:#131a22;--line:#263241;--text:#edf4fb;--muted:#93a4b5;--blue:#4ca8ff;--orange:#ff9d3d}}*{{box-sizing:border-box}}body{{margin:0;overflow-x:hidden;background:var(--bg);color:var(--text);font:14px system-ui,"Microsoft YaHei"}}header{{position:sticky;top:0;z-index:5;background:#0b0f14ee;border-bottom:1px solid var(--line);padding:16px 24px}}h1{{margin:0 0 12px;font-size:22px}}.tools{{display:flex;flex-wrap:wrap;gap:10px}}.tools>*{{flex:0 0 auto;min-width:0}}select,input,textarea,button{{background:#111923;color:var(--text);border:1px solid #334255;border-radius:7px;padding:8px}}button{{cursor:pointer}}button.primary{{background:#1976d2;border-color:#2589e8;white-space:nowrap}}main{{max-width:1500px;margin:auto;padding:20px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:0 0 18px;overflow:hidden}}.meta{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line)}}.pill{{background:#202b38;padding:4px 8px;border-radius:99px;color:#b8c7d6}}.images{{display:grid;grid-template-columns:1fr 1fr 1.35fr;gap:10px;padding:12px;overflow:hidden}}.images>*{{min-width:0}}.images img,.images canvas{{display:block;width:100%;max-width:100%;aspect-ratio:1.3;object-fit:contain;background:#080b0f;border-radius:8px}}.images [hidden]{{display:none}}.review{{display:grid;grid-template-columns:220px 1fr 180px;gap:10px;padding:12px}}.review textarea{{min-height:70px}}.controls{{padding:0 12px 14px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}.empty{{padding:50px;text-align:center;color:var(--muted)}}@media(max-width:800px){{.images,.review{{grid-template-columns:1fr}}}}</style></head><body>
<header><h1>变化教师人工审核</h1><div class='tools'><select id='dataset'><option value=''>全部数据集</option></select><select id='stage'><option value=''>全部阶段</option></select><select id='error'><option value=''>全部错误类型</option></select><select id='sort'><option value='uncertainty'>不确定性高优先</option><option value='f1'>F1 低优先</option><option value='iou'>IoU 低优先</option></select><input id='reviewer' placeholder='审核人'><button class='primary' id='export'>导出 review_decisions.csv</button></div></header><main id='list'></main>
<script>const DATA={payload};const DECISIONS={json.dumps(DECISIONS)};const key=s=>`change-review:{html.escape(run.name)}:${{s}}`;const list=document.querySelector('#list'),datasetEl=document.querySelector('#dataset'),stageEl=document.querySelector('#stage'),errorEl=document.querySelector('#error'),sortEl=document.querySelector('#sort');
function unique(field){{return [...new Set(DATA.flatMap(x=>field==='errors'?x.errors:[x[field]]).filter(Boolean))].sort()}}function fill(id,field){{const el=document.querySelector(id);unique(field).forEach(v=>el.insertAdjacentHTML('beforeend',`<option>${{v}}</option>`))}}fill('#dataset','dataset_name');fill('#stage','stage');fill('#error','errors');
function state(item){{return JSON.parse(localStorage.getItem(key(item.sample_id))||'{{}}')}}function save(item,card){{const value={{sample_id:item.sample_id,dataset_name:item.dataset_name,stage:item.stage,decision:card.querySelector('.decision').value,notes:card.querySelector('.notes').value,reviewer:document.querySelector('#reviewer').value,review_time:new Date().toISOString(),threshold:card.querySelector('.threshold').value,correction_path:''}};localStorage.setItem(key(item.sample_id),JSON.stringify(value));}}
function overlay(card,item){{const canvas=card.querySelector('canvas'),base=card.querySelector('.t2'),mask=card.querySelector('.binaryMask');if(!base.complete||!mask.complete)return;const ctx=canvas.getContext('2d');canvas.width=base.naturalWidth;canvas.height=base.naturalHeight;ctx.drawImage(base,0,0);const off=document.createElement('canvas');off.width=canvas.width;off.height=canvas.height;const ox=off.getContext('2d');ox.drawImage(mask,0,0,off.width,off.height);const data=ox.getImageData(0,0,off.width,off.height),out=ctx.createImageData(off.width,off.height),threshold=+card.querySelector('.threshold').value*2.55,alpha=+card.querySelector('.opacity').value*2.55;for(let i=0;i<data.data.length;i+=4)if(data.data[i]>=threshold){{out.data[i]=76;out.data[i+1]=168;out.data[i+2]=255;out.data[i+3]=alpha}}ctx.putImageData(out,0,0)}}
function card(item){{const saved=state(item),node=document.createElement('section');node.className='card';node.innerHTML=`<div class='meta'><strong>${{item.sample_id}}</strong><span class='pill'>${{item.dataset_name}}</span><span class='pill'>${{item.stage}}</span><span>F1 ${{item.f1||'N/A'}} · IoU ${{item.iou||'N/A'}} · U ${{item.uncertainty||'N/A'}}</span><span>${{item.errors.join(', ')}}</span></div><div class='images'><img src='${{item.t1_path}}' alt='T1'><img class='t2' src='${{item.t2_path}}' alt='T2'><canvas></canvas><img class='binaryMask' hidden src='${{item.binary_probability_path}}'></div><div class='controls'><label>阈值 <input class='threshold' type='range' min='0' max='100' value='${{saved.threshold||50}}'></label><label>透明度 <input class='opacity' type='range' min='0' max='100' value='55'></label><span>蓝色 = binary prediction</span></div><div class='review'><select class='decision'><option value=''>未审核</option>{options}</select><textarea class='notes' placeholder='中文备注'>${{saved.notes||''}}</textarea><button class='save'>保存本条</button></div>`;node.querySelector('.decision').value=saved.decision||'';node.querySelectorAll('img').forEach(x=>x.onload=()=>overlay(node,item));node.querySelectorAll('.threshold,.opacity').forEach(x=>x.oninput=()=>overlay(node,item));node.querySelector('.save').onclick=()=>save(item,node);return node}}
function render(){{let rows=DATA.filter(x=>(!datasetEl.value||x.dataset_name===datasetEl.value)&&(!stageEl.value||x.stage===stageEl.value)&&(!errorEl.value||x.errors.includes(errorEl.value)));const field=sortEl.value;rows.sort((a,b)=>field==='uncertainty'?(+b[field]-+a[field]):(+a[field]-+b[field]));list.replaceChildren(...rows.map(card));if(!rows.length)list.innerHTML='<div class="empty">没有匹配样本</div>'}}document.querySelectorAll('#dataset,#stage,#error,#sort').forEach(x=>x.onchange=render);
document.querySelector('#export').onclick=()=>{{const rows=DATA.map(x=>state(x)).filter(x=>x.decision);const cols=['sample_id','dataset_name','stage','decision','notes','reviewer','review_time','threshold','correction_path'];const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';const csv='\ufeff'+cols.join(',')+'\\n'+rows.map(r=>cols.map(c=>esc(r[c])).join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='review_decisions.csv';a.click()}};render();</script></body></html>"""
    path = review / "index.html"; path.write_text(document, encoding="utf-8")
    return path
