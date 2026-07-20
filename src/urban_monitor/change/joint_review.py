from __future__ import annotations

import html
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


FINAL_STATES = ["stable", "new_building", "demolished", "construction_candidate", "uncertain", "rejected"]
TEACHER_DECISIONS = ["building_teacher_correct", "change_teacher_correct", "both_correct", "both_wrong", "registration_issue", "needs_geometry_edit", "needs_change_mask_edit"]


def build_joint_review(fused_path: str | Path, review_dir: str | Path) -> Path:
    frame = gpd.read_file(fused_path); review_dir = Path(review_dir); review_dir.mkdir(parents=True, exist_ok=True)
    records = []
    display_columns = ["fusion_id", "track_id", "building_teacher_state", "change_teacher_state", "new_probability_mean",
                       "demolition_probability_mean", "any_change_probability_mean", "uncertainty_mean", "teacher_agreement",
                       "fusion_confidence", "review_priority", "fusion_issues", "lineage_event"]
    for _, row in frame.iterrows():
        item = {key: row.get(key, "") for key in display_columns}; item["geometry_wkt"] = row.geometry.wkt if row.geometry else ""; records.append(item)
    payload = json.dumps(records, ensure_ascii=False, default=str).replace("</", "<\\/")
    final_options = "".join(f"<option>{state}</option>" for state in FINAL_STATES)
    teacher_options = "".join(f"<option>{state}</option>" for state in TEACHER_DECISIONS)
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>双教师联合审核</title><style>body{{font:14px system-ui;background:#0d1117;color:#e6edf3;margin:24px}}header{{position:sticky;top:0;background:#0d1117;padding:12px}}article{{border:1px solid #30363d;background:#161b22;border-radius:10px;padding:16px;margin:12px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}}span{{color:#9db0c3}}select,textarea,button{{background:#0d1117;color:#eee;border:1px solid #465467;padding:8px;margin:5px}}textarea{{width:50%}}</style><header><h1>双教师联合审核</h1><select id='priority'><option value=''>全部优先级</option><option>high</option><option>medium</option><option>low</option></select><button id='exportCsv'>导出 joint_review_decisions.csv</button></header><main id='list'></main><script>const DATA={payload},runKey='joint-review:{html.escape(Path(fused_path).stem)}:',priorityEl=document.querySelector('#priority'),listEl=document.querySelector('#list');function state(x){{return JSON.parse(localStorage.getItem(runKey+x.fusion_id)||'{{}}')}}function save(x,node){{localStorage.setItem(runKey+x.fusion_id,JSON.stringify({{fusion_id:x.fusion_id,final_state:node.querySelector('.final').value,teacher_decision:node.querySelector('.teacher').value,notes:node.querySelector('textarea').value,review_time:new Date().toISOString()}}))}}function render(){{const rows=DATA.filter(x=>!priorityEl.value||x.review_priority===priorityEl.value);listEl.replaceChildren(...rows.map(x=>{{const s=state(x),n=document.createElement('article');n.innerHTML=`<h2>${{x.fusion_id}}</h2><div class='metrics'><div>建筑教师<br><span>${{x.building_teacher_state}}</span></div><div>变化教师<br><span>${{x.change_teacher_state}}</span></div><div>new<br><span>${{x.new_probability_mean}}</span></div><div>demolition<br><span>${{x.demolition_probability_mean}}</span></div><div>any-change<br><span>${{x.any_change_probability_mean}}</span></div><div>uncertainty<br><span>${{x.uncertainty_mean}}</span></div><div>agreement<br><span>${{x.teacher_agreement}}</span></div><div>priority<br><span>${{x.review_priority}}</span></div></div><p>${{x.fusion_issues||''}}</p><select class='final'><option value=''>未审核</option>{final_options}</select><select class='teacher'><option value=''>教师判断</option>{teacher_options}</select><textarea placeholder='备注'>${{s.notes||''}}</textarea><button>保存</button>`;n.querySelector('.final').value=s.final_state||'';n.querySelector('.teacher').value=s.teacher_decision||'';n.querySelector('button').onclick=()=>save(x,n);return n}}))}}priorityEl.onchange=render;document.querySelector('#exportCsv').onclick=()=>{{const rows=DATA.map(state).filter(x=>x.final_state),cols=['fusion_id','final_state','teacher_decision','notes','review_time'],q=v=>'"'+String(v||'').replaceAll('"','""')+'"',csv='\ufeff'+cols.join(',')+'\\n'+rows.map(r=>cols.map(c=>q(r[c])).join(',')).join('\\n'),a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv]));a.download='joint_review_decisions.csv';a.click()}};render()</script></html>"""
    output = review_dir / "index.html"; output.write_text(page, encoding="utf-8"); return output


def apply_joint_review(fused_path: str | Path, decisions_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    frame = gpd.read_file(fused_path); decisions = pd.read_csv(decisions_path, dtype=str).fillna("")
    if decisions["fusion_id"].duplicated().any(): raise ValueError("Duplicate fusion_id in joint review decisions")
    mapping = decisions.set_index("fusion_id").to_dict("index")
    for index, row in frame.iterrows():
        decision = mapping.get(str(row["fusion_id"]), {})
        if decision.get("final_state"): frame.loc[index, "state_final"] = decision["final_state"]
        frame.loc[index, "joint_teacher_decision"] = decision.get("teacher_decision", "")
        frame.loc[index, "joint_review_notes"] = decision.get("notes", "")
        frame.loc[index, "joint_review_time"] = decision.get("review_time", "")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    reviewed = output_dir / "fused_reviewed.gpkg"; frame.to_file(reviewed, driver="GPKG")
    student = frame[frame["state_final"].astype(str).str.casefold() != "rejected"].copy()
    student_path = output_dir / "student_fusion_labels.gpkg"; student.to_file(student_path, driver="GPKG")
    report = ["# Joint review fusion report", "", f"- reviewed decisions: {len(decisions)}", f"- final candidates: {len(frame)}",
              f"- rejected: {int((frame['state_final'].astype(str).str.casefold()=='rejected').sum())}", f"- student labels: {len(student)}"]
    (output_dir / "fusion_report_reviewed.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return reviewed, student_path
