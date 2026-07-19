from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import from_bounds

from .manifest import DateRecord
from .tracking import compute_confidence_score


STATE_OPTIONS = ["stable", "new_building", "demolished", "construction_candidate", "uncertain", "rejected"]
DATE_QUALITY_OPTIONS = ["usable", "unusable", "occluded", "registration_issue"]
LINEAGE_OPTIONS = ["confirm", "reject", "needs_geometry_edit"]


def _render_track_chip(record: DateRecord, geometry, output: Path, padding_m: float, *, present: bool) -> None:
    with rasterio.open(record.image_path) as src:
        minx, miny, maxx, maxy = geometry.bounds
        window = from_bounds(minx - padding_m, miny - padding_m, maxx + padding_m, maxy + padding_m, src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(window=window)
        transform = src.window_transform(window)
        rgb = np.moveaxis(data[:3] if data.shape[0] >= 3 else np.repeat(data[:1], 3, axis=0), 0, -1)
        if rgb.dtype != np.uint8:
            lo, hi = np.nanpercentile(rgb, [2, 98])
            rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(image)
        if geometry.geom_type == "MultiPolygon":
            geometry = max(geometry.geoms, key=lambda g: g.area)
        coords = []
        for x, y in geometry.exterior.coords:
            col, row = (~transform) * (x, y)
            coords.append((float(col), float(row)))
        if len(coords) >= 3:
            color = (255, 0, 0) if present else (255, 215, 0)
            draw.line(coords + [coords[0]], fill=color, width=3)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)


def build_review_queue(
    summary_path: str | Path,
    records: list[DateRecord],
    output_dir: str | Path,
    *,
    padding_m: float,
    long_path: str | Path | None = None,
    lineage_path: str | Path | None = None,
) -> Path:
    summary = gpd.read_file(summary_path)
    long_frame = gpd.read_file(long_path) if long_path and Path(long_path).exists() else None
    lineage = gpd.read_file(lineage_path) if lineage_path and Path(lineage_path).exists() else gpd.GeoDataFrame()
    output = Path(output_dir)
    chip_dir = output / "chips"
    rows = []
    for _, row in summary.iterrows():
        track_id = str(row["track_id"])
        by_date = {}
        if long_frame is not None and not long_frame.empty:
            subset = long_frame[long_frame["track_id"].astype(str) == track_id]
            by_date = {str(r["date"]): r.geometry for _, r in subset.iterrows() if r.geometry is not None}
        images = []
        for record in records:
            geometry = by_date.get(record.date, row.geometry)
            present = record.date in by_date if long_frame is not None else True
            chip_path = chip_dir / track_id / f"{record.date}.jpg"
            _render_track_chip(record, geometry, chip_path, padding_m, present=present)
            images.append(
                {
                    "date": record.date,
                    "path": str(chip_path.relative_to(output)).replace("\\", "/"),
                    "present": present,
                    "off_nadir_deg": record.off_nadir_deg,
                    "season": record.season,
                }
            )
        event_refs = []
        if not lineage.empty:
            for _, ev in lineage.iterrows():
                parents = str(ev.get("parent_track_ids", "")).split(";")
                children = str(ev.get("child_track_ids", "")).split(";")
                if track_id in parents or track_id in children:
                    event_refs.append(str(ev.get("event_id", "")))
        rows.append(
            {
                "track_id": track_id,
                "auto_state": str(row.get("state", "uncertain")),
                "confidence": str(row.get("confidence", "low")),
                "mean_score": float(row.get("mean_score", 0.0)),
                "temporal_flag": str(row.get("temporal_flag", "")),
                "event_refs": event_refs,
                "images": images,
                "schema_version": "0.4.0",
            }
        )
    lineage_rows = []
    if not lineage.empty:
        for _, row in lineage.iterrows():
            lineage_rows.append(
                {
                    "event_id": str(row.get("event_id", "")),
                    "event_type": str(row.get("event_type", "")),
                    "date_from": str(row.get("date_from", "")),
                    "date_to": str(row.get("date_to", "")),
                    "parents": str(row.get("parent_track_ids", "")),
                    "children": str(row.get("child_track_ids", "")),
                    "coverage": float(row.get("coverage", 0.0)),
                }
            )

    payload = json.dumps(rows, ensure_ascii=False)
    lineage_payload = json.dumps(lineage_rows, ensure_ascii=False)
    states = json.dumps(STATE_OPTIONS, ensure_ascii=False)
    quality_options = json.dumps(DATE_QUALITY_OPTIONS, ensure_ascii=False)
    lineage_options = json.dumps(LINEAGE_OPTIONS, ensure_ascii=False)
    html = f"""<!doctype html>
<meta charset='utf-8'>
<title>城市更新时序 Ground Truth 审核</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#f5f6f8;color:#1f2937}}
.card{{background:white;border:1px solid #d7dce2;border-radius:8px;margin:14px 0;padding:12px}}
.timeline{{display:flex;gap:8px;overflow-x:auto}}
.frame{{min-width:230px;border:1px solid #e3e6ea;padding:6px;border-radius:6px}} .frame img{{width:220px;height:220px;object-fit:contain;background:#111}}
button{{margin:3px;padding:7px 10px}} button.active{{background:#2563eb;color:white}}
.qbtn{{font-size:11px;padding:4px 6px}} .qbtn.active{{background:#7c3aed}}
textarea{{width:100%;min-height:55px}} #csv{{width:100%;height:220px;font-family:monospace}}
input{{padding:7px 10px;margin:3px 8px;border:1px solid #c7cdd4;border-radius:4px}}
.flag{{background:#fff3cd;padding:4px 8px;border-radius:4px}} .meta{{color:#59636e;font-size:12px}}
</style>
<h1>城市更新时序 Ground Truth 审核</h1>
<p>红线为当期检测，黄线仅表示代表位置。判断拆除前必须先标记不可用、遮挡或配准异常日期；merge/split lineage 独立审核。</p>
<label for='reviewer'>审核人</label><input id='reviewer' placeholder='姓名或稳定 ID'><button onclick='exportCSV()'>导出审核 CSV</button>
<h2>建筑轨迹</h2><div id='root'></div>
<h2>父子 lineage 事件</h2><div id='lineageRoot'></div>
<h2>CSV</h2><textarea id='csv'></textarea>
<script>
const rows={payload}; const lineageRows={lineage_payload}; const states={states}; const qualityOptions={quality_options}; const lineageOptions={lineage_options};
const key='urban-renewal-review-v4'; let labels=JSON.parse(localStorage.getItem(key)||'{{}}');
function save(){{localStorage.setItem(key,JSON.stringify(labels));}}
const root=document.getElementById('root');
for(const r of rows){{
 const card=document.createElement('div'); card.className='card';
 const flag=r.temporal_flag?`<span class='flag'>${{r.temporal_flag}}</span>`:'';
 card.innerHTML=`<h3>${{r.track_id}} | auto=${{r.auto_state}} | confidence=${{r.confidence}} | score=${{r.mean_score.toFixed(3)}} ${{flag}}</h3><div class='meta'>lineage events: ${{r.event_refs.join(', ')||'none'}}</div>`;
 const tl=document.createElement('div'); tl.className='timeline';
 for(const im of r.images){{
   const f=document.createElement('div');f.className='frame';
   f.innerHTML=`<b>${{im.date}}</b> | detected=${{im.present}}<div class='meta'>off-nadir=${{im.off_nadir_deg??''}} season=${{im.season||''}}</div><img src='${{im.path}}'><div class='qcontrols'></div>`;
   const qc=f.querySelector('.qcontrols');
   for(const q of qualityOptions){{const b=document.createElement('button');b.className='qbtn';b.textContent=q;b.onclick=()=>{{labels[r.track_id]=labels[r.track_id]||{{}};labels[r.track_id].date_quality=labels[r.track_id].date_quality||{{}};labels[r.track_id].date_quality[im.date]=q;save();renderQ()}};qc.appendChild(b)}}
   function renderQ(){{for(const b of qc.children)b.classList.toggle('active',b.textContent===(((labels[r.track_id]||{{}}).date_quality||{{}})[im.date]||'usable'))}} renderQ();
   tl.appendChild(f)
 }}
 card.appendChild(tl); const controls=document.createElement('div');
 for(const s of states){{const b=document.createElement('button');b.textContent=s;b.onclick=()=>{{labels[r.track_id]=labels[r.track_id]||{{}};labels[r.track_id].state=s;save();render()}};controls.appendChild(b)}}
 const notes=document.createElement('textarea');notes.placeholder='notes / roof-vs-footprint issue / geometry correction needed';notes.value=(labels[r.track_id]||{{}}).notes||'';notes.oninput=()=>{{labels[r.track_id]=labels[r.track_id]||{{}};labels[r.track_id].notes=notes.value;save()}};
 card.appendChild(controls);card.appendChild(notes);root.appendChild(card);
 function render(){{for(const b of controls.children)b.classList.toggle('active',b.textContent===((labels[r.track_id]||{{}}).state||r.auto_state))}} render();
}}
const lineageRoot=document.getElementById('lineageRoot');
for(const e of lineageRows){{
 const card=document.createElement('div');card.className='card';card.innerHTML=`<h3>${{e.event_id}} | ${{e.event_type}}</h3><p>${{e.date_from}} → ${{e.date_to}} | parents=${{e.parents}} | children=${{e.children}} | coverage=${{e.coverage.toFixed(3)}}</p>`;
 const controls=document.createElement('div');
 for(const opt of lineageOptions){{const b=document.createElement('button');b.textContent=opt;b.onclick=()=>{{labels[e.event_id]=labels[e.event_id]||{{}};labels[e.event_id].lineage_action=opt;save();renderE()}};controls.appendChild(b)}}
 const notes=document.createElement('textarea');notes.placeholder='lineage notes / corrected parent-child IDs';notes.value=(labels[e.event_id]||{{}}).notes||'';notes.oninput=()=>{{labels[e.event_id]=labels[e.event_id]||{{}};labels[e.event_id].notes=notes.value;save()}};
 card.appendChild(controls);card.appendChild(notes);lineageRoot.appendChild(card);
 function renderE(){{for(const b of controls.children)b.classList.toggle('active',b.textContent===((labels[e.event_id]||{{}}).lineage_action||''))}} renderE();
}}
function esc(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}
function exportCSV(){{
 let out=['row_type,record_id,track_id,event_id,reviewed_state,lineage_action,date_quality_json,notes,reviewer,review_time,review_version'];
 const now=new Date().toISOString(); const reviewer=document.getElementById('reviewer').value.trim();
 for(const r of rows){{const x=labels[r.track_id]||{{}};out.push(['track',r.track_id,r.track_id,'',x.state||r.auto_state,'',JSON.stringify(x.date_quality||{{}}),x.notes||'',reviewer,now,'0.4.0'].map(esc).join(','))}}
 for(const e of lineageRows){{const x=labels[e.event_id]||{{}};out.push(['lineage',e.event_id,'',e.event_id,'',x.lineage_action||'', '',x.notes||'',reviewer,now,'0.4.0'].map(esc).join(','))}}
 document.getElementById('csv').value=out.join('\\n')
}}
</script>"""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def apply_reviews(
    summary_path: str | Path,
    review_csv: str | Path,
    output_path: str | Path,
    *,
    lineage_path: str | Path | None = None,
    reviewer: str = "",
    review_version: str = "0.4.0",
    confidence_weights: dict[str, float] | None = None,
) -> Path:
    summary = gpd.read_file(summary_path)
    reviews = pd.read_csv(review_csv, dtype=str, encoding="utf-8-sig").fillna("")
    if "row_type" not in reviews.columns:
        # Backward compatibility with v1 review CSV.
        track_reviews = reviews.rename(columns={"track_id": "record_id"}).copy()
        track_reviews["track_id"] = track_reviews["record_id"]
    else:
        track_reviews = reviews[reviews["row_type"] == "track"].copy()
    if "track_id" not in track_reviews.columns:
        track_reviews["track_id"] = track_reviews.get("record_id", "")
    keep_cols = [c for c in ["track_id", "reviewed_state", "date_quality_json", "notes", "reviewer", "review_time", "review_version"] if c in track_reviews.columns]
    merged = summary.merge(track_reviews[keep_cols], on="track_id", how="left")
    reviewed = merged.get("reviewed_state", pd.Series("", index=merged.index)).fillna("")
    merged["state_final"] = reviewed.where(reviewed != "", merged["state"])
    merged["review_status"] = np.where(reviewed != "", "reviewed", merged["review_status"])
    merged["reviewer"] = merged.get("reviewer", reviewer).fillna(reviewer) if isinstance(merged.get("reviewer"), pd.Series) else reviewer
    merged["review_time"] = merged.get("review_time", "").fillna("") if isinstance(merged.get("review_time"), pd.Series) else ""
    merged.loc[(reviewed != "") & merged["review_time"].eq(""), "review_time"] = datetime.now(timezone.utc).isoformat()
    merged["review_version"] = merged.get("review_version", review_version).fillna(review_version) if isinstance(merged.get("review_version"), pd.Series) else review_version
    if "confidence_components" in merged.columns:
        final_scores = []
        for _, row in merged.iterrows():
            if str(row.get("state_final", "")).lower() == "rejected":
                final_scores.append(0.0)
                continue
            try:
                components = json.loads(str(row.get("confidence_components", "{}")) or "{}")
            except json.JSONDecodeError:
                components = {}
            components["manual_review"] = 1.0 if str(row.get("reviewed_state", "")) else components.get("manual_review", 0.5)
            final_scores.append(compute_confidence_score(components, confidence_weights))
        merged["confidence_score_final"] = final_scores
        merged["confidence_level_final"] = np.where(
            merged["confidence_score_final"] >= 0.75,
            "high",
            np.where(merged["confidence_score_final"] >= 0.50, "medium", "low"),
        )
    merged["schema_version"] = "0.4.0"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output, driver="GPKG")

    quality_rows: list[dict[str, str]] = []
    if "date_quality_json" in track_reviews.columns:
        for _, row in track_reviews.iterrows():
            try:
                values = json.loads(row.get("date_quality_json", "") or "{}")
            except json.JSONDecodeError:
                values = {}
            for date_value, quality in values.items():
                quality_rows.append({"track_id": str(row["track_id"]), "date": str(date_value), "reviewed_quality": str(quality)})
    pd.DataFrame(quality_rows).to_csv(output.with_name(output.stem + "_date_quality.csv"), index=False, encoding="utf-8-sig")

    if lineage_path and Path(lineage_path).exists() and "row_type" in reviews.columns:
        lineage = gpd.read_file(lineage_path)
        lineage_columns = [
            column for column in ["event_id", "lineage_action", "notes", "reviewer", "review_time", "review_version"]
            if column in reviews.columns
        ]
        lineage_reviews = reviews[reviews["row_type"] == "lineage"][lineage_columns].copy()
        lineage_out = lineage.merge(lineage_reviews, on="event_id", how="left")
        lineage_out["review_status"] = np.where(lineage_out["lineage_action"].fillna("") != "", "reviewed", lineage_out["review_status"])
        lineage_out["review_version"] = review_version
        lineage_out["schema_version"] = "0.4.0"
        lineage_out.to_file(output.with_name(output.stem + "_lineage.gpkg"), driver="GPKG")
    return output
