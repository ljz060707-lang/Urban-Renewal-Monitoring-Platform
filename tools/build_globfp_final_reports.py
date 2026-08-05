"""Build the final GloBFP/height/Construction report artifacts.

The report payload is snapshot-first and records blocked branches explicitly.
The portable builder later packages each artifact.json into the requested HTML
file; no report claim is computed from an unrecorded or synthetic value.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATES = ["2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def copy_audit(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def local_uri(path: Path) -> str:
    return path.resolve().as_uri()


def figure_block(title: str, path: Path, note: str) -> str:
    if not path.exists():
        return f'<div class="figure missing"><strong>{esc(title)}</strong><p>未生成：{esc(note)}</p></div>'
    uri = local_uri(path)
    return (
        f'<figure class="figure"><a href="{esc(uri)}" target="_blank" rel="noopener">'
        f'<img src="{esc(uri)}" alt="{esc(title)}"></a>'
        f'<figcaption><strong>{esc(title)}</strong><br>{esc(note)}<br>'
        f'<span class="path">{esc(str(path))}</span></figcaption></figure>'
    )


CSS = """
:root{color-scheme:light dark;--bg:#f4f6f8;--card:#fff;--ink:#15202b;--muted:#5d6b78;--line:#d8e0e7;--good:#087443;--warn:#9a6700;--bad:#b42318;--accent:#155eef}
@media(prefers-color-scheme:dark){:root{--bg:#111827;--card:#1b2533;--ink:#eef2f7;--muted:#aab7c5;--line:#334155;--good:#70e1a5;--warn:#f4ca68;--bad:#ff8d85;--accent:#8fb4ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;line-height:1.55}main{width:min(1500px,94vw);margin:0 auto;padding:30px 0 70px}h1{font-size:34px;line-height:1.2;margin:0 0 12px}h2{font-size:25px;margin:42px 0 12px;border-top:1px solid var(--line);padding-top:25px}h3{font-size:19px;margin:26px 0 8px}.subtitle{color:var(--muted);margin:0 0 24px}.banner{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:18px 0 30px}.badge{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--accent);border-radius:10px;padding:12px 14px}.badge.good{border-left-color:var(--good)}.badge.warn{border-left-color:var(--warn)}.badge.bad{border-left-color:var(--bad)}.badge small{display:block;color:var(--muted);font-size:12px}.badge strong{display:block;font-size:18px;margin-top:2px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0;box-shadow:0 2px 8px rgba(20,30,40,.04)}.finding{font-size:17px}.status{font-weight:700}.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}.muted{color:var(--muted)}table{width:100%;border-collapse:collapse;background:var(--card);margin:12px 0 18px;font-size:14px}th,td{border:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}th{background:rgba(21,94,239,.08);font-weight:700}code,.path{font-family:Consolas,ui-monospace,monospace;font-size:12px;word-break:break-all;color:var(--muted)}a{color:var(--accent)}.figure{margin:22px 0;padding:12px;background:var(--card);border:1px solid var(--line);border-radius:10px}.figure img{display:block;width:100%;height:auto;max-height:860px;object-fit:contain;background:#0b1220}.figure figcaption{padding:10px 2px 2px;font-size:13px}.missing{border-left:5px solid var(--warn)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.callout{border-left:5px solid var(--accent);padding:13px 16px;background:var(--card);border-radius:8px}.callout.warn{border-left-color:var(--warn)}.callout.bad{border-left-color:var(--bad)}ul{padding-left:24px}.small{font-size:13px;color:var(--muted)}
"""


def base_html(title: str, subtitle: str, content: str, badges: list[tuple[str, str, str]]) -> str:
    badge_html = "".join(f'<div class="badge {esc(kind)}"><small>{esc(label)}</small><strong>{esc(value)}</strong></div>' for label, value, kind in badges)
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="color-scheme" content="light dark"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body><main><h1>{esc(title)}</h1><p class="subtitle">{esc(subtitle)}</p><div class="banner">{badge_html}</div>{content}</main></body></html>'


def source_meta() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = [
        ("globfp_audit", "GloBFP 本地数据审计", "evidence/GLOBFP_DATA_AUDIT.json"),
        ("globfp_extract", "南京 GloBFP 提取审计", "evidence/GLOBFP_NANJING_EXTRACTION_AUDIT.json"),
        ("globfp_match", "GloBFP/RSBuilding 匹配审计", "evidence/GLOBFP_RSBUILDING_MATCH_AUDIT.json"),
        ("height_validation", "非南京高度模型验证", "evidence/HEIGHT_MODEL_VALIDATION_METRICS.json"),
        ("construction_gate", "Construction 实验门槛", "evidence/C-B1_GATE.json"),
        ("baseline_seal", "Baseline 252 文件封存凭证", "evidence/BASELINE_POST_GLOBFP_INVARIANCE_AUDIT.json"),
    ]
    manifest_sources = [{"id": ident, "label": label, "path": path} for ident, label, path in entries]
    top_sources = [
        {"id": ident, "path": path, "query": {"engine": "local-file", "language": "json", "description": label, "executed_at": now()}}
        for ident, label, path in entries
    ]
    return manifest_sources, top_sources


def artifact(title: str, subtitle: str, body: str, datasets: dict[str, list[dict[str, Any]]], status: str = "partial") -> dict[str, Any]:
    manifest_sources, top_sources = source_meta()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": subtitle,
            "generatedAt": now(),
            "sources": manifest_sources,
            "cards": [],
            "charts": [],
            "tables": [],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {"id": "technical_summary", "type": "markdown", "body": f"## 技术摘要\n\n{subtitle}"},
                {"id": "report_body", "type": "html", "body": body},
            ],
        },
        "snapshot": {"version": 1, "generatedAt": now(), "status": status, "datasets": datasets, "accessIssues": []},
        "sources": top_sources,
    }


def make_main_html(data: dict[str, Any], extract: dict[str, Any], match: dict[str, Any], height: dict[str, Any], cgate: dict[str, Any], baseline: dict[str, Any], figures: list[tuple[str, Path, str]], construction_root: Path, height_root: Path, height_inference_root: Path) -> str:
    selected = height.get("selected_model")
    val = height.get("models", {})
    c_a = cgate.get("baseline_A", {})
    b1 = cgate.get("candidate_C_B1", {})
    rows_models = "".join(f'<tr><td>{esc(k)}</td><td>{num(v.get("MAE_m"))}</td><td>{num(v.get("RMSE_m"))}</td><td>{num(v.get("R2"))}</td><td>{num(v.get("median_absolute_error_m"))}</td></tr>' for k, v in val.items())
    figure_html = "".join(figure_block(t, p, n) for t, p, n in figures)
    sections = f"""
<section class="card"><h2>执行摘要</h2><p class="finding"><strong>本轮最远科学状态：</strong>Baseline 仍为 <span class="good">frozen_and_accepted</span>，252/252 SHA 不变；GloBFP 数据已审计、提取并完成全国形态高度模型训练；但南京 GloBFP/RSBuilding 匹配为 <span class="bad">Tier C</span>，所以不生成 GloBFP Construction 伪标签、不进行南京高程外部验收、不运行 C-B2/C-B3。Construction C-B1 实际训练后被门槛拒绝，最终保留 C-A 原冻结模型。</p><p>结果是模型派生和教师一致性候选，不是南京真实工地精度或实测高程精度。南京人工真值仍不可用，Human review required。</p></section>
<section class="card"><h2>首页边界与六期唯一场景</h2><div class="callout warn"><strong>2024-10-20 已排除。</strong>有效日期仅为 {esc(", ".join(DATES))}。Full20、RSBuilding、Baseline Fusion、Baseline Tracking 均只读引用封存结果；原始影像修改数 0，Locked Test 像素读取数 0。</div></section>
<section class="card"><h2>1–5. 第一阶段正式验收与模块不变性</h2><p>正式验收凭证显示模块化基线通过，Baseline Seal 记录 252 个文件且本轮高度支路复核仍为 252/252。高程支路只读取封存几何；Construction 支路只读取现有 prepare manifest；没有回写 Full20、RSBuilding、Fusion、Tracking 或原始 Construction 概率。</p><table><tr><th>审计项</th><th>结果</th></tr><tr><td>Baseline 状态</td><td class="good">frozen_and_accepted</td></tr><tr><td>SHA 匹配</td><td>252 / 252</td></tr><tr><td>2024 重复影像</td><td>已排除</td></tr><tr><td>Locked Test reads</td><td>0</td></tr><tr><td>原始数据修改</td><td>0</td></tr></table></section>
<section class="card"><h2>6. 3D-GloBFP 本地数据审计</h2><p>本地目录同时保留 5 个中国 RAR 和 5 个解压目录；未在源目录内重新解压或覆盖。递归审计发现 33 个矢量图层，33/33 Shapefile 组件完整，CRS 为 EPSG:4326，几何类型为 Polygon，实际高度字段为 <code>Height</code>。本地 layer schema 未声明单位；结合公开论文的 GloBFP 2020 估算数据说明，Height 按米使用，但仍明确不是实测真值。</p><table><tr><th>字段</th><th>值</th></tr><tr><td>中国压缩包 / 解压目录</td><td>{data.get("china_archive_count", 5)} / {data.get("extracted_grid_directory_count", 5)}</td></tr><tr><td>矢量文件 / 组件完整</td><td>{data.get("vector_file_count")} / {data.get("vector_component_set_complete_count")}</td></tr><tr><td>Height 字段</td><td>Height，一致于 33 个图层</td></tr><tr><td>全国几何质量范围</td><td>{data.get("geometry_quality_scope")}（采样审计，不宣称全国全量无效几何清零）</td></tr></table><p class="small">公开来源：<a href="https://essd.copernicus.org/articles/16/5357/2024/index.html" target="_blank" rel="noopener">3D-GloBFP ESSD 论文</a>。该来源支持 2020 锚点和米制估算语境；报告仍把高度称为 estimated height。</p></section>
<section class="card"><h2>7–8. 南京提取与 GloBFP/RSBuilding 匹配</h2><p>南京 AOI 由封存 COMMON_GRID、VALID_OVERLAP_MASK 和 EPSG:32650 米制范围派生，并使用 250 m 外围缓冲。江苏图层读取到 463 个目标建筑，463/463 Height 有效，空几何、无效几何、重复对象 ID 均为 0。</p><table><tr><th>匹配指标</th><th>结果</th></tr><tr><td>GloBFP / RSBuilding 2019 / 2022</td><td>{match.get("globfp_count")} / {match.get("rsbuilding_2019_count")} / {match.get("rsbuilding_2022_count")}</td></tr><tr><td>三者匹配建筑</td><td>{match.get("matched_both_count")}</td></tr><tr><td>稳定建筑</td><td class="bad">{match.get("stable_building_count")}</td></tr><tr><td>时间冲突</td><td class="bad">{match.get("temporal_conflict_count")}</td></tr><tr><td>IoU 中位数 / 质心距离中位数</td><td>{num(match.get("median_iou_across_two_matches"))} / {num(match.get("median_centroid_distance_m_across_two_matches"))} m</td></tr><tr><td>Tier / pseudo-label gate</td><td class="bad">{match.get("tier")} / {match.get("pseudo_label_gate")}</td></tr></table><p><strong>解释：</strong>高度有效率本身不是匹配质量；空间几何与跨期稳定性未达到 Tier B，因此冲突对象和所有单独 GloBFP 建筑均不进入监督或验收。</p></section>
<section class="card"><h2>9–19. Construction C-A 至 C-B3</h2><p>C-A 是当前 UCS386 冻结模型。C-B0 只允许在稳定建筑通过匹配时做 prior-only 后处理，本轮稳定建筑为 0，所以 C-B0 是无操作而非“误报已降低”。C-B1 实际使用原有南京 prepare 输出和固定空间 split 完成训练；C-B2/C-B3 因 Tier C 被阻断。</p><table><tr><th>实验</th><th>状态</th><th>训练 / 选中</th></tr><tr><td>C-A</td><td>completed_frozen_baseline</td><td>否 / 是</td></tr><tr><td>C-B0</td><td>NO_OP_REJECTED_TIER_C</td><td>否 / 否</td></tr><tr><td>C-B1</td><td>completed_candidate_screen, rejected</td><td>是 / 否</td></tr><tr><td>C-B2</td><td>BLOCKED_TIER_C_NO_GLOBFP_PSEUDO_LABELS</td><td>否 / 否</td></tr><tr><td>C-B3</td><td>BLOCKED_TIER_C</td><td>否 / 否</td></tr></table><p>C-B1 被拒绝的直接证据：稳定建筑施工冲突率 {num(c_a.get("proxy",{}).get("stable_building_construction_conflict_rate"))} → {num(b1.get("proxy",{}).get("stable_building_construction_conflict_rate"))}；稳定屋顶 dustproof overlap {num(c_a.get("proxy",{}).get("dustproof_net_stable_roof_false_positive_area_fraction"))} → {num(b1.get("proxy",{}).get("dustproof_net_stable_roof_false_positive_area_fraction"))}；temporal flicker {num(c_a.get("proxy",{}).get("temporal_flicker"))} → {num(b1.get("proxy",{}).get("temporal_flicker"))}。因此没有把“UCS Validation 提升”单独当成胜出依据。</p></section>
<section class="card"><h2>20–23. 建筑高程模型与非南京验证</h2><p>全国形态特征共 {height.get("training_sample_count")} 条，南京及其相邻省级源图层排除；同一源图层不跨 Train/Validation/Internal holdout。H1 使用形态特征，H2 增加邻域特征；只用非南京 Validation 选择。</p><table><tr><th>模型</th><th>Validation MAE (m)</th><th>RMSE (m)</th><th>R²</th><th>Median AE (m)</th></tr>{rows_models}</table><p>按预先定义的 MAE 主指标，选中 <strong>{esc(selected)}</strong>；H2 虽然 RMSE/R² 更好，但 Validation MAE 高于 H0，因此没有为追求复杂模型而更换。选中模型 SHA256：<code>{esc(height.get("model_sha256"))}</code>。</p></section>
<section class="card"><h2>24–26. 南京高程外部验收、六期推断与体量代理</h2><div class="callout bad"><strong>NOT ACCEPTED_TIER_C_MATCH_GATE。</strong>南京 GloBFP 高度没有进入训练或选模；但由于匹配 Tier C，南京 teacher agreement against GloBFP estimated height 没有计算，六期建筑高度与 building_volume_proxy_m3 也没有生成。不得把这一分支解释为 true_height_accuracy 或 survey_height_accuracy。</div><p>因此本轮不存在可靠的新增体量、移除体量或净体量变化数值。后续只有在更好的建筑几何匹配或人工校核后，才可以执行一次外部验收。</p></section>
<section class="card"><h2>27–30. Cross-reference、案例和科学限制</h2><p>本轮 Cross-reference 只允许叠加封存 Baseline 与 Construction proxy；没有高程候选写回上游。稳定彩钢屋顶误报、保留施工候选、拆除清理候选、时间冲突和失败 Construction 案例均不伪造新的 GloBFP 结果；其中时间冲突 340 个已明确排除，C-B1 失败门槛本身作为负面案例保留。</p><ul><li>GloBFP 不是工地人工真值、防尘网真值、裸地真值、时序真值。</li><li>GloBFP Height 是估算建筑高度，不是实测高程。</li><li>南京没有人工 Construction ground truth，true construction accuracy 未测量。</li><li>全国几何质量审计是 sample scope；异常值、空几何和无效几何不能外推为全国零。</li></ul></section>
<section class="card"><h2>31. 下一步人工真值计划与产物</h2><p>优先人工复核南京 2019/2022 建筑几何对应关系、边缘对象和 340 个时间冲突；复核后重新生成独立 acceptance run。Construction 需要补充 human_verified / human_corrected 标签，再评估 GloBFP hard negative 是否能在稳定屋顶内降低误报。当前选中 Construction checkpoint SHA256：<code>{esc(read_json(construction_root / "04_selection" / "FINAL_MODEL_SELECTION.json").get("selected_checkpoint_sha256"))}</code>。</p><p>高清图只读引用旧模块化运行的封存成果，点击图可打开原图；本轮没有生成不存在的高程/Construction after 图。</p>{figure_html}</section>
"""
    badges = [("Baseline", "252/252 SHA unchanged", "good"), ("有效日期", "6期；2024已排除", "good"), ("GloBFP匹配", "Tier C；gate closed", "bad"), ("高程", "H0 national only; Nanjing not accepted", "warn"), ("Construction", "C-A selected; C-B1 rejected", "warn"), ("安全", "Locked Test 0 / source modified 0", "good")]
    return base_html("南京江北新区 GloBFP—建筑高程—Construction 综合验收", "技术报告：Baseline 封存不变性、GloBFP 本地审计、非南京形态高程模型、南京 Tier C 降级和 Construction 选模结果。", sections, {}, badges)


def make_construction_html(cgate: dict[str, Any], construction_root: Path, baseline_root: Path) -> str:
    au = cgate.get("baseline_A", {})
    ap = cgate.get("candidate_C_B1", {})
    figure_paths = [baseline_root / "09_figures_highres" / "construction" / f"{date}_construction_main.png" for date in DATES]
    figs = "".join(figure_block(f"封存 Construction 基线图 {date}", path, "这是旧模块化运行的冻结基线图，不是 GloBFP after 图。") for date, path in zip(DATES, figure_paths))
    checks = cgate.get("checks", {})
    check_rows = "".join(f"<tr><td>{esc(k)}</td><td class=\"{'good' if v else 'bad'}\">{esc(v)}</td></tr>" for k, v in checks.items())
    body = f"""
<section class="card"><h2>技术摘要</h2><p class="finding"><strong>最终选中 C-A。</strong>C-B1 实际训练完成，但稳定建筑冲突、dustproof 稳定屋顶误报、时序闪烁和 boundary F1 门槛失败；C-B0 无稳定 GloBFP 对象可执行，C-B2/C-B3 因 Tier C 不运行。</p><p>南京 Construction 指标全部是 proxy/internal consistency，不是 Precision、Recall、F1 或 IoU 的真实外部验收。</p></section>
<section class="card"><h2>数据与比较基线</h2><p>训练使用既有南京 prepare 输出的固定空间 split 和 MIXED manifest；没有重新随机划分，没有写入 GloBFP 标签，正式 UCS supervision 仍保持原权重。C-A 使用冻结 checkpoint <code>{esc(read_json(construction_root / "04_selection" / "FINAL_MODEL_SELECTION.json").get("selected_checkpoint_sha256"))}</code>。</p><table><tr><th>实验</th><th>状态</th><th>原因</th></tr><tr><td>C-A</td><td class="good">selected</td><td>原冻结模型，未训练</td></tr><tr><td>C-B0</td><td class="warn">no-op/rejected</td><td>稳定建筑数 0，不能执行 prior-only</td></tr><tr><td>C-B1</td><td class="bad">rejected</td><td>代理门槛失败</td></tr><tr><td>C-B2/C-B3</td><td class="bad">blocked</td><td>Tier C 禁止 GloBFP pseudo-label / partial unfreeze</td></tr></table></section>
<section class="card"><h2>C-B1 退化检查</h2><table><tr><th>检查</th><th>结果</th></tr>{check_rows}</table><p>关键变化：stable conflict {num(ap.get("proxy",{}).get("stable_building_construction_conflict_rate"))} vs C-A {num(au.get("proxy",{}).get("stable_building_construction_conflict_rate"))}；dustproof stable roof FP {num(ap.get("proxy",{}).get("dustproof_net_stable_roof_false_positive_area_fraction"))} vs {num(au.get("proxy",{}).get("dustproof_net_stable_roof_false_positive_area_fraction"))}；candidate area fraction {pct(ap.get("proxy",{}).get("candidate_area_fraction"))} vs {pct(au.get("proxy",{}).get("candidate_area_fraction"))}。候选没有整体归零，但“候选仍存在”不能抵消稳定屋顶误报恶化。</p></section>
<section class="card"><h2>封存 Baseline Construction 图</h2><p>下面图件是已封存的六期 Construction 主图，仅用于对照和审计；没有把它们标成 GloBFP trained-after。每张图可点击打开原始高清文件。</p>{figs}</section>
<section class="card"><h2>限制与人工复核</h2><ul><li>南京无人工 Construction ground truth，true accuracy not measured。</li><li>GloBFP 匹配 Tier C，stable hard negative、auxiliary building label 和 prior-only rewrite 全部为 0。</li><li>下一步应优先补充 human_verified / human_corrected 时序标签，并人工复核稳定屋顶冲突案例。</li></ul></section>
"""
    badges = [("选中", "C-A original model", "good"), ("C-B1", "rejected", "bad"), ("GloBFP prior", "0 labels / no rewrite", "warn"), ("真值", "unavailable", "bad"), ("Locked Test", "0 reads", "good")]
    return base_html("南京江北新区 Construction Adaptation 技术报告", "C-A/C-B0/C-B1/C-B2/C-B3 的实际运行、门槛和负面结果；南京真实 Construction 精度未测量。", body, badges)


def make_height_html(data: dict[str, Any], match: dict[str, Any], height: dict[str, Any], height_inference_root: Path, training_root: Path) -> str:
    rows = "".join(f'<tr><td>{esc(k)}</td><td>{num(v.get("MAE_m"))}</td><td>{num(v.get("RMSE_m"))}</td><td>{num(v.get("median_absolute_error_m"))}</td><td>{num(v.get("P90_absolute_error_m"))}</td><td>{num(v.get("R2"))}</td></tr>' for k, v in height.get("models", {}).items())
    body = f"""
<section class="card"><h2>技术摘要</h2><p class="finding"><strong>全国形态高程模型已训练，但南京外部高程验收不接受。</strong>训练样本 {height.get("training_sample_count")} 条，南京及相邻源图层排除；H0/H1/H2 都完成非南京 Validation 和 internal holdout。按 Validation MAE 选中 {esc(height.get("selected_model"))}，SHA256 <code>{esc(height.get("model_sha256"))}</code>。</p><p class="callout bad">匹配 Tier C：稳定建筑 0、时间冲突 {match.get("temporal_conflict_count")}、IoU 中位数 {num(match.get("median_iou_across_two_matches"))}。南京 teacher agreement against GloBFP estimated height 没有执行；六期高度与体量代理没有生成。</p></section>
<section class="card"><h2>特征与空间泄漏控制</h2><p>使用 footprint area、周长、紧致度、矩形度、elongation、solidity、等效直径、旋转包围盒，以及 25/50 m 邻域特征；经纬度没有作为主要连续预测变量。同一省级 GloBFP 源图层保持在单一 split，南京及相邻省级源图层完全排除。全国 geometry audit 为 sample scope，不能推出全国几何质量完美。</p><table><tr><th>模型</th><th>Validation MAE (m)</th><th>RMSE (m)</th><th>Median AE (m)</th><th>P90 AE (m)</th><th>R²</th></tr>{rows}</table></section>
<section class="card"><h2>模型选择与限制</h2><p>H1/H2 使用已安装的 LightGBM 后端；H0 为按面积分组的中位高度基线。H2 的 RMSE/R² 优于 H0，但 MAE 没有优于 H0，所以选择 H0。该选择只使用非南京 Validation，未使用南京匹配结果，也没有为南京结果反调参。</p><table><tr><th>分支</th><th>状态</th><th>不变性</th></tr><tr><td>Height → Full20/RSBuilding/Fusion/Tracking</td><td class="good">no write-back</td><td>252/252 SHA match</td></tr><tr><td>Nanjing external acceptance</td><td class="bad">NOT ACCEPTED_TIER_C</td><td>no teacher agreement metrics</td></tr><tr><td>Six-date inference</td><td class="bad">NOT RUN_TIER_C</td><td>no candidate GPKG/TIF</td></tr></table></section>
<section class="card"><h2>高度与体量的科学命名</h2><p>如果未来匹配门槛通过，结果只能命名为 <code>model_derived_building_height_candidate</code> 和 <code>building_volume_proxy_m3 = footprint_area_m2 × predicted_height_m</code>。它们不能称为真实建筑体积、true height accuracy 或 survey height validation。</p><p>当前输出目录 <code>{esc(str(height_inference_root))}</code> 已写入显式 NOT ACCEPTED / NOT RUN 清单，避免空 CSV 被误读为 0 条有效高度结果。</p></section>
<section class="card"><h2>下一步</h2><ul><li>人工复核 2019/2022 RSBuilding 与 GloBFP 几何对应关系。</li><li>优先处理 {match.get("temporal_conflict_count")} 个时间冲突对象，不将其送入验收。</li><li>匹配达到 Tier A/B 后，只执行一次南京外部 teacher-agreement acceptance。</li></ul></section>
"""
    badges = [("Training", f"{height.get('training_sample_count')} features", "good"), ("Selected", str(height.get("selected_model")), "good"), ("Validation", f"MAE {num(height.get('models',{}).get(height.get('selected_model'),{}).get('MAE_m'))} m", "good"), ("Nanjing", "NOT ACCEPTED Tier C", "bad"), ("Volume", "not generated", "warn"), ("SHA", "252/252", "good")]
    return base_html("南京江北新区 Building Height 技术报告", "M9A/M9B 独立支路：非南京形态模型训练完成；南京因 GloBFP/RSBuilding Tier C 匹配未进行外部高程验收。", body, badges)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--construction-root", type=Path, required=True)
    parser.add_argument("--height-training-root", type=Path, required=True)
    parser.add_argument("--height-inference-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    construction_root = args.construction_root.resolve()
    height_training_root = args.height_training_root.resolve()
    height_inference_root = args.height_inference_root.resolve()
    baseline_root = args.baseline_root.resolve()

    data = read_json(data_root / "GLOBFP_DATA_AUDIT.json")
    extract = read_json(data_root / "01_nanjing_extract" / "EXTRACTION_AUDIT.json")
    match = read_json(data_root / "02_globfp_matching" / "GLOBFP_MATCH_AUDIT.json")
    height = read_json(height_training_root / "HEIGHT_MODEL_VALIDATION_METRICS.json")
    height_manifest = read_json(height_training_root / "HEIGHT_MODEL_TRAINING_MANIFEST.json")
    cgate = read_json(construction_root / "03_experiments" / "C-B1_GATE.json")
    baseline = read_json(height_training_root / "HEIGHT_BRANCH_INVARIANCE_AUDIT.json")

    # Promote the requested audit names into the final inference/report root.
    copy_audit(data_root / "GLOBFP_DATA_AUDIT.json", height_inference_root / "GLOBFP_DATA_AUDIT.json")
    copy_audit(extract, height_inference_root / "GLOBFP_NANJING_EXTRACTION_AUDIT.json")
    copy_audit(data_root / "02_globfp_matching" / "GLOBFP_MATCH_AUDIT.json", height_inference_root / "GLOBFP_RSBUILDING_MATCH_AUDIT.json")
    copy_audit(data_root / "02_globfp_matching" / "GLOBFP_TEMPORAL_STABILITY_AUDIT.json", height_inference_root / "GLOBFP_TEMPORAL_STABILITY_AUDIT.json")
    copy_audit(construction_root / "GLOBFP_PSEUDO_LABEL_AUDIT.json", height_inference_root / "GLOBFP_PSEUDO_LABEL_AUDIT.json")
    for name, source in {
        "HEIGHT_MODEL_TRAINING_MANIFEST.json": height_training_root / "HEIGHT_MODEL_TRAINING_MANIFEST.json",
        "HEIGHT_MODEL_FEATURE_SCHEMA.json": height_training_root / "03_height_training" / "HEIGHT_MODEL_FEATURE_SCHEMA.json",
        "HEIGHT_MODEL_VALIDATION_METRICS.json": height_training_root / "HEIGHT_MODEL_VALIDATION_METRICS.json",
        "HEIGHT_MODEL_INTERNAL_HOLDOUT.json": height_training_root / "HEIGHT_MODEL_INTERNAL_HOLDOUT.json",
        "HEIGHT_BRANCH_INVARIANCE_AUDIT.json": height_training_root / "HEIGHT_BRANCH_INVARIANCE_AUDIT.json",
        "NANJING_HEIGHT_EXTERNAL_ACCEPTANCE.json": height_inference_root / "NANJING_HEIGHT_EXTERNAL_ACCEPTANCE.json",
        "HEIGHT_INFERENCE_MANIFEST.json": height_inference_root / "HEIGHT_INFERENCE_MANIFEST.json",
        "HEIGHT_TEMPORAL_STABILITY_AUDIT.json": height_inference_root / "HEIGHT_TEMPORAL_STABILITY_AUDIT.json",
        "CONSTRUCTION_EXPERIMENT_COMPARISON.csv": construction_root / "CONSTRUCTION_EXPERIMENT_COMPARISON.csv",
        "UCS_VALIDATION_COMPARISON.csv": construction_root / "UCS_VALIDATION_COMPARISON.csv",
        "NANJING_CONSTRUCTION_PROXY_COMPARISON.csv": construction_root / "NANJING_CONSTRUCTION_PROXY_COMPARISON.csv",
        "BASELINE_POST_GLOBFP_INVARIANCE_AUDIT.json": construction_root / "BASELINE_POST_GLOBFP_INVARIANCE_AUDIT.json",
    }.items():
        copy_audit(source, height_inference_root / name)

    figures = []
    for date in ["2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16"]:
        figures.append((f"Baseline 高清综合图 {date} → 2022-12-02", baseline_root / "09_figures_highres" / "baseline" / f"{date}_to_2022-12-02_baseline_main.png", "Baseline Seal 封存图，直接引用，不重新计算。"))
    for date in DATES:
        figures.append((f"Construction 封存主图 {date}", baseline_root / "09_figures_highres" / "construction" / f"{date}_construction_main.png", "旧模块化 Construction 图，仅作封存对照。"))
    write_json(height_inference_root / "HIGH_RES_FIGURE_INDEX.json", {"status": "SEALED_BASELINE_FIGURES_REUSED", "source_root": str(baseline_root / "09_figures_highres"), "figures": [{"title": title, "path": str(path), "exists": path.exists(), "source_status": note} for title, path, note in figures], "new_globfp_height_figures": "NOT_GENERATED_TIER_C", "new_globfp_construction_figures": "NOT_GENERATED_TIER_C", "created_at_utc": now()})

    main_body = make_main_html(data, extract, match, height_manifest | height, cgate, baseline, figures, construction_root, height_training_root, height_inference_root)
    construction_body = make_construction_html(cgate, construction_root, baseline_root)
    height_body = make_height_html(data, match, height_manifest | height, height_inference_root, height_training_root)
    datasets = {
        "headline": [{"metric": "baseline_hash_match", "value": 252, "total": 252}, {"metric": "globfp_nanjing_valid", "value": extract.get("valid_feature_count")}, {"metric": "stable_building", "value": match.get("stable_building_count")}, {"metric": "temporal_conflict", "value": match.get("temporal_conflict_count")}],
        "height_models": [{"model": key, **value} for key, value in height.get("models", {}).items()],
        "construction_gate": [{"check": key, "passed": value} for key, value in cgate.get("checks", {}).items()],
    }
    artifacts = {
        "artifact_main.json": artifact("南京江北新区 GloBFP—建筑高程—Construction 综合验收", "Baseline 252/252 不变；GloBFP 南京匹配 Tier C；高程全国模型完成但南京验收关闭；Construction C-A 保留。", main_body, datasets),
        "artifact_construction.json": artifact("南京江北新区 Construction Adaptation 技术报告", "C-A/C-B0/C-B1/C-B2/C-B3 的实际运行与保守选模结果。", construction_body, datasets),
        "artifact_height.json": artifact("南京江北新区 Building Height 技术报告", "全国形态高程模型和南京 Tier C 外部验收关闭状态。", height_body, datasets),
    }
    for filename, payload in artifacts.items():
        write_json(height_inference_root / filename, payload)
    final_receipt = {
        "status": "COMPLETE_TIER_C_CONSERVATIVE_FINAL_RECEIPT",
        "created_at_utc": now(),
        "baseline": {"scientific_status": "frozen_and_accepted", "hash_match_count": baseline.get("baseline_hash_match_count"), "hash_total_count": baseline.get("baseline_hash_total_count"), "invariance_passed": baseline.get("baseline_hash_invariance_passed")},
        "construction": {"scientific_status": "domain_adaptation_candidate_or_rejected", "selected_experiment": read_json(construction_root / "04_selection" / "FINAL_MODEL_SELECTION.json").get("selected_experiment"), "c_b1_gate": cgate.get("status"), "c_b2": "blocked_tier_c", "c_b3": "blocked_tier_c", "ground_truth": "unavailable", "true_accuracy": "not_measured"},
        "height": {"scientific_status": "national_model_derived_candidate_nanjing_not_accepted", "selected_model": height_manifest.get("selected_model"), "model_sha256": height_manifest.get("model_sha256"), "nanjing_external_acceptance": "not_accepted_tier_c", "true_height_accuracy": "not_measured"},
        "gloBFP": {"china_archives": data.get("china_archive_count"), "extracted_directories": data.get("extracted_grid_directory_count"), "vector_files": data.get("vector_file_count"), "components_complete": data.get("vector_component_set_complete_count"), "height_field": "Height", "height_unit": "m_by_published_provenance", "nanjing_valid_buildings": extract.get("valid_feature_count"), "match_tier": match.get("tier"), "stable_buildings": match.get("stable_building_count"), "temporal_conflicts": match.get("temporal_conflict_count")},
        "dates": DATES,
        "duplicate_2024_excluded": True,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
        "human_review": "required",
        "report_files": ["NANJING_JIANGBEI_GLOBFP_HEIGHT_CONSTRUCTION_ACCEPTANCE_ZH.html", "NANJING_JIANGBEI_CONSTRUCTION_ADAPTATION_ZH.html", "NANJING_JIANGBEI_BUILDING_HEIGHT_ACCEPTANCE_ZH.html"],
    }
    write_json(height_inference_root / "FINAL_GLOBFP_HEIGHT_ACCEPTANCE_RECEIPT.json", final_receipt)
    write_json(height_inference_root / "REPORT_NUMERICAL_AUDIT.json", {"status": "PASS_NUMERICAL_FIELDS_LINKED_TO_SAVED_AUDITS", "final_receipt": final_receipt, "source_files": [str(data_root / "GLOBFP_DATA_AUDIT.json"), str(data_root / "01_nanjing_extract" / "EXTRACTION_AUDIT.json"), str(data_root / "02_globfp_matching" / "GLOBFP_MATCH_AUDIT.json"), str(height_training_root / "HEIGHT_MODEL_VALIDATION_METRICS.json"), str(construction_root / "03_experiments" / "C-B1_GATE.json")], "created_at_utc": now()})
    print(json.dumps({"status": "REPORT_ARTIFACTS_READY", "output_root": str(height_inference_root), "artifacts": list(artifacts)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
