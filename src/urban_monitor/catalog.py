"""Authoritative registry and artifact catalog for the urban-renewal project."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY = REPO_ROOT / "configs" / "registry" / "models.yaml"
DATASET_REGISTRY = REPO_ROOT / "configs" / "registry" / "datasets.yaml"
CATALOG_ROOT = REPO_ROOT / "outputs" / "catalog"
RUN_PATTERN = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models() -> list[dict[str, Any]]:
    payload = _read_yaml(MODEL_REGISTRY)
    allowed = set(payload.get("allowed_status", []))
    rows = list(payload.get("models", []))
    ids = [row["model_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model_id in model registry")
    for row in rows:
        if row.get("status") not in allowed:
            raise ValueError(f"Invalid model status: {row.get('model_id')}={row.get('status')}")
    return rows


def load_datasets() -> list[dict[str, Any]]:
    payload = _read_yaml(DATASET_REGISTRY)
    allowed = set(payload.get("allowed_status", []))
    rows = list(payload.get("datasets", []))
    ids = [row["dataset_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dataset_id in dataset registry")
    for row in rows:
        if row.get("status") not in allowed:
            raise ValueError(f"Invalid dataset status: {row.get('dataset_id')}={row.get('status')}")
    return rows


def _model_by_run(models: Iterable[dict[str, Any]]) -> dict[str, str]:
    return {str(row.get("run_id")): row["model_id"] for row in models if row.get("run_id")}


def _json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _epoch_count(run: Path) -> int | None:
    values: list[int] = []
    for path in run.glob("metrics/train_log*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("epoch", "")).strip():
                        values.append(int(float(row["epoch"])))
        except (OSError, ValueError):
            continue
    return max(values) + 1 if values else None


def _find_checkpoint(run: Path, manifest: dict[str, Any]) -> Path | None:
    declared = manifest.get("final_checkpoint") or manifest.get("best_checkpoint")
    if declared:
        candidate = Path(declared)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.is_file():
            return candidate
    candidates = sorted((run / "checkpoints").glob("best*.pth")) if (run / "checkpoints").is_dir() else []
    return candidates[0] if candidates else None


def discover_experiments(models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    models = models or load_models()
    run_map = _model_by_run(models)
    known_sha = {str(row.get("run_id")): row.get("checkpoint_sha256") for row in models}
    rows: list[dict[str, Any]] = []
    root = REPO_ROOT / "outputs" / "pretraining"
    for run in sorted(path for path in root.glob("*/*") if path.is_dir() and RUN_PATTERN.match(path.name)):
        manifest = _json_if_exists(run / "RUN_MANIFEST.json")
        safety = _json_if_exists(run / "TEST_SAFETY_AUDIT.json")
        final_test = _json_if_exists(run / "FINAL_TEST_MANIFEST.json")
        checkpoint = _find_checkpoint(run, manifest)
        model_id = run_map.get(run.name)
        if not model_id:
            family = run.parent.name
            model_id = {"change": "dsifn_binary_v1", "change_s2looking": "s2looking_directional_v1",
                        "change_v2": "full20_multisource_v2"}.get(family, "unregistered")
        metrics = next(iter(sorted((run / "metrics").glob("combined.json"))), None) if (run / "metrics").is_dir() else None
        config = run / "resolved_config.yaml"
        log = next(iter(sorted((run / "metrics").glob("train_log*.csv"))), None) if (run / "metrics").is_dir() else None
        test_read_count = sum(int(value or 0) for key, value in safety.items() if key.endswith("_test_image_reads"))
        test_used = bool(final_test) or bool(safety.get("test_used_for_training")) or bool(safety.get("test_used_for_validation"))
        rows.append({
            "run_id": run.name,
            "model_id": model_id,
            "stage": next(iter(manifest.get("stages", {})), run.parent.name),
            "dataset": ",".join(manifest.get("datasets", [])) if isinstance(manifest.get("datasets"), list) else run.parent.name,
            "start_time": manifest.get("created_at") or run.name,
            "epochs": _epoch_count(run),
            "checkpoint": str(checkpoint.resolve()) if checkpoint else "",
            "checkpoint_sha": known_sha.get(run.name) or "",
            "metrics": str(metrics.resolve()) if metrics else "",
            "status": manifest.get("status", "incomplete" if not manifest else "unknown"),
            "parent_model": manifest.get("source_run_id") or manifest.get("inherited_from") or "",
            "config": str(config.resolve()) if config.is_file() else "",
            "log": str(log.resolve()) if log else "",
            "test_used": test_used,
            "test_read_count": test_read_count,
            "notes": "Registered from RUN_MANIFEST.json" if manifest else "No RUN_MANIFEST.json; incomplete/legacy run",
        })
    return rows


ARTIFACT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".html", ".csv", ".json", ".md", ".tif", ".tiff", ".gpkg", ".geojson"}
SKIP_PARTS = {"checkpoints", "prepared", "normalized_labels", "ai_cache", "cache", "tensorboard"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_IMAGES_PER_DIRECTORY = 20


def _iter_artifact_files(root: Path) -> Iterable[Path]:
    """遍历产物树，并在目录层级跳过大型训练缓存/切片目录。"""
    for directory, child_dirs, filenames in os.walk(root):
        child_dirs[:] = [
            name for name in child_dirs
            if name.casefold() not in SKIP_PARTS
        ]
        parent = Path(directory)
        for filename in sorted(filenames):
            path = parent / filename
            if path.suffix.casefold() in ARTIFACT_EXTENSIONS:
                yield path


def _artifact_type(path: Path) -> str:
    name = path.name.casefold()
    suffix = path.suffix.casefold()
    if name.endswith(".html") and "review" in str(path).casefold():
        return "review_page"
    if name.endswith(".html"):
        return "dashboard"
    if "audit" in name or "manifest" in name or "protocol" in name:
        return "audit"
    if suffix in {".gpkg", ".geojson"}:
        return "vector"
    if suffix in {".tif", ".tiff"} and "prob" in name:
        return "probability_raster"
    if suffix in {".tif", ".tiff"}:
        return "mask" if "mask" in name else "map"
    if suffix in {".csv", ".json"}:
        return "metric_table"
    if "loss" in name or "curve" in name or "training" in name:
        return "training_curve"
    if "preview" in str(path).casefold():
        return "validation_preview"
    if "final_test" in str(path).casefold() or "test_report" in name:
        return "test_report"
    if "inference" in str(path).casefold() or "map" in name:
        return "external_inference"
    return "audit" if suffix == ".md" else "validation_preview"


def _infer_run_id(path: Path) -> str:
    return next((part for part in path.parts if RUN_PATTERN.match(part)), "")


def _infer_dataset(path: Path) -> str:
    value = str(path).casefold()
    for dataset in ("s2looking", "dsifn", "second", "whu", "tongzhou", "beijing"):
        if dataset in value:
            return "beijing_tongzhou" if dataset == "tongzhou" else dataset
    return "multi_source" if "change_v2" in value else "unknown"


def _infer_task(path: Path) -> str:
    name = path.name.casefold()
    if "demolition" in name or "_dem" in name:
        return "demolition"
    if "new" in name:
        return "new_building"
    if "semantic" in name:
        return "semantic_state"
    if "building" in name:
        return "building_geometry"
    if "temporal" in name or "track" in name:
        return "temporal"
    return "any_change"


def discover_artifacts(models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    models = models or load_models()
    run_map = _model_by_run(models)
    roots = [REPO_ROOT / "outputs" / name for name in ("models", "pretraining", "inference", "v2_readiness")]
    rows: list[dict[str, Any]] = []
    image_counts: dict[Path, int] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in _iter_artifact_files(root):
            relative = path.relative_to(REPO_ROOT)
            # 大型数据审计目录可能包含数万张近似预览图。每个目录保留固定数量，
            # 既保留代表性样例，也避免生成几十 MB、无法交互使用的网页。
            if path.suffix.casefold() in IMAGE_EXTENSIONS:
                count = image_counts.get(path.parent, 0)
                if count >= MAX_IMAGES_PER_DIRECTORY:
                    continue
                image_counts[path.parent] = count + 1
            run_id = _infer_run_id(path)
            dataset = _infer_dataset(path)
            is_test = "final_test" in str(path).casefold() or "post_final_test" in str(path).casefold()
            artifact_type = _artifact_type(path)
            family_model = next(
                (
                    model_id
                    for family, model_id in {
                        "change": "dsifn_binary_v1",
                        "change_s2looking": "s2looking_directional_v1",
                        "change_v2": "full20_multisource_v2",
                    }.items()
                    if family in relative.parts
                ),
                None,
            )
            rows.append({
                "model": run_map.get(
                    run_id,
                    "full20_multisource_v2"
                    if "beijing_tongzhou" in str(path).casefold()
                    else (family_model or "unregistered"),
                ),
                "run": run_id,
                "dataset": dataset,
                "stage": "test" if is_test else ("external_inference" if "inference" in relative.parts else "development"),
                "task": _infer_task(path),
                "artifact_type": artifact_type,
                "path": str(path.resolve()),
                "relative_path": relative.as_posix(),
                "ppt_usable": path.suffix.casefold() in {".png", ".jpg", ".jpeg"} and not is_test,
                "from_test": is_test,
                "access_status": "locked_diagnostic_only" if is_test else "available",
                "size_bytes": path.stat().st_size,
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gallery_html(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>城市更新可视化资产库</title>
<style>
:root{{--bg:#07111f;--panel:#0d1b2a;--line:#1f3348;--text:#e6edf5;--muted:#8ca0b3;--accent:#22d3ee;--red:#ef4444;}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#06101c,#0a1727);color:var(--text);font:14px "Segoe UI","Microsoft YaHei",sans-serif}}
header{{padding:30px clamp(20px,5vw,72px);border-bottom:1px solid var(--line)}}h1{{font-size:clamp(28px,4vw,48px);margin:0 0 8px}}.subtitle{{color:var(--muted)}}
.filters{{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:10px;padding:18px clamp(20px,5vw,72px);position:sticky;top:0;background:#07111fe8;backdrop-filter:blur(14px);z-index:2}}
select,input{{background:#0d1b2a;color:var(--text);border:1px solid var(--line);padding:10px;border-radius:8px}}main{{padding:12px clamp(20px,5vw,72px) 60px}}
.count{{color:var(--accent);margin:8px 0 16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px}}article{{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.thumb{{height:180px;background:#08131f;display:grid;place-items:center;color:var(--muted)}}.thumb img{{width:100%;height:100%;object-fit:contain}}.body{{padding:14px}}
.tags{{display:flex;gap:6px;flex-wrap:wrap}}.tag{{background:#12283b;border:1px solid #24435d;border-radius:999px;padding:3px 8px;font-size:11px}}
.locked{{color:#fca5a5}}.path{{color:var(--muted);font:11px Consolas,monospace;word-break:break-all;margin:10px 0}}button,a.btn{{border:1px solid #2a536e;background:#0d2536;color:#b9f5ff;padding:7px 10px;border-radius:7px;text-decoration:none;cursor:pointer}}
@media(max-width:900px){{.filters{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<header><h1>城市更新可视化资产库</h1><div class="subtitle">训练、验证、真实场景与审计资产的离线权威索引。Test 资产只作锁定诊断展示。</div></header>
<div class="filters"><input id="q" placeholder="搜索路径或名称"><select id="model"></select><select id="run"></select><select id="dataset"></select><select id="stage"></select><select id="type"></select></div>
<main><div id="count" class="count"></div><div id="grid" class="grid"></div></main>
<script>
const rows={payload}; const keys=["model","run","dataset","stage","artifact_type"]; const ids=["model","run","dataset","stage","type"]; const labels={{model:"模型",run:"运行",dataset:"数据集",stage:"阶段",artifact_type:"类型"}};
keys.forEach((key,i)=>{{const el=document.getElementById(ids[i]);el.innerHTML=`<option value="">全部${{labels[key]}}</option>`+[...new Set(rows.map(r=>r[key]).filter(Boolean))].sort().map(v=>`<option>${{v}}</option>`).join("")}});
function render(){{const q=document.getElementById("q").value.toLowerCase();const f=Object.fromEntries(ids.map((id,i)=>[keys[i],document.getElementById(id).value]));const matching=rows.filter(r=>(!q||r.path.toLowerCase().includes(q))&&keys.every(k=>!f[k]||r[k]===f[k]));const shown=matching.slice(0,500);document.getElementById("count").textContent=`显示 ${{shown.length}} / ${{matching.length}} 个匹配项（总计 ${{rows.length}}，单次最多500项）`;document.getElementById("grid").innerHTML=shown.map(r=>{{const href=r.relative_path.replace(/^outputs\\//,"../");const image=/\\.(png|jpe?g)$/i.test(r.path)?`<img loading="lazy" src="${{href}}">`:`${{r.artifact_type.toUpperCase()}}`;return `<article><div class="thumb">${{image}}</div><div class="body"><div class="tags"><span class="tag">${{r.model}}</span><span class="tag">${{r.dataset}}</span><span class="tag">${{r.artifact_type}}</span>${{r.from_test?'<span class="tag locked">TEST · LOCKED</span>':""}}</div><div class="path">${{r.path}}</div><a class="btn" href="${{href}}">打开文件</a> <button data-path="${{r.path.replaceAll('"','&quot;')}}">复制路径</button></div></article>`}}).join("");document.querySelectorAll("button[data-path]").forEach(b=>b.onclick=()=>navigator.clipboard.writeText(b.dataset.path))}}
document.querySelectorAll("select,input").forEach(el=>{{el.addEventListener("input",render);el.addEventListener("change",render)}});render();
</script></body></html>"""
    return template.replace("{payload}", payload).replace("{{", "{").replace("}}", "}")


def _dashboard_html(models: list[dict[str, Any]], datasets: list[dict[str, Any]], experiments: list[dict[str, Any]]) -> str:
    active = [m for m in models if m["status"] == "active"]
    archived = [m for m in models if m["status"] == "archived"]
    external = [m for m in models if m["status"] == "external_pretrained"]
    candidates = [m for m in models if m["status"] == "validation_candidate"]
    blocked = [m for m in models if m["status"] == "blocked"]
    locked = [d for d in datasets if d["test_policy"] in {"spent_test", "locked_holdout"}]
    def cards(items, kind):
        return "".join(f'<article><span>{html.escape(kind)}</span><h3>{html.escape(str(x.get("display_name") or x.get("name")))}</h3><p>{html.escape(str(x.get("status")))}</p></article>' for x in items) or "<p class=empty>暂无</p>"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>城市更新模型控制台</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#07111f;color:#ecf5ff;font:15px "Segoe UI","Microsoft YaHei",sans-serif}}header{{padding:56px 7vw 40px;background:radial-gradient(circle at 80% 20%,#123850,#07111f 55%)}}h1{{font-size:52px;max-width:900px;margin:0}}header p{{color:#97afc2;font-size:18px}}main{{padding:30px 7vw 70px}}section{{margin:28px 0}}h2{{font-size:14px;letter-spacing:.18em;color:#22d3ee}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}}article{{border:1px solid #1c3950;background:#0b1a29;padding:18px;border-radius:10px}}article span{{font-size:11px;color:#7dd3fc}}article h3{{margin:10px 0 6px}}article p,.empty{{color:#8ea4b5}}.stats{{display:flex;gap:32px;flex-wrap:wrap}}.stat b{{font-size:32px;display:block}}a{{color:#67e8f9}}</style></head><body>
<header><div class="stats"><div class="stat"><b>{len(models)}</b>模型</div><div class="stat"><b>{len(datasets)}</b>数据源</div><div class="stat"><b>{len(experiments)}</b>已索引实验</div></div><h1>城市更新多源遥感模型控制台</h1><p>从 Change Detection 到 Surface State、Geometry 与 Temporal Event Database 的可审计演进路径。</p></header><main>
<section><h2>ACTIVE MODELS</h2><div class="grid">{cards(active,"ACTIVE")}</div></section>
<section><h2>ARCHIVED TEACHERS</h2><div class="grid">{cards(archived,"ARCHIVED")}</div></section>
<section><h2>VALIDATION CANDIDATES</h2><div class="grid">{cards(candidates,"VALIDATION ONLY")}</div></section>
<section><h2>EXTERNAL PRETRAINED</h2><div class="grid">{cards(external,"NOT INFERENCE READY")}</div></section>
<section><h2>BLOCKED MODELS</h2><div class="grid">{cards(blocked,"BLOCKED")}</div></section>
<section><h2>LOCKED TEST SETS</h2><div class="grid">{cards(locked,"TEST SAFETY")}</div></section>
<section><h2>CURRENT REAL-SCENE OUTPUTS</h2><div class="grid"><article><span>EXTERNAL INFERENCE</span><h3>北京通州多期变化</h3><p>3 组到 2023 年的 candidate building-change objects；并非确认建筑统计。</p></article></div></section>
<section><h2>PENDING TASKS</h2><div class="grid"><article><h3>WHU / RSBuilding</h3><p>修复 CUDA NVRTC 后才能运行 geometry smoke。</p></article><article><h3>Surface State</h3><p>等待经核验的 vacant / construction / building 标签。</p></article><article><h3>Temporal Fusion</h3><p>接口已搭建，等待 surface-state 与 geometry 输出。</p></article></div></section>
<p><a href="visual_gallery.html">打开统一可视化资产库 →</a></p></main></body></html>"""


def _markdown_catalogs(models: list[dict[str, Any]], datasets: list[dict[str, Any]], experiments: list[dict[str, Any]]) -> None:
    docs = REPO_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    model_lines = ["# Model Catalog", "", "| Model | Task | Status | Test | Checkpoint |", "|---|---|---|---|---|"]
    for row in models:
        model_lines.append(f"| `{row['model_id']}` | {', '.join(row['task'])} | `{row['status']}` | `{row['test_status']}` | `{row.get('checkpoint') or 'none'}` |")
    (docs / "MODEL_CATALOG.md").write_text("\n".join(model_lines) + "\n", encoding="utf-8")
    dataset_lines = ["# Dataset Catalog", "", "| Dataset | Role | Train | Val | Test policy | Status |", "|---|---|---:|---:|---|---|"]
    for row in datasets:
        dataset_lines.append(f"| `{row['dataset_id']}` | {row['role']} | {row.get('train_count')} | {row.get('val_count')} | `{row['test_policy']}` | `{row['status']}` |")
    (docs / "DATASET_CATALOG.md").write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")
    statuses: dict[str, int] = {}
    for row in experiments:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    (docs / "EXPERIMENT_STATUS.md").write_text(
        "# Experiment Status\n\n"
        + "\n".join(f"- `{key}`: {value}" for key, value in sorted(statuses.items()))
        + "\n\n机器可读清单位于 `outputs/catalog/experiments.csv` 和 `experiments.json`。\n",
        encoding="utf-8",
    )


def _model_cards(models: list[dict[str, Any]]) -> None:
    for row in models:
        root = REPO_ROOT / "outputs" / "models" / row["model_id"]
        root.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {row['display_name']}", "", "## Purpose", "", ", ".join(row["task"]), "",
            "## Architecture", "", f"- Architecture: `{row['architecture']}`", f"- Encoder: `{row['encoder']}`",
            f"- Heads: {', '.join(row['heads'])}", "", "## Training data", "", ", ".join(row["training_datasets"]) or "None",
            "", "## Validation data", "", ", ".join(row["validation_datasets"]) or "None", "",
            "## Test policy", "", f"`{row['test_status']}`", "", "## Checkpoint", "",
            f"- Path: `{row.get('checkpoint') or 'none'}`", f"- SHA256: `{row.get('checkpoint_sha256') or 'unavailable'}`",
            "", "## Best validation metric", "", f"`{row.get('best_metric') or 'unavailable'}`", "",
            "## Known limitations", "",
            *[f"- {value}" for value in row.get("known_limitations", [])], "",
            "## Recommended usage", "", row.get("notes") or "See registry.", "",
            "## Forbidden usage", "", "- Do not bypass the dataset Test policy.", "- Do not represent candidate objects as ground truth.", "",
            "## Key visualizations", "", f"`{row.get('visualization_root') or 'none'}`",
        ]
        (root / "MODEL_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_catalog() -> dict[str, Any]:
    models = load_models()
    datasets = load_datasets()
    experiments = discover_experiments(models)
    artifacts = discover_artifacts(models)
    visualizations = [row for row in artifacts if row["artifact_type"] in {
        "training_curve", "validation_preview", "test_report", "external_inference", "map", "review_page", "dashboard"
    }]
    CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(CATALOG_ROOT / "experiments.csv", experiments)
    _write_json(CATALOG_ROOT / "experiments.json", experiments)
    _write_csv(CATALOG_ROOT / "artifacts.csv", artifacts)
    _write_csv(CATALOG_ROOT / "visualizations.csv", visualizations)
    (CATALOG_ROOT / "visual_gallery.html").write_text(_gallery_html(visualizations), encoding="utf-8")
    (CATALOG_ROOT / "project_dashboard.html").write_text(_dashboard_html(models, datasets, experiments), encoding="utf-8")
    _markdown_catalogs(models, datasets, experiments)
    _model_cards(models)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": len(models), "datasets": len(datasets), "experiments": len(experiments),
        "artifacts": len(artifacts), "visualizations": len(visualizations),
        "gallery": str((CATALOG_ROOT / "visual_gallery.html").resolve()),
        "dashboard": str((CATALOG_ROOT / "project_dashboard.html").resolve()),
    }
    _write_json(CATALOG_ROOT / "CATALOG_MANIFEST.json", summary)
    return summary


def print_rows(rows: list[dict[str, Any]], columns: list[str]) -> None:
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in columns}
    print("  ".join(key.ljust(widths[key]) for key in columns))
    print("  ".join("-" * widths[key] for key in columns))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key in columns))
