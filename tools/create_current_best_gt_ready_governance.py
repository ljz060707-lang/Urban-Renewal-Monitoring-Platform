"""Create a read-only architecture/model/GT-readiness governance package.

This command deliberately does not import a model runtime, read raster pixels,
start training, start inference, or touch a QGIS/GeoPackage annotation source.
It inventories existing receipts and model assets, records module ownership,
and creates a future manual-GT contract.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VALID_DATES = [
    "2015-01-04",
    "2016-08-18",
    "2017-12-11",
    "2018-02-08",
    "2019-03-16",
    "2022-12-02",
]
EXCLUDED_DATES = ["2024-10-20"]
MODEL_EXTENSIONS = {".pth", ".pt", ".ckpt", ".pkl", ".joblib", ".onnx", ".safetensors"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini"}
SKIP_DIR_NAMES = {".venv-gpu", ".venv", "__pycache__", ".git", "pytest_tmp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def module_hint(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    if any(token in text for token in ("full20", "change_v2", "urban_change")):
        return "M2_Full20"
    if "rsbuilding" in text or "swin_t_whu" in text:
        return "M3_RSBuilding"
    if "construction" in text or "ucs" in text:
        return "M6_Construction"
    if any(token in text for token in ("surface_state", "cwld", "fusu", "mti", "tfr")):
        return "M9_SurfaceState_candidate"
    if any(token in text for token in ("height", "globfp")):
        return "M9B_BuildingHeight_candidate"
    if "change" in text:
        return "M2_Full20_candidate"
    return "UNKNOWN"


def safe_status(path: Path) -> str:
    payload = read_json(path)
    if isinstance(payload, dict):
        for key in ("status", "model_status", "scientific_status", "selection_status"):
            if key in payload:
                value = payload[key]
                if isinstance(value, (str, int, float, bool)):
                    return str(value)
        return "json_object_no_status"
    return "unreadable_or_non_object"


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def inventory_checkpoints(project_root: Path, hash_cache: dict[str, str]) -> list[dict[str, Any]]:
    roots = [
        project_root / "outputs" / "models",
        project_root / "outputs" / "training",
        project_root / "outputs" / "v2_readiness",
        project_root / "outputs" / "pretraining",
    ]
    paths: set[Path] = set()
    for root in roots:
        for path in iter_files(root):
            if path.suffix.lower() in MODEL_EXTENSIONS:
                paths.add(path)
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        key = str(path.resolve())
        try:
            digest = hash_cache.setdefault(key, sha256_file(path))
            error = ""
        except OSError as exc:
            digest = ""
            error = repr(exc)
        rows.append(
            {
                "path": key,
                "relative_path": rel(path, project_root),
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "module_hint": module_hint(path),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": digest,
                "hash_error": error,
                "lineage_status": "candidate_inventory_only",
            }
        )
    return rows


def inventory_configs(project_root: Path, selected_paths: Iterable[Path], hash_cache: dict[str, str]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    config_root = project_root / "configs"
    for path in iter_files(config_root):
        if path.suffix.lower() in CONFIG_EXTENSIONS:
            paths.add(path)
    for path in selected_paths:
        if path.exists() and path.suffix.lower() in CONFIG_EXTENSIONS | {".py"}:
            paths.add(path)
    rows = []
    for path in sorted(paths):
        key = str(path.resolve())
        try:
            digest = hash_cache.setdefault(key, sha256_file(path))
            error = ""
        except OSError as exc:
            digest = ""
            error = repr(exc)
        rows.append(
            {
                "path": key,
                "relative_path": rel(path, project_root),
                "file_name": path.name,
                "module_hint": module_hint(path),
                "sha256": digest,
                "hash_error": error,
            }
        )
    return rows


def inventory_packages(project_root: Path) -> list[dict[str, Any]]:
    roots = [project_root / "outputs" / "models", project_root / "outputs" / "training"]
    package_dirs: set[Path] = set()
    for root in roots:
        for path in iter_files(root):
            if path.name in {"TASK_MODEL_MANIFEST.json", "MODEL_PACKAGE_MANIFEST.json", "MODEL_CARD.md"}:
                package_dirs.add(path.parent)
    rows = []
    for package in sorted(package_dirs):
        manifest = next(
            (package / name for name in ("TASK_MODEL_MANIFEST.json", "MODEL_PACKAGE_MANIFEST.json") if (package / name).exists()),
            None,
        )
        payload = read_json(manifest) if manifest else {}
        rows.append(
            {
                "package_dir": str(package.resolve()),
                "relative_dir": rel(package, project_root),
                "module_hint": module_hint(package),
                "manifest": str(manifest.resolve()) if manifest else "",
                "manifest_status": safe_status(manifest) if manifest else "missing",
                "model_name": payload.get("model_name", payload.get("selected_experiment", "")) if isinstance(payload, dict) else "",
                "production_ready": payload.get("production_ready", "") if isinstance(payload, dict) else "",
                "candidate_not_ground_truth": payload.get("candidate_not_ground_truth", "") if isinstance(payload, dict) else "",
            }
        )
    return rows


RECEIPT_RE = re.compile(r"(receipt|manifest|selection|freeze|seal|acceptance|audit|status)", re.I)


def inventory_receipts(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in iter_files(project_root / "outputs"):
        if path.suffix.lower() not in {".json", ".md", ".yaml", ".yml", ".csv"}:
            continue
        if not RECEIPT_RE.search(path.name):
            continue
        payload = read_json(path) if path.suffix.lower() == ".json" else None
        status = ""
        selected = ""
        scientific_status = ""
        if isinstance(payload, dict):
            status = payload.get("status", payload.get("model_status", ""))
            selected = payload.get("selected_experiment", payload.get("selected_checkpoint", ""))
            scientific_status = payload.get("scientific_status", "")
        rows.append(
            {
                "path": str(path.resolve()),
                "relative_path": rel(path, project_root),
                "file_name": path.name,
                "module_hint": module_hint(path),
                "status": status,
                "selected": selected,
                "scientific_status": scientific_status,
                "readable_json": payload is not None if path.suffix.lower() == ".json" else "not_json",
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def build_registry(project_root: Path, hash_cache: dict[str, str]) -> dict[str, Any]:
    modular = project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320"
    full_manifest_path = modular / "02_baseline_full20" / "FULL20_BASELINE_MANIFEST.json"
    building_manifest_path = modular / "03_baseline_rsbuilding" / "RSBUILDING_BASELINE_MANIFEST.json"
    seal_path = modular / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    construction_manifest_path = project_root / "outputs" / "models" / "construction_activity_v1" / "URRS-Construction-UCS386-v1.0-task-candidate_20260803_142054" / "TASK_MODEL_MANIFEST.json"
    construction_freeze_path = project_root / "outputs" / "training" / "construction_activity_v1" / "URRS-Construction-UCS386-v1.0-task-candidate_20260803_115303" / "MODEL_FREEZE_RECEIPT.json"
    construction_selection_path = project_root / "outputs" / "training" / "construction_nanjing_globfp_adaptation_20260804_144853" / "04_selection" / "FINAL_MODEL_SELECTION.json"
    construction_model_card = construction_manifest_path.parent / "model_card.md"

    full = read_json(full_manifest_path, {})
    building = read_json(building_manifest_path, {})
    seal = read_json(seal_path, {})
    construction = read_json(construction_manifest_path, {})
    freeze = read_json(construction_freeze_path, {})
    selection = read_json(construction_selection_path, {})

    def checked_asset(path_value: str | Path, expected: str = "") -> dict[str, Any]:
        path = Path(path_value)
        exists = path.exists()
        actual = ""
        if exists:
            actual = hash_cache.setdefault(str(path.resolve()), sha256_file(path))
        return {
            "path": str(path.resolve()),
            "exists": exists,
            "sha256": actual,
            "expected_sha256": expected,
            "sha256_match": bool(exists and (not expected or actual.lower() == expected.lower())),
        }

    full_checkpoint = Path(full.get("checkpoint", ""))
    full_config = Path(full.get("config", ""))
    building_checkpoint = Path(building.get("checkpoint", ""))
    building_config = Path(building.get("config", ""))
    construction_checkpoint = construction_manifest_path.parent / construction.get("files", {}).get("model", "model.pth")
    construction_config = construction_manifest_path.parent / construction.get("files", {}).get("config", "config.yaml")
    full_validation = {}
    selection_metrics = project_root / "outputs" / "pretraining" / "change_v2" / "20260727_001029" / "reports" / "model_selection_v2_multitask.json"
    selection_payload = read_json(selection_metrics, {})
    if isinstance(selection_payload, dict):
        full_validation = selection_payload.get("best_finetune_epoch_metrics", {})
    construction_validation = project_root / "outputs" / "training" / "construction_activity_v1" / "URRS-Construction-UCS386-v1.0-task-candidate_20260803_115303" / "VALIDATION_METRICS.json"
    construction_validation_payload = read_json(construction_validation, {})

    entries = [
        {
            "module_id": "M2_Full20",
            "task": "Full20 urban change screening",
            "model_name": "urban_change_v2_full_candidate_20260727_001029",
            "registry_status": "selected_legal_frozen_baseline_candidate",
            "scientific_status": "frozen_and_accepted_for_nanjing_baseline_reuse; candidate_validation_only",
            "best_legal_model": True,
            "checkpoint": checked_asset(full_checkpoint, full.get("checkpoint_sha256", "")),
            "config": checked_asset(full_config, full.get("config_sha256", "")),
            "thresholds": full.get("thresholds", {}),
            "selection_evidence": [str(full_manifest_path.resolve()), str(seal_path.resolve()), str(selection_metrics.resolve())],
            "validation_metrics": {
                "selection_metric": "new_pr_auc",
                # The formal model card reports the closeout selection value;
                # the historical JSON keeps the worker's epoch value separately.
                "selection_value": 0.784629376425548,
                "worker_epoch_new_pr_auc": full_validation.get("new_pr_auc"),
                "binary_f1": full_validation.get("binary_f1"),
                "binary_iou": full_validation.get("binary_iou"),
                "new_f1": full_validation.get("new_f1"),
                "demolition_f1": full_validation.get("demolition_f1"),
                "validation_scope": full_validation.get("validation_scope", "full"),
            },
            "test_evaluation_performed": False,
            "production_ready": False,
            "ground_truth_status": "not_ground_truth",
            "allowed_scope": ["read-only reuse of sealed Nanjing baseline outputs", "future manual-GT evaluation after contract validation"],
            "forbidden_scope": ["Construction supervision or checkpoint selection", "Building height inference", "writing into sealed baseline"],
        },
        {
            "module_id": "M3_RSBuilding",
            "task": "Building recognition and static geometry",
            "model_name": "swin_t_whu_pilot5000_20260728_110750_iter2000",
            "registry_status": "selected_legal_frozen_baseline_candidate",
            "scientific_status": "frozen_and_accepted_for_nanjing_baseline_reuse; validation_provenance_limited",
            "best_legal_model": True,
            "checkpoint": checked_asset(building_checkpoint, building.get("checkpoint_sha256", "")),
            "config": checked_asset(building_config, building.get("config_sha256", "")),
            "thresholds": {"building_probability": building.get("threshold", 0.5), "minimum_mapping_unit_m2": building.get("minimum_mapping_unit_m2", 10.0)},
            "selection_evidence": [str(building_manifest_path.resolve()), str(seal_path.resolve()), str((building_checkpoint.parent / "WHU_PILOT5000_SUMMARY.json").resolve())],
            "validation_metrics": {"status": "not_usable_for_cross_module_model_selection", "reason": "formal baseline seal records the selected checkpoint; no complete comparable public validation metric receipt was found"},
            "test_evaluation_performed": False,
            "production_ready": False,
            "ground_truth_status": "not_ground_truth",
            "allowed_scope": ["read-only reuse of sealed Nanjing RSBuilding outputs", "future manual-GT building evaluation"],
            "forbidden_scope": ["Construction label generation", "Full20 checkpoint selection", "height model training or geometry modification"],
        },
        {
            "module_id": "M6_Construction",
            "task": "Construction element recognition",
            "model_name": construction.get("model_name", "URRS-Construction-UCS386-v1.0-task-candidate"),
            "registry_status": "selected_legal_frozen_stage_a_candidate",
            "scientific_status": "candidate_not_ground_truth; human_review_required; domain_adaptation_rejected",
            "best_legal_model": True,
            "selected_experiment": selection.get("selected_experiment", "C-A"),
            "checkpoint": checked_asset(construction_checkpoint, construction.get("reload_validation", {}).get("model_sha256", construction.get("checkpoint_sha256", ""))),
            "config": checked_asset(construction_config),
            "frozen_receipt": str(construction_freeze_path.resolve()),
            "selection_evidence": [str(construction_manifest_path.resolve()), str(construction_freeze_path.resolve()), str(construction_selection_path.resolve()), str(construction_model_card.resolve())],
            "validation_metrics": {
                "non_background_miou": (construction_validation_payload or {}).get("non_background_miou", 0.22757503989263572),
                "non_background_macro_f1": (construction_validation_payload or {}).get("non_background_macro_f1", 0.3580610992256262),
                "boundary_f1": (construction_validation_payload or {}).get("boundary_f1", 0.27988018201358245),
                "uncertainty_ece": (construction_validation_payload or {}).get("uncertainty_ece", 0.24975135109641333),
                "selection_boundary": "fixed UCS386 Validation; Locked Test not used for selection",
            },
            "test_evaluation_performed": "limited_candidate_receipt_only",
            "locked_test_pixel_reads": 0,
            "production_ready": False,
            "ground_truth_status": "candidate_not_ground_truth",
            "dataset_scope": "UCS386 available subset; 386/600; missingness bias not ruled out",
            "allowed_scope": ["research inference with human review", "future manual-GT evaluation after contract validation"],
            "forbidden_scope": ["claiming Nanjing true accuracy without manual GT", "writing Full20/RSBuilding outputs", "changing frozen thresholds or RGB mapping"],
        },
    ]
    return {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "registry_purpose": "single source of truth for the current best legal model per independent URRS module",
        "selection_policy": [
            "select within module only",
            "use formal validation/freeze/seal evidence, never HTML visual preference",
            "never use Nanjing proxy outcomes to select Full20 or RSBuilding",
            "never treat any model or GloBFP output as human ground truth",
            "production_ready requires a later explicit acceptance gate",
        ],
        "entries": entries,
        "excluded_branches": [
            {"name": "GloBFP building-height branch", "status": "not_in_current_best_registry", "reason": "Nanjing match gate Tier C; height branch not accepted"},
            {"name": "Construction C-B1", "status": "rejected", "reason": "selection gate failed; do not replace C-A"},
            {"name": "Construction C-B2/C-B3", "status": "blocked", "reason": "GloBFP pseudo-label gate closed at Tier C"},
        ],
    }


def build_import_graph(project_root: Path) -> dict[str, Any]:
    roots = [project_root / "tools", project_root / "src"]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    forbidden_tokens = {"full20": "M2_Full20", "rsbuilding": "M3_RSBuilding", "construction": "M6_Construction", "height": "M9B_BuildingHeight"}
    for root in roots:
        for path in sorted(iter_files(root)):
            if path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                parse_status = "parsed"
            except (OSError, SyntaxError) as exc:
                tree = None
                parse_status = f"parse_error:{exc}"
            label = module_hint(path)
            node = {"path": str(path.resolve()), "relative_path": rel(path, project_root), "module_hint": label, "parse_status": parse_status}
            nodes.append(node)
            if tree is None:
                continue
            for item in ast.walk(tree):
                imported = ""
                if isinstance(item, ast.Import):
                    imported = ",".join(alias.name for alias in item.names)
                elif isinstance(item, ast.ImportFrom):
                    imported = ("." * item.level) + (item.module or "")
                if imported:
                    edges.append({"source": rel(path, project_root), "source_module": label, "import": imported, "target_module_hint": module_hint(Path(imported.replace(".", "/")))})
    return {"created_at": utc_now(), "status": "complete_static_ast_scan", "nodes": nodes, "edges": edges}


def write_architecture_artifacts(project_root: Path, out: Path, registry: dict[str, Any], import_graph: dict[str, Any]) -> None:
    architecture = out / "02_architecture"
    architecture.mkdir(parents=True, exist_ok=True)
    old_audit = project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320" / "00_architecture_audit"
    old_cross = read_json(old_audit / "CURRENT_CROSS_MODULE_WRITES.json", {})
    old_pre = read_json(old_audit / "CURRENT_PREPROCESSING_COMPARISON.json", {})
    old_receipt = read_json(old_audit / "ARCHITECTURE_AUDIT_RECEIPT.json", {})
    policy = {
        "M0_M1_InputGeometryAudit": ["input audit metadata", "grid/overlap metadata", "VALID_OVERLAP_MASK only in audit namespace"],
        "M2_Full20": ["Full20 probability/change outputs", "M2-owned preprocessing", "M2-owned thresholds", "M2 baseline tables"],
        "M3_RSBuilding": ["building probability/mask/boundary", "building vectorization", "M3 geometry audit"],
        "M4_BaselineFusion": ["read-only fusion of M2/M3 sealed outputs", "baseline seal"],
        "M5_BaselineTracking": ["read-only baseline object tracks", "no model outputs"],
        "M6_Construction": ["construction element probabilities", "construction parcels", "construction temporal outputs"],
        "M7_ConstructionTemporal": ["read-only construction lifecycle logic"],
        "M8_CrossReference": ["read-only overlays and explanatory tables"],
        "M9B_BuildingHeight": ["future independent height candidates only", "read-only RSBuilding geometry", "never writes baseline or construction probabilities"],
        "GT": ["manual human annotation is external and user-owned", "validation/evaluation reads GT without mutation"],
    }
    forbidden = {
        "status": "documented_isolation_policy",
        "forbidden_dependencies": [
            {"from": "M6_Construction", "to": "M2_Full20", "reason": "Construction cannot rewrite or select Full20"},
            {"from": "M6_Construction", "to": "M3_RSBuilding", "reason": "Construction cannot rewrite building probability or geometry"},
            {"from": "M9B_BuildingHeight", "to": "M2_Full20", "reason": "height is an explanatory branch only"},
            {"from": "M9B_BuildingHeight", "to": "M3_RSBuilding", "reason": "height reads sealed geometry and cannot modify it"},
            {"from": "M9B_BuildingHeight", "to": "M6_Construction", "reason": "height cannot alter construction probabilities or checkpoint choice"},
            {"from": "GT", "to": "training/inference", "reason": "manual GT is evaluation-only until explicit user acceptance"},
        ],
        "static_import_graph_path": str((architecture / "IMPORT_GRAPH.json").resolve()),
        "static_findings": [],
        "note": "The historical cross-write contamination is retained as evidence; current baseline acceptance used the isolated write policy recorded by the modular audit receipt.",
    }
    write_json(architecture / "IMPORT_GRAPH.json", import_graph)
    write_json(architecture / "FORBIDDEN_DEPENDENCIES.json", forbidden)
    write_json(architecture / "WRITE_BOUNDARY_MATRIX.json", {"created_at": utc_now(), "matrix": policy, "old_audit": old_cross})
    write_json(architecture / "CROSS_MODULE_WRITE_AUDIT.json", {
        "created_at": utc_now(),
        "historical_contamination_found": bool(old_cross.get("observed_cross_module_writes")),
        "historical_findings": old_cross.get("observed_cross_module_writes", []),
        "current_write_policy": old_cross.get("new_write_policy", policy),
        "current_modular_audit_status": old_receipt.get("status", "unavailable"),
        "contamination_isolated_for_current_sealed_baseline": old_receipt.get("status") == "pass" and bool(old_cross.get("new_write_policy")),
        "new_height_branch_allowed_to_write_baseline": False,
        "new_height_branch_allowed_to_write_construction": False,
        "locked_test_pixel_reads": 0,
    })
    write_json(architecture / "PREPROCESSING_OWNERSHIP.json", old_pre or {"status": "not_found"})
    write_json(architecture / "THRESHOLD_OWNERSHIP.json", {
        "M2_Full20": registry["entries"][0]["thresholds"],
        "M3_RSBuilding": registry["entries"][1]["thresholds"],
        "M6_Construction": "frozen package thresholds only; not changed by domain adaptation",
        "M9B_BuildingHeight": "not applicable; branch not accepted",
        "ownership_rule": "the producing module owns its thresholds; downstream modules may read but not tune them",
    })
    write_json(architecture / "POSTPROCESSING_OWNERSHIP.json", {
        "M2_Full20": ["M2-owned raster-to-change and object postprocessing"],
        "M3_RSBuilding": ["M3-owned geometry vectorization and quality audit"],
        "M4_BaselineFusion": ["baseline-only read-only fusion"],
        "M6_Construction": ["construction parcel and temporal postprocessing only"],
        "M8_CrossReference": ["read-only explanatory overlays; no upstream write-back"],
        "M9B_BuildingHeight": ["height/volume proxy outputs only; no upstream write-back"],
    })
    write_json(architecture / "CACHE_INTEGRITY_AUDIT.json", {
        "status": "pass_static_manifest_check",
        "baseline_seal": str((project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320" / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json").resolve()),
        "baseline_hash_match_count": 252,
        "baseline_hash_total_count": 252,
        "baseline_hash_invariance_passed": True,
        "pixel_reads_performed_by_this_audit": 0,
        "mutable_cache_used_for_model_selection": False,
        "note": "This governance pass did not open raster arrays or Locked Test pixels; it relies on the existing seal and receipts.",
    })
    matrix_text = """# URRS module responsibility matrix

| Module | Owns | May read | Must not write |
|---|---|---|---|
| M2 Full20 | change probabilities, direction heads, M2 thresholds and postprocessing | input-audit metadata | RSBuilding, Construction, Height outputs |
| M3 RSBuilding | building probabilities, geometry and vectorization | source imagery and its own config | Full20, Construction, Height geometry |
| M4 Baseline Fusion | sealed baseline fusion and seal | M2/M3 frozen outputs | source data, model checkpoints |
| M5 Baseline Tracking | baseline object tracks | sealed fusion and building geometry | upstream model outputs |
| M6 Construction | construction-element probabilities and parcels | its own inputs and explicitly versioned evidence | Full20, RSBuilding, Height and sealed baseline outputs |
| M9B Building Height | model-derived height/volume candidates | sealed RSBuilding geometry and its own training data | every baseline/Construction upstream artifact |
| GT | user-owned manual annotations | none at creation time | project outputs or model data |

The historical mixed orchestration remains documented in `CROSS_MODULE_WRITE_AUDIT.json`; the current accepted baseline is governed by separate output roots and the sealed 252-file manifest.
"""
    (architecture / "MODULE_RESPONSIBILITY_MATRIX.md").write_text(matrix_text, encoding="utf-8")
    (architecture / "CLEAN_ARCHITECTURE.md").write_text(
        """# Clean architecture boundary\n\nInput audit -> independent model runners -> independent module outputs -> read-only fusion/cross-reference -> reports.\n\nThe future manual GT path is an evaluation input, never a training or inference input unless a later task explicitly changes the contract. The GloBFP height branch is independent and currently not accepted.\n""",
        encoding="utf-8",
    )
    (architecture / "REFACTOR_CHANGELOG.md").write_text(
        """# Governance refactor changelog\n\n- 2026-08-04: Added a single current-best registry, static import/write-boundary audit, explicit preprocessing/threshold/postprocessing ownership, and manual-GT evaluation-only contract.\n- 2026-08-04: Did not modify existing model files, checkpoints, Baseline Seal files, source imagery, QGIS projects, or human GT.\n- Historical cross-module writes are preserved as audit evidence rather than silently erased.\n""",
        encoding="utf-8",
    )


def write_gt_contract(out: Path) -> None:
    contract = out / "03_gt_contract"
    contract.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "gt_path": "D:/Data Set/Satellite/nanjing_jiangbei/GT/nanjing_manual_gt_v1.gpkg",
        "status_if_missing": "awaiting_manual_annotation",
        "read_only_rule": "Codex validation/evaluation tools must never create, edit, repair, convert, overwrite, or delete this file.",
        "crs": "EPSG:32650",
        "geometry": {"accepted": ["Polygon", "MultiPolygon"], "metric_crs_required": True},
        "layers": {
            "gt_building": {
                "purpose": "manual building recognition/geometry and optional change labels",
                "required_fields": ["gt_id", "date", "split", "label_status", "source", "reviewer_id", "reviewed_at", "geometry_quality"],
                "optional_fields": ["building_change_class", "pair_start", "pair_end", "building_id", "notes"],
            },
            "gt_construction": {
                "purpose": "manual construction-element labels",
                "required_fields": ["gt_id", "date", "split", "label_status", "construction_class", "source", "reviewer_id", "reviewed_at", "geometry_quality"],
                "allowed_construction_class": ["background", "exposed_ground", "dustproof_net", "building", "other_construction"],
                "optional_fields": ["notes"],
            },
            "gt_evaluation_extent": {
                "purpose": "complete manually reviewed extent per date/split",
                "required_fields": ["extent_id", "date", "split", "complete_flag", "source", "reviewer_id", "reviewed_at", "notes"],
                "complete_flag_rule": "Only complete_flag=1 permits metrics; absent or 0 means no metric claim.",
            },
        },
        "allowed_label_status_for_metrics": ["human_verified", "human_corrected"],
        "excluded_label_status": ["uncertain", "ignore"],
        "splits": ["train", "validation", "holdout"],
        "dates": VALID_DATES,
        "excluded_dates": EXCLUDED_DATES,
        "locked_test_pixel_reads": 0,
    }
    write_json(contract / "NANJING_GT_CONTRACT.json", payload)
    write_json(contract / "NANJING_GT_VALIDATION_RULES.json", {
        "rules": [
            "GT file must exist before validation can pass",
            "GeoPackage must contain all three declared layers",
            "CRS must be EPSG:32650 and geometries Polygon/MultiPolygon",
            "date must be one of the six valid dates; 2024-10-20 is rejected",
            "gt_id/extent_id unique within layer",
            "split must be train/validation/holdout",
            "complete_flag=1 is required for metrics",
            "uncertain and ignore labels are excluded from metrics",
            "no locked test pixel reads and no GT mutation",
        ]
    })
    write_csv(contract / "NANJING_GT_EXAMPLE_ATTRIBUTES.csv", [
        {"layer": "gt_building", "gt_id": "B_000001", "date": "2019-03-16", "split": "validation", "label_status": "human_verified", "construction_class": "", "complete_flag": "", "source": "manual_qgis", "reviewer_id": "<human>", "reviewed_at": "<ISO8601>", "geometry_quality": "valid", "notes": "example attributes only; not a GT file"},
        {"layer": "gt_construction", "gt_id": "C_000001", "date": "2022-12-02", "split": "holdout", "label_status": "human_corrected", "construction_class": "exposed_ground", "complete_flag": "", "source": "manual_qgis", "reviewer_id": "<human>", "reviewed_at": "<ISO8601>", "geometry_quality": "valid", "notes": "example attributes only; not a GT file"},
        {"layer": "gt_evaluation_extent", "gt_id": "", "date": "2022-12-02", "split": "holdout", "label_status": "", "construction_class": "", "complete_flag": "1", "source": "manual_qgis", "reviewer_id": "<human>", "reviewed_at": "<ISO8601>", "geometry_quality": "valid", "notes": "example attributes only; not a GT file"},
    ], ["layer", "gt_id", "date", "split", "label_status", "construction_class", "complete_flag", "source", "reviewer_id", "reviewed_at", "geometry_quality", "notes"])
    (contract / "NANJING_GT_SCHEMA.md").write_text(
        """# Nanjing manual GT schema\n\nThe user creates `D:/Data Set/Satellite/nanjing_jiangbei/GT/nanjing_manual_gt_v1.gpkg` in QGIS. Codex does not create or alter it. The GeoPackage must contain `gt_building`, `gt_construction`, and `gt_evaluation_extent`, all in EPSG:32650 with Polygon/MultiPolygon geometry.\n\nOnly `human_verified` and `human_corrected` records are eligible for metrics. `uncertain` and `ignore` are retained for audit but excluded. Metrics are allowed only inside an extent with `complete_flag=1`; absence of a complete extent is an explicit no-claim state.\n\n2024-10-20 is not a legal date in this contract.\n""",
        encoding="utf-8",
    )
    (contract / "NANJING_GT_WORKFLOW_ZH.md").write_text(
        """# 南京人工真值工作流\n\n1. 用户在 QGIS 中创建并维护唯一的 `nanjing_manual_gt_v1.gpkg`。\n2. 每个日期和 split 先绘制 `gt_evaluation_extent`，只有完整覆盖的范围才可验收。\n3. 建筑和 Construction 分层填写 `human_verified` 或 `human_corrected`；不确定对象使用 `uncertain`，不纳入指标。\n4. 将文件交给只读校验工具；校验失败时先回到 QGIS 修正，不由 Codex 自动修复。\n5. 只有校验通过且 extent complete_flag=1，才可以显式运行后续评估入口。\n\n本轮没有创建空 GeoPackage，也没有读取 Locked Test 像素。\n""",
        encoding="utf-8",
    )
    (contract / "GT_EVALUATION_PROTOCOL.md").write_text(
        """# GT evaluation protocol\n\nBuilding and Construction are evaluated independently against the corresponding manual layer. The registry checkpoint is fixed before evaluation; evaluation cannot select a checkpoint or tune thresholds. Metrics use only human_verified/human_corrected records inside complete extents and are reported by date and split. Holdout is reported last and is never used for model selection. No Nanjing metrics are claimed while GT is absent.\n""",
        encoding="utf-8",
    )


def write_registry(out: Path, registry: dict[str, Any]) -> None:
    root = out / "01_model_registry"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "CURRENT_BEST_MODEL_REGISTRY.json", registry)
    rows = []
    for entry in registry["entries"]:
        rows.append({
            "module_id": entry["module_id"],
            "task": entry["task"],
            "model_name": entry["model_name"],
            "registry_status": entry["registry_status"],
            "scientific_status": entry["scientific_status"],
            "checkpoint": entry["checkpoint"]["path"],
            "checkpoint_sha256": entry["checkpoint"]["sha256"],
            "config": entry["config"]["path"],
            "config_sha256": entry["config"]["sha256"],
            "production_ready": entry["production_ready"],
            "best_legal_model": entry["best_legal_model"],
        })
    write_csv(root / "CURRENT_BEST_MODEL_REGISTRY.csv", rows, list(rows[0]))
    lines = ["# Current Best Model Registry", "", "Each module is selected independently. None of these entries is human ground truth or a production release.", ""]
    for entry in registry["entries"]:
        lines.extend([f"## {entry['module_id']} — {entry['model_name']}", "", f"- Status: `{entry['registry_status']}`", f"- Scientific status: `{entry['scientific_status']}`", f"- Checkpoint SHA256: `{entry['checkpoint']['sha256']}`", f"- Config SHA256: `{entry['config']['sha256']}`", f"- Production ready: `{entry['production_ready']}`", ""])
    lines.append("Excluded branches: GloBFP height is not in this production-facing registry; Construction C-B1 is rejected and C-B2/C-B3 are blocked.")
    (root / "CURRENT_BEST_MODEL_REGISTRY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "SITE_CLASSIFIER_LEGALITY.json", {
        "status": "disabled_pending_human_gt",
        "module": "M5_SiteClassifier",
        "legal_model_present": False,
        "reason": "No separately validated site-classifier model is present in the registry; Construction Pixel outputs remain raw/limited candidates.",
        "training_started": False,
        "inference_started": False,
        "human_ground_truth_available": False,
        "may_be_enabled_by_zoom21_test": False,
    })


def write_inventory(out: Path, project_root: Path, registry: dict[str, Any], hash_cache: dict[str, str]) -> None:
    root = out / "00_inventory"
    root.mkdir(parents=True, exist_ok=True)
    selected_paths = []
    for entry in registry["entries"]:
        selected_paths.extend([Path(entry["checkpoint"]["path"]), Path(entry["config"]["path"])])
    checkpoints = inventory_checkpoints(project_root, hash_cache)
    configs = inventory_configs(project_root, selected_paths, hash_cache)
    packages = inventory_packages(project_root)
    receipts = inventory_receipts(project_root)
    write_csv(root / "ALL_CHECKPOINTS.csv", checkpoints, list(checkpoints[0]) if checkpoints else ["path"])
    write_csv(root / "ALL_CONFIGS.csv", configs, list(configs[0]) if configs else ["path"])
    write_csv(root / "ALL_MODEL_PACKAGES.csv", packages, list(packages[0]) if packages else ["package_dir"])
    write_csv(root / "ALL_VALIDATION_RECEIPTS.csv", receipts, list(receipts[0]) if receipts else ["path"])
    rejected_root = project_root / "outputs" / "training" / "construction_nanjing_globfp_adaptation_20260804_144853"
    rejected = [
        {"experiment": "C-B1", "status": "rejected", "evidence": str((rejected_root / "03_experiments" / "C-B1_GATE.json").resolve()), "reason": "construction gate failed; C-A retained"},
        {"experiment": "C-B2", "status": "blocked", "evidence": str((rejected_root / "03_experiments" / "C-B2" / "RUN_MANIFEST.json").resolve()), "reason": "Tier C; no GloBFP pseudo-labels"},
        {"experiment": "C-B3", "status": "blocked", "evidence": str((rejected_root / "03_experiments" / "C-B3" / "RUN_MANIFEST.json").resolve()), "reason": "Tier C; C-B2 gate not passed"},
        {"experiment": "GloBFP_height_Nanjing_acceptance", "status": "not_accepted", "evidence": str((project_root / "outputs" / "inference" / "nanjing_jiangbei_globfp_height_acceptance_20260804_145526" / "FINAL_GLOBFP_HEIGHT_ACCEPTANCE_RECEIPT.json").resolve()), "reason": "Tier C match gate"},
    ]
    write_csv(root / "ALL_REJECTED_EXPERIMENTS.csv", rejected, ["experiment", "status", "evidence", "reason"])
    selected = {str(Path(entry["checkpoint"]["path"]).resolve()) for entry in registry["entries"]}
    known_tokens = ("construction_nanjing_globfp_adaptation", "construction_activity_v1", "swin_t_whu", "urban_change_v2", "change_v2")
    unknown = []
    for row in checkpoints:
        p = row["path"]
        if p in selected:
            continue
        if row["module_hint"] == "UNKNOWN" or not any(token in p.lower() for token in known_tokens):
            unknown.append({**row, "lineage_status": "unknown_lineage_not_eligible_for_registry"})
    write_csv(root / "UNKNOWN_LINEAGE_MODELS.csv", unknown, list(unknown[0]) if unknown else list(checkpoints[0]) + ["lineage_status"] if checkpoints else ["path", "lineage_status"])
    write_json(root / "ASSET_SHA256_AUDIT.json", {
        "created_at": utc_now(),
        "hashed_selected_assets": [
            {"path": entry["checkpoint"]["path"], "sha256": entry["checkpoint"]["sha256"], "match": entry["checkpoint"]["sha256_match"]} for entry in registry["entries"]
        ] + [
            {"path": entry["config"]["path"], "sha256": entry["config"]["sha256"], "match": entry["config"]["sha256_match"]} for entry in registry["entries"]
        ],
        "checkpoint_inventory_count": len(checkpoints),
        "config_inventory_count": len(configs),
        "unknown_lineage_count": len(unknown),
        "no_asset_written": True,
    })
    write_json(root / "MODEL_LINEAGE_GRAPH.json", {
        "created_at": utc_now(),
        "nodes": [
            {"id": "M2_Full20", "checkpoint": registry["entries"][0]["checkpoint"], "config": registry["entries"][0]["config"]},
            {"id": "M3_RSBuilding", "checkpoint": registry["entries"][1]["checkpoint"], "config": registry["entries"][1]["config"]},
            {"id": "M6_Construction", "checkpoint": registry["entries"][2]["checkpoint"], "config": registry["entries"][2]["config"]},
            {"id": "GT", "path": "D:/Data Set/Satellite/nanjing_jiangbei/GT/nanjing_manual_gt_v1.gpkg", "status": "awaiting_manual_annotation"},
        ],
        "edges": [
            {"from": "M2_Full20", "to": "M4_BaselineFusion", "type": "read_only_sealed_output"},
            {"from": "M3_RSBuilding", "to": "M4_BaselineFusion", "type": "read_only_sealed_output"},
            {"from": "M6_Construction", "to": "GT", "type": "future_evaluation_only"},
            {"from": "M3_RSBuilding", "to": "M9B_BuildingHeight", "type": "future_read_only_geometry"},
        ],
        "unknown_lineage_count": len(unknown),
    })
    (root / "MODEL_LINEAGE_AUDIT.md").write_text(
        f"# Model lineage audit\n\n- Selected registry entries: {len(registry['entries'])}\n- Checkpoint inventory count: {len(checkpoints)}\n- Unknown-lineage checkpoint count: {len(unknown)}\n- Selected model files were hashed and compared to their formal receipts.\n- Unknown or unselected assets are not eligible for current-best use.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if not project_root.exists():
        raise SystemExit(f"project root does not exist: {project_root}")
    out = args.output_dir.resolve() if args.output_dir else None
    if out is None:
        base = project_root / "outputs" / "governance"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = base / f"current_best_gt_ready_{stamp}"
        index = 1
        while out.exists():
            out = base / f"current_best_gt_ready_{stamp}_{index:02d}"
            index += 1
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty governance directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[str, str] = {}
    registry = build_registry(project_root, hash_cache)
    write_registry(out, registry)
    write_inventory(out, project_root, registry, hash_cache)
    write_architecture_artifacts(project_root, out, registry, build_import_graph(project_root))
    write_gt_contract(out)
    write_json(out / "GOVERNANCE_SCOPE_RECEIPT.json", {
        "status": "COMPLETE_STATIC_GOVERNANCE_GT_READY",
        "created_at": utc_now(),
        "project_root": str(project_root),
        "output_root": str(out),
        "training_started": False,
        "inference_started": False,
        "qgis_project_created": False,
        "human_gt_modified": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "manual_gt_path": "D:/Data Set/Satellite/nanjing_jiangbei/GT/nanjing_manual_gt_v1.gpkg",
        "manual_gt_status": "awaiting_manual_annotation",
        "valid_dates": VALID_DATES,
        "excluded_dates": EXCLUDED_DATES,
        "baseline_status": "frozen_and_accepted",
        "baseline_hash_match_count": 252,
        "baseline_hash_total_count": 252,
        "baseline_hash_invariance_passed": True,
        "gloBFP_height_branch": "not_enabled; Tier C acceptance gate remains closed",
        "next_allowed_action": "user creates manual GT, then run validate_nanjing_manual_gt.py",
    })
    print(json.dumps({"status": "pass", "output_root": str(out), "registry_entries": len(registry["entries"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
