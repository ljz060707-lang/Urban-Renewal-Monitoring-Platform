from __future__ import annotations

import json
from pathlib import Path

from urban_renewal.construction.audit_v2 import audit_data, discover_data
from urban_renewal.pipelines.construction_temporal_pipeline import temporal_status
from urban_renewal.pipelines.construction_tongzhou_pipeline import run_tongzhou_construction, tongzhou_status
from urban_renewal.pipelines.construction_training_pipeline import formal_train_guard, run_20_step_smoke, run_dataloader_smoke
from urban_renewal.pipelines.construction_component_smoke import run_component_smoke
from urban_renewal.construction.formal_readiness import build_formal_training_readiness
from urban_renewal.construction.recovery_audit import audit_false_recovery_candidates
from urban_renewal.construction.ucs386_gate import build_ucs386_subset_gate
from urban_renewal.review.seed_temporal import create_seed_temporal_inventory
from urban_renewal.review.transition_workflow import run_transition_review_migration
from urban_renewal.review.transition_preview import diagnose_transition_previews, regenerate_transition_previews
from urban_renewal.review.transition_inventory import build_transition_track_inventory


def add_construction_parser(subparsers):
    parser = subparsers.add_parser("construction", help="construction_activity_v1 experimental candidate")
    actions = parser.add_subparsers(dest="construction_command", required=True)
    discover = actions.add_parser("discover"); discover.add_argument("--data-root", action="append")
    audit = actions.add_parser("audit"); audit.add_argument("--data-root", action="append"); audit.add_argument("--output-dir")
    labels = actions.add_parser("labels"); labels.add_argument("labels_action", choices=["audit"]); labels.add_argument("--data-root", action="append")
    train = actions.add_parser("train"); train.add_argument("--stage", choices=["ucs", "joint"], required=True); train.add_argument("--smoke", action="store_true"); train.add_argument("--device", default="auto")
    parcel = actions.add_parser("parcel"); parcel.add_argument("parcel_action", choices=["train"])
    hard = actions.add_parser("hard-negative"); hard.add_argument("hard_action", choices=["mine"])
    temporal = actions.add_parser("temporal"); temporal.add_argument("temporal_action", choices=["pseudo-label", "train"])
    mti = actions.add_parser("mti"); mti.add_argument("mti_action", choices=["evaluate"])
    actions.add_parser("track")
    tongzhou = actions.add_parser("tongzhou"); tongzhou.add_argument("tongzhou_action", choices=["run"]); tongzhou.add_argument("--device", choices=["cuda", "cuda:0"], default="cuda")
    screening = actions.add_parser("screening"); screening.add_argument("screening_action", choices=["train"])
    smoke = actions.add_parser("smoke"); smoke.add_argument("--device", default="auto"); smoke.add_argument("--dataloader-only", action="store_true")
    component_smoke = actions.add_parser("component-smoke"); component_smoke.add_argument("--output-dir")
    readiness = actions.add_parser("readiness"); readiness.add_argument("--audit-dir"); readiness.add_argument("--output-dir")
    recovery = actions.add_parser("recovery-audit"); recovery.add_argument("--ucs-root", required=True); recovery.add_argument("--output-dir")
    subset_gate = actions.add_parser("subset-gate"); subset_gate.add_argument("--ucs-root", required=True); subset_gate.add_argument("--split-dir"); subset_gate.add_argument("--output-dir"); subset_gate.add_argument("--rgb-mapping", nargs=3, type=int, default=[3, 2, 1]); subset_gate.add_argument("--license-decision", default="confirmed_training_allowed")
    task_model = actions.add_parser("task-model"); task_model.add_argument("task_model_action", choices=["infer"]); task_model.add_argument("--package-dir", required=True); task_model.add_argument("--manifest", required=True); task_model.add_argument("--output-dir", required=True); task_model.add_argument("--sample-id"); task_model.add_argument("--device", default="cuda")
    seed_temporal = actions.add_parser("seed-temporal"); seed_temporal.add_argument("seed_action", choices=["build"]); seed_temporal.add_argument("--review-dir"); seed_temporal.add_argument("--output-dir")
    actions.add_parser("status")
    review = actions.add_parser("review")
    review.add_argument("review_action", choices=["migrate", "transition-diagnose", "transition-preview", "transition-inventory", "status"])
    review.add_argument("--review-dir")
    review.add_argument("--output-dir")
    review.add_argument("--no-regenerate", action="store_true")
    return parser


def _latest_gate(root: Path):
    audits = sorted((root / "outputs" / "urrs" / "audits").glob("construction_activity_*"))
    if not audits:
        return {"status": "not_run", "construction_activity_status": "experimental_in_progress", "next_action": "python tools/urrs.py construction audit"}
    gate_path = audits[-1] / "ONBOARDING_GATE.json"
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload.update({"status": "success", "audit_dir": str(audits[-1].resolve())})
    return payload


def handle_construction(root: Path, args):
    command = args.construction_command
    roots = [Path(path) for path in (getattr(args, "data_root", None) or [])] or None
    if command == "discover":
        output, discovery = discover_data(root, roots=roots)
        return {"status": "success", "output_dir": str(output.resolve()), "discovery": discovery}
    if command in {"audit", "labels"}:
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
        return audit_data(root, output_dir=output_dir, roots=roots)
    if command == "smoke":
        return run_dataloader_smoke(root) if args.dataloader_only else run_20_step_smoke(root, device=args.device)
    if command == "component-smoke":
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
        return run_component_smoke(root, output_dir=output_dir)
    if command == "readiness":
        audits = sorted((root / "outputs" / "urrs" / "audits").glob("construction_activity_*"))
        audit_dir = Path(args.audit_dir) if getattr(args, "audit_dir", None) else audits[-1]
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
        return build_formal_training_readiness(audit_dir, output_dir=output_dir)
    if command == "recovery-audit":
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
        return audit_false_recovery_candidates(root, Path(args.ucs_root), output_dir=output_dir)
    if command == "subset-gate":
        split_dir = Path(args.split_dir) if getattr(args, "split_dir", None) else None
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
        return build_ucs386_subset_gate(
            root,
            Path(args.ucs_root),
            split_dir=split_dir,
            output_dir=output_dir,
            rgb_mapping=tuple(args.rgb_mapping),
            license_status=args.license_decision,
            license_source="explicit_user_confirmation" if args.license_decision == "confirmed_training_allowed" else "cli_decision",
        )
    if command == "task-model":
        if args.task_model_action != "infer":
            raise ValueError(f"unsupported task-model action: {args.task_model_action}")
        from urban_renewal.construction.task_model import infer_task_model

        return infer_task_model(
            Path(args.package_dir),
            Path(args.manifest),
            Path(args.output_dir),
            sample_id=args.sample_id,
            device=args.device,
        )
    if command == "seed-temporal":
        review_dir = Path(args.review_dir) if getattr(args, "review_dir", None) else root / "outputs" / "urrs" / "review" / "tongzhou_20260801"
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else root / "outputs" / "urrs" / "construction" / ("human_seed_labels_" + __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"))
        return create_seed_temporal_inventory(review_dir, output_dir)
    if command == "train":
        if args.smoke:
            return run_20_step_smoke(root, device=args.device)
        return formal_train_guard(root, args.stage)
    if command == "temporal":
        return temporal_status()
    if command == "mti":
        return {**temporal_status(), "evaluation": "PENDING HUMAN LABELS"}
    if command == "tongzhou":
        if getattr(args, "tongzhou_action", None) == "run":
            return run_tongzhou_construction(root, device=args.device)
        return tongzhou_status(root)
    if command == "status":
        return _latest_gate(root)
    if command == "review":
        review_dir = Path(args.review_dir) if getattr(args, "review_dir", None) else root / "outputs" / "urrs" / "review" / "tongzhou_20260801"
        action = args.review_action
        if action == "migrate":
            return run_transition_review_migration(root, review_dir, regenerate=not args.no_regenerate)
        output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else root / "outputs" / "urrs" / "review" / "transition_diagnosis"
        if action == "transition-diagnose":
            return diagnose_transition_previews(review_dir, output_dir)
        if action == "transition-preview":
            payload = regenerate_transition_previews(root, review_dir, output_dir)
            return {"status": "success", "valid_transition_preview_count": payload["valid_transition_preview_count"], "review_allowed_count": payload["review_allowed_count"], "output_dir": str(output_dir.resolve())}
        if action == "transition-inventory":
            payload = json.loads((output_dir / "TRANSITION_REVIEW_INVENTORY.json").read_text(encoding="utf-8"))
            return build_transition_track_inventory(review_dir, payload, output_dir / "track_review")
        state = root / "outputs" / "urrs" / "status" / "CONSTRUCTION_REVIEW_RESUME_STATE.json"
        return json.loads(state.read_text(encoding="utf-8")) if state.exists() else {"status": "not_run"}
    reasons = {
        "parcel": "PENDING HUMAN LABELS: no auditable parcel scene labels",
        "hard-negative": "PENDING DATA: auditable hard-negative parcel labels not onboarded",
        "track": "PENDING MODEL: no formally validated construction pixel predictions",
        "screening": "PENDING DATA: national screening features and labels are not onboarded",
    }
    return {"status": "blocked", "command": command, "reason": reasons[command], "current_urrs_release_unchanged": True, "test_pixel_reads": {"WHU": 0, "SECOND": 0}}
