# 文件索引

按仓库路径列出源码、配置和文档。职责来自源码说明或函数与类名。应用入口、数据与版本状态见[模块状态](MODULE_STATUS.md)。

索引由 tools/maintenance/build_code_index.py 生成。

## 根目录

- [CHANGELOG.md](../CHANGELOG.md) — CHANGELOG
- [README.md](../README.md) — README

## applications

应用交付入口和合成测试

- [README.md](../applications/README.md) — README

### applications/single_date

- [diag_block.py](../applications/single_date/diag_block.py) — Diagnostic: why does a 10% >0.5 block yield only 20 owned objects?
- [equivalence.py](../applications/single_date/equivalence.py) — Equivalence evidence: frozen one-tile-at-a-time protocol vs the batched scheduler.
- [smoke_b0.py](../applications/single_date/smoke_b0.py) — Smoke test: one central and one edge block of the 2018-04-29 epoch.
- [test_epoch.py](../applications/single_date/test_epoch.py) — Two-block end-to-end validation of the frozen epoch path, with timing.
- [write_bindings.py](../applications/single_date/write_bindings.py) — Freeze the inference snapshot and record the real command bindings.
- [write_configs.py](../applications/single_date/write_configs.py) — Write the frozen queue, per-epoch manifests and the command bindings.

### applications/single_date/xian_single_date

- [__init__.py](../applications/single_date/xian_single_date/__init__.py) — XIAN-R5 single-date delivery snapshot (frozen before production block 1).
- [assemble.py](../applications/single_date/xian_single_date/assemble.py) — Per-epoch product assembly, reference linking, delivery matrices and readback.
- [b0.py](../applications/single_date/xian_single_date/b0.py) — B0 retained roof baseline, applied with the frozen V1 protocol.
- [common.py](../applications/single_date/xian_single_date/common.py) — XIAN-R5 single-date delivery: frozen constants and small helpers.
- [epoch.py](../applications/single_date/xian_single_date/epoch.py) — One actual observation epoch: blockwise B0 observation, no temporal model, no GT.
- [rasters.py](../applications/single_date/xian_single_date/rasters.py) — Rasterisation and exact fractions on the frozen reference grid.
- [reference_gt.py](../applications/single_date/xian_single_date/reference_gt.py) — 2019 designated manual roof GT -> reference map product.
- [supervisor.py](../applications/single_date/xian_single_date/supervisor.py) — Single production supervisor: frozen queue, serial GPU, resumable stages.

### applications/xian_r5

- [CLASS_DICTIONARY_V1.json](../applications/xian_r5/CLASS_DICTIONARY_V1.json) — CLASS DICTIONARY V1
- [IMPLEMENTATION_MAP.json](../applications/xian_r5/IMPLEMENTATION_MAP.json) — IMPLEMENTATION MAP
- [MODEL_AND_DATA_ROLES.md](../applications/xian_r5/MODEL_AND_DATA_ROLES.md) — MODEL AND DATA ROLES

### applications/xian_r5/tests

- [test_application.py](../applications/xian_r5/tests/test_application.py) — Explicit synthetic integration fixtures, never real change ground truth.

## configs

运行、训练、标签和注册配置

- [V2_HOLDOUT_PROTOCOL.json](../configs/V2_HOLDOUT_PROTOCOL.json) — V2 HOLDOUT PROTOCOL
- [building_foundation_v2.yaml](../configs/building_foundation_v2.yaml) — building foundation v2
- [building_foundation_v2_smoke.yaml](../configs/building_foundation_v2_smoke.yaml) — building foundation v2 smoke
- [change_public.yaml](../configs/change_public.yaml) — change public
- [change_public_dsifn_windows.yaml](../configs/change_public_dsifn_windows.yaml) — change public dsifn windows
- [change_public_s2looking_demolition_head_epoch2.yaml](../configs/change_public_s2looking_demolition_head_epoch2.yaml) — change public s2looking demolition head epoch2
- [change_public_s2looking_demolition_head_epoch3.yaml](../configs/change_public_s2looking_demolition_head_epoch3.yaml) — change public s2looking demolition head epoch3
- [change_public_s2looking_demolition_head_epoch4.yaml](../configs/change_public_s2looking_demolition_head_epoch4.yaml) — change public s2looking demolition head epoch4
- [change_public_s2looking_demolition_head_epoch5.yaml](../configs/change_public_s2looking_demolition_head_epoch5.yaml) — change public s2looking demolition head epoch5
- [change_public_s2looking_demolition_head_smoke.yaml](../configs/change_public_s2looking_demolition_head_smoke.yaml) — change public s2looking demolition head smoke
- [change_public_s2looking_hard_finetune.yaml](../configs/change_public_s2looking_hard_finetune.yaml) — change public s2looking hard finetune
- [change_public_s2looking_hard_finetune_balanced_smoke.yaml](../configs/change_public_s2looking_hard_finetune_balanced_smoke.yaml) — change public s2looking hard finetune balanced smoke
- [change_public_s2looking_hard_finetune_conservative_smoke.yaml](../configs/change_public_s2looking_hard_finetune_conservative_smoke.yaml) — change public s2looking hard finetune conservative smoke
- [change_public_s2looking_hard_finetune_smoke.yaml](../configs/change_public_s2looking_hard_finetune_smoke.yaml) — change public s2looking hard finetune smoke
- [change_public_s2looking_windows.yaml](../configs/change_public_s2looking_windows.yaml) — change public s2looking windows
- [change_public_v2_full20_inference_windows.yaml](../configs/change_public_v2_full20_inference_windows.yaml) — change public v2 full20 inference windows
- [change_public_v2_multisource_windows.before_mapping_schema_fix_20260726_191612.yaml](../configs/change_public_v2_multisource_windows.before_mapping_schema_fix_20260726_191612.yaml) — change public v2 multisource windows.before mapping schema fix 20260726 191612
- [change_public_v2_multisource_windows.before_second_mapping_20260726_190935.yaml](../configs/change_public_v2_multisource_windows.before_second_mapping_20260726_190935.yaml) — change public v2 multisource windows.before second mapping 20260726 190935
- [change_public_v2_multisource_windows.before_second_mapping_fix_20260726_192910.yaml](../configs/change_public_v2_multisource_windows.before_second_mapping_fix_20260726_192910.yaml) — change public v2 multisource windows.before second mapping fix 20260726 192910
- [change_public_v2_multisource_windows.yaml](../configs/change_public_v2_multisource_windows.yaml) — change public v2 multisource windows
- [change_public_v2_multisource_windows_dev5.yaml](../configs/change_public_v2_multisource_windows_dev5.yaml) — change public v2 multisource windows dev5
- [change_public_v2_multisource_windows_full20.yaml](../configs/change_public_v2_multisource_windows_full20.yaml) — change public v2 multisource windows full20
- [change_public_windows.yaml](../configs/change_public_windows.yaml) — change public windows
- [construction_nanjing_domain_adaptation_v1.yaml](../configs/construction_nanjing_domain_adaptation_v1.yaml) — construction nanjing domain adaptation v1

### configs/datasets

- [beijing_waste_template.yaml](../configs/datasets/beijing_waste_template.yaml) — beijing waste template
- [generic_mti_template.yaml](../configs/datasets/generic_mti_template.yaml) — generic mti template
- [generic_tfr_template.yaml](../configs/datasets/generic_tfr_template.yaml) — generic tfr template

- [example.yaml](../configs/example.yaml) — example

### configs/label_maps

- [beijing_waste_template.yaml](../configs/label_maps/beijing_waste_template.yaml) — beijing waste template

### configs/ontology

- [canonical.yaml](../configs/ontology/canonical.yaml) — canonical

- [phase1.yaml](../configs/phase1.yaml) — phase1

### configs/registry

- [datasets.yaml](../configs/registry/datasets.yaml) — datasets
- [models.yaml](../configs/registry/models.yaml) — models
- [modules.yaml](../configs/registry/modules.yaml) — modules

### configs/temporal

- [default_rules.yaml](../configs/temporal/default_rules.yaml) — default rules
- [example_observations.json](../configs/temporal/example_observations.json) — example observations

### configs/urrs/construction

- [area_accounting.yaml](../configs/urrs/construction/area_accounting.yaml) — area accounting
- [datasets.yaml](../configs/urrs/construction/datasets.yaml) — datasets
- [element_model.yaml](../configs/urrs/construction/element_model.yaml) — element model
- [hard_negative.yaml](../configs/urrs/construction/hard_negative.yaml) — hard negative
- [national_screening.yaml](../configs/urrs/construction/national_screening.yaml) — national screening
- [ontology.yaml](../configs/urrs/construction/ontology.yaml) — ontology
- [parcel_classifier.yaml](../configs/urrs/construction/parcel_classifier.yaml) — parcel classifier
- [state_graph.yaml](../configs/urrs/construction/state_graph.yaml) — state graph
- [temporal.yaml](../configs/urrs/construction/temporal.yaml) — temporal
- [tongzhou.yaml](../configs/urrs/construction/tongzhou.yaml) — tongzhou
- [ucs386_task_candidate.yaml](../configs/urrs/construction/ucs386_task_candidate.yaml) — ucs386 task candidate
- [vectorization.yaml](../configs/urrs/construction/vectorization.yaml) — vectorization

## docs

模块设计、数据准入和历史审计

- [01_PUBLIC_DATASET_TRAINING_WINDOWS.md](../docs/01_PUBLIC_DATASET_TRAINING_WINDOWS.md) — 01 PUBLIC DATASET TRAINING WINDOWS
- [02_PERSONAL_QGIS_GEOOSAM_WORKFLOW.md](../docs/02_PERSONAL_QGIS_GEOOSAM_WORKFLOW.md) — 02 PERSONAL QGIS GEOOSAM WORKFLOW
- [03_FINAL_ACCEPTANCE_REPORT.md](../docs/03_FINAL_ACCEPTANCE_REPORT.md) — 03 FINAL ACCEPTANCE REPORT
- [04_REVIEW_AND_HARD_EXAMPLE_TRAINING.md](../docs/04_REVIEW_AND_HARD_EXAMPLE_TRAINING.md) — 04 REVIEW AND HARD EXAMPLE TRAINING
- [05_GLM_AI_ASSISTED_REVIEW.md](../docs/05_GLM_AI_ASSISTED_REVIEW.md) — 05 GLM AI ASSISTED REVIEW
- [06_REVIEW_GLM_HARD_FINETUNE_ACCEPTANCE.md](../docs/06_REVIEW_GLM_HARD_FINETUNE_ACCEPTANCE.md) — 06 REVIEW GLM HARD FINETUNE ACCEPTANCE
- [BUILDING_FOUNDATION_V2.md](../docs/BUILDING_FOUNDATION_V2.md) — BUILDING FOUNDATION V2
- [CANONICAL_TASK_ONTOLOGY.md](../docs/CANONICAL_TASK_ONTOLOGY.md) — CANONICAL TASK ONTOLOGY
- [CHANGE_TEACHER_TECHNICAL.md](../docs/CHANGE_TEACHER_TECHNICAL.md) — CHANGE TEACHER TECHNICAL
- [CONFIDENCE_MODEL.md](../docs/CONFIDENCE_MODEL.md) — CONFIDENCE MODEL
- [CONSTRUCTION_ACTIVITY_V1.md](../docs/CONSTRUCTION_ACTIVITY_V1.md) — CONSTRUCTION ACTIVITY V1
- [DATASET_CATALOG.md](../docs/DATASET_CATALOG.md) — DATASET CATALOG
- [DATASET_ONBOARDING.md](../docs/DATASET_ONBOARDING.md) — DATASET ONBOARDING
- [EXPERIMENT_STATUS.md](../docs/EXPERIMENT_STATUS.md) — EXPERIMENT STATUS
- [HUMAN_REVIEW_WORKFLOW.md](../docs/HUMAN_REVIEW_WORKFLOW.md) — HUMAN REVIEW WORKFLOW
- [MODEL_CATALOG.md](../docs/MODEL_CATALOG.md) — MODEL CATALOG
- [MODULE_STATUS.md](../docs/MODULE_STATUS.md) — MODULE STATUS
- [PROJECT_OVERVIEW.md](../docs/PROJECT_OVERVIEW.md) — PROJECT OVERVIEW
- [PROJECT_REGISTRY_ACCEPTANCE.md](../docs/PROJECT_REGISTRY_ACCEPTANCE.md) — PROJECT REGISTRY ACCEPTANCE
- [PROJECT_REPOSITORY_AUDIT.md](../docs/PROJECT_REPOSITORY_AUDIT.md) — PROJECT REPOSITORY AUDIT
- [PUBLIC_DATASETS.md](../docs/PUBLIC_DATASETS.md) — PUBLIC DATASETS
- [SUBMETER_SYSTEM.md](../docs/SUBMETER_SYSTEM.md) — SUBMETER SYSTEM
- [TECHNICAL_REFERENCE.md](../docs/TECHNICAL_REFERENCE.md) — TECHNICAL REFERENCE
- [TEST_SAFETY_POLICY.md](../docs/TEST_SAFETY_POLICY.md) — TEST SAFETY POLICY
- [URRS_TONGZHOU_REVIEW_CONSOLE.md](../docs/URRS_TONGZHOU_REVIEW_CONSOLE.md) — URRS TONGZHOU REVIEW CONSOLE
- [V030_AUDIT.md](../docs/V030_AUDIT.md) — V030 AUDIT
- [V040_AUDIT.md](../docs/V040_AUDIT.md) — V040 AUDIT
- [V050_AUDIT.md](../docs/V050_AUDIT.md) — V050 AUDIT
- [V051_AUDIT.md](../docs/V051_AUDIT.md) — V051 AUDIT
- [V052_FIRST_RUN_AUDIT.md](../docs/V052_FIRST_RUN_AUDIT.md) — V052 FIRST RUN AUDIT
- [V2_ARCHITECTURE.md](../docs/V2_ARCHITECTURE.md) — V2 ARCHITECTURE
- [VISUALIZATION_GUIDE.md](../docs/VISUALIZATION_GUIDE.md) — VISUALIZATION GUIDE

### docs/third_party

- [CWLD_LABEL_DEFINITION.md](../docs/third_party/CWLD_LABEL_DEFINITION.md) — CWLD LABEL DEFINITION

## experiments

实验定义

- [change_ablation_plan.yaml](../experiments/change_ablation_plan.yaml) — change ablation plan

## extensions

扩展模块

### extensions/surface_state_v1

- [__init__.py](../extensions/surface_state_v1/__init__.py) — Non-invasive SurfaceState v1 extension package.
- [audit.py](../extensions/surface_state_v1/audit.py) — audit · cwld_codec_audit, fusu_classification

### extensions/surface_state_v1/configs

- [cwld_smoke.yaml](../extensions/surface_state_v1/configs/cwld_smoke.yaml) — cwld smoke
- [cwld_smoke_safe.yaml](../extensions/surface_state_v1/configs/cwld_smoke_safe.yaml) — cwld smoke safe
- [cwld_train.yaml](../extensions/surface_state_v1/configs/cwld_train.yaml) — cwld train
- [fusu_sample_baseline_train.yaml](../extensions/surface_state_v1/configs/fusu_sample_baseline_train.yaml) — fusu sample baseline train
- [fusu_sample_train.yaml](../extensions/surface_state_v1/configs/fusu_sample_train.yaml) — fusu sample train
- [master_overlay_infer.yaml](../extensions/surface_state_v1/configs/master_overlay_infer.yaml) — master overlay infer
- [mini_cwld_train.yaml](../extensions/surface_state_v1/configs/mini_cwld_train.yaml) — mini cwld train
- [mini_fusu_sample_train.yaml](../extensions/surface_state_v1/configs/mini_fusu_sample_train.yaml) — mini fusu sample train
- [mini_joint_probe_check.yaml](../extensions/surface_state_v1/configs/mini_joint_probe_check.yaml) — mini joint probe check
- [mini_tfr_sample_train.yaml](../extensions/surface_state_v1/configs/mini_tfr_sample_train.yaml) — mini tfr sample train
- [mti_sample_calibration.yaml](../extensions/surface_state_v1/configs/mti_sample_calibration.yaml) — mti sample calibration
- [surface_joint_sample_train.yaml](../extensions/surface_state_v1/configs/surface_joint_sample_train.yaml) — surface joint sample train
- [temporal_tfr_sample_train.yaml](../extensions/surface_state_v1/configs/temporal_tfr_sample_train.yaml) — temporal tfr sample train

### extensions/surface_state_v1

- [conftest.py](../extensions/surface_state_v1/conftest.py) — conftest · deterministic_grouped_split_images

### extensions/surface_state_v1/datasets

- [__init__.py](../extensions/surface_state_v1/datasets/__init__.py) — datasets 包入口
- [cwld.py](../extensions/surface_state_v1/datasets/cwld.py) — cwld · CWLDManifestDataset
- [cwld_codec.py](../extensions/surface_state_v1/datasets/cwld_codec.py) — Auditable CWLD RGBA-to-class codec; never assigns ambiguous pixels.
- [fusu.py](../extensions/surface_state_v1/datasets/fusu.py) — fusu · FUSUSampleDataset
- [fusu_split.py](../extensions/surface_state_v1/datasets/fusu_split.py) — Deterministic FUSU sample split with an explicit weak-isolation fallback.

### extensions/surface_state_v1/integration

- [__init__.py](../extensions/surface_state_v1/integration/__init__.py) — integration 包入口
- [master_overlay.py](../extensions/surface_state_v1/integration/master_overlay.py) — Non-invasive evidence overlay; deliberately never imports RSBuilding.
- [overlay_final.py](../extensions/surface_state_v1/integration/overlay_final.py) — overlay final · final_overlay_infer
- [overlay_mti_runtime.py](../extensions/surface_state_v1/integration/overlay_mti_runtime.py) — overlay mti runtime · calibrated_overlay_infer
- [overlay_runtime.py](../extensions/surface_state_v1/integration/overlay_runtime.py) — overlay runtime · alignment_guard, overlay_infer
- [registration.py](../extensions/surface_state_v1/integration/registration.py) — registration · register_candidate
- [registration_final.py](../extensions/surface_state_v1/integration/registration_final.py) — registration final · register_final_candidate
- [tongzhou_acceptance.py](../extensions/surface_state_v1/integration/tongzhou_acceptance.py) — tongzhou acceptance · audit_tongzhou
- [tongzhou_source.py](../extensions/surface_state_v1/integration/tongzhou_source.py) — tongzhou source · inventory_tongzhou_source

### extensions/surface_state_v1/models

- [__init__.py](../extensions/surface_state_v1/models/__init__.py) — models 包入口
- [mti.py](../extensions/surface_state_v1/models/mti.py) — mti · MultiTaskIntegration
- [surface_state_net.py](../extensions/surface_state_v1/models/surface_state_net.py) — surface state net · SurfaceStateNet
- [temporal_change_head.py](../extensions/surface_state_v1/models/temporal_change_head.py) — temporal change head · TemporalChangeHead
- [tfr.py](../extensions/surface_state_v1/models/tfr.py) — tfr · TemporalFeatureRefinement

### extensions/surface_state_v1/tests

- [conftest.py](../extensions/surface_state_v1/tests/conftest.py) — conftest · synthetic_project, mini_run
- [test_alignment_guard.py](../extensions/surface_state_v1/tests/test_alignment_guard.py) — test alignment guard · test_alignment_guard_stops_mismatch
- [test_best_checkpoint_selection.py](../extensions/surface_state_v1/tests/test_best_checkpoint_selection.py) — test best checkpoint selection · test_best_checkpoint_uses_score_then_f1_then_earlier_epoch
- [test_cwld_codec.py](../extensions/surface_state_v1/tests/test_cwld_codec.py) — test cwld codec · test_exact_anchors_and_ambiguous_are_not_background
- [test_final_controls.py](../extensions/surface_state_v1/tests/test_final_controls.py) — test final controls · test_joint_warmstart_uses_selected_shared_and_dataset_heads, test_cwld_class_gate_rejects_single_class_collapse
- [test_frozen_hashes.py](../extensions/surface_state_v1/tests/test_frozen_hashes.py) — test frozen hashes · test_frozen_checkpoint_hashes
- [test_full_epoch_trainer.py](../extensions/surface_state_v1/tests/test_full_epoch_trainer.py) — test full epoch trainer · test_complete_mini_epoch_is_finite
- [test_fusu_spatial_split.py](../extensions/surface_state_v1/tests/test_fusu_spatial_split.py) — test fusu spatial split · test_grouped_fallback_has_no_group_overlap
- [test_joint_batch_scheduler.py](../extensions/surface_state_v1/tests/test_joint_batch_scheduler.py) — test joint batch scheduler · test_joint_cycle_contract
- [test_mti_calibration.py](../extensions/surface_state_v1/tests/test_mti_calibration.py) — test mti calibration · test_mti_output_length_matches_timesteps
- [test_overlay_checkpoint_loading.py](../extensions/surface_state_v1/tests/test_overlay_checkpoint_loading.py) — test overlay checkpoint loading · test_overlay_candidate_checkpoint_loads
- [test_resume_training.py](../extensions/surface_state_v1/tests/test_resume_training.py) — test resume training · test_resume_continues_global_step
- [test_tfr_training_step.py](../extensions/surface_state_v1/tests/test_tfr_training_step.py) — test tfr training step · test_tfr_receives_finite_gradients
- [test_validation_metrics.py](../extensions/surface_state_v1/tests/test_validation_metrics.py) — test validation metrics · test_ignore_pixels_do_not_enter_metrics

### extensions/surface_state_v1/training

- [__init__.py](../extensions/surface_state_v1/training/__init__.py) — training 包入口
- [amp_safe_runtime.py](../extensions/surface_state_v1/training/amp_safe_runtime.py) — Process-local AMP compatibility for probability-space BCE losses.
- [checkpointing.py](../extensions/surface_state_v1/training/checkpointing.py) — checkpointing · sha256, save_checkpoint, select_better, write_hashes
- [complete_post_joint.py](../extensions/surface_state_v1/training/complete_post_joint.py) — Complete registration, final overlay, and Tongzhou audit after Joint/MTI.
- [cwld_finalization.py](../extensions/surface_state_v1/training/cwld_finalization.py) — cwld finalization · finalize_cwld
- [experiment_control.py](../extensions/surface_state_v1/training/experiment_control.py) — experiment control · select_temporal_branch, run_joint_probes, run_joint_from_selection
- [final_pipeline.py](../extensions/surface_state_v1/training/final_pipeline.py) — final pipeline · run_final_pipeline
- [formal_runtime_guards.py](../extensions/surface_state_v1/training/formal_runtime_guards.py) — Process-local guards for formal Surface State V1 training runs.
- [full_trainer.py](../extensions/surface_state_v1/training/full_trainer.py) — Full epoch trainer for the existing SurfaceStateNet extension.
- [losses.py](../extensions/surface_state_v1/training/losses.py) — losses · multiclass_dice, cwld_loss, fusu_loss
- [metrics.py](../extensions/surface_state_v1/training/metrics.py) — metrics · Confusion, BinaryCounts
- [mti_calibration.py](../extensions/surface_state_v1/training/mti_calibration.py) — mti calibration · calibrate
- [mti_final.py](../extensions/surface_state_v1/training/mti_final.py) — mti final · recalibrate_mti
- [mti_selection_guard.py](../extensions/surface_state_v1/training/mti_selection_guard.py) — Validation-only guard that prevents a regressed MTI result from being selected.
- [post_temporal_pipeline.py](../extensions/surface_state_v1/training/post_temporal_pipeline.py) — Resume the formal pipeline after completed FUSU baseline/TFR validation.
- [resume_joint_pipeline.py](../extensions/surface_state_v1/training/resume_joint_pipeline.py) — Resume an interrupted formal Joint run, then finish downstream validation stages.
- [runtime_compat.py](../extensions/surface_state_v1/training/runtime_compat.py) — Narrow compatibility wrapper for the immutable prototype trainer.
- [safe_resume.py](../extensions/surface_state_v1/training/safe_resume.py) — safe resume · resume_cwld
- [scaler_runtime_guard.py](../extensions/surface_state_v1/training/scaler_runtime_guard.py) — Process-local GradScaler resume compatibility for CPU-prepared warm starts.
- [trainer.py](../extensions/surface_state_v1/training/trainer.py) — trainer · smoke_train_cwld

## external

第三方适配说明

- [README.md](../external/README.md) — README

## 根目录

- [pyproject.toml](../pyproject.toml) — pyproject

## reports

历史验收报告

- [BUILDING_FOUNDATION_V2_IMPLEMENTATION.md](../reports/BUILDING_FOUNDATION_V2_IMPLEMENTATION.md) — BUILDING FOUNDATION V2 IMPLEMENTATION
- [CODEX_WINDOWS_ENV_RECOVERY_20260801.md](../reports/CODEX_WINDOWS_ENV_RECOVERY_20260801.md) — CODEX WINDOWS ENV RECOVERY 20260801
- [CONSTRUCTION_ACTIVITY_V1_FIRST_RUN_ACCEPTANCE_20260802.md](../reports/CONSTRUCTION_ACTIVITY_V1_FIRST_RUN_ACCEPTANCE_20260802.md) — CONSTRUCTION ACTIVITY V1 FIRST RUN ACCEPTANCE 20260802
- [NEXT_TRAINING_OPTIMIZATION_REPORT.md](../reports/NEXT_TRAINING_OPTIMIZATION_REPORT.md) — NEXT TRAINING OPTIMIZATION REPORT
- [SURFACE_STATE_V1_EXECUTION_STATUS_20260729_1712.md](../reports/SURFACE_STATE_V1_EXECUTION_STATUS_20260729_1712.md) — SURFACE STATE V1 EXECUTION STATUS 20260729 1712
- [SURFACE_STATE_V1_FINAL_GATE_20260730_034853.md](../reports/SURFACE_STATE_V1_FINAL_GATE_20260730_034853.md) — SURFACE STATE V1 FINAL GATE 20260730 034853
- [SURFACE_STATE_V1_STATUS_20260729.md](../reports/SURFACE_STATE_V1_STATUS_20260729.md) — SURFACE STATE V1 STATUS 20260729
- [SURFACE_STATE_V1_TRAINING_ACCEPTANCE_20260729.md](../reports/SURFACE_STATE_V1_TRAINING_ACCEPTANCE_20260729.md) — SURFACE STATE V1 TRAINING ACCEPTANCE 20260729
- [URBAN_RENEWAL_V2_PREFLIGHT.json](../reports/URBAN_RENEWAL_V2_PREFLIGHT.json) — URBAN RENEWAL V2 PREFLIGHT
- [URRS_FINAL_GATE_20260801_140522.md](../reports/URRS_FINAL_GATE_20260801_140522.md) — URRS FINAL GATE 20260801 140522
- [URRS_FINAL_GATE_20260802_110030.md](../reports/URRS_FINAL_GATE_20260802_110030.md) — URRS FINAL GATE 20260802 110030
- [URRS_FINAL_GATE_20260802_110235.md](../reports/URRS_FINAL_GATE_20260802_110235.md) — URRS FINAL GATE 20260802 110235
- [URRS_FINAL_GATE_20260802_110311.md](../reports/URRS_FINAL_GATE_20260802_110311.md) — URRS FINAL GATE 20260802 110311
- [URRS_FINAL_GATE_20260802_110344.md](../reports/URRS_FINAL_GATE_20260802_110344.md) — URRS FINAL GATE 20260802 110344
- [V2_ARCHITECTURE_ACCEPTANCE.md](../reports/V2_ARCHITECTURE_ACCEPTANCE.md) — V2 ARCHITECTURE ACCEPTANCE
- [V2_FULL_TRAINING_READINESS.md](../reports/V2_FULL_TRAINING_READINESS.md) — V2 FULL TRAINING READINESS

## research

研究与案例脚本

- [README.md](../research/README.md) — README

### research/building_governance/r13_multiview_geometry

- [check_multiview_function.py](../research/building_governance/r13_multiview_geometry/check_multiview_function.py) — Synthetic invariants only; these scores are never actual XiAn accuracy.
- [evaluate_full_geometry.py](../research/building_governance/r13_multiview_geometry/evaluate_full_geometry.py) — Translate complete predicted polygons; no crop loss and no area inflation.
- [make_local_report.py](../research/building_governance/r13_multiview_geometry/make_local_report.py) — make local report
- [multiview_reference_model.py](../research/building_governance/r13_multiview_geometry/multiview_reference_model.py) — Reference-associated conditional position model; not SFM or physical height.
- [run_multiview_conditioned_experiment.py](../research/building_governance/r13_multiview_geometry/run_multiview_conditioned_experiment.py) — All fit/tuning is TRAIN-inner; TUNE once, conditional reference correspondence.

### research/building_governance/r19_loft_native

- [fuse_neighbor_instances.py](../research/building_governance/r19_loft_native/fuse_neighbor_instances.py) — Split individual raw parents only when multiple disjoint LOFT masks support the split.
- [loft_inference.py](../research/building_governance/r19_loft_native/loft_inference.py) — Author LOFT tensors and author offset methods in the installed MMDetection 3.x inference runtime.
- [measure_loft.py](../research/building_governance/r19_loft_native/measure_loft.py) — measure loft · status, assemble, frame_predict, native_source

### research/building_governance/r20_split_alignment_change_diagnostics

- [compare_change.py](../research/building_governance/r20_split_alignment_change_diagnostics/compare_change.py) — Compare binary evidence with prior roof occupancy on the very same supported pixels.
- [diagnose.py](../research/building_governance/r20_split_alignment_change_diagnostics/diagnose.py) — Window-level geometric diagnostics. The reference-guided arms are retrospective only.
- [four_arms.py](../research/building_governance/r20_split_alignment_change_diagnostics/four_arms.py) — Replay four arms in the existing EPSG:32649 polygon and exact fraction evaluators.
- [ground.py](../research/building_governance/r20_split_alignment_change_diagnostics/ground.py) — Estimate cross-date translations using imagery only, on candidate-free texture.
- [local_readback.py](../research/building_governance/r20_split_alignment_change_diagnostics/local_readback.py) — local readback · read, sha, main
- [mamba_pairs.py](../research/building_governance/r20_split_alignment_change_diagnostics/mamba_pairs.py) — Run the authors' WHU tiny binary change weights on actual neighboring observations.
- [train_split.py](../research/building_governance/r20_split_alignment_change_diagnostics/train_split.py) — Learn a visible-roof interior score, then split only inside frozen roof parents.

### research/building_governance/r21_pairwise_roofs

- [candidate_diagnostic.py](../research/building_governance/r21_pairwise_roofs/candidate_diagnostic.py) — Diagnostic maximum map matches from existing image-derived split proposals; no ref-created cuts.
- [evaluate_pairs.py](../research/building_governance/r21_pairwise_roofs/evaluate_pairs.py) — Freeze cut thresholds on native INNER, then read the Jinan and Xi'an evaluations.
- [external.py](../research/building_governance/r21_pairwise_roofs/external.py) — Image-supported map hints. Raw roof pixels are retained; external hints never create a roof.
- [limited.py](../research/building_governance/r21_pairwise_roofs/limited.py) — Evaluate a second operating point selected on Chengdu with an explicit fragmentation allowance.
- [local_readback.py](../research/building_governance/r21_pairwise_roofs/local_readback.py) — local readback · read, sha, main
- [parent_preview.py](../research/building_governance/r21_pairwise_roofs/parent_preview.py) — Show the entire predecessor object and its selected child in the same extent.
- [position.py](../research/building_governance/r21_pairwise_roofs/position.py) — Test split IDs with the original date field, keeping each child's parent translation.
- [prepare.py](../research/building_governance/r21_pairwise_roofs/prepare.py) — Freeze image-only within-parent regions, then join allowed native/map identities.
- [preview.py](../research/building_governance/r21_pairwise_roofs/preview.py) — preview · main
- [public_position.py](../research/building_governance/r21_pairwise_roofs/public_position.py) — Learn how far to translate an observed roof towards corroborated public map anchors.
- [train.py](../research/building_governance/r21_pairwise_roofs/train.py) — Train within-parent same/different-identity affinity; inference has no label input.

### research/building_governance/r22_edge_gap_cuts

- [assess.py](../research/building_governance/r22_edge_gap_cuts/assess.py) — assess · fragments, work, table, main
- [assess_contour.py](../research/building_governance/r22_edge_gap_cuts/assess_contour.py) — assess contour
- [cuts.py](../research/building_governance/r22_edge_gap_cuts/cuts.py) — cuts · choices, generate
- [exact.py](../research/building_governance/r22_edge_gap_cuts/exact.py) — exact · image, supervision, labels, Contour
- [figures.py](../research/building_governance/r22_edge_gap_cuts/figures.py) — figures
- [learn.py](../research/building_governance/r22_edge_gap_cuts/learn.py) — learn · label, Net, arrays, point
- [local_check.py](../research/building_governance/r22_edge_gap_cuts/local_check.py) — local check · sha
- [opportunity.py](../research/building_governance/r22_edge_gap_cuts/opportunity.py) — Post-freeze, manual-reference-conditioned candidate diagnostics for previously missed FIDs.
- [predict.py](../research/building_governance/r22_edge_gap_cuts/predict.py) — predict · main
- [predict_contour.py](../research/building_governance/r22_edge_gap_cuts/predict_contour.py) — predict contour · main
- [rank_review.py](../research/building_governance/r22_edge_gap_cuts/rank_review.py) — rank review

### research/building_governance/r23_cut_selector_retraining

- [final_train.py](../research/building_governance/r23_cut_selector_retraining/final_train.py) — final train · target, main
- [verify_delivery.py](../research/building_governance/r23_cut_selector_retraining/verify_delivery.py) — verify delivery · sha

### research/building_governance/xian_2019_citywide_governance

- [audit_unresolved.py](../research/building_governance/xian_2019_citywide_governance/audit_unresolved.py) — audit unresolved
- [calibrate_facade.py](../research/building_governance/xian_2019_citywide_governance/calibrate_facade.py) — calibrate facade
- [compare_repair_gt.py](../research/building_governance/xian_2019_citywide_governance/compare_repair_gt.py) — compare repair gt · stats, px
- [crop_position_governed.py](../research/building_governance/xian_2019_citywide_governance/crop_position_governed.py) — crop position governed
- [export_good_gt_regions.py](../research/building_governance/xian_2019_citywide_governance/export_good_gt_regions.py) — export good gt regions · profile, write_red
- [infer_models.py](../research/building_governance/xian_2019_citywide_governance/infer_models.py) — infer models · native, native_to_metric, City, flush
- [make_full_recall_review.py](../research/building_governance/xian_2019_citywide_governance/make_full_recall_review.py) — make full recall review
- [make_transparent_overlay.py](../research/building_governance/xian_2019_citywide_governance/make_transparent_overlay.py) — make transparent overlay
- [ml.py](../research/building_governance/xian_2019_citywide_governance/ml.py) — ml · save, status, checkpoint, Conv
- [pilot_native_context.py](../research/building_governance/xian_2019_citywide_governance/pilot_native_context.py) — pilot native context
- [pilot_soft_contours.py](../research/building_governance/xian_2019_citywide_governance/pilot_soft_contours.py) — pilot soft contours
- [prepare_training.py](../research/building_governance/xian_2019_citywide_governance/prepare_training.py) — prepare training · save, status, bucket, item
- [produce.py](../research/building_governance/xian_2019_citywide_governance/produce.py) — produce · save, status, tile
- [refine_alignment.py](../research/building_governance/xian_2019_citywide_governance/refine_alignment.py) — refine alignment
- [repair_field_routing.py](../research/building_governance/xian_2019_citywide_governance/repair_field_routing.py) — repair field routing
- [repair_overlay.py](../research/building_governance/xian_2019_citywide_governance/repair_overlay.py) — repair overlay
- [resources.py](../research/building_governance/xian_2019_citywide_governance/resources.py) — resources
- [summarize_gt_comparison.py](../research/building_governance/xian_2019_citywide_governance/summarize_gt_comparison.py) — summarize gt comparison
- [train_models.py](../research/building_governance/xian_2019_citywide_governance/train_models.py) — train models
- [validate_gt_comparison.py](../research/building_governance/xian_2019_citywide_governance/validate_gt_comparison.py) — validate gt comparison · px
- [validate_repair.py](../research/building_governance/xian_2019_citywide_governance/validate_repair.py) — validate repair
- [verify_local_delivery.py](../research/building_governance/xian_2019_citywide_governance/verify_local_delivery.py) — verify local delivery

### research/building_recognition/r15_native_refiner

- [admit_sam_assets_local.py](../research/building_recognition/r15_native_refiner/admit_sam_assets_local.py) — Only official generic SAM code/checkpoint; no SA1B or other dataset downloads.
- [check_roof_decoder_batch.py](../research/building_recognition/r15_native_refiner/check_roof_decoder_batch.py) — check roof decoder batch
- [compact_diagnostics.py](../research/building_recognition/r15_native_refiner/compact_diagnostics.py) — compact diagnostics
- [join_native_roof_supervision.py](../research/building_recognition/r15_native_refiner/join_native_roof_supervision.py) — Join native roof instances only to already frozen model-generated prompts.
- [native_roof_training_data.py](../research/building_recognition/r15_native_refiner/native_roof_training_data.py) — native roof training data · sha, embedding, instances, valid_map
- [prepare_roof_training_inputs.py](../research/building_recognition/r15_native_refiner/prepare_roof_training_inputs.py) — Image-only predicted prompts and exact frozen SAM FP32 image embeddings.
- [preserve_metric_scope_failure.py](../research/building_recognition/r15_native_refiner/preserve_metric_scope_failure.py) — preserve metric scope failure
- [roi_sam_pilot.py](../research/building_recognition/r15_native_refiner/roi_sam_pilot.py) — Source-INNER only: model-proposed ROI detail vs original whole-image SAM.
- [roof_decoder_evaluation.py](../research/building_recognition/r15_native_refiner/roof_decoder_evaluation.py) — Whole native source frames; inference sees only model-generated prompts.
- [roof_decoder_model.py](../research/building_recognition/r15_native_refiner/roof_decoder_model.py) — Per-image batched SAM decoder; original Meta modules and weights stay intact.
- [sam_native_pilot.py](../research/building_recognition/r15_native_refiner/sam_native_pilot.py) — Official generic SAM with actual model prompts, then identical native source metrics.
- [train_roof_decoder.py](../research/building_recognition/r15_native_refiner/train_roof_decoder.py) — Actual source-native roof mask decoder training; encoder and prompts frozen.

### research/building_recognition/r16_native_roi_roof

- [check_roi_training.py](../research/building_recognition/r16_native_roi_roof/check_roi_training.py) — check roi training · save
- [compact_live_status.py](../research/building_recognition/r16_native_roi_roof/compact_live_status.py) — compact live status · read, process
- [evaluate_xian_map_windows.py](../research/building_recognition/r16_native_roi_roof/evaluate_xian_map_windows.py) — Measure overlap with designated 2019 map polygons on frozen Xi'an TUNE windows.
- [evaluate_xian_position_windows.py](../research/building_recognition/r16_native_roi_roof/evaluate_xian_position_windows.py) — Compare frozen automatic shifts on all designated GT observations in supported TUNE windows.
- [measure_xian_fraction_windows.py](../research/building_recognition/r16_native_roi_roof/measure_xian_fraction_windows.py) — Exact 10 m grid intersections for tested date-field shifts on 72 development windows.
- [prepare_roi_embeddings.py](../research/building_recognition/r16_native_roi_roof/prepare_roi_embeddings.py) — Model-only native ROI definition, separate supervised sampling, exact FP32 cache.
- [reload_waiting_trainer.py](../research/building_recognition/r16_native_roi_roof/reload_waiting_trainer.py) — Reload only the parameter-update-free waiting driver after added preflight.
- [report_xian_reference.py](../research/building_recognition/r16_native_roi_roof/report_xian_reference.py) — Summarize paired Xi'an map-reference comparisons on the original 24 development sites.
- [roi_roof_evaluation.py](../research/building_recognition/r16_native_roi_roof/roi_roof_evaluation.py) — Native full-frame metrics; model sees only its own proposed ROIs and masks.
- [roi_roof_loss.py](../research/building_recognition/r16_native_roi_roof/roi_roof_loss.py) — Two declared loss arms, measured on real native pixels after ROI unwarping.
- [roi_training_data.py](../research/building_recognition/r16_native_roi_roof/roi_training_data.py) — roi training data · sha, instances, valid, ROIRoofDataset
- [train_roi_decoder.py](../research/building_recognition/r16_native_roi_roof/train_roi_decoder.py) — Two fixed native-ROI decoder loss arms; real AdamW steps, immutable selection.
- [xian_roi_selected.py](../research/building_recognition/r16_native_roi_roof/xian_roi_selected.py) — Xi'an TUNE inference after frozen source selection, with one frame of ROI features at a time.

### research/building_recognition/r17_independent_roof_proposals

- [assess_target_observability.py](../research/building_recognition/r17_independent_roof_proposals/assess_target_observability.py) — RGB visibility diagnostics for complete development windows, kept separate from geometry and temporal labels.
- [evaluate_proposals.py](../research/building_recognition/r17_independent_roof_proposals/evaluate_proposals.py) — One-to-one matching of independently proposed interior point and native box to annotated roofs.
- [infer_development_roofs.py](../research/building_recognition/r17_independent_roof_proposals/infer_development_roofs.py) — Evaluate image-only independent building discovery on source masks and Xi'an TUNE; preserve all raw queries.
- [neighbor_prompt_control.py](../research/building_recognition/r17_independent_roof_proposals/neighbor_prompt_control.py) — Matched CPU replay on 32 whole source images: independent effects of a neighboring negative point and padding.
- [prepare_centers.py](../research/building_recognition/r17_independent_roof_proposals/prepare_centers.py) — Build independent whole-frame proposal supervision from known native roof instances.
- [profile_native_misses.py](../research/building_recognition/r17_independent_roof_proposals/profile_native_misses.py) — profile native misses
- [proposal_net.py](../research/building_recognition/r17_independent_roof_proposals/proposal_net.py) — Full-image roof-center, native box and semantic heads. Inference accepts only RGB features.
- [report_development.py](../research/building_recognition/r17_independent_roof_proposals/report_development.py) — report development · read
- [train_independent.py](../research/building_recognition/r17_independent_roof_proposals/train_independent.py) — train independent · status, qualify, rank, domain_guard_test

### research/building_recognition/r18_native_detail_discovery

- [detail_inputs.py](../research/building_recognition/r18_native_detail_discovery/detail_inputs.py) — Prepare native RGB and immutable raw foreground for independent discovery experiments.
- [detail_model.py](../research/building_recognition/r18_native_detail_discovery/detail_model.py) — Controlled native RGB addition and a predeclared extra-building budget.
- [run_detail_masks.py](../research/building_recognition/r18_native_detail_discovery/run_detail_masks.py) — Freeze both discovery arms on identical source and Xi'an windows, then compare actual new native masks.
- [semantic_discovery.py](../research/building_recognition/r18_native_detail_discovery/semantic_discovery.py) — Test the trained independent semantic heads as direct roof masks, bypassing regressed boxes and SAM prompts.
- [train_detail.py](../research/building_recognition/r18_native_detail_discovery/train_detail.py) — train detail · status, selection_key, predictions, measured

### research/historical_xian_reviews

- [xian_aba_visual_export.py](../research/historical_xian_reviews/xian_aba_visual_export.py) — Export a deterministic 100+ ABA candidate visual-QA set on the cloud.
- [xian_candidate_presence_state_overlay.py](../research/historical_xian_reviews/xian_candidate_presence_state_overlay.py) — Candidate-only Presence/State recomputation using an identity overlay.
- [xian_candidate_v2_rebuild.py](../research/historical_xian_reviews/xian_candidate_v2_rebuild.py) — Isolated Xi'an candidate V2 rebuild and full held-edge recovery.
- [xian_final_candidate_rebuild.py](../research/historical_xian_reviews/xian_final_candidate_rebuild.py) — Rebuild Xi'an candidate IdentityGraph and downstream state from scratch.
- [xian_identity_aba_candidate.py](../research/historical_xian_reviews/xian_identity_aba_candidate.py) — Read-only identity continuity audit and ABA candidate scan for Xi'an R4.
- [xian_repair_qa_report.py](../research/historical_xian_reviews/xian_repair_qa_report.py) — xian repair qa report · read_csv, read_json, write_csv, truth
- [xian_rfinal_make_montages.py](../research/historical_xian_reviews/xian_rfinal_make_montages.py) — xian rfinal make montages · main
- [xian_rfinal_phase0_baseline.py](../research/historical_xian_reviews/xian_rfinal_phase0_baseline.py) — xian rfinal phase0 baseline · stat_record, shallow_supplement_inventory, main
- [xian_rfinal_phase1_diagnosis.py](../research/historical_xian_reviews/xian_rfinal_phase1_diagnosis.py) — xian rfinal phase1 diagnosis · pct, save_csv, main
- [xian_rfinal_phase2_candidate_audit.py](../research/historical_xian_reviews/xian_rfinal_phase2_candidate_audit.py) — xian rfinal phase2 candidate audit · main
- [xian_rfinal_phase2_identity_queue.py](../research/historical_xian_reviews/xian_rfinal_phase2_identity_queue.py) — xian rfinal phase2 identity queue · main
- [xian_rfinal_phase2_identity_visual.py](../research/historical_xian_reviews/xian_rfinal_phase2_identity_visual.py) — xian rfinal phase2 identity visual · txt, main
- [xian_rfinal_phase2_observability_diagnostic.py](../research/historical_xian_reviews/xian_rfinal_phase2_observability_diagnostic.py) — xian rfinal phase2 observability diagnostic · pct, main
- [xian_rfinal_phase2_visual_assets.py](../research/historical_xian_reviews/xian_rfinal_phase2_visual_assets.py) — xian rfinal phase2 visual assets · txt, main
- [xian_rfinal_phase2_visual_review_receipt.py](../research/historical_xian_reviews/xian_rfinal_phase2_visual_review_receipt.py) — Record the Phase 2 identity-edge visual triage without promoting edges.
- [xian_rfinal_phase3_absence_diagnostic.py](../research/historical_xian_reviews/xian_rfinal_phase3_absence_diagnostic.py) — Read-only absence evidence diagnostic for unresolved compound residuals.
- [xian_rfinal_phase3_compound_visual.py](../research/historical_xian_reviews/xian_rfinal_phase3_compound_visual.py) — Build a compact visual audit package for high-area unresolved compounds.
- [xian_rfinal_phase3_u05_review.py](../research/historical_xian_reviews/xian_rfinal_phase3_u05_review.py) — xian rfinal phase3 u05 review · main
- [xian_rfinal_targeted_absence_visual.py](../research/historical_xian_reviews/xian_rfinal_targeted_absence_visual.py) — xian rfinal targeted absence visual · main
- [xian_safe_rule_calibration.py](../research/historical_xian_reviews/xian_safe_rule_calibration.py) — Small, interpretable threshold calibration for the held-edge recovery rule.
- [xian_v2_closure_receipt.py](../research/historical_xian_reviews/xian_v2_closure_receipt.py) — xian v2 closure receipt · read_json, main
- [xian_v2_fixed_case_audit.py](../research/historical_xian_reviews/xian_v2_fixed_case_audit.py) — xian v2 fixed case audit · main
- [xian_v2_make_montages.py](../research/historical_xian_reviews/xian_v2_make_montages.py) — xian v2 make montages · main
- [xian_v2_unknown_safety_audit.py](../research/historical_xian_reviews/xian_v2_unknown_safety_audit.py) — xian v2 unknown safety audit · main
- [xian_v2_visual_assets.py](../research/historical_xian_reviews/xian_v2_visual_assets.py) — Generate real raw/overlay/contact-sheet assets for the isolated V2 candidate.
- [xian_v2_visual_review_labels.py](../research/historical_xian_reviews/xian_v2_visual_review_labels.py) — xian v2 visual review labels · label_for, main
- [xian_v4b_regression_audit.py](../research/historical_xian_reviews/xian_v4b_regression_audit.py) — Read-only diagnostics for the Xi'an V4B -> candidate regression.
- [xian_visual_asset_repair.py](../research/historical_xian_reviews/xian_visual_asset_repair.py) — xian visual asset repair · read_json, write_json, sha256, load_font
- [xian_visual_mosaics.py](../research/historical_xian_reviews/xian_visual_mosaics.py) — xian visual mosaics · make_mosaics, main

### research/temporal_tracking/r14_native_roof_association

- [admit_contrastive_method.py](../research/temporal_tracking/r14_native_roof_association/admit_contrastive_method.py) — admit contrastive method
- [admit_dinov2.py](../research/temporal_tracking/r14_native_roof_association/admit_dinov2.py) — Generic upstream code/model only; filter by role before any file bytes download.
- [audit_reference_conflicts.py](../research/temporal_tracking/r14_native_roof_association/audit_reference_conflicts.py) — Real held-development native crop diagnostics; conflicts are not certified errors.
- [check_dino_cpu.py](../research/temporal_tracking/r14_native_roof_association/check_dino_cpu.py) — check dino cpu
- [check_native_pool.py](../research/temporal_tracking/r14_native_roof_association/check_native_pool.py) — check native pool
- [check_native_temporal_contract.py](../research/temporal_tracking/r14_native_roof_association/check_native_temporal_contract.py) — check native temporal contract
- [check_rgb_identity_cpu.py](../research/temporal_tracking/r14_native_roof_association/check_rgb_identity_cpu.py) — check rgb identity cpu
- [check_temporal_hypergraph.py](../research/temporal_tracking/r14_native_roof_association/check_temporal_hypergraph.py) — check temporal hypergraph · directional
- [dino_model.py](../research/temporal_tracking/r14_native_roof_association/dino_model.py) — dino model · load_model, extract
- [evaluate_automatic_position.py](../research/temporal_tracking/r14_native_roof_association/evaluate_automatic_position.py) — Frozen automatic predicted tracks -> frozen TRAIN-fit date/view position model.
- [evaluate_dino_identity.py](../research/temporal_tracking/r14_native_roof_association/evaluate_dino_identity.py) — Frozen FP32 DINOv2 on actual native query ROIs, then TRAIN-inner calibration.
- [legacy_set_matcher.py](../research/temporal_tracking/r14_native_roof_association/legacy_set_matcher.py) — Untrained set-selection head over embeddings, variable K plus one dustbin.
- [make_local_report.py](../research/temporal_tracking/r14_native_roof_association/make_local_report.py) — make local report · read
- [native_temporal_contract.py](../research/temporal_tracking/r14_native_roof_association/native_temporal_contract.py) — Actual native model observations, partial-label loss and exact forward path cover.
- [prepare_native_graph.py](../research/temporal_tracking/r14_native_roof_association/prepare_native_graph.py) — Frozen model caches -> label-blind native observation graph -> separate weak labels.
- [prepare_pair_shape.py](../research/temporal_tracking/r14_native_roof_association/prepare_pair_shape.py) — Prediction-to-prediction centered mask IoU; no GT parameter or label read.
- [prepare_rgb_identity.py](../research/temporal_tracking/r14_native_roof_association/prepare_rgb_identity.py) — Native predicted-roof RGB/mask crops; fit known labels, full inner/TUNE graphs.
- [readback_derived_features.py](../research/temporal_tracking/r14_native_roof_association/readback_derived_features.py) — readback derived features
- [rgb_identity_model.py](../research/temporal_tracking/r14_native_roof_association/rgb_identity_model.py) — rgb identity model · RoofAppearanceEncoder
- [run_hypergraph_experiment.py](../research/temporal_tracking/r14_native_roof_association/run_hypergraph_experiment.py) — Frozen DINO scores -> GT-free track hypotheses -> TRAIN-inner global calibration.
- [temporal_hypergraph.py](../research/temporal_tracking/r14_native_roof_association/temporal_hypergraph.py) — GT-free three-period identity hypotheses with soft cycle evidence.
- [train_bounded_residual.py](../research/temporal_tracking/r14_native_roof_association/train_bounded_residual.py) — Do frozen semantic features fail, or does naive concatenation destabilize geometry?
- [train_dino_pair_residual.py](../research/temporal_tracking/r14_native_roof_association/train_dino_pair_residual.py) — Learn when frozen DINO appearance helps, preserving the trained geometry prior.
- [train_native_temporal.py](../research/temporal_tracking/r14_native_roof_association/train_native_temporal.py) — Two actual CPU AdamW arms; A10 remains dedicated to native instance training.
- [train_rgb_identity.py](../research/temporal_tracking/r14_native_roof_association/train_rgb_identity.py) — Real cross-period weak-reference contrastive fine-tuning of roof appearance.
- [train_shape_residual.py](../research/temporal_tracking/r14_native_roof_association/train_shape_residual.py) — Matched bounded temporal shape residual, plus a stronger soft-shape rule control.

### research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign

- [__init__.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/__init__.py) — Task-aligned temporal preparation. No training or model loading on import.
- [association.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/association.py) — Forward path-cover optimization: unknown long links are not cannot-links.
- [candidates.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/candidates.py) — The same label-blind ragged candidate generator for training and deployment.
- [contracts.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/contracts.py) — Geometry-only observations and explicit candidate/absence boundaries.
- [dataset.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/dataset.py) — Ragged graph-query tensor adapter; identical feature construction at deploy.
- [features.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/features.py) — Budgeted feature cache: one shared image/mask crop path for all labels.
- [metrics.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/metrics.py) — Full provided-observation association scoring with no query-only privilege.
- [model.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/model.py) — Untrained set-selection head over embeddings, variable K plus one dustbin.
- [policy.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/policy.py) — Download roles are checked before any fetch, decode, or payload statistics.
- [presence.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/presence.py) — Research presence wiring. Correspondence dustbin never implies absence.
- [review_replay.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/review_replay.py) — Recount frozen outputs and reproduce the old evaluator's oracle ceiling.
- [scoring.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/scoring.py) — Decode real candidates versus a separate correspondence dustbin.
- [simulation.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/simulation.py) — Named geometry-error simulations, never real birth/death observations.
- [sources.py](../research/temporal_tracking/xian_r5_deployment_candidate_alignment_20261027/taskalign/sources.py) — Read-only metadata adapters. No raster pixel reads, no weight loads.

### research/temporal_tracking/xian_r5_learned_fullspan_association_20261027

- [acquire_bt.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/acquire_bt.py) — Acquire only official TRAIN BT flow/change members; bounded HTTP ranges.
- [association_metrics.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/association_metrics.py) — Association-only IDF1/AssA with explicitly perfect, provided proposals.
- [build_delivery.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/build_delivery.py) — Evidence-linked research report, measured plots and product conservation QA.
- [check_bt_conventions.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/check_bt_conventions.py) — Exhaustive bounded coordinate-contract hypotheses on FIT labels only.
- [cross_epoch_metrics.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/cross_epoch_metrics.py) — Exclude same-image reference controls from cross-temporal pair metrics.
- [evaluate_source.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/evaluate_source.py) — Source-only association and event diagnostics; GT proposals are explicit.
- [evaluate_source_fullspan.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/evaluate_source_fullspan.py) — Source-only association and event diagnostics; GT proposals are explicit.
- [prepare_bt.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/prepare_bt.py) — BANDON BT-vector auxiliary supervision, never fabricate entity ID labels.
- [prepare_sn7.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/prepare_sn7.py) — Source-governed native SN7 identity sequences, no Validation/Holdout reads.
- [prepare_sn7_multiscale.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/prepare_sn7_multiscale.py) — Source-governed native SN7 identity sequences, no Validation/Holdout reads.
- [quarantine_inherited_holdout.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/quarantine_inherited_holdout.py) — Quarantine inherited FINAL_TEST auxiliary bytes acquired with official TRAIN.
- [run_fullspan_training.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/run_fullspan_training.py) — Finalize defined source coverage, freeze, and run three serial local seeds.
- [seal_delivery.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/seal_delivery.py) — Seal research artifacts and create a compact local handoff, no raw datasets.
- [temporal_inference.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/temporal_inference.py) — Cardinality-preserving context adapter; all candidates still get scored.
- [temporal_model.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/temporal_model.py) — One trainable RGB/mask encoder, geometry branch and bidirectional temporal GRU.
- [test_temporal_upgrade.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/test_temporal_upgrade.py) — test temporal upgrade · observation, edge, test_model_all_branches_receive_gradients, test_context_adapter_matches_training_when_three_candidates
- [train_bt.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/train_bt.py) — train bt · main
- [train_temporal.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/train_temporal.py) — Full-epoch training; immutable source groups; serial independent seeds.
- [train_temporal_fullspan.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/train_temporal_fullspan.py) — Full-epoch training; immutable source groups; serial independent seeds.
- [upgrade_application.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/upgrade_application.py) — Reuse the V1 app; source-bound rechecks and learned all-sequence association.
- [verify_bt_coordinates.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/verify_bt_coordinates.py) — Resolve BT coordinate direction against official ST footprint invariance.
- [verify_reloads.py](../research/temporal_tracking/xian_r5_learned_fullspan_association_20261027/verify_reloads.py) — Independent CPU reload against saved CUDA calibration outputs.

## 根目录

- [run.py](../run.py) — run

## scripts

平台调度和操作命令

- [apply_reviews.py](../scripts/apply_reviews.py) — apply reviews · main
- [audit_ucs_readiness.ps1](../scripts/audit_ucs_readiness.ps1) — audit ucs readiness
- [build_coco.py](../scripts/build_coco.py) — build coco · main
- [build_project_manifest.py](../scripts/build_project_manifest.py) — build project manifest · main
- [build_review_queue.py](../scripts/build_review_queue.py) — build review queue · main
- [building_foundation_v2.ps1](../scripts/building_foundation_v2.ps1) — building foundation v2
- [codex_windows_fallback_runner.ps1](../scripts/codex_windows_fallback_runner.ps1) — codex windows fallback runner
- [construction_task_training.ps1](../scripts/construction_task_training.ps1) — construction task training
- [diagnose_alignment.py](../scripts/diagnose_alignment.py) — diagnose alignment · main
- [download_cbers_yizhuang.py](../scripts/download_cbers_yizhuang.py) — download cbers yizhuang · post_stac_search, parse_item_datetime, pick_asset_href, month_score
- [evaluate_phase1.py](../scripts/evaluate_phase1.py) — evaluate phase1 · main
- [export_student_labels.py](../scripts/export_student_labels.py) — export student labels · main
- [generate_quality_masks.py](../scripts/generate_quality_masks.py) — generate quality masks · main
- [infer_timeseries.py](../scripts/infer_timeseries.py) — infer timeseries · main
- [launch_urrs_lite_review.ps1](../scripts/launch_urrs_lite_review.ps1) — launch urrs lite review
- [make_mock_predictions.py](../scripts/make_mock_predictions.py) — make mock predictions · main
- [make_synthetic_example.py](../scripts/make_synthetic_example.py) — make synthetic example · main
- [regularize_predictions.py](../scripts/regularize_predictions.py) — regularize predictions · main
- [run_construction_task_model.ps1](../scripts/run_construction_task_model.ps1) — run construction task model
- [run_nanjing_building_construction_integrated.ps1](../scripts/run_nanjing_building_construction_integrated.ps1) — run nanjing building construction integrated
- [run_nanjing_domain_adaptation_stable_tracking.ps1](../scripts/run_nanjing_domain_adaptation_stable_tracking.ps1) — run nanjing domain adaptation stable tracking
- [run_nanjing_jiangbei_acceptance.ps1](../scripts/run_nanjing_jiangbei_acceptance.ps1) — run nanjing jiangbei acceptance
- [run_nanjing_jiangbei_modular_rebuild.ps1](../scripts/run_nanjing_jiangbei_modular_rebuild.ps1) — run nanjing jiangbei modular rebuild
- [run_nanjing_stable_building_construction.ps1](../scripts/run_nanjing_stable_building_construction.ps1) — run nanjing stable building construction
- [run_nanjing_zoom21_current_best.ps1](../scripts/run_nanjing_zoom21_current_best.ps1) — run nanjing zoom21 current best
- [run_phase1_smoke.py](../scripts/run_phase1_smoke.py) — run phase1 smoke · main
- [run_tongzhou_construction_images_best.ps1](../scripts/run_tongzhou_construction_images_best.ps1) — run tongzhou construction images best
- [run_tongzhou_construction_task_model.ps1](../scripts/run_tongzhou_construction_task_model.ps1) — run tongzhou construction task model
- [second_building_mapping_review.py](../scripts/second_building_mapping_review.py) — second building mapping review · parts_lower, is_test, is_validation, is_label_candidate
- [surface_state_v1.ps1](../scripts/surface_state_v1.ps1) — surface state v1
- [surface_state_v1_continue_after_cwld.ps1](../scripts/surface_state_v1_continue_after_cwld.ps1) — surface state v1 continue after cwld
- [surface_state_v1_pipeline.ps1](../scripts/surface_state_v1_pipeline.ps1) — surface state v1 pipeline
- [surface_state_v1_post_cwld_chain.ps1](../scripts/surface_state_v1_post_cwld_chain.ps1) — surface state v1 post cwld chain
- [surface_state_v1_train.ps1](../scripts/surface_state_v1_train.ps1) — surface state v1 train
- [tongzhou_rsbuilding_geometry.ps1](../scripts/tongzhou_rsbuilding_geometry.ps1) — tongzhou rsbuilding geometry
- [track_states.py](../scripts/track_states.py) — track states · main
- [train_teacher.py](../scripts/train_teacher.py) — train teacher · main
- [urrs_construction.ps1](../scripts/urrs_construction.ps1) — urrs construction
- [urrs_review.ps1](../scripts/urrs_review.ps1) — urrs review
- [v2_bootstrap.ps1](../scripts/v2_bootstrap.ps1) — v2 bootstrap
- [validate_alignment.py](../scripts/validate_alignment.py) — validate alignment · main
- [validate_crs.py](../scripts/validate_crs.py) — validate crs · main
- [validate_schemas.py](../scripts/validate_schemas.py) — validate schemas · main
- [verify_second_building_mapping.py](../scripts/verify_second_building_mapping.py) — verify second building mapping · sha256_file, main

### scripts/windows

- [01_check_environment.ps1](../scripts/windows/01_check_environment.ps1) — 01 check environment
- [02_public_preflight.ps1](../scripts/windows/02_public_preflight.ps1) — 02 public preflight
- [03_public_quick.ps1](../scripts/windows/03_public_quick.ps1) — 03 public quick
- [04_public_full.ps1](../scripts/windows/04_public_full.ps1) — 04 public full
- [05_personal_preflight.ps1](../scripts/windows/05_personal_preflight.ps1) — 05 personal preflight
- [06_personal_all.ps1](../scripts/windows/06_personal_all.ps1) — 06 personal all
- [07_threshold_scan.ps1](../scripts/windows/07_threshold_scan.ps1) — 07 threshold scan
- [08_glm_ai_review.ps1](../scripts/windows/08_glm_ai_review.ps1) — 08 glm ai review
- [09_hard_finetune.ps1](../scripts/windows/09_hard_finetune.ps1) — 09 hard finetune
- [10_compare_models.ps1](../scripts/windows/10_compare_models.ps1) — 10 compare models

## src

核心包和可复用算法

### src/urban_monitor

- [__init__.py](../src/urban_monitor/__init__.py) — Urban renewal phase-1 starter.
- [alignment.py](../src/urban_monitor/alignment.py) — alignment · RasterSignature, inspect_raster, validate_alignment
- [alignment_diagnostics.py](../src/urban_monitor/alignment_diagnostics.py) — alignment diagnostics · diagnose_alignment, write_alignment_report

### src/urban_monitor/building_foundation_v2

- [__init__.py](../src/urban_monitor/building_foundation_v2/__init__.py) — URRS Building Foundation v2 static building candidate components.
- [checkpoint.py](../src/urban_monitor/building_foundation_v2/checkpoint.py) — Checkpoint eligibility and deterministic selection helpers.
- [data.py](../src/urban_monitor/building_foundation_v2/data.py) — WHU Building data adapter with a hard guard against Test discovery.
- [losses.py](../src/urban_monitor/building_foundation_v2/losses.py) — Multitask losses with finite-value protection.
- [metrics.py](../src/urban_monitor/building_foundation_v2/metrics.py) — Finite segmentation/object metrics for static Building Foundation v2.
- [model.py](../src/urban_monitor/building_foundation_v2/model.py) — Small, shape-compatible static building model used by the v2 extension.

### src/urban_monitor

- [catalog.py](../src/urban_monitor/catalog.py) — Authoritative registry and artifact catalog for the urban-renewal project.

### src/urban_monitor/change

- [__init__.py](../src/urban_monitor/change/__init__.py) — Independent bi-temporal change-teacher branch (v0.5).

### src/urban_monitor/change/adapters

- [__init__.py](../src/urban_monitor/change/adapters/__init__.py) — adapters 包入口
- [base.py](../src/urban_monitor/change/adapters/base.py) — base · split_directories, named_directory, files_by_stem, require_same_stems
- [dsifn.py](../src/urban_monitor/change/adapters/dsifn.py) — dsifn · DSIFNAdapter
- [s2looking.py](../src/urban_monitor/change/adapters/s2looking.py) — s2looking · S2LookingAdapter
- [second.py](../src/urban_monitor/change/adapters/second.py) — second · load_color_mapping, semantic_ids, SECONDAdapter

### src/urban_monitor/change

- [ai_review.py](../src/urban_monitor/change/ai_review.py) — ai review · run_ai_review
- [augmentations.py](../src/urban_monitor/change/augmentations.py) — augmentations · ChangePairTransform
- [checkpoint_merge_v2.py](../src/urban_monitor/change/checkpoint_merge_v2.py) — checkpoint merge v2 · MergeRecord, merge_v2_checkpoints
- [cli.py](../src/urban_monitor/change/cli.py) — cli · change_doctor, build_parser, main
- [compare.py](../src/urban_monitor/change/compare.py) — compare · compare_runs
- [corrections.py](../src/urban_monitor/change/corrections.py) — corrections · sha256, save_correction, apply_review_corrections
- [datasets.py](../src/urban_monitor/change/datasets.py) — datasets · PublicChangeDataset, change_collate
- [diagnostics.py](../src/urban_monitor/change/diagnostics.py) — diagnostics · binary_stage_diagnostic
- [evaluation.py](../src/urban_monitor/change/evaluation.py) — evaluation · evaluate_change_checkpoint
- [evaluation_cache.py](../src/urban_monitor/change/evaluation_cache.py) — evaluation cache · sha256_file, validation_probabilities
- [final_test.py](../src/urban_monitor/change/final_test.py) — One-time, fixed-threshold evaluation of the held-out change-detection test set.
- [fusion.py](../src/urban_monitor/change/fusion.py) — fusion · fuse_teachers
- [fusion_v2.py](../src/urban_monitor/change/fusion_v2.py) — fusion v2 · FusionResult, fuse_teacher_outputs
- [geospatial.py](../src/urban_monitor/change/geospatial.py) — geospatial · RasterGrid, grid_from_dataset, same_grid, aligned_pair
- [hard_examples.py](../src/urban_monitor/change/hard_examples.py) — hard examples · build_hard_examples, build_demolition_head_sampling_index
- [hard_examples_v2.py](../src/urban_monitor/change/hard_examples_v2.py) — hard examples v2 · mine_v2_hard_examples
- [inference_v2.py](../src/urban_monitor/change/inference_v2.py) — inference v2 · run_urban_renewal_inference
- [joint_review.py](../src/urban_monitor/change/joint_review.py) — joint review · build_joint_review, apply_joint_review
- [local_labels.py](../src/urban_monitor/change/local_labels.py) — local labels · generate_local_change_candidates
- [losses.py](../src/urban_monitor/change/losses.py) — losses · binary_dice_loss, multiclass_dice_loss, boundary_target, boundary_bce_loss
- [manual_diagnostic_review.py](../src/urban_monitor/change/manual_diagnostic_review.py) — Offline DSIFN final-test diagnostic review; deliberately has no model dependency.
- [metrics.py](../src/urban_monitor/change/metrics.py) — metrics · ChangeMetricAccumulator

### src/urban_monitor/change/models

- [__init__.py](../src/urban_monitor/change/models/__init__.py) — models 包入口
- [changemamba_adapter.py](../src/urban_monitor/change/models/changemamba_adapter.py) — changemamba adapter · build_changemamba
- [siamese_unet.py](../src/urban_monitor/change/models/siamese_unet.py) — siamese unet · ConvBlock, MultiTaskSiameseUNetBase, build_change_model

### src/urban_monitor/change

- [mudi_v2.py](../src/urban_monitor/change/mudi_v2.py) — mudi v2 · write_mudi_features
- [optimization.py](../src/urban_monitor/change/optimization.py) — optimization · profile_training, audit_training_signal, audit_gradients, benchmark_loader
- [preflight.py](../src/urban_monitor/change/preflight.py) — preflight · run_change_preflight
- [preflight_v2.py](../src/urban_monitor/change/preflight_v2.py) — preflight v2 · run_v2_preflight, format_v2_preflight
- [preparation.py](../src/urban_monitor/change/preparation.py) — preparation · create_change_run, prepare_public_change, normalized_for
- [preview_v2.py](../src/urban_monitor/change/preview_v2.py) — preview v2 · build_preview
- [registry.py](../src/urban_monitor/change/registry.py) — registry · load_change_config, enabled_adapters
- [review.py](../src/urban_monitor/change/review.py) — review · build_static_review
- [review_app.py](../src/urban_monitor/change/review_app.py) — review app · launch_review_app
- [rsbuilding_adapter.py](../src/urban_monitor/change/rsbuilding_adapter.py) — rsbuilding adapter · load_rsbuilding_exports
- [runtime_config.py](../src/urban_monitor/change/runtime_config.py) — Shared runtime normalization for non-training change-model consumers.
- [samplers.py](../src/urban_monitor/change/samplers.py) — samplers · DatasetBalancedSamplerBase, DemolitionPatchCompositionSampler, V2DirectionalSampler
- [schema.py](../src/urban_monitor/change/schema.py) — schema · ChangeSample, validate_sample
- [second_dev_split.py](../src/urban_monitor/change/second_dev_split.py) — Frozen, auditable development validation split for SECOND Train only.
- [splits.py](../src/urban_monitor/change/splits.py) — splits · resolve_train_validation_test_splits, quick_subset
- [supervision_v2.py](../src/urban_monitor/change/supervision_v2.py) — supervision v2 · supervision_matrix, active_heads, TestReadReceipt, TestReadGuard
- [teacher_registry.py](../src/urban_monitor/change/teacher_registry.py) — teacher registry · TeacherStatus, TeacherMetadata, TeacherOutputs, sha256_file
- [thresholds.py](../src/urban_monitor/change/thresholds.py) — thresholds · scan_thresholds
- [train_review.py](../src/urban_monitor/change/train_review.py) — train review · generate_train_review
- [training.py](../src/urban_monitor/change/training.py) — training · seed_everything, resolve_device, stage_loss_config, configure_demolition_head_only
- [vectorize_v2.py](../src/urban_monitor/change/vectorize_v2.py) — vectorize v2 · vectorize_mask
- [visualization.py](../src/urban_monitor/change/visualization.py) — visualization · build_training_visualizations

### src/urban_monitor

- [cli.py](../src/urban_monitor/cli.py) — cli · file_sha256, compute_data_hash, complete_run, fail_run
- [coco.py](../src/urban_monitor/coco.py) — coco · build_coco, build_spatial_coco
- [config.py](../src/urban_monitor/config.py) — config · merge_config, load_config, load_project_config, resolve_path

### src/urban_monitor/contracts

- [__init__.py](../src/urban_monitor/contracts/__init__.py) — contracts 包入口
- [model_output.py](../src/urban_monitor/contracts/model_output.py) — Canonical prediction contract; unsupported outputs are explicit, never fake zeros.

### src/urban_monitor

- [coordinate_encoding.py](../src/urban_monitor/coordinate_encoding.py) — coordinate encoding · validate_coordinate_encoding, require_training_safe_encoding, conversion_provenance
- [crs.py](../src/urban_monitor/crs.py) — crs · validate_source_crs, get_aoi_centroid_lonlat, select_local_utm_crs, ensure_metric_crs
- [dataset.py](../src/urban_monitor/dataset.py) — dataset · CocoBuildingDataset, collate_fn

### src/urban_monitor/datasets

- [__init__.py](../src/urban_monitor/datasets/__init__.py) — datasets 包入口

### src/urban_monitor/datasets/adapters

- [__init__.py](../src/urban_monitor/datasets/adapters/__init__.py) — adapters 包入口
- [base.py](../src/urban_monitor/datasets/adapters/base.py) — Dataset onboarding protocol for single-date, paired and temporal sources.
- [beijing_waste_adapter.py](../src/urban_monitor/datasets/adapters/beijing_waste_adapter.py) — beijing waste adapter · BeijingWasteAdapter
- [generic.py](../src/urban_monitor/datasets/adapters/generic.py) — generic · PatternDatasetAdapter
- [generic_mti_adapter.py](../src/urban_monitor/datasets/adapters/generic_mti_adapter.py) — generic mti adapter · GenericMTIAdapter
- [generic_tfr_adapter.py](../src/urban_monitor/datasets/adapters/generic_tfr_adapter.py) — generic tfr adapter · GenericTFRAdapter

### src/urban_monitor/datasets

- [onboarding.py](../src/urban_monitor/datasets/onboarding.py) — Metadata-first dataset onboarding; no Test pixels and no invented label semantics.

### src/urban_monitor

- [downscale.py](../src/urban_monitor/downscale.py) — downscale · export_student_change_labels
- [evaluation.py](../src/urban_monitor/evaluation.py) — evaluation · evaluate_phase1
- [geometry.py](../src/urban_monitor/geometry.py) — geometry · polygon_iou, intersection_over_area, buffered_iou, bbox_iou
- [ground_truth.py](../src/urban_monitor/ground_truth.py) — ground truth · read_building_labels, GroundTruthResult, normalize_label_frame, normalize_label_file
- [inference.py](../src/urban_monitor/inference.py) — inference · regularize_prediction_frame, infer_raster, infer_full_timeseries
- [manifest.py](../src/urban_monitor/manifest.py) — manifest · DateRecord, load_manifest
- [model.py](../src/urban_monitor/model.py) — model · checkpoint_sha256, build_model
- [outputs.py](../src/urban_monitor/outputs.py) — outputs · prepare_output
- [preflight.py](../src/urban_monitor/preflight.py) — preflight · PreflightError, run_personal_preflight, run_preflight
- [preparation.py](../src/urban_monitor/preparation.py) — preparation · prepare_project
- [project_discovery.py](../src/urban_monitor/project_discovery.py) — project discovery · DiscoveredProject, parse_date_from_filename, discover_images, discover_labels
- [provenance.py](../src/urban_monitor/provenance.py) — provenance · file_sha256, git_commit, utc_now, write_run_manifest

### src/urban_monitor/providers

- [__init__.py](../src/urban_monitor/providers/__init__.py) — Imagery providers kept outside temporal business logic.
- [local_raster.py](../src/urban_monitor/providers/local_raster.py) — local raster · inspect_local_raster

### src/urban_monitor

- [quality.py](../src/urban_monitor/quality.py) — quality · geometry_quality_fraction, generate_quality_mask, compose_quality_masks
- [review.py](../src/urban_monitor/review.py) — review · build_review_queue, apply_reviews

### src/urban_monitor/safety

- [__init__.py](../src/urban_monitor/safety/__init__.py) — safety 包入口
- [test_guard.py](../src/urban_monitor/safety/test_guard.py) — Code-level holdout protection shared by onboarding and future training APIs.

### src/urban_monitor

- [schema.py](../src/urban_monitor/schema.py) — schema · validate_manifest_schema, validate_annotation_schema, validate_prediction_schema, validate_tracking_schema
- [splitting.py](../src/urban_monitor/splitting.py) — splitting · build_spatial_blocks, assign_geometry_splits, split_for_tile, write_split_report

### src/urban_monitor/surface_state

- [__init__.py](../src/urban_monitor/surface_state/__init__.py) — surface_state 包入口
- [audit.py](../src/urban_monitor/surface_state/audit.py) — Read-only FUSU/CWLD discovery, pairing, label and leakage audits.
- [codec.py](../src/urban_monitor/surface_state/codec.py) — Native label codecs built only from observed Train label values.
- [metrics.py](../src/urban_monitor/surface_state/metrics.py) — metrics · surface_state_metrics
- [model.py](../src/urban_monitor/surface_state/model.py) — Surface-state model interface: background, vacant, construction, building, unknown.
- [training.py](../src/urban_monitor/surface_state/training.py) — Training entrypoint guard for future verified surface-state datasets.

### src/urban_monitor/temporal

- [__init__.py](../src/urban_monitor/temporal/__init__.py) — temporal 包入口
- [schemas.py](../src/urban_monitor/temporal/schemas.py) — schemas · QualityState, TemporalState, TemporalObservation, StateSegment
- [state_engine.py](../src/urban_monitor/temporal/state_engine.py) — state engine · TemporalStateEngine
- [tracking.py](../src/urban_monitor/temporal/tracking.py) — Object-level candidate matching with spatial indexing and explicit lineage.
- [transitions.py](../src/urban_monitor/temporal/transitions.py) — transitions · derive_transition

### src/urban_monitor

- [tracking.py](../src/urban_monitor/tracking.py) — tracking · Observation, assign_tracks, repair_short_gaps, repair_isolated_gaps
- [training.py](../src/urban_monitor/training.py) — training · resolve_runtime, evaluate_model, train_teacher
- [unified_inference.py](../src/urban_monitor/unified_inference.py) — Canonical run layout and manifest; absent modules remain pending/unsupported.
- [validation.py](../src/urban_monitor/validation.py) — validation · evaluate_checkpoint

### src/urban_renewal

- [__init__.py](../src/urban_renewal/__init__.py) — Urban Renewal Remote Sensing System (URRS) canonical package.
- [cli.py](../src/urban_renewal/cli.py) — cli · models_verify, system_status, inventory, registry_validate

### src/urban_renewal/construction

- [__init__.py](../src/urban_renewal/construction/__init__.py) — Construction activity evidence, parcel and lifecycle services.
- [aggregation.py](../src/urban_renewal/construction/aggregation.py) — aggregation · aggregate_site_mask
- [area_accounting.py](../src/urban_renewal/construction/area_accounting.py) — area accounting · AreaAudit, is_metric_crs, audit_area, lifecycle_areas
- [audit.py](../src/urban_renewal/construction/audit.py) — audit · discover_data, audit_data
- [audit_v2.py](../src/urban_renewal/construction/audit_v2.py) — audit v2 · audit_data
- [cli.py](../src/urban_renewal/construction/cli.py) — cli · add_construction_parser, handle_construction
- [datasets.py](../src/urban_renewal/construction/datasets.py) — datasets · UCSCodec, UCSManifestDataset, CWLDConstructionDataset
- [formal_readiness.py](../src/urban_renewal/construction/formal_readiness.py) — formal readiness · build_formal_training_readiness
- [hard_negatives.py](../src/urban_renewal/construction/hard_negatives.py) — hard negatives · eligible_hard_negative
- [json_io.py](../src/urban_renewal/construction/json_io.py) — json io · load_json, atomic_write_text, dump_json
- [ontology.py](../src/urban_renewal/construction/ontology.py) — ontology · evidence_projection
- [parcel_features.py](../src/urban_renewal/construction/parcel_features.py) — parcel features · calculate_parcel_features
- [partial_labels.py](../src/urban_renewal/construction/partial_labels.py) — partial labels · multiclass_dice_loss, boundary_target, PartialLabelLoss
- [recovery_audit.py](../src/urban_renewal/construction/recovery_audit.py) — recovery audit · audit_false_recovery_candidates
- [scene_rules.py](../src/urban_renewal/construction/scene_rules.py) — scene rules · classify_scene
- [spatial_context.py](../src/urban_renewal/construction/spatial_context.py) — spatial context · distance_feature, urban_context
- [task_model.py](../src/urban_renewal/construction/task_model.py) — task model · load_task_model, infer_task_model
- [ucs386_gate.py](../src/urban_renewal/construction/ucs386_gate.py) — ucs386 gate · build_ucs386_subset_gate
- [vectorization.py](../src/urban_renewal/construction/vectorization.py) — vectorization · repair_geometry, vectorize_mask

### src/urban_renewal/core

- [__init__.py](../src/urban_renewal/core/__init__.py) — core 包入口
- [hashing.py](../src/urban_renewal/core/hashing.py) — hashing · sha256_file, canonical_json_hash
- [receipts.py](../src/urban_renewal/core/receipts.py) — receipts · utc_now, write_json, read_json
- [types.py](../src/urban_renewal/core/types.py) — types · SceneRecord, AlignmentReport, PipelineReceipt, SystemState

### src/urban_renewal/geometry

- [__init__.py](../src/urban_renewal/geometry/__init__.py) — geometry 包入口
- [service.py](../src/urban_renewal/geometry/service.py) — service · audit_geometry, vectorize_geometry, audit_alignment, geometry_status

### src/urban_renewal/inference

- [__init__.py](../src/urban_renewal/inference/__init__.py) — inference 包入口
- [overlay.py](../src/urban_renewal/inference/overlay.py) — overlay · run_geometry_overlay

### src/urban_renewal/models/construction

- [__init__.py](../src/urban_renewal/models/construction/__init__.py) — construction 包入口
- [boundary_head.py](../src/urban_renewal/models/construction/boundary_head.py) — boundary head · BoundaryHead
- [checkpoint.py](../src/urban_renewal/models/construction/checkpoint.py) — checkpoint · save_construction_checkpoint, load_construction_checkpoint
- [dual_branch.py](../src/urban_renewal/models/construction/dual_branch.py) — dual branch
- [element_model.py](../src/urban_renewal/models/construction/element_model.py) — element model · ConstructionElementNet
- [encoder.py](../src/urban_renewal/models/construction/encoder.py) — encoder · ConvBlock, DualBranchEncoder
- [native_heads.py](../src/urban_renewal/models/construction/native_heads.py) — native heads · NativeSegmentationHead, UnifiedEvidenceProjector
- [parcel_classifier.py](../src/urban_renewal/models/construction/parcel_classifier.py) — parcel classifier · ConstructionParcelClassifier
- [predictor.py](../src/urban_renewal/models/construction/predictor.py) — predictor · ConstructionPredictor
- [uncertainty_head.py](../src/urban_renewal/models/construction/uncertainty_head.py) — uncertainty head · UncertaintyHead

### src/urban_renewal/modular

- [__init__.py](../src/urban_renewal/modular/__init__.py) — Modular, inference-only URRS orchestration contracts.
- [contracts.py](../src/urban_renewal/modular/contracts.py) — Immutable data contracts for the URRS modular rebuild.

### src/urban_renewal/pipelines

- [construction_component_smoke.py](../src/urban_renewal/pipelines/construction_component_smoke.py) — construction component smoke · run_component_smoke
- [construction_external_pipeline.py](../src/urban_renewal/pipelines/construction_external_pipeline.py) — construction external pipeline · run_external_construction
- [construction_temporal_pipeline.py](../src/urban_renewal/pipelines/construction_temporal_pipeline.py) — construction temporal pipeline · temporal_status
- [construction_tongzhou_pipeline.py](../src/urban_renewal/pipelines/construction_tongzhou_pipeline.py) — construction tongzhou pipeline · run_tongzhou_construction, tongzhou_status
- [construction_training_pipeline.py](../src/urban_renewal/pipelines/construction_training_pipeline.py) — construction training pipeline · run_dataloader_smoke, run_20_step_smoke, formal_train_guard
- [national_screening_pipeline.py](../src/urban_renewal/pipelines/national_screening_pipeline.py) — national screening pipeline · NationalConstructionScreening

### src/urban_renewal/reporting

- [human_review.py](../src/urban_renewal/reporting/human_review.py) — human review

### src/urban_renewal/review

- [__init__.py](../src/urban_renewal/review/__init__.py) — review 包入口
- [app.py](../src/urban_renewal/review/app.py) — app · prepare
- [construction_corrections.py](../src/urban_renewal/review/construction_corrections.py) — construction corrections · export_qgis_corrections
- [construction_inventory.py](../src/urban_renewal/review/construction_inventory.py) — construction inventory · build_construction_inventory
- [construction_inventory_v2.py](../src/urban_renewal/review/construction_inventory_v2.py) — construction inventory v2 · build_construction_inventory
- [construction_schema.py](../src/urban_renewal/review/construction_schema.py) — construction schema · normalize_construction_decision
- [construction_store.py](../src/urban_renewal/review/construction_store.py) — construction store · ConstructionReviewStore
- [finalization.py](../src/urban_renewal/review/finalization.py) — finalization · finalize_review
- [inventory.py](../src/urban_renewal/review/inventory.py) — inventory · build_inventory
- [preview.py](../src/urban_renewal/review/preview.py) — preview · generate_previews
- [review_migration.py](../src/urban_renewal/review/review_migration.py) — review migration · migrate_existing_review
- [review_schema.py](../src/urban_renewal/review/review_schema.py) — review schema · normalize_decision
- [review_store.py](../src/urban_renewal/review/review_store.py) — review store · ReviewStore
- [reviewed_regions.py](../src/urban_renewal/review/reviewed_regions.py) — reviewed regions · write_reviewed_region_index
- [security.py](../src/urban_renewal/review/security.py) — security · safe_path, allowed_origin
- [seed_temporal.py](../src/urban_renewal/review/seed_temporal.py) — seed temporal · create_seed_temporal_inventory, build_seed_temporal_inventory, SeedTemporalLabelStore
- [server.py](../src/urban_renewal/review/server.py) — server · ReviewServer, serve
- [transition_inventory.py](../src/urban_renewal/review/transition_inventory.py) — transition inventory · build_transition_track_inventory
- [transition_preview.py](../src/urban_renewal/review/transition_preview.py) — transition preview · diagnose_transition_previews, regenerate_transition_previews
- [transition_preview_v2.py](../src/urban_renewal/review/transition_preview_v2.py) — transition preview v2 · diagnose_transition_previews, regenerate_transition_previews
- [transition_validation.py](../src/urban_renewal/review/transition_validation.py) — transition validation · PreviewQuality, validate_preview
- [transition_workflow.py](../src/urban_renewal/review/transition_workflow.py) — transition workflow · run_transition_review_migration

### src/urban_renewal/temporal

- [construction_change_head.py](../src/urban_renewal/temporal/construction_change_head.py) — construction change head · ConstructionChangeHead
- [construction_labeler.py](../src/urban_renewal/temporal/construction_labeler.py) — construction labeler · pseudo_label, promote_human_label
- [construction_mti.py](../src/urban_renewal/temporal/construction_mti.py) — construction mti · ConstructionMTI
- [construction_no_mti.py](../src/urban_renewal/temporal/construction_no_mti.py) — construction no mti · ConstructionNoMTI
- [construction_state_graph.py](../src/urban_renewal/temporal/construction_state_graph.py) — construction state graph · ConstructionStateGraph
- [construction_tfr.py](../src/urban_renewal/temporal/construction_tfr.py) — construction tfr · ConstructionTemporalFeatureService
- [construction_tracks.py](../src/urban_renewal/temporal/construction_tracks.py) — construction tracks · summarize_track

### src/urban_renewal/tracking

- [construction_lifecycle.py](../src/urban_renewal/tracking/construction_lifecycle.py) — construction lifecycle
- [construction_matching.py](../src/urban_renewal/tracking/construction_matching.py) — construction matching · match_construction_objects
- [split_merge.py](../src/urban_renewal/tracking/split_merge.py) — split merge · detect_split_merge

### src/xian_r5

- [README.md](../src/xian_r5/README.md) — README
- [__init__.py](../src/xian_r5/__init__.py) — Isolated R5 candidate components for Xi'an three-epoch identity work.
- [acceptance.py](../src/xian_r5/acceptance.py) — S4.14 read-only acceptance primitives; never fit or apply a transform.

### src/xian_r5/alignment

- [__init__.py](../src/xian_r5/alignment/__init__.py) — Image-source and registration evidence for R5.
- [adaptive_local.py](../src/xian_r5/alignment/adaptive_local.py) — GT-free image-grid baseline; unverified candidates never become roof GT.
- [base_tie.py](../src/xian_r5/alignment/base_tie.py) — Local image-to-image tie candidates on an already shared raster grid.
- [common.py](../src/xian_r5/alignment/common.py) — Small dependency-light helpers used by the isolated R5 pipeline.
- [comparison.py](../src/xian_r5/alignment/comparison.py) — Candidate-only comparison of alternate registration paths into one grid.
- [component_tie_consensus.py](../src/xian_r5/alignment/component_tie_consensus.py) — Export exact-source-component, multi-reference tie diagnostics.
- [composition.py](../src/xian_r5/alignment/composition.py) — Compose local actual-date imagery ties with annual-base pose candidates.
- [feature_recovery.py](../src/xian_r5/alignment/feature_recovery.py) — Mask attribution and explicit ground-evidence gates; no GT or saved shifts.
- [field.py](../src/xian_r5/alignment/field.py) — Traceable local image-registration fields with a fail-closed apply gate.
- [field_io.py](../src/xian_r5/alignment/field_io.py) — Read traceable image-alignment candidates from paired receipts and vectors.
- [frozen_candidate_audit.py](../src/xian_r5/alignment/frozen_candidate_audit.py) — Fixed-cohort, frozen-vector numerical replay; not independent validation.
- [ground_registration.py](../src/xian_r5/alignment/ground_registration.py) — SIFT ground-registration baseline for R5.
- [ground_trial.py](../src/xian_r5/alignment/ground_trial.py) — Explicit ground candidate support and whole-structure independence.
- [image_block_replay.py](../src/xian_r5/alignment/image_block_replay.py) — Spatial-block consistency of one frozen image-vector, never a metric bound.
- [instance_adaptive.py](../src/xian_r5/alignment/instance_adaptive.py) — Instance alignment admission seam; missing learned weights never imply zero.
- [local_field.py](../src/xian_r5/alignment/local_field.py) — Estimate read-only, cell-local displacement candidates from masked ties.
- [mosaic_grid.py](../src/xian_r5/alignment/mosaic_grid.py) — Exact source-tile to stitched-raster grid mapping for Xi'an z20 mosaics.
- [numerical_closure.py](../src/xian_r5/alignment/numerical_closure.py) — Convention and same-support design audits, never an alignment acceptance adapter.
- [point_roof_alignment.py](../src/xian_r5/alignment/point_roof_alignment.py) — Point-RGB fixed-source replay; experimental image-grid evidence only.
- [pose_graph.py](../src/xian_r5/alignment/pose_graph.py) — Solve a local translation graph without promoting its coordinates.
- [preview.py](../src/xian_r5/alignment/preview.py) — Render explicitly unaccepted image-grid ties as review-only raster previews.
- [registry.py](../src/xian_r5/alignment/registry.py) — Actual-date/source registry for the isolated Xi'an R5 candidate.
- [relative_offsets.py](../src/xian_r5/alignment/relative_offsets.py) — Summarize spatially local, relative building-shift candidates.
- [relative_roof_candidate_products.py](../src/xian_r5/alignment/relative_roof_candidate_products.py) — Export spatially local relative-roof alignment as candidate-only products.
- [review_packet.py](../src/xian_r5/alignment/review_packet.py) — Export a fail-closed, traceable package for human alignment review.
- [reviewed_field.py](../src/xian_r5/alignment/reviewed_field.py) — Fail-closed human review adapter for local image-grid tie cells.
- [roof_candidate.py](../src/xian_r5/alignment/roof_candidate.py) — Image-grid roof previews and fail-closed reviewed alignment application.
- [roof_overlay_preview.py](../src/xian_r5/alignment/roof_overlay_preview.py) — Transparent, source-clipped roof envelope visualization; never training GT.
- [source_template_alignment.py](../src/xian_r5/alignment/source_template_alignment.py) — Source-anchored equal-population search against reference image context.
- [spatial_composition.py](../src/xian_r5/alignment/spatial_composition.py) — Compose spatially overlapping actual-date ties into an annual-base grid.
- [subpixel_phase.py](../src/xian_r5/alignment/subpixel_phase.py) — Fixed-grid local inverse DFT refinement after the integer peak is removed.
- [supported_ground_features.py](../src/xian_r5/alignment/supported_ground_features.py) — Support-closed directional features and internal spatial-check candidates.

### src/xian_r5

- [application.py](../src/xian_r5/application.py) — XIAN-R5 application V1: serial local inference and explicit candidate products.
- [application_contract.py](../src/xian_r5/application_contract.py) — Application contracts and read-only, content-addressed stage commits.
- [application_geometry.py](../src/xian_r5/application_geometry.py) — Metric geometry adapters; classes, coverage and unknown remain separate.
- [application_models.py](../src/xian_r5/application_models.py) — Inference-only serial workers. Never construct an optimizer or read labels.
- [application_temporal.py](../src/xian_r5/application_temporal.py) — R4 candidate tracking, bounded rechecks and existing R5 formal reconstruction.
- [cached_roof_products.py](../src/xian_r5/cached_roof_products.py) — Minimal sealed B0/V2 -> existing I interchange adapter; no model execution.

### src/xian_r5/cloud_joint

- [alignment.py](../src/xian_r5/cloud_joint/alignment.py) — Small GT-free learned alignment candidate; no production admission implied.
- [segmentation.py](../src/xian_r5/cloud_joint/segmentation.py) — Native B0 roof + independent facade auxiliary adapter (torch only).

### src/xian_r5

- [coherent_support.py](../src/xian_r5/coherent_support.py) — Derived union of edge-adjacent supports sharing one frozen experimental field.
- [fraction_shadow.py](../src/xian_r5/fraction_shadow.py) — Exact polygon-intersection shadow fractions; never supervision permission.
- [instance_evidence.py](../src/xian_r5/instance_evidence.py) — Unshifted source-component/legacy-object overlay, not identity acceptance.
- [integration.py](../src/xian_r5/integration.py) — Versioned handoff of existing R5 artifacts; evidence gates stay upstream.
- [label_transfer.py](../src/xian_r5/label_transfer.py) — Shadow-only label transport: exact target-grid fractions, never ground GT.
- [manual_roof_reference.py](../src/xian_r5/manual_roof_reference.py) — User-confirmed manual roof reference, without inventing metric accuracy.
- [native_fragment_evidence.py](../src/xian_r5/native_fragment_evidence.py) — Native-grid serialization/strip-fragment evidence, never ground calibration.

### src/xian_r5/reconstruction

- [__init__.py](../src/xian_r5/reconstruction/__init__.py) — Evidence-gated, actual-date building reconstruction.
- [core.py](../src/xian_r5/reconstruction/core.py) — Small public contract for date, geometry, identity and presence evidence.
- [external_candidates.py](../src/xian_r5/reconstruction/external_candidates.py) — Load external polygons as explicitly unverified footprint candidates.
- [ground_review_packet.py](../src/xian_r5/reconstruction/ground_review_packet.py) — Build a bounded, read-only-source packet for human roof/ground review.
- [products.py](../src/xian_r5/reconstruction/products.py) — One EPSG:32649 grid for ground footprints, IDs, changes and spatial QA.
- [rbf_sam_candidates.py](../src/xian_r5/reconstruction/rbf_sam_candidates.py) — Ingest RBF-SAM roof-mask/offset proposals as unverified ground candidates.
- [rbf_sam_handoff.py](../src/xian_r5/reconstruction/rbf_sam_handoff.py) — Prepare a fixed-case RBF-SAM bundle for the reconstruction importer.
- [rbf_sam_native.py](../src/xian_r5/reconstruction/rbf_sam_native.py) — Bridge upstream RBF-SAM JSON predictions into the audited R5 bundle.
- [registration.py](../src/xian_r5/reconstruction/registration.py) — Ground-masked tile registration with explicit candidate/accepted separation.
- [registration_worker.py](../src/xian_r5/reconstruction/registration_worker.py) — Subprocess adapter for the optional OpenCV registration implementation.
- [reviewed_absences.py](../src/xian_r5/reconstruction/reviewed_absences.py) — Fail-closed adapter for reviewed full-coverage absence evidence.
- [reviewed_candidates.py](../src/xian_r5/reconstruction/reviewed_candidates.py) — Attach human-drawn ground-boundary candidates without promoting them.
- [reviewed_footprints.py](../src/xian_r5/reconstruction/reviewed_footprints.py) — Fail-closed input adapter for independently reviewed ground footprints.
- [reviewed_identities.py](../src/xian_r5/reconstruction/reviewed_identities.py) — Fail-closed adapter for identity review independent of ground geometry.
- [source_dates.py](../src/xian_r5/reconstruction/source_dates.py) — Resolve candidate source dates from exact acquisition dates of source tiles.

### src/xian_r5

- [reference_core_cohort.py](../src/xian_r5/reference_core_cohort.py) — Fixed source-position reference corroboration, not offset-shaped target masks.
- [reference_label_filter.py](../src/xian_r5/reference_label_filter.py) — Conservative manual-reference corroboration of an existing pseudo-label.
- [reference_preflight.py](../src/xian_r5/reference_preflight.py) — Reference readiness census, not matching, split selection or GT admission.

### src/xian_r5/resolver

- [__init__.py](../src/xian_r5/resolver/__init__.py) — Evidence-resolver cascade components.
- [merged_mask_resolver.py](../src/xian_r5/resolver/merged_mask_resolver.py) — Track-guided merged-mask resolver contract.

### src/xian_r5

- [roof_group_correspondence.py](../src/xian_r5/roof_group_correspondence.py) — Direct-overlap reference groups, not physical identities or registration.
- [roof_identity.py](../src/xian_r5/roof_identity.py) — Non-mutating, multi-feature identity hypotheses, never physical-ID creation.
- [roof_observation.py](../src/xian_r5/roof_observation.py) — Model-digitized source roof surfaces with exact pixel/coordinate provenance.
- [roof_pair_translation.py](../src/xian_r5/roof_pair_translation.py) — Internal translation consistency of frozen roof-shape hypotheses.
- [roof_reference_candidates.py](../src/xian_r5/roof_reference_candidates.py) — Frozen, geometry-only roof correspondences; not ground ties or physical IDs.
- [sam_source_candidates.py](../src/xian_r5/sam_source_candidates.py) — Exact source-frame polygonization of unselected official-SAM masks.
- [sam_source_prompts.py](../src/xian_r5/sam_source_prompts.py) — Source-only segmentation prompts, explicitly never registration ties.
- [shadow_pipeline.py](../src/xian_r5/shadow_pipeline.py) — Live RGB adapter over existing A/I/Presence/fraction seams, shadow-only.
- [single_date_identity.py](../src/xian_r5/single_date_identity.py) — Evidence gates for observation-to-existing-object provenance only.
- [single_year_experiment.py](../src/xian_r5/single_year_experiment.py) — User-authorized nominal-year experiment, separate from measured GT gates.
- [source_mask_retention.py](../src/xian_r5/source_mask_retention.py) — Explicit source-image mask retention, separate from formal roof admission.
- [source_partition.py](../src/xian_r5/source_partition.py) — Source-specific candidate transport, never a registration approval gate.
- [source_prompt_refinement.py](../src/xian_r5/source_prompt_refinement.py) — Bounded source-only SAM prompt correction, not GT-guided geometry editing.
- [source_roof_batch_scope.py](../src/xian_r5/source_roof_batch_scope.py) — Complete-ROI source-only SAM proposal worklist, no GT-based selection.

### src/xian_r5/tracking

- [__init__.py](../src/xian_r5/tracking/__init__.py) — Track-aware candidate identity components.

### src/xian_r5/tracking/tbia

- [__init__.py](../src/xian_r5/tracking/tbia/__init__.py) — Small, no-training TBIA prototype used before any model training.
- [attention.py](../src/xian_r5/tracking/tbia/attention.py) — Handcrafted local cross-attention prototype for R5 Stage 1.
- [features.py](../src/xian_r5/tracking/tbia/features.py) — Track-query and neighborhood features for the R5 no-training prototype.

### src/xian_r5/training

- [__init__.py](../src/xian_r5/training/__init__.py) — Training contracts kept separate from inference-only R5 Stage 1.

### src/xian_r5/training/temporal_consistency

- [__init__.py](../src/xian_r5/training/temporal_consistency/__init__.py) — SemiBCD-inspired consistency metadata, not a replacement change detector.
- [consistency.py](../src/xian_r5/training/temporal_consistency/consistency.py) — Uncertainty-aware temporal consistency utilities for R5.

### src/xian_r5

- [transfer_admission.py](../src/xian_r5/transfer_admission.py) — Whole-window disposition for experimental roof transfer, not formal GT.

## tests

单元和集成测试

- [test_alignment.py](../tests/test_alignment.py) — test alignment · test_alignment_diagnostic_dynamic_buffer, test_reproject_raster_matches_reference_grid
- [test_change_adapters_v050.py](../tests/test_change_adapters_v050.py) — test change adapters v050 · test_s2looking_normalizes_binary_and_directional_labels, test_second_derives_change_without_fabricating_input_path, test_second_rgb_mapping_must_be_explicit, test_dsifn_keeps_directional_supervision_missing
- [test_change_cli_v050.py](../tests/test_change_cli_v050.py) — test change cli v050 · test_change_cli_commands_exist_and_v040_parser_still_routes, test_failed_change_run_is_auditable_and_does_not_create_latest
- [test_change_corrections_v050.py](../tests/test_change_corrections_v050.py) — test change corrections v050 · test_human_correction_overrides_original_and_exclusions_are_removed
- [test_change_final_test_v057.py](../tests/test_change_final_test_v057.py) — test change final test v057 · test_final_test_reads_only_test_and_forwards_once, test_final_test_requires_confirmation_and_refuses_repeat, test_final_test_normalizes_auto_and_explicit_runtime_values, test_invalid_runtime_config_fails_before_constructing_test_dataset
- [test_change_fusion_v050.py](../tests/test_change_fusion_v050.py) — test change fusion v050 · test_double_teacher_fusion_flags_disagreement
- [test_change_gpu_v050.py](../tests/test_change_gpu_v050.py) — test change gpu v050 · test_real_cuda_three_stage_smoke
- [test_change_joint_review_v050.py](../tests/test_change_joint_review_v050.py) — test change joint review v050 · test_joint_review_updates_state_and_rejected_never_enters_student_labels
- [test_change_local_labels_v050.py](../tests/test_change_local_labels_v050.py) — test change local labels v050 · test_local_labels_are_candidates_requiring_review
- [test_change_loss_v050.py](../tests/test_change_loss_v050.py) — test change loss v050 · test_masked_loss_ignores_missing_directional_and_semantic_targets, test_masked_loss_rejects_true_flag_with_missing_target
- [test_change_metrics_v050.py](../tests/test_change_metrics_v050.py) — test change metrics v050 · test_metrics_report_binary_directional_and_semantic, test_missing_supervision_is_unavailable_not_fake_zero, test_static_training_dashboard_and_required_charts
- [test_change_model_v050.py](../tests/test_change_model_v050.py) — test change model v050 · test_multitask_siamese_unet_forward_shapes, test_pretrained_never_downloads_silently, test_changemamba_has_clear_platform_gate
- [test_change_preparation_v050.py](../tests/test_change_preparation_v050.py) — test change preparation v050 · test_prepare_writes_index_statistics_and_previews
- [test_change_review_v050.py](../tests/test_change_review_v050.py) — test change review v050 · test_static_review_contains_filters_local_storage_and_csv_export
- [test_change_sampler_v050.py](../tests/test_change_sampler_v050.py) — test change sampler v050 · test_weighted_sampler_is_deterministic_and_balances_dataset_size, test_round_robin_and_distributed_views_cover_every_dataset
- [test_change_training_optimization_v056.py](../tests/test_change_training_optimization_v056.py) — test change training optimization v056 · test_change_aware_crop_hits_requested_demolition_region, test_ablation_plan_starts_with_one_epoch_and_isolates_test, test_threshold_selection_rejects_test_before_inference, test_binary_only_threshold_scan_skips_unsupported_heads_without_reading_test
- [test_change_training_v050.py](../tests/test_change_training_v050.py) — test change training v050 · test_formal_training_rejects_cpu, test_stage_selection_matches_required_public_datasets, test_local_finetune_schedule_freezes_then_unfreezes_encoder, test_tiny_cpu_epoch_writes_best_last_and_loadable_checkpoint
- [test_chinese_paths.py](../tests/test_chinese_paths.py) — test chinese paths · test_chinese_and_space_manifest_path, test_utf8_bom_manifest_roundtrip
- [test_cli_lifecycle_v040.py](../tests/test_cli_lifecycle_v040.py) — test cli lifecycle v040 · test_failed_run_does_not_update_latest, test_successful_run_updates_latest_without_symlink, test_label_change_invalidates_data_hash, test_force_flag_exists_on_all_user_commands
- [test_coco_v040.py](../tests/test_coco_v040.py) — test coco v040 · test_spatial_coco_is_stable_and_preserves_hole, test_train_and_val_tiles_share_no_block, test_empty_label_date_can_supply_hard_negative_tiles, test_dataset_reads_window_generated_real_tile
- [test_conservative_hard_finetune_v054.py](../tests/test_conservative_hard_finetune_v054.py) — test conservative hard finetune v054 · test_conservative_smoke_uses_fixed_source_thresholds_and_small_hard_weight
- [test_crs.py](../tests/test_crs.py) — test crs · test_china_city_utm_selection, test_missing_crs_is_explicit_error, test_geographic_metric_operation_rejected, test_web_mercator_metric_operation_rejected
- [test_demolition_head_only_v055.py](../tests/test_demolition_head_only_v055.py) — test demolition head only v055 · test_demolition_head_smoke_config_is_strictly_isolated, test_demolition_sampler_uses_train_only_balanced_subset, test_only_demolition_head_is_trainable_when_torch_is_available
- [test_determinism.py](../tests/test_determinism.py) — test determinism · test_track_ids_stable_when_input_order_changes, test_equal_candidates_are_marked_ambiguous
- [test_downscale.py](../tests/test_downscale.py) — test downscale · test_mmu_199_201_boundary_and_fraction_range, test_missing_date_is_reported_not_invented
- [test_evaluation.py](../tests/test_evaluation.py) — test evaluation · test_exact_object_match_scores_one, test_missing_ground_truth_is_not_computable
- [test_geometry.py](../tests/test_geometry.py) — test geometry · test_multipolygon_is_preserved, test_courtyard_hole_is_preserved, test_l_shape_is_not_forced_to_rectangle, test_regularization_area_guard
- [test_ground_truth_v040.py](../tests/test_ground_truth_v040.py) — test ground truth v040 · test_geometry_only_ground_truth_gets_defaults, test_chinese_field_aliases_are_reported, test_invalid_polygon_is_repaired, test_points_and_lines_are_rejected
- [test_inference_v040.py](../tests/test_inference_v040.py) — test inference v040 · test_checkpoint_hash_is_stable, test_legacy_inference_defaults_to_best_checkpoint
- [test_lineage.py](../tests/test_lineage.py) — test lineage · test_split_lineage_component, test_complex_three_to_two_lineage_component, test_lineage_event_id_is_stable
- [test_manual_diagnostic_review_v058.py](../tests/test_manual_diagnostic_review_v058.py) — test manual diagnostic review v058 · test_manual_diagnostic_review_uses_existing_assets_only
- [test_nanjing_jiangbei_acceptance.py](../tests/test_nanjing_jiangbei_acceptance.py) — test nanjing jiangbei acceptance · test_parse_date_token_supports_compact_scene_dates, test_pair_specs_has_adjacent_and_each_to_final_pairs, test_state_ontology_is_stable_and_model_derived, test_real_nanjing_input_audit_regression
- [test_nanjing_jiangbei_modular_rebuild.py](../tests/test_nanjing_jiangbei_modular_rebuild.py) — test nanjing jiangbei modular rebuild · test_frozen_contract_cannot_be_mutated, test_module_matrix_is_write_isolated, test_architecture_audit_writes_complete_read_only_audit_set, test_report_metric_separation_allows_provenance_flags
- [test_outputs_v040.py](../tests/test_outputs_v040.py) — test outputs v040 · test_force_is_required_to_replace_output, test_force_removes_only_explicit_output
- [test_preparation_v040.py](../tests/test_preparation_v040.py) — test preparation v040 · test_prepare_selects_utm_and_preserves_sources, test_prepare_rejects_missing_label_crs
- [test_project_discovery.py](../tests/test_project_discovery.py) — test project discovery · test_parse_supported_dates, test_duplicate_image_date_lists_conflicts, test_image_without_label_is_an_error, test_invalid_calendar_date_is_an_error
- [test_project_registry_architecture.py](../tests/test_project_registry_architecture.py) — test project registry architecture · test_registry_has_authoritative_models_and_test_policies, test_canonical_output_marks_missing_heads_unsupported, test_spent_and_locked_tests_are_blocked_during_development, test_temporal_engine_requires_usable_stable_observations
- [test_quality.py](../tests/test_quality.py) — test quality · test_quality_mask_marks_bright_patch, test_quality_composition_manual_overrides_provider, test_registration_issue_invalidates_whole_scene
- [test_readme_v040.py](../tests/test_readme_v040.py) — test readme v040 · test_readme_is_short_chinese_plug_and_train_guide, test_documented_commands_exist_in_parser
- [test_retrain_index_inheritance_v053.py](../tests/test_retrain_index_inheritance_v053.py) — test retrain index inheritance v053 · test_candidate_inherits_identical_baseline_index
- [test_review.py](../tests/test_review.py) — test review · test_apply_reviews_preserves_auto_and_chinese_notes, test_apply_reviews_is_idempotent, test_review_html_contains_chinese_ui
- [test_review_hard_training_v052.py](../tests/test_review_hard_training_v052.py) — test review hard training v052 · test_threshold_metrics_select_expected_threshold, test_static_review_uses_offscreen_overlay_and_keeps_gt, test_hard_examples_never_promotes_validation_or_test, test_hard_rank_and_max_samples_are_deterministic
- [test_review_student_export_v040.py](../tests/test_review_student_export_v040.py) — test review student export v040 · test_reviewed_rejected_new_building_never_enters_student_labels, test_effective_state_prefers_state_final
- [test_schema.py](../tests/test_schema.py) — test schema · test_legacy_manifest_returns_warning, test_strict_manifest_requires_v030_columns, test_manifest_rejects_bad_date, test_annotation_schema_accepts_valid_frame
- [test_second_dev_validation_v2.py](../tests/test_second_dev_validation_v2.py) — test second dev validation v2 · test_frozen_second_train_split_is_reused_and_quick_is_a_subset, test_second_directional_strata_ignore_void_pixels, test_second_manifest_audit_hash_mismatch_is_not_silently_accepted
- [test_splitting_v040.py](../tests/test_splitting_v040.py) — test splitting v040 · test_spatial_split_is_deterministic, test_same_spatial_block_is_same_split_across_dates, test_boundary_crossing_tile_is_excluded, test_retained_train_and_val_tiles_do_not_overlap
- [test_surface_state_amp_safe.py](../tests/test_surface_state_amp_safe.py) — test surface state amp safe · test_probability_bce_is_finite_inside_autocast_and_restored, test_probability_bce_expands_single_channel_target
- [test_surface_state_formal_runtime.py](../tests/test_surface_state_formal_runtime.py) — test surface state formal runtime · test_requested_run_directory_gets_all_training_subdirectories
- [test_surface_state_mti_registration_guard.py](../tests/test_surface_state_mti_registration_guard.py) — test surface state mti registration guard · test_regressed_mti_is_not_selected, test_registration_supports_role_specific_key
- [test_surface_state_resume_selection.py](../tests/test_surface_state_resume_selection.py) — test surface state resume selection · test_restore_selection_and_patience, test_restore_selection_handles_warmstart_without_logs
- [test_surface_state_scaler_guard.py](../tests/test_surface_state_scaler_guard.py) — test surface state scaler guard · test_empty_disabled_scaler_state_is_skipped_and_method_restored
- [test_tracking.py](../tests/test_tracking.py) — test tracking · test_new_building_interval, test_off_nadir_shift_keeps_track, test_gap_can_be_occlusion_or_construction, test_merge_lineage_event
- [test_tracking_v040.py](../tests/test_tracking_v040.py) — test tracking v040 · test_spatial_index_bounds_pair_metric_calls, test_ambiguity_uses_previous_and_current_margins, test_complete_observation_status_matrix_and_confidence
- [test_training_readiness_v051.py](../tests/test_training_readiness_v051.py) — test training readiness v051 · test_validation_isolated_from_test_and_test_fallback_is_opt_in, test_quick_subset_is_per_dataset_and_excludes_test, test_s2looking_verified_rgb_binary_is_allowed_but_generic_binary_rejects_it, test_pipeline_skips_unsupervised_stages_and_keeps_binary_as_final
- [test_training_v040.py](../tests/test_training_v040.py) — test training v040 · test_tiny_cpu_training_best_reload_inference_and_resume
- [test_v2_boundary_loss.py](../tests/test_v2_boundary_loss.py) — test v2 boundary loss · test_boundary_target_only_marks_edge_band
- [test_v2_checkpoint_merge.py](../tests/test_v2_checkpoint_merge.py) — test v2 checkpoint merge · Tiny, test_checkpoint_merge_has_explicit_provenance
- [test_v2_dataset_sampling.py](../tests/test_v2_dataset_sampling.py) — test v2 dataset sampling · test_dataset_sampler_is_deterministic_and_keeps_each_dataset
- [test_v2_directional_sampler_source_guard.py](../tests/test_v2_directional_sampler_source_guard.py) — test v2 directional sampler source guard · test_v2_directional_sampler_rejects_non_s2looking_directional_pool
- [test_v2_geotiff_alignment.py](../tests/test_v2_geotiff_alignment.py) — test v2 geotiff alignment · test_alignment_and_output_preserve_reference_grid
- [test_v2_missing_teacher.py](../tests/test_v2_missing_teacher.py) — test v2 missing teacher · test_missing_optional_teachers_do_not_block_partial
- [test_v2_partial_inference.py](../tests/test_v2_partial_inference.py) — test v2 partial inference · test_partial_fusion_marks_missing_teachers_without_fake_arrays
- [test_v2_semantic_gating.py](../tests/test_v2_semantic_gating.py) — test v2 semantic gating · test_second_mapping_pending_keeps_semantic_without_directional
- [test_v2_supervision_routing.py](../tests/test_v2_supervision_routing.py) — test v2 supervision routing · test_explicit_supervision_routing
- [test_v2_teacher_registry.py](../tests/test_v2_teacher_registry.py) — test v2 teacher registry · test_registry_reports_ready_and_missing
- [test_v2_test_isolation.py](../tests/test_v2_test_isolation.py) — test v2 test isolation · test_test_guard_and_hard_mining_never_select_test
- [test_v2_test_isolation_adapters.py](../tests/test_v2_test_isolation_adapters.py) — test v2 test isolation adapters · test_allowed_splits_never_returns_test
- [test_v2_vectorization.py](../tests/test_v2_vectorization.py) — test v2 vectorization · test_vectorization_applies_mmu_and_preserves_crs
- [test_xian_acceptance.py](../tests/test_xian_acceptance.py) — Synthetic contract regressions; no image, checkpoint, or protected test reads.
- [test_xian_adaptive_local_alignment.py](../tests/test_xian_adaptive_local_alignment.py) — Public fit/predict seams explicitly requested by the R5-A specification.
- [test_xian_additive_post_inference.py](../tests/test_xian_additive_post_inference.py) — Public seams for the additive Xi'an post-inference layer.
- [test_xian_additive_watcher.py](../tests/test_xian_additive_watcher.py) — Tests for the non-invasive post-inference watcher.
- [test_xian_alignment_closure.py](../tests/test_xian_alignment_closure.py) — Coordinate conventions and same-region independent structures, not invented ties.
- [test_xian_cached_roof_products.py](../tests/test_xian_cached_roof_products.py) — Meaningful fail-closed tests for the cached B0/V2 I seam; synthetic data only.
- [test_xian_cloud_a_selection.py](../tests/test_xian_cloud_a_selection.py) — Synthetic reproduction of unknown-exterior selection leakage (no files).
- [test_xian_cloud_geo_partition_recovery.py](../tests/test_xian_cloud_geo_partition_recovery.py) — Synthetic-only partition32/8 differential QA, no GT/models/cloud or disk data.
- [test_xian_cloud_joint_alignment.py](../tests/test_xian_cloud_joint_alignment.py) — Synthetic only: numerical interface, geometry, gradients and strict restore.
- [test_xian_cloud_public_final.py](../tests/test_xian_cloud_public_final.py) — Synthetic only: no public payloads, real labels, real models or cloud reads.
- [test_xian_cloud_reference_queue.py](../tests/test_xian_cloud_reference_queue.py) — Regression tests for the resume-safe Xi'an cloud queue.
- [test_xian_cloud_release_geo.py](../tests/test_xian_cloud_release_geo.py) — Synthetic geometry/GTiff/GPKG QA; use geo env, no torch or real GT reads.
- [test_xian_cloud_release_inference.py](../tests/test_xian_cloud_release_inference.py) — Synthetic torch/geometric input-contract QA. No real inference or GT reads.
- [test_xian_cloud_s_train.py](../tests/test_xian_cloud_s_train.py) — Synthetic-only trainer checks. All training I/O is intercepted in memory.
- [test_xian_cloud_segmentation.py](../tests/test_xian_cloud_segmentation.py) — Synthetic-only S contract checks; no dataset or protected evaluation reads.
- [test_xian_coherent_support.py](../tests/test_xian_coherent_support.py) — test xian coherent support · CoherentSupportTests
- [test_xian_feature_recovery.py](../tests/test_xian_feature_recovery.py) — Public mask-attribution and evidence-group gates; no mocked matching.
- [test_xian_final_temporal_package.py](../tests/test_xian_final_temporal_package.py) — Public-seam tests for the Xi'an actual-date temporal package.
- [test_xian_fraction_shadow.py](../tests/test_xian_fraction_shadow.py) — test xian fraction shadow · FractionShadowTests
- [test_xian_frozen_candidate_audit.py](../tests/test_xian_frozen_candidate_audit.py) — Frozen-vector replay checks; no tests claim geographic accuracy.
- [test_xian_gt10m_transfer.py](../tests/test_xian_gt10m_transfer.py) — Synthetic contract tests; not actual geographic validation.
- [test_xian_image_block_replay.py](../tests/test_xian_image_block_replay.py) — test xian image block replay · BlockReplayTests
- [test_xian_instance_adaptive.py](../tests/test_xian_instance_adaptive.py) — No-checkpoint safety contract, not learned predictor accuracy tests.
- [test_xian_instance_evidence.py](../tests/test_xian_instance_evidence.py) — test xian instance evidence · InstanceEvidenceTests
- [test_xian_integration.py](../tests/test_xian_integration.py) — Integration regressions at existing geometry, identity, state and GT seams.
- [test_xian_legacy_evidence_csv.py](../tests/test_xian_legacy_evidence_csv.py) — test xian legacy evidence csv · LegacyCsvTests
- [test_xian_live_shadow.py](../tests/test_xian_live_shadow.py) — Small live adapter invariants; synthetic data, no test imagery or models.
- [test_xian_manual_roof_reference.py](../tests/test_xian_manual_roof_reference.py) — test xian manual roof reference · ManualReferenceTests
- [test_xian_materialize_transform.py](../tests/test_xian_materialize_transform.py) — Regression tests for Xi'an VRT geotransform conversion.
- [test_xian_native_fragment_evidence.py](../tests/test_xian_native_fragment_evidence.py) — test xian native fragment evidence · NativeFragmentTests
- [test_xian_point_roof_alignment.py](../tests/test_xian_point_roof_alignment.py) — test xian point roof alignment · PointRoofTests
- [test_xian_reference_core_cohort.py](../tests/test_xian_reference_core_cohort.py) — test xian reference core cohort · ReferenceCoreTests
- [test_xian_reference_label_filter.py](../tests/test_xian_reference_label_filter.py) — test xian reference label filter · ReferenceFilterTests
- [test_xian_reference_preflight.py](../tests/test_xian_reference_preflight.py) — test xian reference preflight · ReferencePreflightTests
- [test_xian_roof_group_correspondence.py](../tests/test_xian_roof_group_correspondence.py) — test xian roof group correspondence · RoofGroupTests
- [test_xian_roof_identity.py](../tests/test_xian_roof_identity.py) — test xian roof identity · RoofIdentityTests
- [test_xian_roof_observation.py](../tests/test_xian_roof_observation.py) — test xian roof observation · RoofObservationTests
- [test_xian_roof_overlay_preview.py](../tests/test_xian_roof_overlay_preview.py) — Pixel-display transport safety, not roof accuracy validation.
- [test_xian_roof_pair_translation.py](../tests/test_xian_roof_pair_translation.py) — test xian roof pair translation · PairTranslationTests
- [test_xian_roof_reference_candidates.py](../tests/test_xian_roof_reference_candidates.py) — test xian roof reference candidates · RoofReferenceTests
- [test_xian_sam_source_candidates.py](../tests/test_xian_sam_source_candidates.py) — test xian sam source candidates · SamSourceCandidateTests
- [test_xian_sam_source_prompts.py](../tests/test_xian_sam_source_prompts.py) — test xian sam source prompts · SamSourcePromptTests
- [test_xian_single_date_identity.py](../tests/test_xian_single_date_identity.py) — test xian single date identity · SingleDateSourceLinkTests
- [test_xian_single_year_experiment.py](../tests/test_xian_single_year_experiment.py) — test xian single year experiment · SingleYearTests
- [test_xian_source_mask_retention.py](../tests/test_xian_source_mask_retention.py) — test xian source mask retention · SourceMaskRetentionTests
- [test_xian_source_partition.py](../tests/test_xian_source_partition.py) — Synthetic source-transport checks, not registration accuracy tests.
- [test_xian_source_prompt_refinement.py](../tests/test_xian_source_prompt_refinement.py) — test xian source prompt refinement · SourcePromptRefinementTests
- [test_xian_source_roof_batch_scope.py](../tests/test_xian_source_roof_batch_scope.py) — test xian source roof batch scope · BatchScopeTests
- [test_xian_source_template_alignment.py](../tests/test_xian_source_template_alignment.py) — test xian source template alignment · SourceTemplateTests
- [test_xian_subpixel_phase.py](../tests/test_xian_subpixel_phase.py) — Inverse Fourier peak localization uses phase only, not known shift input.
- [test_xian_supported_ground_features.py](../tests/test_xian_supported_ground_features.py) — Frozen failure loop: modifying excluded pixels must not change a tie.
- [test_xian_transfer_admission.py](../tests/test_xian_transfer_admission.py) — test xian transfer admission · TransferAdmissionTests

### tests/unit/building_v2

- [test_building_training_gate.py](../tests/unit/building_v2/test_building_training_gate.py) — test building training gate · test_test_reads_block_candidate
- [test_checkpoint_selection.py](../tests/unit/building_v2/test_checkpoint_selection.py) — test checkpoint selection · test_class_collapse_is_ineligible, test_baseline_retained_when_candidate_does_not_improve
- [test_dataset_leakage.py](../tests/unit/building_v2/test_dataset_leakage.py) — test dataset leakage · test_dataset_refuses_test_rows
- [test_final_qa_queue.py](../tests/unit/building_v2/test_final_qa_queue.py) — test final qa queue · test_final_qa_entrypoint_exists
- [test_globfp_audit.py](../tests/unit/building_v2/test_globfp_audit.py) — test globfp audit · test_globfp_audit_entrypoint_exists
- [test_globfp_pairing.py](../tests/unit/building_v2/test_globfp_pairing.py) — test globfp pairing · test_globfp_pairing_entrypoint_exists
- [test_height_quality.py](../tests/unit/building_v2/test_height_quality.py) — test height quality · test_height_is_not_enabled_by_default
- [test_model_outputs.py](../tests/unit/building_v2/test_model_outputs.py) — test model outputs · test_static_outputs_and_feature_bundle_are_finite, test_height_head_is_explicit
- [test_model_package.py](../tests/unit/building_v2/test_model_package.py) — test model package · test_package_script_exists, test_static_inference_tool_exists
- [test_multitask_losses.py](../tests/unit/building_v2/test_multitask_losses.py) — test multitask losses · test_multitask_loss_finite
- [test_objectization.py](../tests/unit/building_v2/test_objectization.py) — test objectization · test_objectization_counts_components
- [test_static_feature_bundle.py](../tests/unit/building_v2/test_static_feature_bundle.py) — test static feature bundle · test_feature_bundle_keys

### tests/unit/construction

- [test_audit_ucs_readiness.py](../tests/unit/construction/test_audit_ucs_readiness.py) — test audit ucs readiness · test_seed_progress_supports_declared_inventory_shapes, test_seed_progress_unknown_structure_is_explicit_error, test_seed_progress_parses_real_seed_inventory_without_modifying_it
- [test_component_smoke.py](../tests/unit/construction/test_component_smoke.py) — test component smoke · test_component_smoke_receipt_is_development_only
- [test_construction_activity_v1.py](../tests/unit/construction/test_construction_activity_v1.py) — test construction activity v1 · test_ucs_codec_and_test_read_guard, test_dataset_specific_head_and_partial_label_loss_are_finite, test_missing_native_evidence_is_not_background, test_tfr_preserves_time_shape_and_missing_timestep
- [test_json_io.py](../tests/unit/construction/test_json_io.py) — test json io · test_load_json_accepts_bom_and_non_bom, test_dump_json_is_utf8_without_bom_and_atomic, test_atomic_json_reads_are_never_partial_under_concurrent_writes, test_corrupt_json_reports_path
- [test_recovery_audit.py](../tests/unit/construction/test_recovery_audit.py) — test recovery audit · test_recovery_audit_rejects_numeric_cross_dataset_candidates_without_reading_pixels
- [test_tongzhou_construction_images_best.py](../tests/unit/construction/test_tongzhou_construction_images_best.py) — test tongzhou construction images best · load_tool, test_contract_has_only_four_frozen_dates_and_no_nanjing_paths, test_nanjing_guard_rejects_both_english_and_chinese_paths, test_recipe_registry_requires_validation_only
- [test_ucs386_gate.py](../tests/unit/construction/test_ucs386_gate.py) — test ucs386 gate · test_ucs386_gate_keeps_manifest_copies_and_locks_test_pixels
- [test_ucs_package_merge.py](../tests/unit/construction/test_ucs_package_merge.py) — test ucs package merge · test_package_merge_deduplicates_identical_and_keeps_complementary, test_package_merge_reports_conflicting_same_stem, test_review_page_declares_independent_construction_tab

### tests/unit/governance

- [test_cross_city_building_stability.py](../tests/unit/governance/test_cross_city_building_stability.py) — test cross city building stability · test_stable_objects_form_multidate_tracks, test_one_gap_bridge_does_not_create_middle_object, test_split_candidates_are_lineage_not_forced_matches, test_metrics_keep_scientific_boundaries
- [test_current_best_gt_ready.py](../tests/unit/governance/test_current_best_gt_ready.py) — test current best gt ready · latest_governance_root, test_current_best_registry_has_three_independent_entries, test_site_classifier_is_disabled_without_gt, test_historical_rejected_experiments_are_registered
- [test_nanjing_building_construction_integrated.py](../tests/unit/governance/test_nanjing_building_construction_integrated.py) — Acceptance tests for the read-only Nanjing integrated run.
- [test_zoom20_2019_gt_comparison.py](../tests/unit/governance/test_zoom20_2019_gt_comparison.py) — Regression tests for the frozen-model Zoom20/2019-GT acceptance run.
- [test_zoom21_current_best.py](../tests/unit/governance/test_zoom21_current_best.py) — test zoom21 current best · latest_zoom21_root, test_date_and_claimed_zoom_parsing, test_actual_nanjing_source_has_no_zoom21_candidate, test_zoom21_gate_receipt_is_honest_and_registry_only

### tests/unit/review

- [test_geometry_editing.py](../tests/unit/review/test_geometry_editing.py) — test geometry editing · test_self_intersection_blocked, test_closed_polygon_is_valid
- [test_holdout_seal.py](../tests/unit/review/test_holdout_seal.py) — test holdout seal · test_holdout_setting_is_sealed
- [test_label_export.py](../tests/unit/review/test_label_export.py) — test label export · test_geojson_export_is_atomic
- [test_review_api.py](../tests/unit/review/test_review_api.py) — test review api · test_server_declares_local_only
- [test_review_database.py](../tests/unit/review/test_review_database.py) — test review database · test_new_database_and_autosave, test_repair_from_jsonl, test_human_draw_and_delete_reason
- [test_seed_temporal.py](../tests/unit/review/test_seed_temporal.py) — test seed temporal · test_seed_inventory_preserves_source_and_is_model_independent, test_seed_store_requires_vocab_and_keeps_verified_seed_non_ground_truth
- [test_split_lock.py](../tests/unit/review/test_split_lock.py) — test split lock · test_split_is_deterministic_and_locked
- [test_training_command_whitelist.py](../tests/unit/review/test_training_command_whitelist.py) — test training command whitelist · test_arbitrary_shell_command_rejected, test_whitelisted_token_only
- [test_training_gate.py](../tests/unit/review/test_training_gate.py) — test training gate · test_empty_project_gate_counts_are_explicit
- [test_transition_preview_v2.py](../tests/unit/review/test_transition_preview_v2.py) — test transition preview v2 · test_manifest_backed_transition_preview_is_date_specific_and_gate_safe, test_construction_inventory_groups_regenerated_transitions
- [test_urrs_review_console.py](../tests/unit/review/test_urrs_review_console.py) — test urrs review console · test_inventory_is_deterministic, test_store_atomic_undo_and_export, test_path_traversal_rejected

### tests/xian_r5

- [test_alignment_preview.py](../tests/xian_r5/test_alignment_preview.py) — test alignment preview · make_cell, make_field, CandidateAlignmentPreviewTests
- [test_alignment_review_packet.py](../tests/xian_r5/test_alignment_review_packet.py) — test alignment review packet · AlignmentReviewPacketTests
- [test_base_mosaic_pair.py](../tests/xian_r5/test_base_mosaic_pair.py) — test base mosaic pair · BaseMosaicPairDiagnosticTests
- [test_base_tie_diagnostic.py](../tests/xian_r5/test_base_tie_diagnostic.py) — test base tie diagnostic · BaseTieDiagnosticTests
- [test_component_tie_consensus.py](../tests/xian_r5/test_component_tie_consensus.py) — test component tie consensus · ComponentTieConsensusTests
- [test_composition.py](../tests/xian_r5/test_composition.py) — test composition · CompositionTests
- [test_displacement_field.py](../tests/xian_r5/test_displacement_field.py) — test displacement field · write_lightglue_fixture, attach_synthetic_stitched_mosaics, downgrade_mosaic_receipt_to_legacy, cell
- [test_ground_review_packet.py](../tests/xian_r5/test_ground_review_packet.py) — test ground review packet · sha256, GroundReviewPacketTests
- [test_grounded_reconstruction.py](../tests/xian_r5/test_grounded_reconstruction.py) — Interface-level tests; synthetic geometry is not Xi'an accuracy evidence.
- [test_lightglue_patch_windows.py](../tests/xian_r5/test_lightglue_patch_windows.py) — test lightglue patch windows · LightGluePatchWindowTests
- [test_local_field.py](../tests/xian_r5/test_local_field.py) — test local field · LocalFieldTests
- [test_mosaic_grid.py](../tests/xian_r5/test_mosaic_grid.py) — test mosaic grid · MosaicTileGridTests
- [test_mosaic_registration.py](../tests/xian_r5/test_mosaic_registration.py) — test mosaic registration · MosaicRegistrationTests
- [test_multi_anchor_base_tie.py](../tests/xian_r5/test_multi_anchor_base_tie.py) — test multi anchor base tie · MultiAnchorBaseTieTests
- [test_pose_graph.py](../tests/xian_r5/test_pose_graph.py) — test pose graph · PoseGraphTests
- [test_r5_relative_alignment_contract.py](../tests/xian_r5/test_r5_relative_alignment_contract.py) — test r5 relative alignment contract · RelativeAlignmentContractTests
- [test_rbf_sam_candidates.py](../tests/xian_r5/test_rbf_sam_candidates.py) — Contract tests for unverified RBF-SAM roof-to-footprint proposals.
- [test_rbf_sam_native_bridge.py](../tests/xian_r5/test_rbf_sam_native_bridge.py) — test rbf sam native bridge · RbfSamNativeBridgeTests
- [test_reconstruction_qa.py](../tests/xian_r5/test_reconstruction_qa.py) — test reconstruction qa · ReconstructionQaSchemaTests
- [test_registration_backend.py](../tests/xian_r5/test_registration_backend.py) — test registration backend · RegistrationBackendTests
- [test_registration_worker.py](../tests/xian_r5/test_registration_worker.py) — test registration worker · SubprocessRegistrationBackendTests
- [test_relative_offsets.py](../tests/xian_r5/test_relative_offsets.py) — Contract tests for spatially binned relative-shift candidates.
- [test_relative_roof_candidate_products.py](../tests/xian_r5/test_relative_roof_candidate_products.py) — Contract tests for grid-aligned, explicitly relative roof candidates.
- [test_reviewed_absences.py](../tests/xian_r5/test_reviewed_absences.py) — Absence evidence interface tests; synthetic inputs are not Xi'an accuracy evidence.
- [test_reviewed_footprint_candidates.py](../tests/xian_r5/test_reviewed_footprint_candidates.py) — test reviewed footprint candidates · sha256, ReviewedFootprintCandidateTests
- [test_reviewed_ground_footprints.py](../tests/xian_r5/test_reviewed_ground_footprints.py) — Strict input-contract tests; synthetic records are not Xi'an accuracy evidence.
- [test_reviewed_image_alignment.py](../tests/xian_r5/test_reviewed_image_alignment.py) — test reviewed image alignment · sha256, write_relative_tie_sidecar, candidate_field, spatial_composition_candidate_field
- [test_roof_candidate_alignment.py](../tests/xian_r5/test_roof_candidate_alignment.py) — test roof candidate alignment · field, cell, RoofCandidateAlignmentTests
- [test_source_dates.py](../tests/xian_r5/test_source_dates.py) — test source dates · SourceTileDateAssociationTests
- [test_spatial_composition.py](../tests/xian_r5/test_spatial_composition.py) — test spatial composition · SpatialCompositionTests
- [test_tbia.py](../tests/xian_r5/test_tbia.py) — test tbia · TBIATests
- [test_temporal_consistency.py](../tests/xian_r5/test_temporal_consistency.py) — test temporal consistency · TemporalConsistencyTests

## tools

诊断、数据准备、执行和回读工具

- [README.md](../tools/README.md) — README
- [accelerated_ge_filters.py](../tools/accelerated_ge_filters.py) — Resume-safe one-pass GE3/GE5 filtering for the Xi'an primary package.
- [acquire_b04_reference_evidence.py](../tools/acquire_b04_reference_evidence.py) — Read-only source audit and official Sentinel catalogue evidence, not registration.
- [acquire_xian_sam_candidate_runtime.py](../tools/acquire_xian_sam_candidate_runtime.py) — Isolated official SAM dependency acquisition; not RBF-SAM, no environment edits.
- [analyze_xian_b04_full_mosaic_relative_offsets.py](../tools/analyze_xian_b04_full_mosaic_relative_offsets.py) — Expand a bounded fixed-site static-vector relative-shift check to its 11x12 AOI.
- [analyze_xian_candidate_relative_offsets.py](../tools/analyze_xian_candidate_relative_offsets.py) — Read-only relative-shift diagnostic using repeated external polygon IDs.
- [analyze_xian_r5_stage2_date_pair_alternatives.py](../tools/analyze_xian_r5_stage2_date_pair_alternatives.py) — Enumerate exact-date spatial overlaps and fixed-case support for Stage-2.
- [assemble_xian_temporal.py](../tools/assemble_xian_temporal.py) — Assemble Xi'an GEID regions into date-aware epoch references.
- [assess_xian_external_footprints.py](../tools/assess_xian_external_footprints.py) — Bounded cross-date comparison to the undocumented local building layer.
- [attribute_full20_support_to_building_objects.py](../tools/attribute_full20_support_to_building_objects.py) — Attribute frozen Full20 change support to sealed Building objects.
- [audit_building_v2_leakage.py](../tools/audit_building_v2_leakage.py) — Audit grouped leakage for a Building Foundation v2 run manifest.
- [audit_globfp_dataset.py](../tools/audit_globfp_dataset.py) — Audit a local 3D-GloBFP distribution without modifying source data.
- [audit_shenzhen_tif_quality.py](../tools/audit_shenzhen_tif_quality.py) — Read-only quality and analysis-ready contract for the four Shenzhen TIFs.
- [audit_ucs_readiness.py](../tools/audit_ucs_readiness.py) — URRS UCS formal-readiness audit.
- [audit_xian_frozen_source_candidates.py](../tools/audit_xian_frozen_source_candidates.py) — Audit only the five frozen candidates. No search or actual roof transport.
- [audit_xian_instance_evidence.py](../tools/audit_xian_instance_evidence.py) — Bounded existing-object/track/source audit; no fitting or label mutation.
- [audit_xian_native_fragments.py](../tools/audit_xian_native_fragments.py) — Close strip-fragment ambiguity, legacy edge links, and bounded native evidence.
- [audit_xian_r5_acceptance.py](../tools/audit_xian_r5_acceptance.py) — Read-only S4.14 postflight of an explicit frozen, unaligned Shadow run.
- [audit_xian_r5_mosaic_tile_grid.py](../tools/audit_xian_r5_mosaic_tile_grid.py) — Audit fixed-case source JPEG windows against stitched and 10m raster grids.
- [audit_xian_supplement_cache.py](../tools/audit_xian_supplement_cache.py) — Register Xi'an supplemental Google Earth cache evidence.
- [audit_xian_window_fraction_reference.py](../tools/audit_xian_window_fraction_reference.py) — All-window fixed-block annotation-fraction audit, without sample selection.
- [bind_xian_manual_roof_reference.py](../tools/bind_xian_manual_roof_reference.py) — Append-only user reference binding and exact 10m annotation fractions.
- [build_artifact_index.py](../tools/build_artifact_index.py) — build artifact index
- [build_building_v2_diagnostics.py](../tools/build_building_v2_diagnostics.py) — Audit existing Nanjing/Tongzhou static outputs after candidate freeze.
- [build_globfp_final_reports.py](../tools/build_globfp_final_reports.py) — Build the final GloBFP/height/Construction report artifacts.
- [build_globfp_final_reports_compat.py](../tools/build_globfp_final_reports_compat.py) — Compatibility runner for the final report builder's path-copy typo.
- [build_globfp_final_reports_compat2.py](../tools/build_globfp_final_reports_compat2.py) — Idempotent compatibility runner for the final GloBFP report builder.
- [build_globfp_final_reports_compat3.py](../tools/build_globfp_final_reports_compat3.py) — Resume the final report build with deterministic template compatibility fixes.
- [build_globfp_final_reports_compat4.py](../tools/build_globfp_final_reports_compat4.py) — Resume report generation with the portable report chart schema enabled.
- [build_globfp_final_reports_compat5.py](../tools/build_globfp_final_reports_compat5.py) — Idempotently build final reports with the required native chart block.
- [build_globfp_height_model.py](../tools/build_globfp_height_model.py) — build globfp height model
- [build_globfp_height_model_compat.py](../tools/build_globfp_height_model_compat.py) — Compatibility entry point for the first height-model script revision.
- [build_globfp_height_model_compat2.py](../tools/build_globfp_height_model_compat2.py) — In-memory repair runner for the malformed final dictionary lookup.
- [build_globfp_height_model_compat3.py](../tools/build_globfp_height_model_compat3.py) — Geometry-degenerate compatibility runner for morphology model training.
- [build_primary_two_period_results.py](../tools/build_primary_two_period_results.py) — Build the bounded two-period primary raster package.
- [build_shenzhen_epoch_manifest.py](../tools/build_shenzhen_epoch_manifest.py) — Create an immutable Shenzhen epoch manifest from the GEID completion receipt.
- [build_shenzhen_integrated_preflight.py](../tools/build_shenzhen_integrated_preflight.py) — Build a read-only Shenzhen four-epoch integrated-system preflight.
- [build_shenzhen_scene_manifest.py](../tools/build_shenzhen_scene_manifest.py) — Build a four-scene Shenzhen manifest without guessing tile acquisition dates.
- [build_visual_gallery.py](../tools/build_visual_gallery.py) — build visual gallery
- [build_xian_boundary_temporal_outputs.py](../tools/build_xian_boundary_temporal_outputs.py) — Build boundary-scoped Xi'an temporal accounting from immutable geometry.
- [build_xian_current_source_roof_bank.py](../tools/build_xian_current_source_roof_bank.py) — Versioned source-roof bank, retaining old attempts and uncertain roles.
- [build_xian_exact_three_state.py](../tools/build_xian_exact_three_state.py) — Build exact, source-grid-aligned Xi'an three-state temporal products.
- [build_xian_model_roof_observations.py](../tools/build_xian_model_roof_observations.py) — Bind raw-model roof surfaces; never rasterize them as whole-roof labels.
- [build_xian_r5_stage2_alignment_frame.py](../tools/build_xian_r5_stage2_alignment_frame.py) — Build bounded, actual-date-coherent alignment candidate pools for R5 Stage-2.
- [build_xian_supplemental_coverage_remote.py](../tools/build_xian_supplemental_coverage_remote.py) — Build residual-Unknown priority and supplemental-coverage evidence.
- [building_foundation_v2_cli.py](../tools/building_foundation_v2_cli.py) — Unified local CLI for the Building Foundation v2 audit/training flow.
- [check_b04_closure_tests.py](../tools/check_b04_closure_tests.py) — Persist bounded closure tests, including environment errors without hiding them.
- [check_b04_evidence_environment.py](../tools/check_b04_evidence_environment.py) — Audit base and isolated imports without mutating the base runtime.
- [check_xian_fullroi_roof_translation.py](../tools/check_xian_fullroi_roof_translation.py) — Recompute all frozen 267 pair vectors; internal per-source uniformity check.
- [check_xian_reference_core_alignment.py](../tools/check_xian_reference_core_alignment.py) — Fixed corroborated-source cohort replay; no new displacement application.
- [check_xian_reference_core_alignment_v2.py](../tools/check_xian_reference_core_alignment_v2.py) — Fixed corroborated-source cohort replay; no new displacement application.
- [check_xian_roof_reference_groups.py](../tools/check_xian_roof_reference_groups.py) — Append direct group geometry diagnostics without modifying source retention.
- [close_b04_reviewed_registration.py](../tools/close_b04_reviewed_registration.py) — Bounded B04 zero-acceptance closure; consumes existing candidates only.
- [codex_windows_fallback_runner.py](../tools/codex_windows_fallback_runner.py) — Process-level fallback bridge for the URRS recovery workflow.
- [compare_xian_local_building_layer.py](../tools/compare_xian_local_building_layer.py) — Compare the local WGS84 building layer to R4 outlines on exact-date tiles.
- [compile_xian_b04_component_tie_consensus.py](../tools/compile_xian_b04_component_tie_consensus.py) — Compile bounded B04 multi-reference, exact-source-tile tie candidates.
- [construction_finalize.py](../tools/construction_finalize.py) — construction finalize · finalize_existing_run
- [construction_nanjing_domain_adaptation_worker.py](../tools/construction_nanjing_domain_adaptation_worker.py) — construction nanjing domain adaptation worker · now, sha256, read_json, write_json
- [construction_training_worker.py](../tools/construction_training_worker.py) — construction training worker · evaluate_model, prepare_run, train_run, status_run
- [convert_xian_rbf_sam_native_output.py](../tools/convert_xian_rbf_sam_native_output.py) — Convert official RBF-SAM JSON output to the fixed-case R5 candidate bundle.
- [create_building_v2_qa_queue.py](../tools/create_building_v2_qa_queue.py) — Create a fixed 200-case Building Foundation v2 QA queue without changing Lite Review.
- [create_clean_transition_maps.py](../tools/create_clean_transition_maps.py) — create clean transition maps · render_clean_map, main
- [create_current_best_gt_ready_governance.py](../tools/create_current_best_gt_ready_governance.py) — Create a read-only architecture/model/GT-readiness governance package.
- [create_globfp_height_gate_outputs.py](../tools/create_globfp_height_gate_outputs.py) — Create explicit NOT-ACCEPTED height outputs for a Tier-C match gate.
- [create_xian_publication_figures.py](../tools/create_xian_publication_figures.py) — Create publication-style Xi'an map and workflow figures from frozen outputs.
- [crop_xian_fixed_blocks.py](../tools/crop_xian_fixed_blocks.py) — Cloud-side crop of the frozen Xi'an 1 km blocks.
- [crossvalidate_xian_relative_displacement_field.py](../tools/crossvalidate_xian_relative_displacement_field.py) — LOO diagnostic: does a spatial plane explain static-FID relative shifts?
- [deliver_xian_reference_screened_window.py](../tools/deliver_xian_reference_screened_window.py) — Whole current-window reference-screened experiment delivery, never formal GT.
- [ensure_globfp_report_chart.py](../tools/ensure_globfp_report_chart.py) — Ensure generated report artifacts contain the required native chart block.
- [evaluate_cross_city_building_stability.py](../tools/evaluate_cross_city_building_stability.py) — Cross-city building tracking stability evaluator.
- [evaluate_nanjing_manual_gt.py](../tools/evaluate_nanjing_manual_gt.py) — Future explicit evaluation entry point; never trains or runs inference.
- [export_nanjing_building_gt.py](../tools/export_nanjing_building_gt.py) — Export accepted building GT attributes without modifying the source GT.
- [export_nanjing_construction_gt.py](../tools/export_nanjing_construction_gt.py) — Export accepted Construction GT attributes without modifying the source GT.
- [export_whu_val32_comparison.py](../tools/export_whu_val32_comparison.py) — export whu val32 comparison · parse_args, sha256_file, read_text_auto, resolve_manifest_path
- [export_xian_r5_alignment_review_packet.py](../tools/export_xian_r5_alignment_review_packet.py) — Export a read-only-source, fail-closed image-alignment review package.
- [export_xian_r5_ground_review_packet.py](../tools/export_xian_r5_ground_review_packet.py) — Export an actual-date, candidate-only ground-footprint review packet.
- [export_xian_sam_refinement_candidates.py](../tools/export_xian_sam_refinement_candidates.py) — Dynamic-count source mask exporter for the independently frozen refinement.
- [finalize_b04_closure_receipt.py](../tools/finalize_b04_closure_receipt.py) — Add final test limitations and delivery inventory to this new closure only.
- [finalize_b04_evidence_manifest.py](../tools/finalize_b04_evidence_manifest.py) — Freeze hashes for the completed, metadata-only B04 evidence acquisition run.
- [finalize_building_v2_run.py](../tools/finalize_building_v2_run.py) — Finalize a Building Foundation v2 run receipt without changing registries.
- [finalize_nanjing_domain_review_readiness.py](../tools/finalize_nanjing_domain_review_readiness.py) — Write the auditable latest status for the Nanjing review-ready milestone.
- [finalize_primary_two_period_results.py](../tools/finalize_primary_two_period_results.py) — Finalize the Xi'an two-period GE3/GE5 package without CC rescans.
- [finalize_xian_adaptive_alignment.py](../tools/finalize_xian_adaptive_alignment.py) — Readback, immutable-source verification and append-only R5-A archive.
- [finalize_xian_ground_repair.py](../tools/finalize_xian_ground_repair.py) — Actual-image contamination replay, roof-fragment provenance and partial bundle.
- [finalize_xian_instance_adaptive_admission.py](../tools/finalize_xian_instance_adaptive_admission.py) — Resume publication only after timezone-data failure; no data/model rerun.
- [finalize_xian_point_roof_alignment.py](../tools/finalize_xian_point_roof_alignment.py) — Deliver all-pixel dispositions, fraction effects, lineage, and a safe bundle.
- [finalize_xian_r5_reconstruction_freeze.py](../tools/finalize_xian_r5_reconstruction_freeze.py) — Create an append-only freeze receipt from bounded B03/B04 case outputs.
- [finalize_xian_r5_stage2.py](../tools/finalize_xian_r5_stage2.py) — Finalize bounded Xi'an R5 Stage-2 evidence without touching R4 production.
- [finalize_xian_selection_after_smoke.py](../tools/finalize_xian_selection_after_smoke.py) — Promote only diagnostic-smoke-passing candidates into the final freeze table.
- [finalize_xian_single2019_experiment.py](../tools/finalize_xian_single2019_experiment.py) — Append guarded recommendations and a raw-grid experimental baseline.
- [finalize_xian_template_experiment.py](../tools/finalize_xian_template_experiment.py) — Seal an append-only fixed-template experiment, never promote it to GT.
- [flatten_globfp_report_chart_dataset.py](../tools/flatten_globfp_report_chart_dataset.py) — Make the report chart dataset scalar while preserving audited summaries.
- [globfp_data_audit_runner.py](../tools/globfp_data_audit_runner.py) — Read-only, bounded audit for the local 3D-GloBFP distribution.
- [globfp_data_audit_v2.py](../tools/globfp_data_audit_v2.py) — Second-generation bounded 3D-GloBFP data audit entry point.
- [globfp_height_pipeline.py](../tools/globfp_height_pipeline.py) — Auditable preparation entry point for the local 3D-GloBFP branch.
- [globfp_nanjing_extract.py](../tools/globfp_nanjing_extract.py) — Extract the Nanjing-area slice from the observed Jiangsu GloBFP layer.
- [globfp_nanjing_extract_compat.py](../tools/globfp_nanjing_extract_compat.py) — Compatibility wrapper for pyarrow-free and older Rasterio environments.
- [globfp_nanjing_extract_identity.py](../tools/globfp_nanjing_extract_identity.py) — Deterministic identity wrapper for the Jiangsu slice.
- [globfp_nanjing_extract_no_arrow.py](../tools/globfp_nanjing_extract_no_arrow.py) — Compatibility wrapper for environments without pyarrow.
- [globfp_nanjing_match.py](../tools/globfp_nanjing_match.py) — Match the extracted 2020 GloBFP slice to sealed RSBuilding 2019/2022.
- [import_nanjing_domain_adaptation_review.py](../tools/import_nanjing_domain_adaptation_review.py) — Import the current Nanjing 66-block label package into URRS Lite Review.
- [infer_building_foundation_v2.py](../tools/infer_building_foundation_v2.py) — Run auditable static Building Foundation v2 inference on one raster.
- [infer_rsbuilding_static_geometry.py](../tools/infer_rsbuilding_static_geometry.py) — infer rsbuilding static geometry · utc_now, timestamp, sha256_file, write_json
- [infer_shenzhen_sparse_geometry.py](../tools/infer_shenzhen_sparse_geometry.py) — Run frozen B0 geometry on Shenzhen without materialising huge probability TIFFs.
- [infer_xian_fullroi_sam_cpu.py](../tools/infer_xian_fullroi_sam_cpu.py) — Resumable full-ROI weak source proposals, per-group hash-verified receipts.
- [infer_xian_sam_roof_pilot_cpu.py](../tools/infer_xian_sam_roof_pilot_cpu.py) — Official SAM inference in the unchanged Python3.9 model environment, CPU only.
- [inspect_xian_independent_building_data.py](../tools/inspect_xian_independent_building_data.py) — Read-only inventory of the local Xi'an building polygon layer.
- [inspect_xian_negative_alignment_blocks.py](../tools/inspect_xian_negative_alignment_blocks.py) — Native imagery and exact fixed-cohort traces for known-negative blocks.

### tools/maintenance

- [build_code_index.py](../tools/maintenance/build_code_index.py) — Build a linked inventory of versioned source, configurations and documentation.

- [materialize_globfp_report_chart_sql.py](../tools/materialize_globfp_report_chart_sql.py) — Materialize the report chart rows through a recorded SQLite query.
- [materialize_xian_reference.py](../tools/materialize_xian_reference.py) — Materialise one audited Xi'an VRT into an independent GeoTIFF.
- [materialize_xian_reference_tiles.py](../tools/materialize_xian_reference_tiles.py) — Materialise an Xi'an VRT by streaming its source JPEG tiles.
- [nanjing_gt_common.py](../tools/nanjing_gt_common.py) — Shared, read-only helpers for the future Nanjing manual-GT workflow.
- [nanjing_gt_pipeline.py](../tools/nanjing_gt_pipeline.py) — Unified explicit manual-GT validation/evaluation/export entry point.
- [nanjing_jiangbei_acceptance.py](../tools/nanjing_jiangbei_acceptance.py) — nanjing jiangbei acceptance · utc_now, stamp, sha256_file, write_json
- [nanjing_jiangbei_acceptance_retraining.py](../tools/nanjing_jiangbei_acceptance_retraining.py) — nanjing jiangbei acceptance retraining · sha256_file, load_json, write_json, run_acceptance
- [nanjing_jiangbei_domain_adaptation_prepare.py](../tools/nanjing_jiangbei_domain_adaptation_prepare.py) — nanjing jiangbei domain adaptation prepare · now, sha256, read_json, write_json
- [nanjing_jiangbei_modular_rebuild.py](../tools/nanjing_jiangbei_modular_rebuild.py) — nanjing jiangbei modular rebuild · utc_now, run_stamp, sha256_file, write_json
- [package_building_foundation_v2.py](../tools/package_building_foundation_v2.py) — Package a frozen v2 candidate or an explicit baseline-retained result.
- [pair_globfp_with_local_imagery.py](../tools/pair_globfp_with_local_imagery.py) — Create an auditable, conservative GloBFP-to-imagery pairing manifest.
- [plot_xian_identity_panel.py](../tools/plot_xian_identity_panel.py) — Render actual GE pixels and original polygons; no synthetic imagery.
- [plot_xian_numerical_closure.py](../tools/plot_xian_numerical_closure.py) — Measured synthetic calibration and same-support evidence, no corrected roof fiction.
- [postprocess_xian_b0_area.py](../tools/postprocess_xian_b0_area.py) — Metric-area threshold exports for cloud B0 polygon candidates.
- [prepare_b04_a13_human_review_packet.py](../tools/prepare_b04_a13_human_review_packet.py) — Prepare the frozen, a13-only R5 human FIT/HOLDOUT evidence packet.
- [prepare_building_v2_audit_artifacts.py](../tools/prepare_building_v2_audit_artifacts.py) — Attach immutable GloBFP audit provenance to a v2 run.
- [prepare_xian_case_figure_data.py](../tools/prepare_xian_case_figure_data.py) — Read existing scientific products into a plotting-only snapshot.
- [prepare_xian_cloud_joint.py](../tools/prepare_xian_cloud_joint.py) — One bounded increment over existing registries; no protected test evaluation.
- [prepare_xian_dev_canary_selection.py](../tools/prepare_xian_dev_canary_selection.py) — Cloud-side, low-resolution selection of fixed 1 km Xi'an DEV-CANARY blocks.
- [prepare_xian_fullroi_sam_proposals.py](../tools/prepare_xian_fullroi_sam_proposals.py) — Full frozen-ROI source-prior SAM proposal plan; all eligible parents included.
- [prepare_xian_roof_observation_packets.py](../tools/prepare_xian_roof_observation_packets.py) — Raw, unshifted native GE packets for model-only roof observation candidates.
- [prepare_xian_sam_roof_pilot.py](../tools/prepare_xian_sam_roof_pilot.py) — Freeze an isolated official-SAM source-only pilot before inference.
- [prepare_xian_sam_source_refinement.py](../tools/prepare_xian_sam_source_refinement.py) — Freeze two source-image prompt corrections, no GT-guided refitting.
- [prepare_xian_satellite_overlay.py](../tools/prepare_xian_satellite_overlay.py) — Plotting-only real imagery / frozen polygon transfer; never writes labels.
- [process_xian_single_epoch.py](../tools/process_xian_single_epoch.py) — Process one completed GEID Xi'an epoch into an auditable URRS package.
- [qa_xian_r5_grounded_reconstruction.py](../tools/qa_xian_r5_grounded_reconstruction.py) — Bounded structural QA for one isolated actual-imagery engineering output.
- [quarantine_xian_reference_core_risks.py](../tools/quarantine_xian_reference_core_risks.py) — Remove known-negative image-check blocks; keep source/target risk frames apart.
- [record_shenzhen_geometry_interruption.py](../tools/record_shenzhen_geometry_interruption.py) — Record an interrupted, resumable full-raster geometry attempt.
- [record_sparse_geometry_failure.py](../tools/record_sparse_geometry_failure.py) — Record a resumable sparse-geometry failure without touching raw inputs.
- [record_xian_reference_runtime.py](../tools/record_xian_reference_runtime.py) — Record a verified remote Xi'an reference-job transition.
- [redraw_nanzhan_redevelopment_case.py](../tools/redraw_nanzhan_redevelopment_case.py) — redraw nanzhan redevelopment case · sha256, json_dump, percentile_rgb, read_rgb
- [render_xian_fullroi_source_sheets.py](../tools/render_xian_fullroi_source_sheets.py) — Source-only compact mask sheets; no score rank, review verdict or coordinates.
- [render_xian_r5_method_figure.py](../tools/render_xian_r5_method_figure.py) — Render the non-result R5 Stage-1 architecture figure.
- [render_xian_roof_candidate_overlay.py](../tools/render_xian_roof_candidate_overlay.py) — User-authorized display-only roof candidate preview. Never training labels.
- [restore_xian_cloud_joint.py](../tools/restore_xian_cloud_joint.py) — Strict F-drive model/numerical and geometry replay, without rereading GT.
- [rsbuilding_external_vectorize.py](../tools/rsbuilding_external_vectorize.py) — rsbuilding external vectorize · utc_now, write_json, local_utm_epsg, main
- [rsbuilding_runtime_bootstrap.py](../tools/rsbuilding_runtime_bootstrap.py) — rsbuilding runtime bootstrap
- [run_aliyun_postprocess_controller.py](../tools/run_aliyun_postprocess_controller.py) — Wait for the final4 cloud geometry queue and run bounded post-processing.
- [run_globfp_construction_experiments.py](../tools/run_globfp_construction_experiments.py) — Run isolated C-A/C-B0/C-B1 and record Tier-C blocked C-B2/C-B3.
- [run_nanjing_building_construction_integrated.py](../tools/run_nanjing_building_construction_integrated.py) — Build the read-only Nanjing Building + Construction integrated acceptance run.
- [run_nanjing_domain_adaptation_stable_tracking.py](../tools/run_nanjing_domain_adaptation_stable_tracking.py) — Nanjing Building domain-adaptation and stable-tracking controller.
- [run_nanjing_stable_building_construction.py](../tools/run_nanjing_stable_building_construction.py) — Nanjing stable Building tracking and Construction output controller.
- [run_nanjing_zoom20_2019_gt_comparison.py](../tools/run_nanjing_zoom20_2019_gt_comparison.py) — Run the frozen Current Best models on Nanjing Zoom 20 and validate 2019 GT.
- [run_nanjing_zoom21_current_best.py](../tools/run_nanjing_zoom21_current_best.py) — Audit and, when data exists, gate a Registry-only Zoom 21 stress test.
- [run_tongzhou_construction_images_best.py](../tools/run_tongzhou_construction_images_best.py) — run tongzhou construction images best · now, sha256, load_json, write_json
- [run_xian_adaptive_alignment.py](../tools/run_xian_adaptive_alignment.py) — Bounded R5-A audit / GT-free baseline / fail-closed delivery; CPU only.
- [run_xian_arosics_trial.py](../tools/run_xian_arosics_trial.py) — Isolated AROSICS: synthetic calibration before bounded real diagnostics.
- [run_xian_base_mosaic_pair_diagnostic.py](../tools/run_xian_base_mosaic_pair_diagnostic.py) — Measure local image-frame ties among annual 10 m base rasters in one case.
- [run_xian_base_tie_diagnostic.py](../tools/run_xian_base_tie_diagnostic.py) — Render a bounded submeter-to-10m tie diagnostic for one fixed case.
- [run_xian_boundary_after_queue.py](../tools/run_xian_boundary_after_queue.py) — Run final Xi'an boundary accounting after the existing geometry queue drains.
- [run_xian_cloud_followup.py](../tools/run_xian_cloud_followup.py) — Continue the explicit bounded job after its independently frozen S phase.
- [run_xian_cloud_stage_s.py](../tools/run_xian_cloud_stage_s.py) — Frozen phase-S launch, not a scheduler. Existing instance only.
- [run_xian_coherent_support_transfer.py](../tools/run_xian_coherent_support_transfer.py) — Replay frozen equal fields on derived unions; no refitting or scope expansion.
- [run_xian_composed_alignment_candidates.py](../tools/run_xian_composed_alignment_candidates.py) — Compose B04 actual-date-to-base ties with the six-year base pose graph.
- [run_xian_fraction_shadow.py](../tools/run_xian_fraction_shadow.py) — Frozen TRAINROI fraction shadow from native teacher pixels, not preview RGB.
- [run_xian_gt10m_pilot.py](../tools/run_xian_gt10m_pilot.py) — Reproducible, append-only B04 shadow pilot. Never writes protected sources.
- [run_xian_instance_adaptive_admission.py](../tools/run_xian_instance_adaptive_admission.py) — Frozen real-data A2 admission, unresolved inference and vector handoff.
- [run_xian_mosaic_lightglue_field_diagnostic.py](../tools/run_xian_mosaic_lightglue_field_diagnostic.py) — Build a bounded, date-aware ALIKED/LightGlue local-tile field diagnostic.
- [run_xian_mosaic_registration_diagnostic.py](../tools/run_xian_mosaic_registration_diagnostic.py) — Estimate bounded, date-coherent high-resolution mosaic ties (candidate only).
- [run_xian_point_roof_alignment.py](../tools/run_xian_point_roof_alignment.py) — Purposeful raw-roof RGB replay with a frozen cross-descriptor conflict gate.
- [run_xian_r2_aoi.py](../tools/run_xian_r2_aoi.py) — Execute the bounded Xi'an R2 repair set on five frozen AOIs.
- [run_xian_r5_grounded_reconstruction.py](../tools/run_xian_r5_grounded_reconstruction.py) — Read Stage-2 fixed-case imagery and emit isolated R5 reconstruction products.
- [run_xian_r5_integration.py](../tools/run_xian_r5_integration.py) — Freeze explicit module handoffs, then replay them into a fresh R5 run.
- [run_xian_r5_stage1.py](../tools/run_xian_r5_stage1.py) — Run the isolated Xi'an R5 Stage-1 evidence and no-training prototype.
- [run_xian_r5_stage2.py](../tools/run_xian_r5_stage2.py) — Build isolated Stage-2 actual-date evidence for the Xi'an R5 candidate.
- [run_xian_r5_stage2_alignment_diagnostics.py](../tools/run_xian_r5_stage2_alignment_diagnostics.py) — Run SIFT and official ALIKED+LightGlue on identical Stage-2 tile proposals.
- [run_xian_reference_preflight.py](../tools/run_xian_reference_preflight.py) — Append-only current-ROI reference census; no matching, fitting or label writes.
- [run_xian_roof_image_alignment.py](../tools/run_xian_roof_image_alignment.py) — Roof-interior image-grid experiment, explicitly not a ground displacement field.
- [run_xian_roof_reference_candidates.py](../tools/run_xian_roof_reference_candidates.py) — Bounded geometry-only roof correspondence inventory under frozen rules.
- [run_xian_single2019_experiment.py](../tools/run_xian_single2019_experiment.py) — Bounded single-2019 engineering run under explicit nominal-year policy.
- [run_xian_source_partition_alignment.py](../tools/run_xian_source_partition_alignment.py) — Bounded, image-only candidate alignment; never rewrites source or GT.
- [run_xian_source_roof_retention.py](../tools/run_xian_source_roof_retention.py) — Commit source-only semantic retention before any manual-roof shape comparison.
- [run_xian_source_template_alignment.py](../tools/run_xian_source_template_alignment.py) — Frozen source-population roof template replay against unmasked target context.
- [run_xian_spatial_alignment_composition.py](../tools/run_xian_spatial_alignment_composition.py) — Write candidate-only, spatially composed B04 alignment evidence.
- [run_xian_stabilized_reconstruction.py](../tools/run_xian_stabilized_reconstruction.py) — Build an append-only Xi'an three-epoch stabilization workspace.
- [run_xian_supplemental_pixel_audit.py](../tools/run_xian_supplemental_pixel_audit.py) — Audit Xi'an supplemental pixel payloads without modifying any source data.
- [run_xian_supported_ground_repair.py](../tools/run_xian_supported_ground_repair.py) — One frozen support-closed feature repair replay; no threshold/radius sweep.
- [run_xian_temporal_geometry.py](../tools/run_xian_temporal_geometry.py) — Resume-safe queue for Xi'an sparse B0 geometry packages.
- [scan_urrs_dataset_leakage_exhaustive.py](../tools/scan_urrs_dataset_leakage_exhaustive.py) — Exhaustive, audit-first leakage scan for URRS training assets.
- [screen_xian_candidate_smoke.py](../tools/screen_xian_candidate_smoke.py) — Cloud-side B0 smoke screening for candidate 1 km blocks.
- [smoke_xian_cloud_joint.py](../tools/smoke_xian_cloud_joint.py) — Actual TRAIN RGB+native-label CUDA smoke, never an optimizer update.
- [stage_xian_cloud_followup.py](../tools/stage_xian_cloud_followup.py) — Append-only snapshot of this run's followup adapters and local restore job.
- [stage_xian_cloud_native.py](../tools/stage_xian_cloud_native.py) — Stage the existing16 native RGB packets for GT-free release inference.
- [stage_xian_cloud_reference.py](../tools/stage_xian_cloud_reference.py) — Training/evaluation-only reference polygons; never in GT-free input view.
- [surface_state_v1_amp_safe_pipeline_cli.py](../tools/surface_state_v1_amp_safe_pipeline_cli.py) — Resume the formal Surface State V1 pipeline with AMP-safe change BCE.
- [surface_state_v1_cli.py](../tools/surface_state_v1_cli.py) — Independent extension CLI; does not modify the project's existing run.py.
- [surface_state_v1_complete_post_joint_cli.py](../tools/surface_state_v1_complete_post_joint_cli.py) — surface state v1 complete post joint cli · main
- [surface_state_v1_continue_cli.py](../tools/surface_state_v1_continue_cli.py) — surface state v1 continue cli · main
- [surface_state_v1_experiment_cli.py](../tools/surface_state_v1_experiment_cli.py) — surface state v1 experiment cli · main
- [surface_state_v1_final_overlay_cli.py](../tools/surface_state_v1_final_overlay_cli.py) — surface state v1 final overlay cli · main
- [surface_state_v1_final_pipeline_cli.py](../tools/surface_state_v1_final_pipeline_cli.py) — surface state v1 final pipeline cli · main
- [surface_state_v1_guarded_pipeline_cli.py](../tools/surface_state_v1_guarded_pipeline_cli.py) — Run the formal Surface State V1 pipeline with audited runtime guards.
- [surface_state_v1_overlay_mti_cli.py](../tools/surface_state_v1_overlay_mti_cli.py) — surface state v1 overlay mti cli · main
- [surface_state_v1_overlay_smoke.py](../tools/surface_state_v1_overlay_smoke.py) — surface state v1 overlay smoke
- [surface_state_v1_post_temporal_cli.py](../tools/surface_state_v1_post_temporal_cli.py) — surface state v1 post temporal cli · main
- [surface_state_v1_resume_joint_cli.py](../tools/surface_state_v1_resume_joint_cli.py) — surface state v1 resume joint cli · main
- [surface_state_v1_runtime_cli.py](../tools/surface_state_v1_runtime_cli.py) — surface state v1 runtime cli
- [surface_state_v1_tongzhou_cli.py](../tools/surface_state_v1_tongzhou_cli.py) — surface state v1 tongzhou cli · main
- [surface_state_v1_tongzhou_source_cli.py](../tools/surface_state_v1_tongzhou_source_cli.py) — surface state v1 tongzhou source cli · main
- [surface_state_v1_train_cli.py](../tools/surface_state_v1_train_cli.py) — Training CLI for the existing non-invasive SurfaceState v1 extension.
- [tongzhou_full_pipeline.py](../tools/tongzhou_full_pipeline.py) — tongzhou full pipeline · sha256, parse_date, safe_tag, ensure_dirs
- [track_xian_buildings.py](../tools/track_xian_buildings.py) — Auditable same-date canonicalization and cross-date identity tracking.
- [train_building_foundation_v2.py](../tools/train_building_foundation_v2.py) — Train the project-native Building Foundation v2 refinement smoke/pilot.
- [train_globfp_height_from_existing_features.py](../tools/train_globfp_height_from_existing_features.py) — Train H0/H1/H2 from an existing, spatially split morphology feature CSV.
- [train_globfp_height_h0_fallback.py](../tools/train_globfp_height_h0_fallback.py) — Dependency-limited H0 height baseline from completed morphology features.
- [update_xian_stabilized_progress.py](../tools/update_xian_stabilized_progress.py) — Write an evidence-backed progress snapshot for the Xi'an stabilized run.
- [urrs.py](../tools/urrs.py) — urrs · main

### tools/urrs_lite_review

- [__init__.py](../tools/urrs_lite_review/__init__.py) — URRS Lite Review: local-only, resumable review and training-prep tools.
- [check_training_gate.py](../tools/urrs_lite_review/check_training_gate.py) — check training gate · evaluate, main
- [core.py](../tools/urrs_lite_review/core.py) — core · now_iso, stable_hash, atomic_write_text, atomic_write_json
- [export_verified_labels.py](../tools/urrs_lite_review/export_verified_labels.py) — export verified labels · export_verified, main
- [launch_controlled_training.py](../tools/urrs_lite_review/launch_controlled_training.py) — launch controlled training · is_allowed_command, launch, main
- [prepare_review_queues.py](../tools/urrs_lite_review/prepare_review_queues.py) — prepare review queues · prepare, status, main
- [server.py](../tools/urrs_lite_review/server.py) — server · Handler, run, main

### tools/urrs_lite_review/static

- [app.js](../tools/urrs_lite_review/static/app.js) — app
- [index.html](../tools/urrs_lite_review/static/index.html) — index
- [styles.css](../tools/urrs_lite_review/static/styles.css) — styles

- [urrs_review.py](../tools/urrs_review.py) — urrs review · default_review_dir, build_review, main
- [validate_building_foundation_v2.py](../tools/validate_building_foundation_v2.py) — Validate a Building Foundation v2 checkpoint on WHU validation only.
- [validate_nanjing_manual_gt.py](../tools/validate_nanjing_manual_gt.py) — Validate the user-owned Nanjing manual GT GeoPackage, read-only.
- [vectorize_xian_sam_source_candidates.py](../tools/vectorize_xian_sam_source_candidates.py) — Preserve unselected source SAM alternatives, with lossless pixel provenance.
- [verify_xian_cloud_bundle.py](../tools/verify_xian_cloud_bundle.py) — Verify a private, explicit cloud payload before exclusive extraction.
- [verify_xian_feature_delivery.py](../tools/verify_xian_feature_delivery.py) — Append-only audit of returned delivery and prospective rerun guards.
- [verify_xian_fraction_shadow.py](../tools/verify_xian_fraction_shadow.py) — Read-back QA of a completed fraction shadow; no calibration/GT certification.
- [verify_xian_identity_minimal.py](../tools/verify_xian_identity_minimal.py) — Finalize actual-data source-link evidence after regression and read-back.
- [verify_xian_r4_freeze_readonly.py](../tools/verify_xian_r4_freeze_readonly.py) — Read-only SHA recheck of the Stage-2 frozen R4 candidate inputs.
- [verify_xian_r5_acceptance.py](../tools/verify_xian_r5_acceptance.py) — Append-only independent checks of S4.14 serialization and V2 regressions.
- [verify_xian_r5_live_shadow.py](../tools/verify_xian_r5_live_shadow.py) — Append-only post-run checks, including actual downstream deterministic replay.
- [verify_xian_reference_screened_window.py](../tools/verify_xian_reference_screened_window.py) — Fresh readback, complete zip manifest and pixel-rule verification.
- [verify_xian_satellite_overlay.py](../tools/verify_xian_satellite_overlay.py) — Read-only regression checks and append-only receipt for photograph overlays.
- [xian_additive_post_inference.py](../tools/xian_additive_post_inference.py) — Additive, resume-safe post-inference checks for the Xi'an canonical core.
- [xian_alignment_robustness.py](../tools/xian_alignment_robustness.py) — Fixed CPU preprocessing comparison; no roofs, GT, real ties or displacement application.
- [xian_area_statistics.py](../tools/xian_area_statistics.py) — Area statistics for the Xi'an current-best candidate delivery.
- [xian_case_sheets.py](../tools/xian_case_sheets.py) — Render tracking-case contact sheets from the real imagery and real geometry.
- [xian_cloud_a_train.py](../tools/xian_cloud_a_train.py) — Bounded A synthetic/real weak training using the existing torch loop idiom.
- [xian_cloud_bundle.py](../tools/xian_cloud_bundle.py) — Allowlisted immutable training bundle creation and content verification.
- [xian_cloud_export_manifest.py](../tools/xian_cloud_export_manifest.py) — Freeze completed phase outputs for resumable copy; never include raw data.
- [xian_cloud_geo_block_probe.py](../tools/xian_cloud_geo_block_probe.py) — Fixed-window differential performance test; predictions only, no GT.
- [xian_cloud_geo_probe.py](../tools/xian_cloud_geo_probe.py) — Bounded, GT-free performance repro of frozen geometry export calls.
- [xian_cloud_joint_phase.py](../tools/xian_cloud_joint_phase.py) — Small matched S/A joint continuation; native S supervision remains anchored.
- [xian_cloud_public_final.py](../tools/xian_cloud_public_final.py) — One-shot PUBLIC native S evaluation after model freeze; never train/select.
- [xian_cloud_reference_crosswalk.py](../tools/xian_cloud_reference_crosswalk.py) — Keep original manual GT FIDs distinct from derived GPKG feature IDs.
- [xian_cloud_s_train.py](../tools/xian_cloud_s_train.py) — Budget-bound native-roof training. No training or data reads on import.
- [xian_current_best_reporting.py](../tools/xian_current_best_reporting.py) — Build the non-destructive reporting layer for the current Xi'an candidate.
- [xian_figure_tools.py](../tools/xian_figure_tools.py) — Shared rendering toolkit for Xi'an tracking cases and publication figures.
- [xian_geom_cache_build.py](../tools/xian_geom_cache_build.py) — Build a compact on-disk geometry cache for cases, figures and ranking.
- [xian_geometry_source_audit.py](../tools/xian_geometry_source_audit.py) — Audit the three-epoch Geometry sources actually used by the rasterizer.
- [xian_identity_input_audit.py](../tools/xian_identity_input_audit.py) — Read-only identity input inventory; never repairs source databases.
- [xian_identity_minimal.py](../tools/xian_identity_minimal.py) — Bounded single-date source-link audit; all output is append-only.
- [xian_identity_minimal_links.py](../tools/xian_identity_minimal_links.py) — Actual bounded native-grid replay of existing source provenance.
- [xian_identity_panel_data.py](../tools/xian_identity_panel_data.py) — Prepare factual RGB crops and supplementary topology diagnostics.
- [xian_identity_panel_revision.py](../tools/xian_identity_panel_revision.py) — Append-only visual QA revision: distinct contrast cases, original geometry.
- [xian_identity_verify.py](../tools/xian_identity_verify.py) — Regression and read-back verification, append-only to completed run.
- [xian_local_numerical_closure.py](../tools/xian_local_numerical_closure.py) — CPU-only append-only numerical closure. Truth is scorer-only, never estimator input.
- [xian_r5_live_worker.py](../tools/xian_r5_live_worker.py) — Existing local Torch runtime worker for inference and nontraining readback.
- [xian_r5_registration_worker.py](../tools/xian_r5_registration_worker.py) — One-request JSON worker for an OpenCV-capable Python runtime.
- [xian_raster_completeness_audit.py](../tools/xian_raster_completeness_audit.py) — Completeness / QA scan for candidate raster products.
- [xian_raster_fast_run.py](../tools/xian_raster_fast_run.py) — In-memory geometry rasterizer for the Xi'an current-best candidate delivery.
- [xian_raster_resume_run.py](../tools/xian_raster_resume_run.py) — Resumable runner for the Xi'an current-best-candidate rasterization.
- [xian_raster_v2_singlepass.py](../tools/xian_raster_v2_singlepass.py) — Single-pass rasterizer for the Xi'an current-best candidate delivery.
- [xian_rasterize_current_best.py](../tools/xian_rasterize_current_best.py) — Rasterize the latest Xi'an candidate Geometry and IdentityGraph locally.
- [xian_roof_identity_closure.py](../tools/xian_roof_identity_closure.py) — Append-only, bounded relationship view of existing roof/building IDs.
- [xian_status_report.py](../tools/xian_status_report.py) — Generate the live RUN_STATUS.json / RUN_PROGRESS.md for the delivery.
- [xian_three_state_combine.py](../tools/xian_three_state_combine.py) — Combine the per-state interval rasters into the delivered three-state product.
- [xian_tracking_case_pool.py](../tools/xian_tracking_case_pool.py) — Build the tracking-case candidate pool for the delivery.
- [xian_tracking_case_rank.py](../tools/xian_tracking_case_rank.py) — Rank tracking-case candidates by GEOMETRY QUALITY, not by raw size.

## web

交互审阅与静态预览

### web/urrs_review

- [app.js](../web/urrs_review/app.js) — app
- [index.html](../web/urrs_review/index.html) — index
- [index_v2.html](../web/urrs_review/index_v2.html) — index v2
- [review_corrections.js](../web/urrs_review/review_corrections.js) — review corrections
- [review_v2.css](../web/urrs_review/review_v2.css) — review v2
- [review_v2.js](../web/urrs_review/review_v2.js) — review v2
- [seed_temporal.js](../web/urrs_review/seed_temporal.js) — seed temporal
- [styles.css](../web/urrs_review/styles.css) — styles

## work

R3/R4 应用依赖入口

- [run_xian_r3_vision_guided_repair.py](../work/run_xian_r3_vision_guided_repair.py) — Bounded, read-only-input Xi'an R3 vision-guided repair.
- [run_xian_r4_tracking_candidate.py](../work/run_xian_r4_tracking_candidate.py) — Run one bounded Xi'an R4 Tracking-only development candidate.
