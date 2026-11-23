"""Write the frozen queue, per-epoch manifests and the command bindings."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r'F:\20261028西安市区结果\XIAN_R5_SINGLE_DATE_20261028_01\CODE_SNAPSHOT')

import yaml

from xian_single_date import b0, common

ROOT = common.RUN_ROOT
CFG = ROOT / 'configs'
CFG.mkdir(parents=True, exist_ok=True)


def manifest(epoch):
    with __import__('rasterio').open(epoch['mosaic']) as ds:
        width, height = ds.width, ds.height
        crs = str(ds.crs)
    return {
        'schema_version': 'XIAN_R5_APPLICATION_V1',
        'project_id': 'XIAN-R5-GROUNDREG-FREEZE-0929',
        'run_name': 'SINGLE_DATE_%s' % epoch['epoch_label'],
        'output_root': str(ROOT / 'runs'),
        'product_level': 'research_candidate',
        'run_scope': 'manifest_only',
        'single_date_adapter': {
            'branch': 'GT_FREE_SINGLE_DATE_CITY_SCALE',
            'executed_by': 'CODE_SNAPSHOT/xian_single_date (block adapter)',
            'why_not_the_application_driver':
                'the delivered application driver holds the whole epoch raster in one process and '
                'evaluates per-object source support against the full array; at 15.66 Gpx that is '
                'not memory-safe, so the same frozen protocol is applied blockwise with a '
                'guard band and single-owner assignment',
            'reused_unchanged': ['B0 checkpoint', 'B0 preprocessing', 'tile lattice 256/192',
                                 'triangular fusion weights', 'threshold 0.5',
                                 '8-connectivity components', 'EPSG:32649 metric conversion',
                                 'exact full-cell fraction definition'],
            'candidate_id_rule': 'per-block component index + deterministic observation hash',
        },
        'execution': {
            'code_snapshot': str(ROOT / 'CODE_SNAPSHOT'),
            'device': 'local_cuda', 'gpu_workers': 1, 'module_workers_serial': True,
            'encoding': 'utf-8', 'resume': True, 'overwrite': False,
            'training_enabled': False, 'cloud_enabled': False,
            'download_models_automatically': False,
            'max_input_pixels': int(width) * int(height),
            'max_wallclock_seconds': None,
            'block_policy': {'core_px': common.CORE, 'guard_px': common.GUARD,
                             'edge_margin_px': common.EDGE_MARGIN,
                             'ownership': 'component centroid inside the core block',
                             'tile_batch': common.BATCH},
        },
        'model_routing': {
            'profile': 'best_available_research',
            'routing_manifest': str(common.ROUTING),
            'roof': {'preferred_role': 'B0_RETAINED_TARGET_DOMAIN_BASELINE',
                     'expected_checkpoint_sha256':
                         'f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7',
                     'preprocess_from_manifest': True, 'threshold_from_manifest': True},
            'facade': {'enabled': False, 'reason': 'not executed in this pass'},
            'alignment': {'enabled': False, 'reason': 'V2 not executed in this pass'},
            'tracking': {'runtime_profile': 'DISABLED_FOR_SINGLE_DATE'},
        },
        'inputs': {
            'source_registry': str(ROOT / 'SINGLE_DATE_JOB_QUEUE.json'),
            'prior_observations': None, 'prior_tracks': None,
            'epochs': [{
                'epoch_id': epoch['epoch_id'],
                'nominal_label': epoch['epoch_label'],
                'actual_date': epoch['actual_observation_date'],
                'actual_time_window': None,
                'lr_image': str(common.TEMPLATE),
                'parents': [{
                    'parent_id': epoch['epoch_label'] + '_mosaic',
                    'hr_image': epoch['mosaic'],
                    'hr_building_tif': None,
                    'source_acquisition_id': '%s:%s' % (epoch['epoch_label'],
                                                        epoch['nominal_input_date']),
                    'actual_date': epoch['actual_observation_date'],
                    'source_window': [0, 0, int(width), int(height)],
                    'parent_raw_source': epoch['mosaic'],
                    'date_source': epoch['date_source'],
                    'date_status': epoch['date_status'],
                }],
            }],
        },
        'grid': {'target_lr_grid_template': str(common.TEMPLATE),
                 'inherit_full_geotransform': True, 'native_outputs': True,
                 'map_fine_grid': {'enabled': True,
                                   'subdivision_of_lr_axes': common.FINE_SUBDIVISION},
                 'area_units': 'm2', 'derive_metric_crs_from_inputs': True,
                 'metric_crs': common.METRIC_CRS},
        'm1_recognition': {'use_existing_masks_when_compatible': True,
                           'tile_fusion_from_model_protocol': True,
                           'preserve_full_source_instances': True,
                           'facade_role': 'NOT_RUN_THIS_PASS',
                           'aspect_screen': {'enabled': True, 'mode': 'screen_only',
                                             'measurement': 'metric_minimum_rotated_rectangle',
                                             'truncated_object_action': 'unknown',
                                             'preserve_excluded_layer': True,
                                             'make_nonroof_training_labels': False}},
        'm2_position_area': {'prediction_uses_gt': False, 'compose_transforms_once': True,
                             'source_and_target_frames_required': True,
                             'physical_area_recovery': {
                                 'mode': 'auto_if_evidence',
                                 'target_quantity': 'ROOF_REFERENCE_PLANE_COVERAGE',
                                 'no_camera_geometry_action': 'null_physical_correction_continue_pipeline',
                                 'preserve_raw_area_and_shape': True}},
        'm3_temporal': {'enabled': False, 'status': 'DISABLED_FOR_SINGLE_DATE',
                        'active_recheck': {'enabled': False},
                        'missing_detection_is_absence': False},
        'reference_constrained': {
            'enabled': True, 'mode': 'POST_PROCESS_LINK_ONLY',
            'product': 'ROOF_GT2019_REFERENCE', 'geometry_role': common.GT_ROLE,
            'geometry_copy_into_epoch_product': False,
            'snap_to_reference': False,
            'fitted_variant': 'REFERENCE_FITTED_DEVELOPMENT',
            'identity_proof_from_distance_or_iou': False,
        },
        'products': {'single_date_metric_vectors': True, 'single_date_area_table': True,
                     'exact_fraction': True, 'support_date_quality_layers': True,
                     'formal_supervision_weight': True,
                     'numerical_candidates_not_automatically_qualified': True},
        'evaluation': {'enabled': False, 'gt_mounted_in_prediction_process': False,
                       'reference_manifest': None, 'predictions_must_be_sealed_first': True},
    }


queue = {'schema': 'XIAN_R5_SINGLE_DATE_JOB_QUEUE_V1',
         'run_id': ROOT.name,
         'frozen_utc': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
         'target_aoi': {'template': str(common.TEMPLATE), 'area_km2': 1198.91,
                        'crs': common.METRIC_CRS,
                        'note': 'frozen 10 m LR template inherited from the delivered application'},
         'grid_policy': {'fine_subdivision': common.FINE_SUBDIVISION,
                         'common_anchor_all_epochs': True,
                         'no_cross_epoch_gap_filling': True},
         'reference_job': {'product': 'ROOF_GT2019_REFERENCE',
                           'source': str(common.GT_SHAPEFILE),
                           'geometry_role': common.GT_ROLE, 'reference_year': 2019,
                           'acquisition_date_status': 'UNVERIFIED'},
         'epochs': []}
for e in common.EPOCHS:
    with __import__('rasterio').open(e['mosaic']) as ds:
        w, h = ds.width, ds.height
    plan = b0.block_plan(e['mosaic'])
    queue['epochs'].append(dict(
        epoch_id=e['epoch_id'], epoch_label=e['epoch_label'],
        nominal_input_date=e['nominal_input_date'],
        actual_observation_date=e['actual_observation_date'],
        date_source=e['date_source'], date_status=e['date_status'],
        dominant_share_pct=e['dominant_share_pct'], mixing_severity=e['mixing_severity'],
        assumption=e['assumption'], mosaic=e['mosaic'],
        mosaic_pixels=w * h, mosaic_width=w, mosaic_height=h,
        registered_tiles=e['verified_tile_count'] + e['missing_tile_count'],
        tiles_without_recovered_date=e['missing_tile_count'],
        blocks=len(plan), block_policy={'core_px': common.CORE, 'guard_px': common.GUARD},
        status='QUEUED'))
    (CFG / ('epoch_%s.yaml' % e['actual_observation_date'])).write_text(
        yaml.safe_dump(manifest(e), allow_unicode=True, sort_keys=False), encoding='utf-8')

(CFG / '2019_reference.json').write_text(json.dumps({
    'schema': 'XIAN_R5_REFERENCE_DERIVED_2019_V1',
    'branch': 'REFERENCE_DERIVED_2019',
    'source': str(common.GT_SHAPEFILE),
    'reference_year': 2019,
    'acquisition_date_status': 'UNVERIFIED_NO_2019_HR_IMAGE_IN_REGISTERED_QUEUE',
    'geometry_role': common.GT_ROLE,
    'geometry_source': 'ORIGINAL_GT_FID',
    'alignment_to_designated_reference': 'BY_CONSTRUCTION',
    'independent_absolute_accuracy': 'NOT_MEASURED',
    'grid': {'template': str(common.TEMPLATE), 'crs': common.METRIC_CRS,
             'fine_subdivision': common.FINE_SUBDIVISION},
    'operations_allowed': ['CRS transform', 'AOI clip', 'grid rasterisation'],
    'operations_forbidden': ['V2 translation', 'RGB roof snapping', 'aspect deletion',
                             'buffering', 'simplification', 'model prediction'],
    'outputs': ['ROOF_GT2019_REFERENCE.gpkg', 'ROOF_GT2019_REFERENCE.tif',
                'ROOF_GT2019_INSTANCE_ID.tif', 'ROOF_GT2019_FRACTION_10M.tif',
                'GT2019_OBJECT_AREA.csv', 'GT2019_AREA_SUMMARY.csv',
                'REFERENCE_GEOMETRY_CONSISTENCY.json', 'PRODUCT_MANIFEST.json'],
    'gt_free_inference': False,
    'may_be_used_to_score_itself': False,
}, ensure_ascii=False, indent=2), encoding='utf-8')

(ROOT / 'SINGLE_DATE_JOB_QUEUE.json').write_text(
    json.dumps(queue, ensure_ascii=False, indent=2), encoding='utf-8')

print('configs written:', sorted(p.name for p in CFG.iterdir()))
print('queue epochs:', [e['actual_observation_date'] for e in queue['epochs']])
print('blocks per epoch:', [e['blocks'] for e in queue['epochs']])
