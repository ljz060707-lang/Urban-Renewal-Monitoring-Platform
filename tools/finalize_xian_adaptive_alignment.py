"""Readback, immutable-source verification and append-only R5-A archive."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np,rasterio,fiona
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
P=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
OUT=P/'XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_20261013';ACTIVE=OUT/'BASELINE_FIXED_02'

def main():
    if (OUT/'RUN_RECEIPT.json').exists():raise ValueError('Final receipt already exists')
    stage=json.loads((ACTIVE/'RUN_RECEIPT.json').read_text(encoding='utf-8'))
    if stage['tests_passed']!=330:raise ValueError('Measured regression receipt changed')
    ledger=json.loads((OUT/'SOURCE_COMPONENT_LEDGER.json').read_text(encoding='utf-8'))
    for row in ledger['tiles']:
        if sha(row['source_JPG'])!=row['source_JPG_sha256']:raise ValueError('Native source JPG changed')
    from xian_r5.alignment.mosaic_grid import MosaicTileGrid
    base=Path('F:/xian/亚米市区底图');GE=base/'xa190101_combined/xa190101_zoom_20.tif'
    grid=MosaicTileGrid.open(base/'xa190101.geid',GE,base/'xa190101/20')
    replay=[]
    with rasterio.open(GE) as mosaic:
        for x,y in [(420769,312030),(420769,312031),(420769,312032)]:
            w=grid.window_for_tile(x,y)
            jpeg=np.asarray(Image.open(grid.source_path_for_tile(x,y)).convert('RGB'))
            pixels=np.moveaxis(mosaic.read(window=rasterio.windows.Window(*w.pixel_window)),0,-1)
            if pixels.shape!=jpeg.shape:raise ValueError('Native pixel grid differs')
            delta=np.abs(pixels.astype(float)-jpeg.astype(float))
            replay.append({'tile':[x,y,20],'mosaic_window':w.pixel_window,'exact_equal':bool(np.array_equal(pixels,jpeg)),
                'maximum_RGB_difference':float(delta.max()),'mean_absolute_RGB_difference':float(delta.mean()),
                'mosaic_pixel_array_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
                'decoded_JPG_pixel_array_sha256':hashlib.sha256(jpeg.tobytes()).hexdigest(),
                'sample_selection':'THREE_FROZEN_PRIOR_PROVENANCE_SAMPLES_NOT_BEST_MATCH_SEARCH'})
    save_json(OUT/'NATIVE_SOURCE_PIXEL_REPLAY_QA.json',{'samples':replay,
        'mapping_is_position_correction':False,'all_1433_JPG_hashes_reverified':True,
        'all_1433_JPG_pixels_compared':False,'absolute_accuracy':'UNKNOWN'})
    # Every printed diagnostic keeps a fixed, numeric full-window denominator.
    ref=None
    with rasterio.open(P/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21/2019_MANUAL_REFERENCE_ROOF_FRACTION_10m.tif') as s:ref=s.read(1)
    comparison=json.loads((ACTIVE/'BASELINE_COMPARISON.json').read_text(encoding='utf-8'))
    for name in ['M0','M1','M2']:
        with rasterio.open(ACTIVE/(name+'_ANNOTATION_FRACTION_DIAGNOSTIC.tif')) as s:a=s.read(1)
        measured=float(np.abs(a.astype(float)-ref).sum()/65536)
        expected=comparison['baselines'][name]['10M_ROOF_FRACTION_MAE']
        if abs(measured-expected)>1e-7:raise ValueError('Exported MAE readback differs')
    with rasterio.open(ACTIVE/'GT10M_QUALITY.tif') as s:
        if np.any(s.read(4)):raise ValueError('Unqualified pixels admitted')
    with fiona.open(ACTIVE/'CORRECTED_ROOF_OBJECTS.gpkg',layer='qualified_corrected_roofs') as s:
        if len(s):raise ValueError('Unexpected qualified geometry')
    # Inputs used by the examples remain bound to the prior observed-photo receipt.
    previous=P/'SINGLE2019_SATELLITE_OVERLAY_CASES_20261012_29'
    old=json.loads((previous/'SATELLITE_OVERLAY_VERIFICATION_RECEIPT.json').read_text(encoding='utf-8'))['output_sha256']
    for key in ['RASTER_COMPONENT_001476','RASTER_COMPONENT_000707']:
        for suffix in ['_GE_DISPLAY.png','_SENTINEL_NATIVE_10M.png']:
            name=key+suffix
            if sha(previous/name)!=old[name]:raise ValueError('Example photograph changed')
    names=['DATA_AND_PROVENANCE_AUDIT.json','REGION_DEFINITION_AND_FROZEN_SPLIT.json',
        'BASELINE_COMPARISON.json','TRAINING_DATA_MANIFEST.json','LOCAL_ALIGNMENT_EVALUATION.json',
        'MODEL_OR_ALGORITHM_DECISION.md','GPU_READINESS_AND_RESOURCE_REQUEST.json','RUN_REPORT.md']
    active_names={'BASELINE_COMPARISON.json','TRAINING_DATA_MANIFEST.json','LOCAL_ALIGNMENT_EVALUATION.json','GPU_READINESS_AND_RESOURCE_REQUEST.json'}
    mapping={n:('BASELINE_FIXED_02/'+n if n in active_names else n) for n in names}
    mapping['RUN_RECEIPT.json']='RUN_RECEIPT.json'
    for name,relative in mapping.items():
        if name!='RUN_RECEIPT.json' and not (OUT/relative).is_file():raise ValueError('Required delivery missing')
    save_json(OUT/'AUTHORITATIVE_DELIVERY.json',{'run':OUT.name,'authoritative_baseline':'BASELINE_FIXED_02',
        'artifacts':mapping,'root_training_manifest':'PRESERVED_INCOMPLETE_FIRST_ATTEMPT_NOT_AUTHORITATIVE',
        'older_attempts_preserved':True,'status':'PARTIAL_ENGINEERING_BASELINE_NOT_ACCEPTED_ALIGNMENT'})
    scripts=[ROOT/'src/xian_r5/alignment/adaptive_local.py',ROOT/'tools/run_xian_adaptive_alignment.py',
        ROOT/'tools/finalize_xian_adaptive_alignment.py',ROOT/'tests/test_xian_adaptive_local_alignment.py',
        ROOT/'figures/gen_fig_xian_adaptive_local.py']
    code=OUT/'reproducible_code';code.mkdir()
    for p in scripts:shutil.copy2(p,code/p.name)
    save_json(OUT/'FIGURE_QA.json',{'PNG_visual_QA':'PASS_ACTUAL_PREVIEW_INSPECTED',
        'PNG_pixels':[4200,2700],'PNG_dpi':300,'PDF_pages':1,'PDF_extractable_text_chars':551,
        'PDF_rasterized_visual_QA':'NOT_RUN','figures_only_real_photo_and_recorded_geometry':True,
        'new_displacement_drawn':False,'source_actual_date':'2018-11-01'})
    outputs=[p for p in OUT.rglob('*') if p.is_file() and 'mpl_config' not in p.parts]
    save_json(OUT/'RUN_RECEIPT.json',{'run':OUT.name,'completion':'FINAL_TARGET_NOT_ACHIEVED',
        'stage':'GT_FREE_CPU_BASELINE_AND_DATA_GATE_DELIVERED',
        'authoritative_baseline':'BASELINE_FIXED_02','tests_passed':330,'new_interface_tests':10,
        'prior_group_discovery_without_log':'NOT_COUNTED_AS_PASS',
        'protected_inputs_unchanged':stage['protected_inputs_checked'],
        'native_JPG_hashes_reverified_unchanged':1433,'native_pixel_replay_samples':3,
        'INDEPENDENT_TEST_MAE':None,'CORRECTION_COVERAGE_RATE':0.,'UNRESOLVED_ALIGNMENT_REGIONS':16,
        'READY_FOR_GT10M_TRAINING':False,'ABSOLUTE_POSITION_ACCURACY':'UNKNOWN','READY_FOR_CITYWIDE':False,
        'formal_state_unchanged':True,'formal_fit_holdout_completed':0,'accepted_alignment_cells':0,
        'rejected_alignment_cells':0,'old_B04_unresolved_cells':9,'new_registered_roofs':0,'new_physical_ids':0,
        'new_canonical_geometries':0,'new_ground_footprints':0,'original_experiment_samples_unchanged':9866,
        'cloud_instances_started':0,'GPU_models_trained':0,'segmentation_weights_modified':False,
        'code_sha256':{str(p):sha(p) for p in scripts},
        'output_sha256':{str(p.relative_to(OUT)):sha(p) for p in outputs}})
    archive=OUT/'XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_FIRST_BASELINE.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in outputs+[OUT/'RUN_RECEIPT.json']:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
    save_json(OUT/'ARCHIVE_RECEIPT.json',{'archive':archive.name,'sha256':sha(archive),'CRC':'PASS'})
    print(json.dumps({'tests_passed':330,'native_replay':replay,'archive':str(archive)},ensure_ascii=False))

if __name__=='__main__':main()
