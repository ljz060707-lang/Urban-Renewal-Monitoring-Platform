"""Append guarded recommendations and a raw-grid experimental baseline."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import rasterio
from affine import Affine
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import save_json,sha
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.fraction_shadow import QUALITY_BANDS
from xian_r5.single_year_experiment import alignment_experiment_readiness,experimental_pseudolabel,experiment_batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);out=p.parse_args().run.resolve()
    if out.parent!=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment').resolve():
        raise ValueError('Dedicated experiment run required')
    if (out/'2019_EXPERIMENT_DATA_MANIFEST.json').exists():raise FileExistsError('Append-only finalization')
    def read(name):
        with rasterio.open(out/name) as s:return s.read(),Grid(s.transform,s.width,s.height,str(s.crs))
    raw,grid=read('2019_ROOF_FRACTION_RAW.tif');scope,_=read('2019_SCOPE_FRACTION.tif')
    quality,_=read('2019_QUALITY.tif');inputs,_=read('2019_S1S2_INPUT_WINDOW.tif')
    smoke=json.loads((out/'2019_DATALOADER_SMOKE.json').read_text(encoding='utf-8'))
    if smoke['invalid_input_pixels']!=0:raise ValueError('Need explicit original input-validity mask; cannot guess')
    field=json.loads((out/'2019_SOURCE_ALIGNMENT_CANDIDATES.json').read_text(encoding='utf-8'))
    guards=alignment_experiment_readiness(field['cells'])
    flags=np.zeros((256,256),np.uint8);candidate_allow=np.zeros((256,256),bool)
    for cell in guards:
        rr,cc=(int(s[1:]) for s in cell['region_id'].split('_')[1:]);sl=np.s_[rr*64:(rr+1)*64,cc*64:(cc+1)*64]
        if cell['candidate_experiment_supported']:flags[sl]=1;candidate_allow[sl]=True
        elif 'NO_NUMERIC_VECTOR' in cell['reasons']:flags[sl]=0
        else:flags[sl]=4 if len(cell['reasons'])>1 else (3 if 'PEAK_AT_SEARCH_LIMIT' in cell['reasons'] else 2)
    provisional_weight,_=read('2019_EXPERIMENTAL_SAMPLE_WEIGHT.tif')
    guarded_weight=provisional_weight[0]*candidate_allow
    # Independent grid-only baseline, not a guessed zero for unknown candidates.
    base_quality=quality.copy();base_quality[2]=0
    yy,xx=np.indices((256,256));boundary=(xx%64==0)|(xx%64==63)|(yy%64==0)|(yy%64==63)
    base_quality[5]|=boundary.astype(np.uint16)
    full=scope[0]>=1-1e-6
    base_quality[0]=np.where(full,1,2)
    base_quality[0,full&(base_quality[5].astype(bool)|base_quality[6].astype(bool))]=3
    base_quality[0,base_quality[7].astype(bool)]=4
    pseudo=experimental_pseudolabel(raw[0].astype(np.float64),scope[0].astype(np.float64),base_quality,
        np.ones((256,256),bool),allow_unverified_geometry=True,allow_assumed_background=True)
    batch=experiment_batch(inputs,pseudo['label'],pseudo['experiment_weight'])
    extra={'ALIGNMENT_MODE':'RAW_GRID_ONLY_BASELINE_NO_SHIFT','RECOMMENDATION':'DEFAULT_ENGINEERING_BASELINE_NOT_CORRECTED_GT'}
    write(out,'2019_RAW_BASELINE_PSEUDOLABEL_TAU043.tif',pseudo['label'],grid,['EXPERIMENTAL_BASELINE_LABEL'],nodata=255,
        role='RAW_BASELINE_EXPERIMENTAL_PSEUDOLABEL',extra=extra)
    write(out,'2019_RAW_BASELINE_SAMPLE_WEIGHT.tif',pseudo['experiment_weight'],grid,['EXPERIMENT_SAMPLE_SELECTION'],
        role='RAW_BASELINE_EXPERIMENTAL_SELECTION',extra=extra)
    write(out,'2019_RAW_BASELINE_QUALITY.tif',base_quality,grid,QUALITY_BANDS,extra=extra)
    write(out,'2019_CANDIDATE_GUARDED_WEIGHT.tif',guarded_weight,grid,['CANDIDATE_WEIGHT_AFTER_NUMERICAL_RISK_GUARD'],
        extra={'PROVISIONAL_WEIGHT_NOT_RECOMMENDED':'YES'})
    write(out,'2019_ALIGNMENT_NUMERICAL_RISK_FLAGS.tif',flags,grid,['ALIGNMENT_RISK_CODE'],
        extra={'CODES':'0=no_vector;1=numerically_supported_candidate_not_formal;2=no_complete_kernel;3=search_limit;4=both'})
    manifest={'run':out.name,'default_experiment_branch':'RAW_GRID_ONLY_ENGINEERING_BASELINE',
        'label':str(out/'2019_RAW_BASELINE_PSEUDOLABEL_TAU043.tif'),
        'weight':str(out/'2019_RAW_BASELINE_SAMPLE_WEIGHT.tif'),
        'quality':str(out/'2019_RAW_BASELINE_QUALITY.tif'),'inputs':str(out/'2019_S1S2_INPUT_WINDOW.tif'),
        'year':2019,'year_policy':'USER_AUTHORIZED_NOMINAL_YEAR','source_observed_date_retained':'2018-11-01',
        'experimental_counts':pseudo['counts'],'raw_baseline_shift_applied':False,
        'raw_baseline_unknown_candidate_vectors_coerced_to_zero':False,
        'candidate_guards':guards,'candidate_selected_after_guard':int((guarded_weight>0).sum()),
        'candidate_original_label_and_weight':'PROVISIONAL_DIAGNOSTIC_ONLY; use guarded weight, not original provisional weight',
        'scientifically_qualified_gt':0,'independent_accuracy_validation':False,
        'raw_baseline_dataloader_smoke':{'input_shape':list(batch['inputs'].shape),'target_shape':list(batch['target'].shape),
            'ignore_index':-100,'optimization_steps':0,'normalization':'NOT_RUN','model_weights_loaded':False},
        'model_training_performed':False,'existing_formal_alignment_contract_waived':False}
    save_json(out/'2019_EXPERIMENT_DATA_MANIFEST.json',manifest)
    print(json.dumps({'baseline_counts':pseudo['counts'],'guarded_candidate_samples':manifest['candidate_selected_after_guard'],
        'unsupported_numeric_cells':sum('NO_COMPLETE_FEATURE_KERNEL_SUPPORT' in c['reasons'] for c in guards),
        'search_limit_cells':sum('PEAK_AT_SEARCH_LIMIT' in c['reasons'] for c in guards)},ensure_ascii=True))

if __name__=='__main__':main()
