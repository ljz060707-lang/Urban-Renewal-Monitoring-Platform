"""Replay four arms in the existing EPSG:32649 polygon and exact fraction evaluators."""
import hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/xian20261111_r16_roi_roof/code')
import evaluate_xian_position_windows as pos
import measure_xian_fraction_windows as frac
R=Path(__file__).resolve().parent
ARMS=['RAW','GROUND','DATE','BOTH']
def one(args):
    r,d,g=args
    boundary,obj,ref=pos.load_window_geometry(r)
    keys=[r['id']+'_q'+str(k) for k in obj]
    # Only coordinate differences are used by the existing evaluator.
    coords={a:{} for a in ARMS}
    for k in keys:
        dt=np.array(d.get(k,(0.,0.)))
        for a,value in zip(ARMS,[(0.,0.),g,dt,dt+g]):coords[a][k]=np.asarray(value)
    pos.ARMS=ARMS
    o=pos.window((r,coords))
    o['no_date_shift']=sum(k not in d for k in keys)
    fractions={}
    for a in ARMS:
        frac.DIRECTORY=R/'four_arm_fraction'/a
        fractions[a]=frac.measure((r,coords[a]))
    return dict(objects=o,fractions=fractions)
def main():
    ground=json.loads((R/'GROUND_IMAGE_ONLY_FROZEN.json').read_text())
    moves={v['moving']:(v['dx_m'],v['dy_m']) for v in ground if v['anchor'].startswith('TUNE_xa180429_')}
    p=pos.SHIFT/'ALL_POSITION_PREDICTIONS.json'
    pos.read_verified(p,json.loads((pos.SHIFT/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json').read_text())['prediction_sha256'])
    src=json.loads(p.read_text())['methods']
    dates={q:np.array(x)-src['RAW'][q] for q,x in src['GEOMETRY__DATE_FIELD'].items()}
    rows=[r for line in (pos.R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))['role']=='TUNE']
    tasks=[(r,{k:v for k,v in dates.items() if k.startswith(r['id']+'_q')},moves.get(r['id'],(0.,0.))) for r in rows]
    with ProcessPoolExecutor(8) as pool:
        results=list(pool.map(one,tasks))
    assert len(results)==72
    summary={}
    for ep in sorted({r['epoch'] for r in rows}):
        rr=[w for w in results if w['objects']['epoch']==ep];summary[ep]={}
        for a in ARMS:
            obs=[o for w in rr for o in w['objects']['results'][a]]
            fs=[w['fractions'][a] for w in rr]
            b=[o for w in rr for o in w['objects']['results']['RAW']]
            matched=[o['distance_m'] for o in obs if o['matched_query'] is not None]
            dist=[o['distance_m'] for o,v in zip(obs,b) if o['matched_query'] is not None and o['matched_query']==v['matched_query']]
            summary[ep][a]=dict(n=len(obs),matched=len(matched),med_on_matches=float(np.median(matched)),p95_on_matches=float(np.quantile(matched,.95)),
                shared_raw_match_n=len(dist),shared_med=float(np.median(dist)),shared_p95=float(np.quantile(dist,.95)),
                mean_best_IoU=float(np.mean([o['best_IoU'] for o in obs])),
                fraction_MAE=sum(f['after_shift_sum_abs_fraction_difference_on_positive_reference_cells'] for f in fs)/sum(f['positive_reference_cells'] for f in fs),
                gained=sum(u['matched_query'] is not None and v['matched_query'] is None for u,v in zip(obs,b)),lost=sum(u['matched_query'] is None and v['matched_query'] is not None for u,v in zip(obs,b)))
    (R/'FOUR_ARMS.json').write_text(json.dumps(summary))
    (R/'FOUR_ARMS_DETAIL.json').write_text(json.dumps(results))
if __name__=='__main__':main()

