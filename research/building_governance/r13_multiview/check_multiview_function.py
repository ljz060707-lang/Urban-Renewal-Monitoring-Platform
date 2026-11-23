"""Synthetic invariants only; these scores are never actual XiAn accuracy."""
from pathlib import Path
import numpy as np,json
from multiview_reference_model import fit,infer
R=Path(__file__).parent;rng=np.random.default_rng(20261111);a=np.array([[1.,2.],[-1,0],[1,-2]]);v=np.array([[2.,0],[0,3],[-1,-1]]);dates=['2018-04-29','2018-11-01','2019-12-10'];tracks=[]
for i in range(200):
 g=rng.uniform([290000,3770000],[330000,3810000]);h=rng.normal();points={d:g+a[j]+h*v[j]+rng.normal(0,.01,2) for j,d in enumerate(dates)};tracks.append(dict(reference_id='SYNTHETIC_'+str(i),points=points,g=g,location=np.mean(list(points.values()),axis=0)))
model=fit(tracks,False,True,1.,40);errors=[];raw=[]
for i in range(50):
 g=rng.uniform([290000,3770000],[330000,3810000]);h=rng.normal();p={d:g+a[j]+h*v[j] for j,d in enumerate(dates)};out,info=infer(p,model,'consensus');errors.extend(np.linalg.norm(q-g) for q in out.values());raw.extend(np.linalg.norm(q-g) for q in p.values())
assert np.mean(errors)<.3 and np.mean(errors)<np.mean(raw)*.25
single={dates[0]:np.array([5.,8.])};out,_=infer(single,model);assert np.array_equal(out[dates[0]],single[dates[0]])
unknown={'2099-12-31':np.array([5.,8.])};out,_=infer(unknown,model);assert np.array_equal(out['2099-12-31'],unknown['2099-12-31'])
same=dict(model,a=np.zeros_like(np.array(model['a'])).tolist(),v=np.tile([[[1.,2.]]],(3,1,1)).tolist());_,info=infer({d:np.array([10.,10.]) for d in dates},same);assert info['view_contrast_m2']==0 and info['latent_factor']==0
rec=dict(status='PASS_SYNTHETIC_MULTIVIEW_POSITION_INVARIANTS',synthetic_mean_error=float(np.mean(errors)),synthetic_raw_error=float(np.mean(raw)),single_view_raw_fallback=True,unseen_date_raw_fallback=True,identical_view_vectors_no_height_inference=True,real_XIAN_queries=0,real_GT_reads=0,source_pixel_reads=0,synthetic_not_accuracy=True);(R/'SYNTHETIC_FUNCTION_RECEIPT.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(json.dumps(rec,indent=2))
