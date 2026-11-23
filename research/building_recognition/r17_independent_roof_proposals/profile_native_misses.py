from pathlib import Path
import json
import numpy as np
from prepare_centers import D,T,save
from evaluate_proposals import ious

data=json.loads((D/'READY.json').read_text())
result={}
for role in ['FIT','INNER']:
    rr=[r for r in data['rows'] if r['role']==role]
    observed=[]
    for r in rr:
        with np.load(r['targets']) as z:
            gt=z['gt_boxes']; raw=z['raw_boxes'];miss=z['no_raw_pixel_overlap'];known=z['center_owner']
            wh=gt[:,4:6]-gt[:,2:4]
            best=ious(gt[:,2:],raw[:,2:]).max(1) if len(raw) else np.zeros(len(gt))
            for size,no_overlap,biou in zip(wh.min(1),miss,best):
                observed.append((size,no_overlap,biou))
    a=np.asarray(observed)
    out={}
    for low,hi in [(0,8),(8,16),(16,32),(32,64),(64,float('inf'))]:
        use=(a[:,0]>low)&(a[:,0]<=hi)
        subset=a[use];misses=int(subset[:,1].sum())
        out[f'{low}-{hi}']=dict(native_observations=len(subset),completely_uncovered=misses,
            uncovered_rate=misses/max(1,len(subset)),any_raw_bbox_IoU50=int((subset[:,2]>=.5).sum()))
    result[role]=dict(by_min_native_box_side_px=out,
        median_min_side_px_completely_uncovered=float(np.median(a[a[:,1]>0,0])),
        median_min_side_px_any_coverage=float(np.median(a[a[:,1]==0,0])),
        at_least_one_matching_raw_box_in_spite_of_zero_raw_pixels=int(((a[:,1]>0)&(a[:,2]>=.50)).sum()))
save(T/'SOURCE_COMPLETE_MISS_PROFILE.json',dict(results=result,
    scope='COCO visible roof instances with >=1 known/source-supported pixel; complete miss means no raw foreground on any such pixel; bbox overlap diagnostic only'))
print(json.dumps(result))

