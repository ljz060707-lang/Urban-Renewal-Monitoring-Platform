"""Exhaustive bounded coordinate-contract hypotheses on FIT labels only."""
import json,warnings
from pathlib import Path
import numpy as np,cv2,rasterio
from prepare_bt import ROOT,PAY,OLD
def main():
    warnings.filterwarnings('ignore',category=rasterio.errors.NotGeoreferencedWarning)
    prior=json.loads((ROOT/'evidence/BT_COORDINATE_CONTRACT_CHECK.json').read_text());allrec=json.loads((ROOT/'evidence/BANDON_T1_DATA_RECEIPT.json').read_text())['records'];byidx={r['pair']:r for r in allrec}
    rows={r['id']:r for r in json.loads((PAY/'config/segmentation_data.json').read_text())['records']};pairs=(OLD/'official_sources/lists/list_BANDON_train.txt').read_text().splitlines()
    stats={};samples={}
    for entry in prior['records']:
        r=byidx[entry['pair']];parts=pairs[r['pair']].split();labs=[cv2.imread(str(PAY/rows[i]['label']),0) for i in r['ids']]
        with rasterio.open(r['flow']) as d:bt=d.read()
        offsets=[]
        for i in [5,6]:
            with rasterio.open(ROOT/'data/bt_direction_offsets/train'/parts[i].lstrip('/')) as d:offsets.append(d.read())
        for anchor in [0,1]:
            for roof_code in [1,2]:
                roof=(labs[anchor][4::8,4::8]==roof_code).astype('uint8');interior=cv2.erode(roof,np.ones((3,3),np.uint8))>0
                valid=interior&np.isfinite(bt[:2]).all(0)&(abs(bt[:2]).max(0)<400)&(np.linalg.norm(bt[:2],axis=0)>8)
                gy,gx=np.where(valid);pos=np.stack([gx*8+4,gy*8+4],1);off0=offsets[anchor][:2,gy,gx].T
                for swap in [False,True]:
                    vec=bt[:2,gy,gx].T
                    if swap:vec=vec[:,::-1]
                    for sign in [-1,1]:
                        key=f'anchor{anchor}_roof{roof_code}_swap{int(swap)}_sign{sign}';dest=pos+sign*vec;xx,yy=np.rint(dest).astype(int).T;tx,ty=np.floor(dest/8).astype(int).T
                        inside=(xx>=0)&(xx<2048)&(yy>=0)&(yy<2048)&(tx>=0)&(tx<256)&(ty>=0)&(ty<256)
                        xx,yy,tx,ty=xx[inside],yy[inside],tx[inside],ty[inside];o0=off0[inside];o1=offsets[1-anchor][:2,ty,tx].T
                        hit=labs[1-anchor][yy,xx]==roof_code;finite=np.isfinite(o0).all(1)&np.isfinite(o1).all(1)&(abs(o0).max(1)<400)&(abs(o1).max(1)<400)
                        error=np.linalg.norm((pos[inside]-o0)-(dest[inside]-o1),axis=1);keep=hit&finite
                        stats.setdefault(key,[]).extend(error[keep].tolist());samples.setdefault(key,[]).extend(hit.tolist())
    result={key:dict(n=len(v),median_footpoint_error_px=float(np.median(v)) if len(v) else None,roof_hit_rate=float(np.mean(samples[key])) if samples[key] else None) for key,v in stats.items()}
    eligible=[(k,v) for k,v in result.items() if v['n']>=100 and v['median_footpoint_error_px']<4 and v['roof_hit_rate']>.8]
    (ROOT/'evidence/BT_EXHAUSTIVE_COORDINATE_CHECK.json').write_text(json.dumps(dict(FIT_only=True,source_parents=32,hypotheses=result,passing_contracts=eligible,status='RESOLVED' if len(eligible)==1 else 'UNRESOLVED'),indent=2))
    print(json.dumps(dict(best=sorted(result.items(),key=lambda x:x[1]['median_footpoint_error_px'] or 1e9)[:5],passing=eligible)),flush=True)
if __name__=='__main__':main()
