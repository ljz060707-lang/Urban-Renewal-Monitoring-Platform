"""Resolve BT coordinate direction against official ST footprint invariance.
Only 32 inherited FIT parents; this is label-contract checking, not target tuning.
"""
import json,struct,zlib,hashlib
from pathlib import Path
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
import requests,numpy as np,rasterio,cv2
from acquire_bt import ROOT,OLD,Form
from prepare_bt import PAY
def main():
    members=json.loads((OLD/'BANDON_TRAIN_ZIP_MEMBER_INVENTORY.json').read_text());idx={m['name']:m for m in members['members']}
    receipt=json.loads((ROOT/'evidence/BANDON_T1_DATA_RECEIPT.json').read_text());pairs=(OLD/'official_sources/lists/list_BANDON_train.txt').read_text().splitlines()
    selected=[];parents=set()
    for r in receipt['records']:
        if r['role']=='FIT' and r['parent'] not in parents and len(r['samples'])>=8:
            selected.append(r);parents.add(r['parent'])
        if len(selected)==32:break
    names=sorted({'train/'+pairs[r['pair']].split()[i].lstrip('/') for r in selected for i in [5,6]})
    f=Form();f.feed((OLD/'TRAIN_RANGE_PROBE.bin').read_text(encoding='utf8'));url=f.action+'?'+urlencode(f.values)
    dest=ROOT/'data/bt_direction_offsets';dest.mkdir(exist_ok=True);cache={}
    assert sum(idx[n]['compressed_bytes']+1200 for n in names)<128*1024**2
    def fetch(n):
        m=idx[n];out=dest/n
        if out.exists():return n,str(out)
        a=m['offset'];b=a+30+len(n.encode())+1024+m['compressed_bytes']-1
        with requests.get(url+'&range_start='+str(a),headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(20,60)) as response:
            response.raise_for_status();assert response.status_code==206 and response.headers['Content-Range']==f'bytes {a}-{b}/{members["archive_bytes"]}'
            raw=b''.join(response.iter_content(1024**2));assert len(raw)==b-a+1
        v=struct.unpack('<4s5H3I2H',raw[:30]);p=30+v[-2]+v[-1];payload=raw[p:p+m['compressed_bytes']]
        if v[3]==8:payload=zlib.decompress(payload,-15)
        else:assert v[3]==0
        assert len(payload)==m['bytes'] and zlib.crc32(payload)&0xffffffff==m['CRC'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(payload)
        return n,str(out)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,p in pool.map(fetch,names):cache[n]=p
    records=[];all_errors={s:[] for s in [-1,1]};roof_hits={s:[] for s in [-1,1]}
    payload_rows={r['id']:r for r in json.loads((PAY/'config/segmentation_data.json').read_text())['records']}
    for r in selected:
        parts=pairs[r['pair']].split()
        with rasterio.open(r['flow']) as d:bt=d.read()
        offsets=[]
        for i in [5,6]:
            with rasterio.open(cache['train/'+parts[i].lstrip('/')]) as d:offsets.append(d.read())
        second=payload_rows[r['ids'][1]];lab=cv2.imread(str(PAY/second['label']),0)
        row=dict(pair=r['pair'],parent=r['parent'],hypotheses={})
        for sign in [-1,1]:
            errors=[];hits=[]
            for s in r['samples']:
                gx,gy=s['grid_xy'];x,y=s['native_xy'];vector=bt[:2,gy,gx];destxy=np.array([x,y])+sign*vector
                xx,yy=np.rint(destxy).astype(int);tx,ty=np.floor(destxy/8).astype(int)
                if not (0<=xx<2048 and 0<=yy<2048 and 0<=tx<256 and 0<=ty<256):continue
                hit=bool(lab[yy,xx]==1);hits.append(hit)
                o1=offsets[0][:2,gy,gx];o2=offsets[1][:2,ty,tx]
                if hit and np.isfinite(o1).all() and np.isfinite(o2).all() and abs(o1).max()<400 and abs(o2).max()<400 and np.linalg.norm(vector)>8:
                    err=float(np.linalg.norm((np.array([x,y])-o1)-(destxy-o2)));errors.append(err)
            all_errors[sign].extend(errors);roof_hits[sign].extend(hits)
            row['hypotheses'][str(sign)]=dict(n=len(errors),footpoint_error_median_px=float(np.median(errors)) if errors else None,roof_hit_rate=float(np.mean(hits)) if hits else None)
        records.append(row)
    stats={str(s):dict(n=len(all_errors[s]),median_footpoint_error_px=float(np.median(all_errors[s])),roof_hit_rate=float(np.mean(roof_hits[s]))) for s in [-1,1]}
    winner=min([-1,1],key=lambda s:stats[str(s)]['median_footpoint_error_px'])
    best=stats[str(winner)];other=stats[str(-winner)]
    resolved=best['n']>=100 and best['median_footpoint_error_px']<4 and best['median_footpoint_error_px']<other['median_footpoint_error_px']*.25 and best['roof_hit_rate']>.8
    result=dict(status='RESOLVED_BY_OFFICIAL_FOOTPOINT_CONSISTENCY' if resolved else 'UNRESOLVED_DO_NOT_WARP_FOR_TRAINING',
        formula=f'target_native_xy = source_native_xy + ({winner}) * BT_xy' if resolved else None,sign=winner if resolved else None,
        source_fit_only=True,target_pixels_read=False,stats=stats,records=records,
        offset_files=[dict(member=n,path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for n,p in cache.items()],
        provenance='official flow_utils roof->foot uses native point minus ST offset; choose BT orientation only when footpoints agree; no target-based sign selection')
    (ROOT/'evidence/BT_COORDINATE_CONTRACT_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','formula','stats']}),flush=True)
if __name__=='__main__':main()
