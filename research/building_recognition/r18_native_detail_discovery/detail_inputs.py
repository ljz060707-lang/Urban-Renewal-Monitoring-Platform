"""Prepare native RGB and immutable raw foreground for independent discovery experiments."""
import hashlib,json,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
P=Path('/root/xian20261111_r17_proposals')
sys.path.insert(0,str(P/'code'))
from prepare_centers import sha,save
R=Path('/root/xian20261111_r18_detail');D=R/'data';O=R/'output'
D.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
def pack(item):
    source,old=item;folder=D/source['role']/source['frame_id'];folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'RECEIPT.json'
    if receipt.exists():
        r=json.loads(receipt.read_text());assert sha(r['rgb'])==r['rgb_sha'] and sha(r['raw'])==r['raw_sha'];return r
    assert sha(source['image_path'])==source['image_sha256'] and sha(source['cache_path'])==source['cache_sha256']
    rgb=np.asarray(Image.open(source['image_path']).convert('RGB'));assert rgb.shape==(1024,1024,3)
    with np.load(source['cache_path']) as z:raw=z['instances']>0;support=z['support'].astype(bool)
    with np.load(old['targets']) as z:
        pos=z['center_owner'];gt=z['gt_boxes'];miss=z['no_raw_pixel_overlap'];weights=np.ones_like(pos,np.float32);seen=set()
        for j in sorted(range(len(gt)),key=lambda k:(gt[k,4]-gt[k,2])*(gt[k,5]-gt[k,3])):
            ix,iy=np.floor(gt[j,:2]/8).astype(int)
            if (ix,iy) in seen:continue
            assert pos[iy,ix];seen.add((ix,iy))
            if miss[j]:weights[iy,ix]=4.
        assert len(seen)==int(pos.sum())
    img=folder/'RGB.npy';extras=folder/'RAW_AND_WEIGHTS.npz'
    np.save(img,rgb)
    np.savez_compressed(extras,raw=np.packbits(raw),support=np.packbits(support),weights=weights)
    rec=dict(**old,rgb=str(img),raw=str(extras),rgb_sha=sha(img),raw_sha=sha(extras),source_rgb_sha256=source['image_sha256'])
    save(receipt,rec);return rec
def main():
    source=json.loads(Path('/root/xian20261111_r15/data/native_roof_decoder/ALL_IMAGE_ONLY_INPUTS_FROZEN.json').read_text())['records']
    old=json.loads((P/'data/READY.json').read_text())['rows'];by={r['id']:r for r in old}
    assert {v['frame_id'] for v in source}==set(by)
    rows=[]
    with ProcessPoolExecutor(8) as pool:
        for r in pool.map(pack,[(v,by[v['frame_id']]) for v in source]):
            rows.append(r)
            if len(rows)%50==0:save(D/'STATUS.json',dict(event='PACK_NATIVE_RGB_AND_MODEL_FOREGROUND',frames=len(rows),time=time.time()))
    save(D/'READY.json',dict(rows=rows,R17_sha256=sha(P/'data/READY.json'),protected_reads=0))
if __name__=='__main__':
    try:main()
    except Exception:
        save(D/'FAILURE.json',dict(traceback=traceback.format_exc()))
        raise

