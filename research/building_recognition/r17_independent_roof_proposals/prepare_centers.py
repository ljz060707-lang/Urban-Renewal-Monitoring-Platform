"""Build independent whole-frame proposal supervision from known native roof instances."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import traceback

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cm
from scipy import ndimage

R=Path('/root/xian20261111_r17_proposals')
D=R/'data'
T=R/'output'
S=Path('/root/xian20261111_r15/data/native_roof_decoder')
COCO=Path('/root/xian20261111_r10/instance_data/EXPANDED1004.json')
D.mkdir(parents=True,exist_ok=True)
T.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(data,allow_nan=False,indent=2))
    tmp.replace(path)
def anchor(mask):
    yy,xx=np.nonzero(mask)
    x0,x1=int(xx.min()),int(xx.max()+1)
    y0,y1=int(yy.min()),int(yy.max()+1)
    dist=ndimage.distance_transform_edt(np.pad(mask[y0:y1,x0:x1],1))[1:-1,1:-1]
    iy,ix=np.unravel_index(dist.argmax(),dist.shape)
    return [float(x0+ix+.5),float(y0+iy+.5),x0,y0,x1,y1]
def reduce8(x):
    return x.reshape(128,8,128,8).mean(axis=(1,3),dtype='float32')
def build(item):
    cv2.setNumThreads(0)
    record,role,anns=item
    name=record['frame_id']
    folder=D/record['role']/name
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/'FRAME.json'
    if done.exists():
        v=json.loads(done.read_text())
        for k in ['features','targets']:
            assert sha(v[k])==v[k+'_sha256']
        return v
    for p,h in [('label_path','label_sha256'),('query_cache_path','query_cache_sha256')]:
        assert sha(role[p])==role[h]
    assert sha(record['embedding_path'])==record['embedding_sha256'] and record['GT_used_for_generation'] is False
    with np.load(record['embedding_path']) as z:
        embedding=z['image_embedding'][0]
    assert embedding.shape==(256,64,64) and embedding.dtype==np.float32 and np.isfinite(embedding).all()
    feat=folder/'image.npy'
    np.save(feat,embedding)
    with np.load(role['query_cache_path']) as z:
        old=z['instances']
        supported=z['support'].astype(bool)
    label=np.asarray(Image.open(role['label_path']))
    assert set(np.unique(label)).issubset({0,1,255})
    valid=(label!=255)&supported
    for a in anns:
        if a.get('iscrowd'):
            valid &= ~cm.decode(cm.merge(cm.frPyObjects(a['segmentation'],1024,1024))).astype(bool)
    gt_masks=[]
    gt_rows=[]
    heat=np.zeros((128,128),np.float32)
    owner=np.full((128,128),-1,np.int32)
    zero=[]
    gt_union=np.zeros_like(valid)
    box_reg=np.zeros((6,128,128),np.float32)
    no_valid=0
    for a in anns:
        if a.get('iscrowd'):continue
        mask=cm.decode(cm.merge(cm.frPyObjects(a['segmentation'],1024,1024))).astype(bool)&valid
        if not mask.any():
            no_valid+=1
            continue
        v=anchor(mask)
        gt_rows.append(v)
        zero.append(not old[mask].any())
        gt_masks.append(cm.encode(np.asfortranarray(mask.astype('uint8'))))
        gt_union|=mask
    for idx in sorted(range(len(gt_rows)),key=lambda i:(gt_rows[i][4]-gt_rows[i][2])*(gt_rows[i][5]-gt_rows[i][3])):
        x,y,x0,y0,x1,y1=gt_rows[idx]
        iy,ix=int(y//8),int(x//8)
        sigma=float(np.clip(min(x1-x0,y1-y0)/48.,.8,4.))
        radius=int(np.ceil(3*sigma))
        a,b=max(0,iy-radius),min(128,iy+radius+1)
        c,d=max(0,ix-radius),min(128,ix+radius+1)
        yy,xx=np.ogrid[a:b,c:d]
        np.maximum(heat[a:b,c:d],np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*sigma**2)),out=heat[a:b,c:d])
        if owner[iy,ix]>=0:continue
        owner[iy,ix]=idx
        box_reg[:,iy,ix]=[x/8-ix,y/8-iy,*np.log1p(np.array([x-x0,y-y0,x1-x,y1-y])/8)]
    # Only native labelled positives and known background outside every old query.
    # Unlabelled predicted building pixels contribute no negative signal.
    known=valid & (gt_union|((label==0)&(old==0)))
    pos=owner>=0
    known_fraction=reduce8(known);roof_fraction=reduce8(known&gt_union)
    center_valid=(known_fraction==1)|pos
    negative_excluded=int(np.count_nonzero(valid&(old>0)&~gt_union))
    old_rows=[anchor(old==qid) for qid in np.unique(old) if qid]
    tg=folder/'targets.npz'
    np.savez_compressed(tg,heat=heat,center_valid=center_valid,center_owner=pos,box_reg=box_reg,
        known=known_fraction,roof=np.divide(roof_fraction,known_fraction,out=np.zeros_like(roof_fraction),where=known_fraction>0),
        gt_boxes=np.asarray(gt_rows,np.float32).reshape(-1,6),raw_boxes=np.asarray(old_rows,np.float32).reshape(-1,6),
        no_raw_pixel_overlap=np.asarray(zero,bool),valid_native=np.packbits(valid),
        known_native=np.packbits(known),gt_counts=np.asarray([g['counts'] for g in gt_masks],dtype=np.bytes_))
    rec=dict(id=name,role=record['role'],image_sha256=record['image_sha256'],features=str(feat),features_sha256=sha(feat),
        targets=str(tg),targets_sha256=sha(tg),annotated=len(gt_rows),no_valid_pixel_GT=no_valid,
        zero_raw_overlap=sum(zero),score_cell_collisions=len(gt_rows)-int(pos.sum()),excluded_raw_nonGT_native_pixels=negative_excluded,
        source_embedding_sha256=record['embedding_sha256'],role_label_sha256=role['label_sha256'])
    save(done,rec)
    return rec
def main():
    records=json.loads((S/'ALL_IMAGE_ONLY_INPUTS_FROZEN.json').read_text())['records']
    roles={r['image_path']:r for r in json.loads((S/'ROLES_FROZEN_BEFORE_PIXEL_READ.json').read_text())['rows']}
    coco=json.loads(COCO.read_text())
    ids={v['file_name']:v['id'] for v in coco['images']}
    anns=defaultdict(list)
    for a in coco['annotations']:anns[a['image_id']].append(a)
    assert len(records)==1004 and {x['role'] for x in records}=={'FIT','INNER'}
    assert len({r['image_sha256'] for r in records})<=len(records)
    assert not ({r['image_sha256'] for r in records if r['role']=='FIT'} & {r['image_sha256'] for r in records if r['role']=='INNER'})
    inputs=[]
    for r in records:
        role=roles[r['image_path']]
        assert r['role']==role['decoder_role'] and r['frame_id']==role['id']
        inputs.append((r,role,anns[ids[r['image_path']]]))
    save(D/'PROTOCOL.json',dict(roles={'FIT':972,'INNER':32},role_sha256=sha(S/'ROLES_FROZEN_BEFORE_PIXEL_READ.json'),
        source_COCO_sha256=sha(COCO),input_only='generic frozen whole-frame SAM RGB features',no_raw_or_GT_prompts=True,
        annotation='visible roof, ignore 255/crowd; no unlabelled query negatives',center='native interior EDT max; stride 8',
        bbox='native known visible roof half-open extent',native_roof_and_facade_not_map_GT=True,protected_reads=0))
    rows=[]
    with ProcessPoolExecutor(8) as pool:
        for v in pool.map(build,inputs,chunksize=1):
            rows.append(v)
            if len(rows)%25==0:
                save(D/'STATUS.json',dict(event='PREPARE_NATIVE_INDEPENDENT_PROPOSALS',frames=len(rows),pid=os.getpid()))
    summary={}
    for role in ['FIT','INNER']:
        subset=[x for x in rows if x['role']==role]
        summary[role]={k:sum(x[k] for x in subset) for k in ['annotated','zero_raw_overlap','score_cell_collisions','no_valid_pixel_GT','excluded_raw_nonGT_native_pixels']}
        summary[role]['frames']=len(subset)
    save(D/'READY.json',dict(rows=rows,summary=summary,protocol_sha256=sha(D/'PROTOCOL.json')))
    save(D/'STATUS.json',dict(event='COMPLETE',summary=summary))
if __name__=='__main__':
    try:main()
    except Exception:
        save(D/'FAILURE.json',dict(traceback=traceback.format_exc()))
        raise

