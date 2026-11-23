"""BANDON BT-vector auxiliary supervision, never fabricate entity ID labels."""
import json,hashlib,math,warnings
from pathlib import Path
import cv2,numpy as np,rasterio
from shapely.geometry import box
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('F:/xian/XIAN_R5_S4_CALIBRATION_FACADE_PREPARATION_20261019_01')
PAY=Path('F:/xian/XIAN_R5_CLOUD_JOINT_V1_20261020_01/payload')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def native_patch(im,lab,cx,cy):
    x=int(round(cx-64));y=int(round(cy-64));a=max(x,0);b=max(y,0);c=min(x+128,2048);d=min(y+128,2048)
    rgb=np.zeros((128,128,3),np.uint8);mask=np.zeros((128,128),np.uint8)
    rgb[b-y:d-y,a-x:c-x]=im[b:d,a:c];mask[b-y:d-y,a-x:c-x]=(lab[b:d,a:c]==1)*255
    return np.concatenate([cv2.resize(rgb,(32,32),interpolation=cv2.INTER_AREA),cv2.resize(mask,(32,32),interpolation=cv2.INTER_NEAREST)[...,None]],-1).transpose(2,0,1)
def main():
    cv2.setNumThreads(1);warnings.filterwarnings('ignore',category=rasterio.errors.NotGeoreferencedWarning)
    rows=json.loads((PAY/'config/segmentation_data.json').read_text())['records'];idx={r['id']:r for r in rows if r['dataset']=='BANDON'}
    arrays={r:{k:[] for k in ['a','b','flow']} for r in ['FIT','TUNE','CALIBRATION']};records=[];excluded={};parents={}
    for pi,line in enumerate((OLD/'official_sources/lists/list_BANDON_train.txt').read_text().splitlines()):
        parts=line.split(); ids=['BANDON_train_'+p.lstrip('/').rsplit('.',1)[0].replace('/','_') for p in parts[:2]]
        if any(i not in idx for i in ids):excluded['MISSING_PAYLOAD_RECORD']=excluded.get('MISSING_PAYLOAD_RECORD',0)+1;continue
        rr=[idx[i] for i in ids];roles={r['split'] for r in rr}
        if len(roles)!=1 or next(iter(roles)) not in ['TRAIN','TUNE','CALIBRATION']:
            key='EXCLUDED_'+','.join(sorted(roles));excluded[key]=excluded.get(key,0)+1;continue
        role={'TRAIN':'FIT','TUNE':'TUNE','CALIBRATION':'CALIBRATION'}[rr[0]['split']]
        assert rr[0]['parent_scene']==rr[1]['parent_scene'];parent=rr[0]['parent_scene']
        assert parent not in parents or parents[parent]==role;parents[parent]=role
        ims=[];labels=[]
        for r in rr:
            p=PAY/r['image'];l=PAY/r['label'];assert sha(p)==r['image_sha256'] and sha(l)==r['label_sha256']
            ims.append(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB));labels.append(cv2.imread(str(l),0))
        fp=ROOT/'data/bandon_bt/train'/parts[7].lstrip('/')
        with rasterio.open(fp) as src:flow=src.read()
        assert flow.shape==(3,256,256)
        # Exactly mirror official loading's ignore transformation: 255 -> 0,
        # every remaining positive third-band value is ignored (127 is invalid).
        ignore=flow[2].copy();ignore[ignore==255]=0
        roof=cv2.resize((labels[0]==1).astype('uint8'),(256,256),interpolation=cv2.INTER_NEAREST)>0
        valid=(ignore==0)&roof&np.isfinite(flow[:2]).all(0)&(abs(flow[:2]).max(0)<400)
        accepted=[]
        for gy in range(4):
            for gx in range(4):
                ys,xs=np.where(valid[gy*64:(gy+1)*64,gx*64:(gx+1)*64])
                if not len(xs):continue
                j=int(np.argmin((xs-31.5)**2+(ys-31.5)**2)); x=int(xs[j]+gx*64); y=int(ys[j]+gy*64)
                cx=(x+.5)*8;cy=(y+.5)*8
                pair=[native_patch(im,lab,cx,cy) for im,lab in zip(ims,labels)]
                arrays[role]['a'].append(pair[0]);arrays[role]['b'].append(pair[1]);arrays[role]['flow'].append(flow[:2,y,x]/128.)
                accepted.append(dict(grid_xy=[x,y],native_xy=[cx,cy],native_vector=flow[:2,y,x].tolist()))
        records.append(dict(pair=pi,parent=parent,role=role,ids=ids,samples=accepted,flow_sha256=sha(fp),flow=str(fp)))
        if pi%100==0:print(json.dumps(dict(pair=pi,eligible_pairs=len(records),samples={r:len(v['a']) for r,v in arrays.items()})),flush=True)
    for r,a in arrays.items():
        assert a['a'];np.savez(ROOT/'data'/('bt_'+r+'.npz'),a=np.stack(a['a']),b=np.stack(a['b']),flow=np.asarray(a['flow'],np.float32))
    receipt=dict(status='COMPLETE_PREDECLARED_ELIGIBLE_PAIR_COVERAGE',source='official TRAIN pairs, filtered by inherited parent roles',
        acquired_raw_all_official_train=True,raw_download_included_inherited_excluded_roles=True,
        protected_role_training_or_pixel_decode=0,excluded_pairs=excluded,pairs=len(records),samples={r:len(v['a']) for r,v in arrays.items()},records=records,
        semantics='BT vector regression auxiliary only, not entity SAME/DIFFERENT labels; no guessed flow warp direction',
        units='native vectors unchanged by official 256->2048 field interpolation; normalized by128 only for loss',
        patch='one supported source roof point per occupied 4x4 spatial grid cell, same-coordinate paired 128 native pixel crops resized32',
        SpaceNet4='NOT_AVAILABLE_IN_BOUNDED_LOCAL_INVENTORY; not substituted with temporal demolition labels')
    (ROOT/'evidence/BANDON_T1_DATA_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
