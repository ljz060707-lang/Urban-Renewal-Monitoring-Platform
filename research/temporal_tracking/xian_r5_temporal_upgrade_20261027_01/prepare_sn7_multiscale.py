"""Source-governed native SN7 identity sequences, no Validation/Holdout reads."""
import csv,json,hashlib,re,math,time,argparse
from functools import lru_cache
from shapely import wkb
from pathlib import Path
import numpy as np
import cv2,rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape,mapping,box
ROOT=Path(__file__).resolve().parents[1]
SPLIT=Path('F:/URRS Output/temporal_foundation_v8_20260809_120303/00_data_audit/SPACENET7_V8_SPLIT.csv')
DATA=Path('E:/Data Set/SN7_buildings_train/train')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(maxsize=20000)
def crop_mask(wkb_bytes,side,x,y):
    poly=wkb.loads(wkb_bytes)
    mask=rasterize([(mapping(poly),255)],out_shape=(side,side),transform=Affine.translation(x,y),fill=0,dtype='uint8')
    return cv2.resize(mask,(32,32),interpolation=cv2.INTER_NEAREST)
def crop(rgb,poly,center=None,side=None):
    if center is None: center=(poly.centroid.x,poly.centroid.y)
    if side is None: side=max(16.,max(poly.bounds[2]-poly.bounds[0],poly.bounds[3]-poly.bounds[1])*1.8)
    side=int(math.ceil(side)); x=int(round(center[0]-side/2)); y=int(round(center[1]-side/2))
    # Same deterministic extraction and area interpolation for all candidate labels.
    patch=np.zeros((side,side,3),np.uint8); a=max(x,0); b=max(y,0); c=min(x+side,rgb.shape[1]); d=min(y+side,rgb.shape[0])
    if c>a and d>b: patch[b-y:d-y,a-x:c-x]=rgb[b:d,a:c]
    mask=crop_mask(poly.wkb,side,x,y)
    out=np.concatenate([cv2.resize(patch,(32,32),interpolation=cv2.INTER_AREA),mask[...,None]],-1)
    return out.transpose(2,0,1)
def geometry(a,b,gap,valid):
    scale=max(math.sqrt(a.area),4.)
    return np.array([(b.centroid.x-a.centroid.x)/scale,(b.centroid.y-a.centroid.y)/scale,
        math.log(max(b.area,1)/max(a.area,1)),a.intersection(b).area/max(a.union(b).area,1),
        (b.length-a.length)/max(a.length,1),gap,valid,1.],np.float32).clip(-10,10)
def main():
    cv2.setNumThreads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1);args=parser.parse_args()
    dest=ROOT/'data/sn7_multiscale'; dest.mkdir(exist_ok=True)
    source=list(csv.DictReader(SPLIT.open()))
    train=sorted([r['aoi_id'] for r in source if r['split']=='train'],key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    assert len(train)==42
    roles={a:('TUNE' if i<5 else 'CALIBRATION' if i<10 else 'FIT') for i,a in enumerate(train)}
    protocol=dict(schema='XIAN_TEMPORAL_RESEARCH_V2',split_source=str(SPLIT),split_sha256=sha(SPLIT),
        roles=roles,original_validation_reads=0,locked_holdout_reads=0,sequence_length=5,window_stride=4,
        sampling='four predeclared five-date full-span/large-gap schedules; one building per occupied128px cell PLUS every alpha-supported labelled ID exit/return; union with frozen contiguous shards',
        crop='native RGB+candidate mask -> identical 32px area RGB / nearest mask for every class; no class-specific preprocessing',
        negative='two nearest centroid confirmed different canonical IDs at each current date; vectorized exact centroid ordering; no resampling a pool of 3 negatives',
        unknown='alpha<.98 excludes ID and labelled-presence losses; alpha support is not a cloud/shadow certification',
        presence='canonical label-record presence; not direct roof visibility, not physical demolition',
        historical_exposure='all original TRAIN AOIs were exposed in old research; these are new-run internal source groups, not an independent final test',
        training=dict(max_epochs=30,min_epochs=8,patience=5,batch=32,lr=.0003,weight_decay=.01,seeds=[20261027,20261028,20261029]),
        model='trainable RGB+mask CNN / geometry MLP / bidirectional temporal GRU; no pretrained foundation added',
        target_domain_labels='none; XiAn candidates are not ground truth',
        calibration='CALIBRATION group only, fixed association operating point: maximum 5 percent upper Wilson error bound, >=30 accepted samples; if none pass, reject all and retain U')
    (ROOT/'evidence/TRAINING_PROTOCOL_MULTISCALE.json').write_text(json.dumps(protocol,indent=2))
    count=0; manifests=[]
    for ai,aoi in enumerate(train):
        if ai%args.workers!=args.worker:continue
        out=dest/(aoi+'.npz'); mp=dest/(aoi+'.json')
        if out.exists() and mp.exists(): manifests.append(json.loads(mp.read_text())); continue
        folder=DATA/aoi; labs=sorted((folder/'labels_match_pix').glob('*.geojson'))
        frames=[]; records=[]
        for p in labs:
            stem=p.name.replace('_Buildings.geojson','')
            ip=folder/'images_masked'/(stem+'.tif'); assert ip.is_file(),ip
            feats=json.loads(p.read_text())['features']; polys={}
            for f in feats:
                if f.get('geometry') and f.get('properties',{}).get('Id') is not None:
                    g=shape(f['geometry'])
                    if not g.is_valid:g=g.buffer(0)
                    if not g.is_empty and g.area>=4:polys[int(f['properties']['Id'])]=g
            with rasterio.open(ip) as src: im=src.read()
            assert im.shape[0]>=4
            date=re.search(r'monthly_(\d{4})_(\d{2})',p.name); month=int(date[1])*12+int(date[2])
            keys=np.asarray(list(polys),np.int64); centers=np.array([[polys[k].centroid.x,polys[k].centroid.y] for k in keys]).reshape(-1,2)
            frames.append((im[:3].transpose(1,2,0),im[3]>0,polys,month,keys,centers))
            records.append(dict(image=str(ip),image_sha256=sha(ip),labels=str(p),label_sha256=sha(p),identities=len(polys)))
        qlist=[]; cs=[]; gs=[]; ss=[]; ys=[]; ps=[]; vs=[]; ts=[]; rows=[]
        nframes=len(frames)
        schedules=sorted(set([tuple(np.rint(np.linspace(0,nframes-1,5)).astype(int)),(0,1,nframes//2,nframes-2,nframes-1),(0,1,2,3,nframes-1),(0,nframes-4,nframes-3,nframes-2,nframes-1)]))
        for indices in schedules:
            start=int(indices[0]); fs=[frames[i] for i in indices]; ids=sorted(set().union(*(set(f[2]) for f in fs)))
            selected={}
            for identity in ids:
                fi=next(i for i,f in enumerate(fs) if identity in f[2]); poly=fs[fi][2][identity]; x,y=poly.centroid.x,poly.centroid.y
                if not (0<=x<1024 and 0<=y<1024):continue
                cell=(int(x)//128,int(y)//128); dist=(x-(cell[0]+.5)*128)**2+(y-(cell[1]+.5)*128)**2
                if cell not in selected or (dist,identity)<selected[cell][:2]:selected[cell]=(dist,identity,fi,poly)
            # Add every labelled exit/return with valid alpha support at the event.
            chosen_ids={v[1] for v in selected.values()}
            for identity in ids:
                present=[identity in f[2] for f in fs]
                exits=[j for j in range(4) if present[j] and not present[j+1]]
                supported=False
                for j in exits:
                    gg=fs[j][2][identity];x0,y0,x1,y1=map(int,gg.bounds)
                    slices=(slice(max(0,y0),min(1024,y1+1)),slice(max(0,x0),min(1024,x1+1)))
                    vv=[f[1][slices] for f in [fs[j],fs[j+1]]]
                    if all(v.size and v.mean()>=.98 for v in vv):supported=True
                if supported and identity not in chosen_ids:
                    fi=next(i for i,p in enumerate(present) if p);poly=fs[fi][2][identity]
                    selected[('EXIT',identity)]=(0.,identity,fi,poly);chosen_ids.add(identity)
            for cell,(_,identity,refi,anchor) in sorted(selected.items(),key=lambda item:str(item[0])):
                rgb=fs[refi][0]; q=crop(rgb,anchor); c=[]; g=[]; s=[]; y=[]; pr=[]; va=[]; ti=[]; candidate_ids=[]
                for j,(image,valid,polys,month,keys,centers) in enumerate(fs):
                    minx,miny,maxx,maxy=map(int,anchor.bounds); vv=valid[max(0,miny):min(1024,maxy+1),max(0,minx):min(1024,maxx+1)]
                    vf=float(vv.mean()) if vv.size else 0.; gap=(month-fs[refi][3])/12.
                    distances=((centers-np.array([anchor.centroid.x,anchor.centroid.y]))**2).sum(1)
                    distances[keys==identity]=np.inf
                    different=[int(keys[k]) for k in np.argsort(distances,kind='stable') if keys[k]!=identity][:2]
                    current=[(identity,polys[identity],0)] if identity in polys else [(-1,anchor,2)]
                    current.extend((i,polys[i],1) for i in different)
                    while len(current)<3: current.append((-2,anchor,-100))
                    # Permutation is source keyed, identical class preprocessing.
                    seed=int(hashlib.sha256(f'{aoi}:{start}:{identity}:{j}'.encode()).hexdigest()[:8],16)
                    order=np.random.default_rng(seed).permutation(3); current=[current[k] for k in order]
                    c.append(np.stack([crop(image,p) for _,p,_ in current]));g.append(np.stack([geometry(anchor,p,gap,vf) for _,p,_ in current]))
                    y.append([label if vf>=.98 else -100 for _,_,label in current]); candidate_ids.append([i for i,_,_ in current])
                    s.append(crop(image,anchor,side=max(32.,max(anchor.bounds[2]-anchor.bounds[0],anchor.bounds[3]-anchor.bounds[1])*2.5)))
                    pr.append(float(identity in polys));va.append(vf>=.98);ti.append([gap,float(refi==j)])
                if len(c)!=5:continue
                qlist.append(q);cs.append(c);gs.append(g);ss.append(s);ys.append(y);ps.append(pr);vs.append(va);ts.append(ti)
                rows.append(dict(aoi=aoi,window=start,frame_indices=list(map(int,indices)),reference_local_index=refi,canonical_id=identity,reference_frame=int(indices[refi]),candidate_ids=candidate_ids,months=[f[3] for f in fs]))
        assert qlist,aoi
        np.savez(out,q=np.stack(qlist),c=np.asarray(cs,dtype=np.uint8),g=np.asarray(gs,dtype=np.float32),s=np.asarray(ss,dtype=np.uint8),
            y=np.asarray(ys,dtype=np.int64),p=np.asarray(ps,dtype=np.float32),v=np.asarray(vs,dtype=np.float32),times=np.asarray(ts,dtype=np.float32))
        meta=dict(aoi=aoi,role=roles[aoi],sequences=len(rows),frames=records,rows=rows,path=str(out),sha256=sha(out))
        mp.write_text(json.dumps(meta)); manifests.append(meta); count+=len(rows)
        print(json.dumps(dict(aoi=ai+1,total=42,sequences=len(rows),role=roles[aoi])),flush=True)
    manifests=[json.loads((dest/(a+'.json')).read_text()) for a in train if (dest/(a+'.json')).exists() and (dest/(a+'.npz')).exists()]
    if len(manifests)!=42:return
    (ROOT/'evidence/SN7_MULTISCALE_DATASET_RECEIPT.json').write_text(json.dumps(dict(status='COMPLETE_DEFINED_SAMPLE_COVERAGE',aoi_count=len(manifests),
        sequences=sum(m['sequences'] for m in manifests),roles={r:sum(m['sequences'] for m in manifests if m['role']==r) for r in ('FIT','TUNE','CALIBRATION')},
        all_native_frames_included_in_windows=False,contiguous_dataset_is_required_for_full_frame_coverage=True,all_buildings_claim=False,original_validation_reads=0,locked_holdout_reads=0,
        shard_records=[{k:m[k] for k in ('aoi','role','sequences','path','sha256')} for m in manifests]),indent=2))
if __name__=='__main__': main()
