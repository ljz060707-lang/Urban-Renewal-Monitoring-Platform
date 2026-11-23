from pathlib import Path
import sys,json,os
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np
P=Path(__file__).resolve().parent
OLD=Path('/root/xian20261111_r21_pair')
sys.path.insert(0,str(OLD))
from prepare import digest,verify,save
D=P/'data';D.mkdir(exist_ok=True)
Y,X=np.indices((64,64))
axes=[]
for ang in (0,30,60,90,120,150):
    a=np.deg2rad(ang)
    u=np.rint(X*np.cos(a)+Y*np.sin(a)).astype(int);u-=u.min()
    v=X*(-np.sin(a))+Y*np.cos(a)
    bands=((v-v.min())*8/(v.max()-v.min()+1)).astype(int)
    axes.append((u,bands))
def choices(img,box):
    mask=img[:,:,3]>0
    if mask.sum()<48:return []
    gray=cv2.cvtColor(img[:,:,:3],cv2.COLOR_RGB2GRAY).astype('float32')/255
    base=cv2.GaussianBlur(gray,(9,9),2)
    dark=np.maximum(base-gray,0)
    edge=np.minimum(1,np.hypot(cv2.Sobel(gray,-1,1,0),cv2.Sobel(gray,-1,0,1))*.7)
    weight=.55-.55*edge-1.5*dark
    yy,xx=np.where(mask);total=len(xx);out=[]
    for j,(u,bands) in enumerate(axes):
        uc=u[mask];nb=int(u.max())+1
        n=np.bincount(uc,minlength=nb);w=np.bincount(uc,weight[mask],minlength=nb)
        cost=np.convolve(w,[.25,.5,.25],'same');cs=n.cumsum()/total
        for mode in (0,1):
            ok=(cs>.13)&(cs<.87)&(n>0)
            if mode==1:ok &= (np.abs(np.arange(nb)-anchor)>max(5,nb*.13))
            if not ok.any():break
            anchor=int(np.argmin(np.where(ok,cost+1.5*np.abs(cs-.5),1e6)))
            # Curve follows a narrow, locally supported roof edge/gap at eight cross sections.
            vals=np.arange(max(0,anchor-5),min(nb,anchor+6))
            bands_cost=np.bincount((bands[mask]*nb+uc),weight[mask],minlength=8*nb).reshape(8,nb)
            path=[]
            for b in range(8):
                prev=path[-1] if path else anchor
                path.append(vals[np.argmin(bands_cost[b,vals]+.18*abs(vals-anchor)+.60*abs(vals-prev))])
            cutline=np.interp(bands,np.arange(8),path)
            part=mask&(u<=cutline);a=part.sum();b=total-a
            if min(a,b)<.10*total:continue
            if any(min(np.count_nonzero(part!=old),total-np.count_nonzero(part!=old))<.05*total for old,_,_ in out):continue
            line=mask&(np.abs(u-cutline)<=1.3)
            if not line.any():continue
            s=gray[part];t=gray[mask&~part]
            def geom(m):
                y,x=np.where(m);return [len(y)/((np.ptp(y)+1)*(np.ptp(x)+1)),x.mean(),y.mean()]
            g,h=geom(part),geom(mask&~part)
            fh=box[2]-box[1];fw=box[4]-box[3]
            feat=[min(a,b)/total,total/4096,line.sum()/np.sqrt(total),gray[line].mean(),edge[line].mean(),dark[line].mean(),edge[mask].mean(),dark[mask].mean(),abs(s.mean()-t.mean()),s.std()+t.std(),min(g[0],h[0]),max(g[0],h[0]),np.hypot(g[1]-h[1],g[2]-h[2])/64,np.log1p(fw*fh)/10,min(fw,fh)/max(fw,fh),np.std(path)/5,abs(np.cos(np.deg2rad(j*30))),abs(cs[anchor]-.5),gray[mask].std()]
            out.append((part,feat,(j,anchor)))
    return out
def generate(rec):
    cv2.setNumThreads(1)
    key=rec['id'];src=OLD/'data'/(key+'.npz');verify(src,rec['sha'])
    with np.load(src) as f:imgs=f['imgs'];boxes=f['boxes']
    groups=[];feats=[];masks=[];tags=[]
    for k,(img,box) in enumerate(zip(imgs,boxes)):
        for part,feat,tag in choices(img,box):
            groups.append(k);feats.append(feat);masks.append(np.packbits(part));tags.append(tag)
    p=D/(key+'.npz')
    np.savez_compressed(p,groups=np.array(groups,np.int32),feat=np.array(feats,np.float32).reshape(-1,19),masks=np.array(masks,np.uint8).reshape(-1,512),tags=np.array(tags,np.uint8).reshape(-1,2))
    return dict(id=key,role=rec['role'],n=len(groups),parents=len(set(groups)),sha=digest(p))
if __name__=='__main__':
    rows=json.loads((OLD/'IMAGE_ONLY_FROZEN.json').read_text())
    with ProcessPoolExecutor(14) as pool:
        rec=[]
        for i,r in enumerate(pool.map(generate,rows,chunksize=1)):
            rec.append(r)
            if i%75==0:print(i,r['role'],flush=True)
    save(P/'CANDIDATES_FROZEN.json',rec)
