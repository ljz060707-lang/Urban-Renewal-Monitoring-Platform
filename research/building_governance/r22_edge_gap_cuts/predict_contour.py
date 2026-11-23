import csv
import numpy as np,torch,cv2
from cuts import *
from exact import Contour as Net, E, B
from prepare import rows
BASE=OLD/'selective_split/limited_fragmentation'
def main():
    rec=json.loads((P/'CANDIDATES_FROZEN.json').read_text())
    reg=rows();prior=json.loads((BASE/'BEFORE_GT.json').read_text())
    model_file=B/'SELECTOR.pt';verify(model_file,json.loads((B/'SELECTOR_FROZEN.json').read_text())['sha256'])
    ck=torch.load(model_file,weights_only=True);net=Net().cuda().eval();net.load_state_dict(ck['model'])
    mani={};logs=[];dest=B/'predictions';dest.mkdir(exist_ok=True)
    for r in rec:
        if r['role']!='XIAN':continue
        key=r['id'];rr=reg[key];v=prior[key];verify(v['path'],v['sha256'])
        with np.load(v['path']) as f:out=f['SPLIT'].copy()
        verify(rr['query_cache_path'],rr['query_cache_sha256'])
        with np.load(rr['query_cache_path']) as f:raw=f['instances'].astype(np.int32)
        with np.load(OLD/'data'/(key+'.npz')) as f:boxes=f['boxes']
        fp=D/(key+'.npz');verify(fp,r['sha'])
        with np.load(fp) as f:feat=f['feat'];g=f['groups'];masks=f['masks'];tags=f['tags']
        feat=np.load(E/(key+'.npy'))
        with torch.no_grad():s=net(((torch.tensor(feat)-ck['mu'])/ck['sd']).cuda()).cpu().numpy()
        mx=int(out.max());preserved=set(np.unique(raw[raw!=out]).tolist())
        for k in np.unique(g):
            q,y0,y1,x0,x1=map(int,boxes[k]);ii=np.where(g==k)[0];best=ii[s[ii].argmax()]
            score=float(s[best]);tag=tags[best]
            action='retain'
            if q in preserved:action='prior_split'
            elif score>=ck['cut']:
                win=np.s_[y0:y1,x0:x1];parent=raw[win]==q
                a=cv2.resize(np.unpackbits(masks[best]).reshape(64,64),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST).astype(bool)&parent
                b=parent&~a
                if min(a.sum(),b.sum())>=max(32,.1*parent.sum()):
                    mx+=1;out[win][a]=mx;action='split'
                else:action='reject_native_area'
            logs.append(dict(window=key,parent=q,candidates=len(ii),score=score,angle=int(tag[0])*30,anchor=int(tag[1]),action=action))
        assert np.array_equal(raw>0,out>0)
        fp=dest/(key+'.npz');np.savez_compressed(fp,SPLIT=out);mani[key]=dict(path=str(fp),sha256=digest(fp))
    with open(B/'DECISIONS.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=logs[0]);w.writeheader();w.writerows(logs)
    save(B/'PREDICTIONS_FROZEN_BEFORE_MAP.json',mani)
if __name__=='__main__':main()
