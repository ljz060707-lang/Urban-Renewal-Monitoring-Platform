from cuts import *
import csv,torch
from learn import Net
from exact import Contour,B,E
from collections import defaultdict,Counter
rec=list(csv.DictReader((P/'POTENTIAL_CUTS.csv').open()))
out={}
for folder,net,name in [(P,Net(),'edge_gap'),(B,Contour(),'contour')]:
    ck=torch.load(folder/'SELECTOR.pt',weights_only=True);net.load_state_dict(ck['model']);net.eval()
    first=set();passed=set();total=set();windows=defaultdict(list)
    for r in rec:windows[r['window']].append(r)
    for key,items in windows.items():
        with np.load(D/(key+'.npz')) as f:group=f['groups'];feat=f['feat']
        if name=='contour':feat=np.load(E/(key+'.npy'))
        with torch.no_grad():z=net((torch.tensor(feat)-ck['mu'])/ck['sd']).numpy()
        for r in items:
            i=int(r['candidate']);g=group[i];ids=np.flatnonzero(group==g);f=(key,r['fid']);total.add(f)
            if i==ids[z[ids].argmax()]:
                first.add(f)
                if z[i]>=ck['cut']:passed.add(f)
    out[name]=dict(local_possible=len(total),top_ranked=len(first),top_ranked_and_above_cut=len(passed))
save(P/'RANK_REVIEW.json',out)

