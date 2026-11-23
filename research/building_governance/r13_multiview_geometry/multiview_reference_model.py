"""Reference-associated conditional position model; not SFM or physical height."""
import numpy as np,hashlib,collections

def feature(p,center,scale,spatial):
 p=np.asarray(p);return np.column_stack([np.ones(len(p)),(p-center)/scale]) if spatial else np.ones((len(p),1))

def prepare(rows):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['reference_id']].append(r)
 out=[]
 for ref,rr in groups.items():
  dd=collections.defaultdict(list)
  for r in rr:dd[r['actual_date']].append(r)
  points={d:np.median([r['p'] for r in a],axis=0) for d,a in dd.items()};truth=np.median([r['g'] for r in rr],axis=0);assert max(np.linalg.norm(np.asarray([r['g'] for r in rr])-truth,axis=1))<1e-4
  out.append(dict(reference_id=ref,rows=rr,points=points,g=truth,location=np.mean(list(points.values()),axis=0),reference_source=rr[0]['reference_source']))
 return out

def weighted_fit(X,Y,w,ridge=.1):
 return np.linalg.solve(X.T@(X*w[:,None])+ridge*np.eye(X.shape[1]),X.T@(Y*w[:,None]))

def fit(tracks,spatial=False,rank=False,lam=10.,iterations=40):
 dates=sorted(d for d,n in collections.Counter(d for t in tracks for d in t['points']).items() if n>=25);di={d:i for i,d in enumerate(dates)};p=np.array([t['location'] for t in tracks]);center=p.mean(0);scale=np.maximum(p.std(0),1000);B=feature(p,center,scale,spatial);k=B.shape[1];a=np.zeros((len(dates),k,2));v=np.zeros_like(a);D=np.zeros((len(tracks),len(dates),2));known=np.zeros(D.shape[:2],bool)
 for i,t in enumerate(tracks):
  for d,q in t['points'].items():
   if d in di:D[i,di[d]]=q-t['g'];known[i,di[d]]=True
 for j in range(len(dates)):
  ids=np.where(known[:,j])[0];w=np.ones(len(ids));X=B[ids];Y=D[ids,j]
  for _ in range(10):a[j]=weighted_fit(X,Y,w);err=np.linalg.norm(Y-X@a[j],axis=1);w=np.minimum(1,3/np.maximum(err,1e-9))
 baseline=a.copy();multi=known.sum(1)>=2;h=np.zeros(len(tracks));history=[]
 if rank:
  matrix=np.zeros((multi.sum(),len(dates)*2));mi=np.where(multi)[0]
  for jj,i in enumerate(mi):
   for j in np.where(known[i])[0]:matrix[jj,2*j:2*j+2]=D[i,j]-B[i]@a[j]
  u,s,vt=np.linalg.svd(matrix,full_matrices=False);h[mi]=u[:,0]*np.sqrt(len(mi))
  for j in range(len(dates)):v[j,0]=vt[0,2*j:2*j+2]*s[0]/max(1,np.sqrt(len(mi)))
  w=known.astype(float)
  for step in range(iterations):
   av=np.einsum('ik,dkc->idc',B,a);vv=np.einsum('ik,dkc->idc',B,v);h=((w[:,:,None]*vv*(D-av)).sum((1,2)))/((w[:,:,None]*vv*vv).sum((1,2))+lam);h[~multi]=0
   active=h[multi];mean=float(active.mean());sd=max(float(active.std()),1e-6);h[multi]=(active-mean)/sd;a+=mean*v;v*=sd
   for j in range(len(dates)):
    ids=np.where(known[:,j]&multi)[0]
    if len(ids)<10:continue
    X=np.column_stack([B[ids],B[ids]*h[ids,None]]);coef=weighted_fit(X,D[ids,j],w[ids,j],ridge=.5);a[j]=coef[:k];v[j]=coef[k:]
   pred=np.einsum('ik,dkc->idc',B,a)+h[:,None,None]*np.einsum('ik,dkc->idc',B,v);err=np.linalg.norm(D-pred,axis=2);w=known*np.minimum(1,3/np.maximum(err,1e-9));res=err[known&multi[:,None]];history.append(dict(iteration=step+1,mean_reference_residual_m=float(res.mean()),p95_reference_residual_m=float(np.quantile(res,.95)),h_std=float(h[multi].std())))
 bounds=np.quantile(h[multi],[.01,.99]).tolist() if rank else [0.,0.]
 return dict(dates=dates,a=a.tolist(),v=v.tolist(),baseline_a=baseline.tolist(),center=center.tolist(),scale=scale.tolist(),spatial=spatial,rank=rank,lambda_h=lam,h_bounds=bounds,history=history,TRAIN_tracks=len(tracks),TRAIN_multiview_tracks=int(multi.sum()),view_model='ROBUST_DATE_FIELD_PLUS_ONE_LATENT_VIEW_FACTOR_NOT_PHYSICAL_HEIGHT')

def infer(points,model,mode='corrected'):
 # This function cannot receive any reference coordinate or building identity.
 dates=model['dates'];known=[d for d in points if d in dates];result={d:np.asarray(p).copy() for d,p in points.items()};info=dict(supported_dates=len(known),available_dates=len(points),latent_factor=None)
 if not known:return result,info
 loc=np.mean(list(points.values()),axis=0);b=feature([loc],np.asarray(model['center']),np.asarray(model['scale']),model['spatial'])[0];aa=np.asarray([b@np.asarray(model['a'])[dates.index(d)] for d in known]);q=np.asarray([points[d] for d in known]);y=q-aa
 if mode=='raw_mean':g=np.mean(list(points.values()),axis=0);return {d:g.copy() for d in points},info
 if not model['rank']:
  proposed=np.tile(y.mean(0),(len(known),1)) if mode=='consensus' else y
 else:
  if len(known)<2:return result,info
  vv=np.asarray([b@np.asarray(model['v'])[dates.index(d)] for d in known]);dv=vv-vv.mean(0);dy=y-y.mean(0);den=float((dv*dv).sum());h=float((dv*dy).sum()/(den+model['lambda_h']));h=float(np.clip(h,*model['h_bounds']));corrected=y-h*vv;proposed=np.tile(corrected.mean(0),(len(known),1)) if mode=='consensus' else corrected;info.update(latent_factor=h,view_contrast_m2=den)
 for d,p in zip(known,proposed):
  if np.linalg.norm(p-points[d])<=25:result[d]=p
 return result,info

def predict(tracks,model,mode):
 outputs={};diagnostics=[]
 for t in tracks:
  p,info=infer(t['points'],model,mode)
  for r in t['rows']:
   # Corrected mode preserves differences between duplicate-window observations.
   out=p[r['actual_date']] if mode in ['consensus','raw_mean'] else np.asarray(r['p'])+(p[r['actual_date']]-t['points'][r['actual_date']])
   outputs[r['id']]=out
  diagnostics.append(dict(reference_id=t['reference_id'],**info))
 return outputs,diagnostics

def summary(rows,outputs):
 result={}
 for group,rr in [('all',rows),('manual',[r for r in rows if r['reference_source']=='DESIGNATED_GT2019_ANCHOR']),('public',[r for r in rows if r['reference_source']!='DESIGNATED_GT2019_ANCHOR'])]:
  errors=np.array([np.linalg.norm(outputs[r['id']]-r['g']) for r in rr]);result[group]=dict(n=len(rr),mean_error_m=float(errors.mean()) if len(rr) else None,median_error_m=float(np.median(errors)) if len(rr) else None,P95_error_m=float(np.quantile(errors,.95)) if len(rr) else None)
 return result

def inner_holdout(t):
 key=str(int(np.floor(t['location'][0]/1000)))+':'+str(int(np.floor(t['location'][1]/1000)))
 return int(hashlib.sha256(('R13_TRAIN_INNER1KM:'+key).encode()).hexdigest()[:8],16)%5==0
