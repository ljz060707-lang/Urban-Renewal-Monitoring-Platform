"""Source-only association and event diagnostics; GT proposals are explicit."""
import json,sys,math
from pathlib import Path
import numpy as np,torch,rasterio
from shapely.geometry import shape
from upgrade_application import ROOT
from xian_r5.application_geometry import map_geometry
from train_temporal_fullspan import Dataset,metrics,write
from urban_monitor.building_tracker.checkpoint import load_checkpoint
from urban_monitor.building_tracker.contracts import BuildingInstance
from urban_monitor.building_tracker.candidate_gate import GateProfile,_edge
from urban_monitor.building_tracker.features import AssociationFeatureBuilder,standardize_features
def source_b2(role):
    ds=Dataset(role);probs=np.zeros((*ds.arr['y'].shape,3),np.float32);probs[...,2]=1
    path=ROOT/'models/RECOVERED/B2.pth'
    model,mean,std,meta=load_checkpoint(path);profile=GateProfile('G1_balanced',24.,8.,.02,3.,16);fb=AssociationFeatureBuilder(profile=profile)
    cache={};n=0;xs=[];indices=[];gate_rejections=0
    for shard in json.loads((ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json').read_text())['shard_records']:
        if shard['role']!=role:continue
        m=json.loads(Path(shard['path']).with_suffix('.json').read_text());frames=[]
        for f in m['frames']:
            with rasterio.open(f['image']) as src:aff=src.transform;crs=str(src.crs)
            g={int(r['properties']['Id']):map_geometry(shape(r['geometry']),aff) for r in json.loads(Path(f['labels']).read_text())['features'] if r.get('geometry') and r.get('properties',{}).get('Id') is not None}
            frames.append((g,crs))
        for row in m['rows']:
            anchor=frames[row['reference_frame']][0][row['canonical_id']];refmonth=row['months'][row.get('reference_local_index',row['reference_frame']-row['window'])]
            def date(month):return f'{(month-1)//12:04d}-{(month-1)%12+1:02d}-01'
            left=BuildingInstance('SOURCE',m['aoi'],date(refmonth),'Q',None,anchor,frames[0][1],source_kind='gt')
            for j,candidates in enumerate(row['candidate_ids']):
                frame=row.get('frame_indices',list(range(row['window'],row['window']+5)))[j]
                for k,identity in enumerate(candidates):
                    if ds.arr['y'][n,j,k]<0:continue
                    poly=frames[frame][0][identity] if identity>=0 else anchor
                    right=BuildingInstance('SOURCE',m['aoi'],date(row['months'][j]),str(identity),None,poly,frames[0][1],source_kind='gt')
                    e=_edge(left,right,profile)
                    if e is None:gate_rejections+=1;continue
                    xs.append(fb.build_edge(e));indices.append((n,j,k))
            n+=1
    assert n==len(ds)
    x,_,_=standardize_features(xs,mean,std)
    with torch.no_grad():p=model(torch.from_numpy(x)).softmax(-1).numpy()
    for idx,pr in zip(indices,p):probs[idx]=pr
    pre=probs[:,:,:,0].max(2)
    m=metrics(probs,ds.arr['y'],pre,ds.arr['p'],ds.arr['v']);m.update(gate_rejections=gate_rejections,checkpoint=meta['checkpoint_sha256'],proposal_mode='same oracle-proposal sample set, original G1 gate; rejected pair is NO_MATCH prediction; no null flag from GT')
    return ds,dict(probs=probs,y=ds.arr['y'],pre=pre,p=ds.arr['p'],v=ds.arr['v']),m
def sequence_diagnostics(a,ds):
    probabilities=a['probs'];known=a['v'].astype(bool);truth=a['p'].astype(int);candidate=probabilities[:,:,:,0].argmax(2)
    confidence=np.take_along_axis(probabilities[:,:,:,0],candidate[...,None],2)[...,0];pred=probabilities.argmax(-1)
    accepted=(confidence>=.5)&known
    actual_ids=np.array([r['candidate_ids'] for r in ds.meta]);identities=np.array([r['canonical_id'] for r in ds.meta])[:,None]
    chosen_ids=np.take_along_axis(actual_ids,candidate[...,None],2)[...,0]
    correct=(chosen_ids==identities)&accepted
    topclass=np.take_along_axis(pred,candidate[...,None],2)[...,0]
    strict_accept=(topclass==0)&known
    strict_correct=(chosen_ids==identities)&strict_accept
    idtp=int(strict_correct.sum());idfp=int((strict_accept&~strict_correct).sum());idfn=int(((truth==1)&known&~strict_correct).sum())
    id_switches=int(((chosen_ids[:,1:]!=chosen_ids[:,:-1])&accepted[:,1:]&accepted[:,:-1]&known[:,1:]&known[:,:-1]).sum())
    fragments=0
    for row,kt in zip(correct,known):
        segments=0;last=False
        for x,k in zip(row,kt):
            if k and x and not last:segments+=1
            last=bool(k and x)
        fragments+=max(segments-1,0)
    # Presence head is label-record supervision, class thresholds predeclared .5.
    pp=(a['pre']>=.5).astype(int)
    true_event=np.full(truth[:,:-1].shape,3,int);pred_event=np.full(truth[:,:-1].shape,3,int)
    for code,(before,after) in enumerate([(1,1),(0,1),(1,0)]):
        true_event[(truth[:,:-1]==before)&(truth[:,1:]==after)]=code
        pred_event[(pp[:,:-1]==before)&(pp[:,1:]==after)]=code
    # A model positive needs accepted identity support; negatives remain diagnostic.
    positive_supported=(pp==0)|accepted
    pred_event[~(positive_supported[:,:-1]&positive_supported[:,1:])]=4
    valid=known[:,:-1]&known[:,1:];cm=np.zeros((4,5),int);np.add.at(cm,(true_event[valid],pred_event[valid]),1)
    prec=np.diag(cm[:3,:3])/np.maximum(cm[:,:3].sum(0),1);rec=np.diag(cm[:3,:3])/np.maximum(cm[:3].sum(1),1)
    return dict(query_strict_argmax_IDF1=2*idtp/max(2*idtp+idfp+idfn,1),query_strict_argmax_counts=dict(IDTP=idtp,IDFP=idfp,IDFN=idfn),common_half_probability_ID_switches=id_switches,query_window_fragments=fragments,truth_classes=['STABLE','ADDITION','DEMOLITION','NEITHER'],prediction_classes=['STABLE','ADDITION','DEMOLITION','NEITHER','UNKNOWN'],
        event_confusion=cm.tolist(),event_precision=prec.tolist(),event_recall=rec.tolist(),event_support=cm[:3].sum(1).tolist(),
        scope='overlapping five-date query windows with GT candidate proposals; not global end-to-end SCOT/HOTA, not XiAn ground truth',
        full_roof_recall='NOT_MEASURED_SN7_FOOTPRINT_NOT_VISIBLE_ROOF',change_area_error=None)
def main():
    torch.set_num_threads(2);out={}
    ds,a,m=source_b2('CALIBRATION');np.savez(ROOT/'evidence/B2_FULLSPAN_SOURCE_PREDICTIONS.npz',**a);out['B2_BASELINE']=dict(classification=m,sequence=sequence_diagnostics(a,ds),event_head_available=False,event_metrics_status='COMMON_HALF_POINT_PROXY_ONLY_NOT_A_BASELINE_PRESENCE_HEAD')
    for seed in [20261027,20261028,20261029]:
        p=ROOT/'runs/T2_FULL_SPAN'/('seed_'+str(seed))
        with np.load(p/'calibration_predictions.npz') as f:a={k:f[k] for k in f.files}
        out[str(seed)]=dict(classification=json.loads((p/'TRAINING_COMPLETE.json').read_text())['calibration'],sequence=sequence_diagnostics(a,ds))
    write(ROOT/'evidence/SOURCE_COMPARISON_FULLSPAN.json',out)
if __name__=='__main__':main()
