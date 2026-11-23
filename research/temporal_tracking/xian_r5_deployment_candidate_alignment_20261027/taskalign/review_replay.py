"""Recount frozen outputs and reproduce the old evaluator's oracle ceiling."""
import ast,importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
from .contracts import read,write,sha

def old_evaluator(old_root):
    p=old_root/'source/association_metrics.py';spec=importlib.util.spec_from_file_location('frozen_old_association_metrics',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def frozen_score_summary(run):
    edges=[__import__('json').loads(s) for s in (run/'LEARNED_EDGES.jsonl').read_text().splitlines() if s.strip()]
    counts=np.zeros(3,int);mutual=0;minimum=[]
    for e in edges:
        f=np.asarray(e['forward']);r=np.asarray(e['reverse']);counts[f.argmax()]+=1;counts[r.argmax()]+=1
        mutual+=int(f.argmax()==0 and r.argmax()==0);minimum.append(min(float(f[0]),float(r[0])))
    controls=[]
    for line in (run/'LABEL_RECORD_PRESENCE_HEAD.jsonl').read_text().splitlines():
        p=__import__('json').loads(line)
        if 'same_image_self_control' in p:controls.append(p['same_image_self_control'])
    cc=np.bincount(np.argmax(controls,axis=1),minlength=3).tolist() if controls else None
    return dict(candidate_pairs=len(edges),directional_scores=int(counts.sum()),directional_argmax=dict(zip(['SAME_ID','DIFFERENT_ID','NO_LABELLED_COUNTERPART'],counts.tolist())),
        bidirectional_SAME_argmax=mutual,max_min_directional_SAME=max(minimum) if minimum else None,threshold=edges[0]['threshold'] if edges else None,self_control_argmax=cc,
        source_files={str(p):sha(p) for p in [run/'LEARNED_EDGES.jsonl',run/'LABEL_RECORD_PRESENCE_HEAD.jsonl']},model_forwards=0)

def synthetic_query_coverage(old):
    def example(queries):
        rows=[dict(aoi='SYNTHETIC_ONLY',months=[1,2,3,4,5],canonical_id=q,reference_frame=0,window=0,candidate_ids=[[0,1,2]]*5) for q in range(queries)]
        p=np.zeros((queries,5,3,3),np.float32)
        for i in range(queries):
            for k in range(3):p[i,:,k,0 if i==k else 1]=1
        return old.evaluate(p,np.ones((queries,5),bool),rows,0.)
    return dict(status='SYNTHETIC_FUNCTION_TEST_NOT_REAL_ACCURACY',one_query=example(1),all_three_queries=example(3))

def legacy_real_oracle(old_root,old,portable_dir=None):
    receipt=read(old_root/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json');rows=[];metadata=[]
    for r in receipt['shard_records']:
        if r['role']!='CALIBRATION':continue
        p=Path(r['path']).with_suffix('.json');m=read(p);rows.extend(m['rows']);metadata.append(dict(path=str(p),sha256=sha(p),row_count=len(m['rows'])))
    prediction=old_root/'evidence/B2_FULLSPAN_SOURCE_PREDICTIONS.npz'
    with np.load(prediction) as a:y=a['y'];v=a['v'];original_p=a['probs']
    assert len(rows)==len(y)
    p=np.zeros((*y.shape,3),np.float32)
    for i in range(3):p[...,i]=y==i
    p[y<0,2]=1
    for i,r in enumerate(rows):
        for j,ids in enumerate(r['candidate_ids']):
            for k,cid in enumerate(ids):
                if y[i,j,k]<0:continue
                expected=0 if cid==r['canonical_id'] else 2 if cid==-1 else 1
                assert int(y[i,j,k])==expected
    result=old.evaluate(p,v,rows,0.)
    reference=read(old_root/'evidence/ORACLE_PROPOSAL_ASSOCIATION_METRICS.json')['B2_BASELINE']
    reproduced=old.evaluate(original_p,v,rows,0.)
    for key in ['nodes','IDTP','IDFP','IDFN','IDF1','AssA','ID_switches','identity_fragments']:
        if abs(reproduced[key]-reference[key])>1e-12:raise ValueError('Original metadata does not reproduce the sealed baseline: '+key)
    portable={}
    if portable_dir is not None:
        import gzip,json,shutil
        portable_dir.mkdir(parents=True,exist_ok=True)
        rp=portable_dir/'ORIGINAL_QUERY_POOL_ROWS.json.gz'
        with gzip.open(rp,'wt',encoding='utf8') as f:json.dump(rows,f,ensure_ascii=False)
        ap=portable_dir/'ORIGINAL_QUERY_POOL_ARRAYS.npz'
        np.savez_compressed(ap,y=y,v=v,probs=original_p)
        sp=portable_dir/'original_association_metrics_frozen.py'
        shutil.copyfile(old_root/'source/association_metrics.py',sp)
        portable={p.name:sha(p) for p in [rp,ap,sp]}
        write(portable_dir/'MANIFEST.json',dict(scope='EXPOSED_OLD_CAL_CACHED_LABEL_METADATA_ONLY',files=portable,
            raw_images_or_patches_included=False,original_model_predictions=True,new_training_use=False,reference_metrics=reference))
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['aoi'],tuple(r['months']))].append(i)
    coverage=[]
    for (aoi,months),indices in groups.items():
        queries=[i for i in indices if v[i,rows[i].get('reference_local_index',rows[i]['reference_frame']-rows[i]['window'])]]
        qids={rows[i]['canonical_id'] for i in queries}
        nodes={(j,cid) for i in indices for j,cands in enumerate(rows[i]['candidate_ids']) if v[i,j] for cid in cands if cid>=0}
        covered={(j,cid) for j,cid in nodes if cid in qids}
        coverage.append(dict(aoi=aoi,months=list(months),scored_nodes=len(nodes),query_id_covered_nodes=len(covered),
            query_rows=len(queries),unique_query_identities=len(qids),distinct_gold_identities=len({cid for _,cid in nodes}),
            query_id_node_coverage=len(covered)/len(nodes) if nodes else None))
    total=sum(r['scored_nodes'] for r in coverage);covered=sum(r['query_id_covered_nodes'] for r in coverage)
    return dict(status='LEGACY_EXPOSED_CAL_METADATA_REANALYSIS_ONLY',oracle_classifier_original_selector=result,
        total_scored_nodes=total,query_id_covered_nodes=covered,query_id_node_coverage=covered/total,coverage_by_window=coverage,
        rows=len(rows),metadata_sources=metadata,cached_label_array_source=str(prediction),cached_label_array_sha256=sha(prediction),
        sealed_B2_metrics_reproduced_exactly=True,sealed_B2_IDF1=reproduced['IDF1'],
        portable_reproduction_files=portable,
        source_raster_pixels_read=0,original_GT_vector_files_read=0,new_blind_test_claim=False,
        interpretation='Perfect pair-classification attainable result under the frozen sparse-query selector, not a mathematical optimum over every possible tracker')

def old_missing_edge_counterexample(old_root):
    p=old_root/'source/upgrade_application.py';s=p.read_text();tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='graph_partition')
    function='\n'.join(s.splitlines()[node.lineno-1:node.end_lineno]);env=dict(np=np,math=__import__('math'),itertools=__import__('itertools'),
        make_tracks=lambda groups:[dict(observation_ids=[r['observation_id'] for r in g]) for g in groups])
    exec(compile(function,str(p),'exec'),env)
    oo=[dict(observation_id=str(i),epoch_id=str(i)) for i in range(3)]
    ee=[dict(source='0',target='1',same=.99,different=.005,null=.005,eligible=True),dict(source='1',target='2',same=.99,different=.005,null=.005,eligible=True)]
    tracks,edges,status=env['graph_partition'](oo,ee)
    return dict(scope='SYNTHETIC_MISSING_LONG_EDGE_ONLY',source=str(p),source_sha256=sha(p),function_sha256=__import__('hashlib').sha256(function.encode()).hexdigest(),
        serializer_stub_only=True,old_tracks=tracks,old_selected_edges=len(edges),old_solver=status,ground_truth_or_model_calls=0)
