"""Forward path-cover optimization: unknown long links are not cannot-links."""
from collections import defaultdict
import itertools
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from .contracts import digest

def solve_paths(observations,supports,cannot_links=(),time_limit_seconds=60,max_cut_rounds=256):
    byid={r.uid:r for r in observations}
    if len(byid)!=len(observations):raise ValueError('Duplicate nodes')
    dates={r.uid:r.time_key for r in observations};edges=[];seen=set()
    for e in supports:
        a,b=e['source'],e['target']
        if a not in byid or b not in byid:raise ValueError('Unknown edge endpoint')
        if dates[a]==dates[b]:raise ValueError('Same-epoch support is not a temporal link')
        if dates[a]>dates[b]:a,b=b,a
        if (a,b) in seen:raise ValueError('Deduplicate directional scores before path optimization')
        seen.add((a,b));w=float(e['utility'])
        if not np.isfinite(w):raise ValueError('Nonfinite uncalibrated utility')
        if w>0:edges.append({**e,'source':a,'target':b,'utility':w})
    edges.sort(key=lambda e:(dates[e['source']],e['source'],dates[e['target']],e['target']))
    forbidden=[]
    for row in cannot_links:
        if row.get('qualification') not in ('EXPLICIT_VERIFIED_DIFFERENT','SYNTHETIC_TEST_CANNOT_LINK'):
            raise ValueError('A classifier DIFFERENT prediction is not a qualified cannot-link')
        if row['a'] not in byid or row['b'] not in byid:raise ValueError('Unknown cannot-link endpoint')
        forbidden.append((row['a'],row['b']))
    def assemble(selected):
        successor={};predecessor={};selected_map={}
        for k in selected:
            e=edges[k];successor[e['source']]=e['target'];predecessor[e['target']]=e['source'];selected_map[(e['source'],e['target'])]=k
        paths=[]
        for uid in sorted(byid,key=lambda u:(dates[u],u)):
            if uid in predecessor:continue
            path=[uid]
            while path[-1] in successor:path.append(successor[path[-1]])
            if len({byid[u].epoch for u in path})!=len(path):raise AssertionError('Same epoch repeated in track')
            paths.append(path)
        if sum(map(len,paths))!=len(byid):raise AssertionError('Lost observations or cycle')
        return paths,selected_map
    if not edges:paths,_=assemble([]);return _result(paths,[],[],0,0)
    n=len(edges);rows=[];cols=[];vals=[];bounds=[]
    for uid in byid:
        for kind in ['source','target']:
            indexes=[k for k,e in enumerate(edges) if e[kind]==uid]
            if indexes:
                ri=len(bounds);bounds.append(1.)
                for k in indexes:rows.append(ri);cols.append(k);vals.append(1.)
    for a,b in forbidden:
        for k,e in enumerate(edges):
            if {a,b}=={e['source'],e['target']}:
                ri=len(bounds);bounds.append(0.);rows.append(ri);cols.append(k);vals.append(1.)
    cuts=[]
    for iteration in range(max_cut_rounds):
        matrix=coo_matrix((vals,(rows,cols)),shape=(len(bounds),n)).tocsc()
        result=milp(-np.array([e['utility'] for e in edges]),integrality=np.ones(n),bounds=Bounds(0,1),
            constraints=LinearConstraint(matrix,-np.inf,np.array(bounds)),options={'time_limit':time_limit_seconds})
        if not result.success:raise RuntimeError('No proved feasible optimum: '+result.message)
        selected=[k for k,x in enumerate(result.x) if x>.5];paths,selected_map=assemble(selected);violations=[]
        for path in paths:
            position={u:i for i,u in enumerate(path)}
            for a,b in forbidden:
                if a in position and b in position:
                    lo,hi=sorted([position[a],position[b]])
                    indexes=[selected_map[(path[j],path[j+1])] for j in range(lo,hi)]
                    if not indexes:raise ValueError('Cannot-link self endpoint')
                    violations.append(indexes)
        if not violations:
            inferred=[];direct={(e['source'],e['target']) for e in edges}
            for path in paths:
                for i,j in itertools.combinations(range(len(path)),2):
                    a,b=path[i],path[j]
                    if (a,b) not in direct:
                        inferred.append(dict(source=a,target=b,status='RESEARCH_PATH_INFERRED_UNKNOWN_DIRECT_EDGE',
                            support_path=path[i:j+1],independent_confirmation=False,physical_identity_certified=False))
            return _result(paths,[edges[k] for k in selected],inferred,iteration+1,len(cuts))
        for indexes in violations:
            cut=tuple(indexes)
            if cut in cuts:raise RuntimeError('Repeated invalid path after cannot-link cut')
            cuts.append(cut);ri=len(bounds);bounds.append(len(indexes)-1.)
            for k in indexes:rows.append(ri);cols.append(k);vals.append(1.)
    raise RuntimeError('Cannot-link cut budget reached; no unsafe tracks exported')

def _result(paths,edges,inferred,iterations,cuts):
    return dict(tracks=[dict(track_id='TA_'+digest(p)[:24],observation_ids=p,physical_id=None,identity_level='RESEARCH_PATH_SUPPORT') for p in paths],
        accepted_supports=edges,inferred_relations=inferred,solver=dict(status='OPTIMAL' if iterations else 'ALL_SINGLETON_NULL_EXITS',cut_rounds=iterations,cannot_link_path_cuts=cuts),
        missing_edge_semantics='UNKNOWN_NOT_DIFFERENT',all_observations_retained=True,probability_calibration_claim=False)
