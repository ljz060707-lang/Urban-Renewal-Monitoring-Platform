"""Real held-development native crop diagnostics; conflicts are not certified errors."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,collections,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from native_temporal_contract import O,save,sha
D=O/'reference_conflict_audit';D.mkdir(exist_ok=True);nodes=json.loads((O/'OBSERVATIONS_ALL.json').read_text());labels=json.loads((O/'SUPERVISION_SEPARATE.json').read_text())['labels'];rgb=O/'rgb_identity';tracks=json.loads((rgb/'training/CONTRASTIVE64_TUNE_TRACKS.json').read_text());z=np.load(rgb/'training/CONTRASTIVE64_TUNE_FEATURES.npz');emb={int(i):v for i,v in zip(z['indices'],z['features'])};manifest=json.loads((rgb/'CROP_MANIFEST.json').read_text());crop_files={r['frame']:r for r in manifest['records']};cache={};window={i:w for w in tracks['windows'] for t in w['tracks'] for i in t};bad=[]
def crop(i):
 n=nodes[i];assert n['role']=='TUNE';frame=n['frame']
 if frame not in cache:
  m=crop_files[frame];p=rgb/m['path'];assert sha(p)==m['sha256'];a=np.load(p);cache[frame]={int(k):v for k,v in zip(a['indices'],a['crops'])}
 return cache[frame][i]
for edge in tracks['accepted_supports']:
 i,j=edge['source'],edge['target'];a=labels.get(str(i));b=labels.get(str(j))
 if not a or not b or a['reference_id']==b['reference_id']:continue
 assert nodes[i]['role']==nodes[j]['role']=='TUNE';w=window[i];expected=[k for t in w['tracks'] for k in t if nodes[k]['period']==nodes[j]['period'] and labels.get(str(k),{}).get('reference_id')==a['reference_id']];assert len(expected)<=1;positive=expected[0] if expected else None;cos=lambda k:float(emb[i]@emb[k]);distance=lambda k:float(np.linalg.norm(np.array(nodes[i]['center_utm'])-nodes[k]['center_utm']));row=dict(source=i,selected=j,same_admitted_reference_candidate=positive,source_uid=nodes[i]['uid'],selected_uid=nodes[j]['uid'],source_reference=a['reference_id'],selected_reference=b['reference_id'],source_date=nodes[i]['actual_date'],selected_date=nodes[j]['actual_date'],source_reference_provider=a['reference_source'],selected_reference_provider=b['reference_source'],selected_cosine=cos(j),selected_distance_m=distance(j),selected_area_ratio=nodes[j]['area_native_pixels']/nodes[i]['area_native_pixels'],selected_is_distinct_map_reference=True,physical_identity_verified=False)
 if positive is not None:row.update(same_ref_cosine=cos(positive),same_ref_distance_m=distance(positive),same_ref_area_ratio=nodes[positive]['area_native_pixels']/nodes[i]['area_native_pixels'],appearance_margin_wrong_minus_same_ref=cos(j)-cos(positive))
 bad.append(row)
save(D/'REFERENCE_CONFLICTS.json',dict(conflicts=bad,count=len(bad),scope='13_DIRECT_EDGES_WITH_TWO_KNOWN_MAP_REFERENCE_LABELS_CONFLICTING_NOT_VERIFIED_PHYSICAL_ID_ERRORS',evaluation_GT_semantics='GEOMETRIC_ADMISSION_CAN_BE_WRONG_OR_AMBIGUOUS_UNDER_PARALLAX_NOT_INDEPENDENT_TRUE_TEMPORAL_ID',protected_reads=0,TUNE_visual_diagnostics_not_blind=True,training_updates=0,source_RGB='HASH_VERIFIED_SAVED_ACTUAL_PREDICTED_QUERY_CROPS'))
selected=sorted([r for r in bad if r['same_admitted_reference_candidate'] is not None],key=lambda r:-r['appearance_margin_wrong_minus_same_ref'])[:6];fig,axes=plt.subplots(len(selected),3,figsize=(10.5,3.3*len(selected)),squeeze=False,constrained_layout=True)
for line,row in enumerate(selected):
 ids=[row['source'],row['selected'],row['same_admitted_reference_candidate']]
 for col,i in enumerate(ids):
  a=axes[line,col];c=crop(i);a.imshow(c[:3].transpose(1,2,0));a.contour(c[3]>0,levels=[.5],colors=['#ff4e38'],linewidths=.65);a.axis('off');prefix=['Source','Model-selected: reference conflict','Same admitted map reference'][col];reference=labels[str(i)]['reference_id'];a.set_title(f"{prefix}\n{nodes[i]['actual_date']} / {reference}",fontsize=9)
  if col:a.text(.5,-.04,f"cos={float(emb[ids[0]]@emb[i]):.3f}; distance={np.linalg.norm(np.array(nodes[ids[0]]['center_utm'])-nodes[i]['center_utm']):.1f} m",transform=a.transAxes,ha='center',fontsize=8)
fig.suptitle('Actual predicted-roof crops; red = model mask, labels = geometrically admitted map references\nDevelopment diagnostics: these references do not independently certify cross-date physical identity',fontsize=11);fig.savefig(D/'ACTUAL_REFERENCE_CONFLICT_CASES.png',dpi=150);plt.close(fig);print('REFERENCE_CONFLICT_AUDIT',len(bad),len(selected),[dict(margin=r['appearance_margin_wrong_minus_same_ref'],wrong_d=r['selected_distance_m'],same_ref_d=r['same_ref_distance_m']) for r in selected])
