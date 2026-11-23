"""Append-only visual QA revision: distinct contrast cases, original geometry."""
import math
from collections import defaultdict
from shapely.geometry import mapping
from xian_identity_minimal import OUT,PREV,read,write,graph_rows

panel=read(OUT/'PANEL_CASES.json');cases=panel['cases']
obs=graph_rows('roof_instance');objects=graph_rows('building_object_reference');refs=graph_rows('manual_roof_reference')
used={c['observation_id'] for i,c in enumerate(cases) if i!=2}
records=read(OUT/'AMBIGUOUS_RECOVERY_DIAGNOSTIC.json')['records']
pool=[r for r in records if r['adjudication']=='COMPETING_LINKS' and r['observation_id'] not in used]
def clarity(r):
    g=obs[r['observation_id']][1]
    return g.area/g.minimum_rotated_rectangle.area if 150<g.area<5000 else -1
r=max(pool,key=lambda r:(clarity(r),r['observation_id']))
oid=r['observation_id'];g=obs[oid][1];x0,y0,x1,y1=g.bounds
side=max(90,x1-x0+50,y1-y0+50);cx=(x0+x1)/2;cy=(y0+y1)/2
ids=[p['building_object_id'] for p in r['source_object_records']]
fids=r['manual_reference_fids']
cases[2]={'title':'竞争几何关联','observation_id':oid,'building_object_ids':ids,'manual_reference_fids':fids,
    'bounds':[cx-side/2,cy-side/2,cx+side/2,cy+side/2],'observed_geometry':mapping(g),
    'object_geometries':{uid:mapping(objects[uid][1]) for uid in ids},
    'manual_geometries':{fid:mapping(refs[fid][1]) for fid in fids},'result':r['adjudication'],
    'evidence_level':r['link_level'],'physical_id':None,'GT_semantics':'2019_MANUAL_ROOF_REFERENCE_NOT_2018_PRESENCE_PROOF'}
assert len({c['observation_id'] for c in cases})==4
write('PANEL_CASES_FIXED_02.json',panel)
write('FIGURE_REVISION_NOTE.json',{'superseded':'IDENTITY_LINK_EVIDENCE_PANEL.png',
    'authoritative':'figure_fixed_02/IDENTITY_LINK_EVIDENCE_PANEL.png',
    'reasons':['FIRST_RENDER_LEGEND_OVERLAPPED_FIRST_TITLE','THIRD_CASE_DUPLICATED_SECOND_CASE'],
    'scientific_results_changed':False})
