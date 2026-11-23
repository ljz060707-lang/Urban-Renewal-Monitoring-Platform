"""Actual bounded native-grid replay of existing source provenance."""
import csv
import hashlib
import math
import sqlite3
from collections import Counter,defaultdict
import json
from pathlib import Path
import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely import STRtree
from shapely.geometry import shape,mapping,box
from shapely.ops import transform
from pyproj import Transformer
from xian_r5.single_date_identity import adjudicate,validate_links
from xian_identity_minimal import OUT,PREV,S,ROOT,GE,OBJECTS,COMP,REGISTRY,DATES,CANON,TRACK,read,write,sha,graph_rows


def main():
    audit=read(OUT/'EXISTING_ID_CONNECTION_AUDIT.json'); frozen=read(OUT/'FROZEN_INPUTS.json')
    observations=graph_rows('roof_instance'); objects=graph_rows('building_object_reference')
    gts=graph_rows('manual_roof_reference')
    feature=read(PREV/'IDENTITY_FEATURE_EVIDENCE.json'); candidates=defaultdict(list)
    for row in feature['existing_object_edges']:candidates[row['roof_obs_id']].append(row)
    with (PREV/'ROOF_IDENTITY_MATCH_TABLE.csv').open(encoding='utf-8-sig',newline='') as stream:old_table=list(csv.DictReader(stream))
    old_status={}
    for row in old_table:
        oid=row['roof_obs_id']; state=row['match_status']
        if oid not in old_status or state in ('PLAUSIBLE','UNRESOLVED'):old_status[oid]=state
    assert Counter(old_status.values())=={'PLAUSIBLE':41,'AMBIGUOUS':1047,'UNRESOLVED':574}
    possible=[]
    for oid,status in old_status.items():
        if status!='AMBIGUOUS':continue
        g=observations[oid][1]; overlaps=[]
        for row in candidates[oid]:
            uid=row['reference_id']; h=objects[uid][1]; inter=g.intersection(h).area
            if inter>0:overlaps.append((inter/max(g.area,h.area),uid))
        possible.append({'observation_id':oid,'max_unshifted_overlap_ratio':max([x[0] for x in overlaps],default=0),
            'overlap_object_ids':[uid for ratio,uid in overlaps if ratio>.15],
            'selection_only_not_source_proof':True})
    strongest=sorted((r for r in possible if r['max_unshifted_overlap_ratio']>.98),key=lambda r:r['observation_id'])[:12]
    selected={r['observation_id'] for r in strongest}
    compound=sorted((r for r in possible if len(r['overlap_object_ids'])>1 and r['observation_id'] not in selected),
                    key=lambda r:r['observation_id'])[:12]
    selected.update(r['observation_id'] for r in compound)
    scope=audit['plausible_ids']+sorted(selected)
    write('BOUNDED_SCOPE_FREEZE.json',{'plausible_41':audit['plausible_ids'],'ambiguous_source_replay':strongest,
        'ambiguous_compound_cases':compound,'total_audited_observations':len(scope),
        'selection_before_native_replay':True,'geometry_overlap_is_not_acceptance':True,
        'remaining_ambiguous_not_reprocessed':1047-len(selected),'unresolved_untouched':574})
    reg=read(REGISTRY); tiles=reg['tiles']; sup=[shape(r['support_geometry']) for r in tiles];tree=STRtree(sup)
    tiles_by_obs={}; selected_tile_ids=set()
    for oid in scope:
        g=observations[oid][1]
        ids=[int(i) for i in tree.query(g) if g.intersection(sup[int(i)]).area>1e-8]
        tiles_by_obs[oid]=ids;selected_tile_ids.update(ids)
    needed={tiles[i]['download_tile_id'] for i in selected_tile_ids}; date_rows=defaultdict(list)
    with DATES.open(encoding='utf-8-sig',newline='') as stream:
        for line,row in enumerate(csv.DictReader(stream),2):
            key=f"{row['tile_x']}:{row['tile_y']}:{row['zoom']}"
            if row['task_name']=='xa190101' and key in needed:date_rows[key].append({'line':line,**row})
    checked_tiles={}
    for i in sorted(selected_tile_ids):
        tile=tiles[i];key=tile['download_tile_id'];records=date_rows[key]
        actual=sha(tile['source_JPG'])
        dates={r['actual_capture_date'] for r in records}
        checked_tiles[key]={'source_JPG':tile['source_JPG'],'actual_sha256':actual,
            'expected_sha256':tile['source_JPG_sha256'],'hash_matches':actual==tile['source_JPG_sha256'],
            'registry_date':tile['actual_date'],'date_records':records,
            'date_matches':bool(records) and dates=={tile['actual_date']} and all(r['date_conflict'] in ('0','False','false','') for r in records)}
    write('SOURCE_DATE_HASH_VERIFICATION.json',{'tiles':checked_tiles,'date_csv_sha256':sha(DATES),
        'source_registry_sha256':sha(REGISTRY),'checked_tile_count':len(checked_tiles)})
    canonical=audit['canonical_records_in_existing_scope']; tracking=audit['tracking_records_in_existing_scope']
    to_native=Transformer.from_crs(32649,4326,always_xy=True).transform
    canonical_count_ok=audit['canonical_csv_rows_read_once']==audit['rasterization_manifest']['selection']['total_canonical_objects']
    source_tags=audit['teacher']['tags']; teacher=Path(audit['teacher']['path'])
    same_declared=source_tags.get('geometry_source')==CANON.with_suffix('.gpkg').name and source_tags.get('rasterization_all_touched')=='FALSE' and source_tags.get('area_filter_m2')=='100.0'
    with rasterio.open(GE) as ge,rasterio.open(teacher) as mask:
        grid_same=ge.crs==mask.crs and ge.shape==mask.shape and ge.transform==mask.transform
    results=[]; source_record_ledger=[]; query_cache={}
    with rasterio.open(teacher) as raster,fiona.open(OBJECTS,layer='buildings') as src:
        for oid in scope:
            meta,g=observations[oid]; candidates_for_obs=candidates[oid]
            old_best=next((r['reference_id'] for r in candidates_for_obs if r['is_best_diagnostic']),None)
            native_geom=transform(to_native,g)
            bb=native_geom.bounds; inv=~raster.transform
            points=[inv*(bb[0],bb[1]),inv*(bb[2],bb[3])]
            c0=math.floor(min(p[0] for p in points))-3;r0=math.floor(min(p[1] for p in points))-3
            c1=math.ceil(max(p[0] for p in points))+3;r1=math.ceil(max(p[1] for p in points))+3
            win=Window(c0,r0,c1-c0,r1-r0);aff=raster.window_transform(win);dims=(int(win.height),int(win.width))
            obs_mask=rasterize([(mapping(native_geom),1)],out_shape=dims,transform=aff,all_touched=False,dtype='uint8')>0
            raw=raster.read(1,window=win);positive_ok=bool(obs_mask.any() and np.all(raw[obs_mask]==1))
            # Query source objects spatially, including objects omitted by prior shape scoring.
            source_features=list(src.filter(bbox=g.buffer(1).bounds)); replay=[];exact_ids=[];contributors=[]
            for original in source_features:
                props=dict(original['properties']);uid='xa190101::'+str(props['object_id']);h=shape(original['geometry'])
                if h.area<100:continue
                rec=canonical.get(uid,{}).get('row',{})
                if rec and rec.get('canonical_keep')!='1':continue
                pix=rasterize([(mapping(transform(to_native,h)),1)],out_shape=dims,transform=aff,all_touched=False,dtype='uint8')>0
                inter=int((pix&obs_mask).sum())
                if not inter:continue
                digest=hashlib.sha256(h.wkb).hexdigest(); outside=h.difference(transform(Transformer.from_crs(4326,32649,always_xy=True).transform,box(*rasterio.windows.bounds(win,raster.transform)))).area
                exact=bool(np.array_equal(pix,obs_mask) and outside<1e-6)
                row={'building_object_id':uid,'source_feature_id':original['id'],'source_path':str(OBJECTS),
                    'original_object_id':str(props['object_id']),'original_geometry_sha256':digest,
                    'canonical_geometry_sha256':rec.get('geometry_hash'),'canonical_hash_matches':rec.get('geometry_hash')==digest,
                    'canonical_record_line':canonical.get(uid,{}).get('line_number'),
                    'canonical_record_exists':bool(rec),'overlapping_native_pixels':inter,
                    'observation_native_pixels':int(obs_mask.sum()),'object_native_pixels_in_window':int(pix.sum()),
                    'object_extends_outside_replay_window':outside>1e-6,'native_pixels_exact':exact,
                    'pixel_difference':int((pix!=obs_mask).sum()),'canonical_date':rec.get('local_actual_date'),
                    'object_uid_consistent':rec.get('object_uid')==uid and rec.get('source_object_id')==str(props['object_id'])}
                replay.append(row)
                if exact:exact_ids.append(uid)
                contributors.append(uid)
                query_cache[uid]=(dict(original['properties']),h,original['id'])
            chosen=exact_ids[0] if len(exact_ids)==1 else old_best
            chosen_record=next((r for r in replay if r['building_object_id']==chosen),{})
            cr=canonical.get(chosen,{}).get('row',{});tr=tracking.get(chosen,{}).get('row',{})
            source_tile_records=[checked_tiles[tiles[i]['download_tile_id']] for i in tiles_by_obs[oid]]
            tile_ok=bool(source_tile_records) and all(r['hash_matches'] and r['date_matches'] and r['registry_date']==meta['source_date'] for r in source_tile_records)
            conflicts=[]
            if cr and cr.get('local_actual_date')!=meta['source_date']:conflicts.append('CANONICAL_DATE_DIFFERS_FROM_OBSERVATION')
            if tr and tr.get('local_actual_date')!=meta['source_date']:conflicts.append('TRACKING_DATE_DIFFERS_FROM_OBSERVATION')
            if chosen_record and not chosen_record['canonical_hash_matches']:conflicts.append('ORIGINAL_GEOMETRY_HASH_DIFFERS_FROM_CANONICAL_RECORD')
            if not positive_ok:conflicts.append('OBSERVED_GEOMETRY_NOT_SUPPORTED_BY_TEACHER_PIXELS')
            evidence={'original_id_and_geometry_verified':bool(chosen_record) and chosen in objects and
                    objects[chosen][0]['source_fid']==chosen_record['source_feature_id'] and
                    query_cache[chosen][1].equals_exact(objects[chosen][1],0),
                'canonical_record_geometry_verified':bool(chosen_record.get('canonical_hash_matches')) and chosen_record.get('object_uid_consistent',False),
                'canonical_export_complete':canonical_count_ok,'declared_rasterization_source_matches':same_declared,
                'same_source_grid':grid_same,'date_and_source_hash_verified':tile_ok and cr.get('local_actual_date')==meta['source_date'],
                'tracking_record_consistent':bool(tr) and tr.get('source_object_id')==cr.get('source_object_id') and
                    tr.get('geometry_hash')==cr.get('geometry_hash') and tr.get('local_actual_date')==meta['source_date'] and
                    tr.get('geometry_modified')=='0',
                'native_pixels_exact':chosen in exact_ids and positive_ok,
                'all_relevant_source_objects_checked':all(r['canonical_record_exists'] and r['canonical_hash_matches'] for r in replay),
                'unique_native_object':len(exact_ids)==1 and len(contributors)==1,
                'native_exact_object_count':len(exact_ids),'multiple_contributors':len(contributors)>1,
                'geometric_plausible':old_status[oid]=='PLAUSIBLE','source_conflicts':conflicts}
            decision=adjudicate(evidence)
            record={'observation_id':oid,'roof_obs_id':oid,'building_object_id':chosen,'physical_id':None,
                'historical_candidate_track_id':tr.get('track_id') or None,'baseline_status':old_status[oid],
                'source_actual_date':meta['source_date'],'GT_USED_FOR_IDENTITY':True,'GT_USED_FOR_GEOMETRY':False,
                'manual_reference_fids':sorted({r['roof_gt_id'] for r in old_table if r['roof_obs_id']==oid and r['roof_gt_id']}),
                'evidence':evidence,**decision,'source_object_records':replay,
                'previous_geometric_candidates':[r['reference_id'] for r in candidates_for_obs],
                'replay_native_window':[c0,r0,c1-c0,r1-r0],'teacher_window_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                'source_tile_ids':[tiles[i]['download_tile_id'] for i in tiles_by_obs[oid]],
                'geometry_lineage':'EXISTING_CANONICAL_OBJECT -> NATIVE_BINARY_RASTER -> EXISTING_CONNECTED_COMPONENT',
                'proof_scope':'SINGLE_DATE_SOURCE_ASSOCIATION_ONLY_NOT_ROOF_PURITY_OR_PHYSICAL_ID'}
            results.append(record)
            print(oid,decision['adjudication'],'exact',len(exact_ids),'contributors',len(contributors),flush=True)
    validate_links(results,set(observations),set(objects))
    write('PLAUSIBLE_41_ADJUDICATION.json',{'records':[r for r in results if r['baseline_status']=='PLAUSIBLE'],
        'adjudication_counts':dict(Counter(r['adjudication'] for r in results if r['baseline_status']=='PLAUSIBLE')),
        'GT_role':'MANUAL_ROOF_OUTLINE_GT_REFERENCE_ONLY','physical_ids_created':0})
    write('AMBIGUOUS_RECOVERY_DIAGNOSTIC.json',{'original_count':1047,'processed_count':len(selected),
        'records':[r for r in results if r['baseline_status']=='AMBIGUOUS'],
        'unprocessed_count':1047-len(selected),'unresolved_574_unchanged':True})
    decided={r['observation_id']:r for r in results};ledger=[]
    for oid in sorted(observations):
        row=decided.get(oid); options=candidates[oid]
        by_id={r['reference_id']:r for r in options}
        if row:
            for original in row['source_object_records']:by_id.setdefault(original['building_object_id'],{})
        for uid in sorted(by_id) or [None]:
            trace=bool(row and row['link_level']=='TRACEABLE_SINGLE_DATE_LINK' and row['building_object_id']==uid)
            origin=objects[uid][0] if uid in objects else {}
            ledger.append({'observation_id':oid,'roof_obs_id':oid,'building_object_id':uid,'physical_id':None,
                'historical_candidate_track_id':tracking.get(uid,{}).get('row',{}).get('track_id'),
                'original_object_id':origin.get('original_id'),'source_feature_id':origin.get('source_fid'),
                'source_path':origin.get('source_path'),'source_actual_date':observations[oid][0]['source_date'],
                'baseline_status':old_status[oid],
                'relationship_type':'TRACEABLE_SOURCE_LINK' if trace else row['adjudication'] if row and uid==row['building_object_id'] else
                    'COMPETING_LINKS' if old_status[oid]=='AMBIGUOUS' else 'INSUFFICIENT_EVIDENCE',
                'link_level':'TRACEABLE_SINGLE_DATE_LINK' if trace else 'CANDIDATE_ONLY',
                'evidence_level':'SOURCE_RECORD_AND_EXACT_NATIVE_REPLAY' if trace else 'CANDIDATE_OR_UNAUDITED',
                'audited_this_run':row is not None,'GT_USED_FOR_GEOMETRY':False,
                'manual_reference_fids':json.dumps(row['manual_reference_fids'] if row else sorted({r['roof_gt_id'] for r in old_table if r['roof_obs_id']==oid and r['roof_gt_id']})),
                'source_evidence':'PLAUSIBLE_41_ADJUDICATION.json' if old_status[oid]=='PLAUSIBLE' else 'AMBIGUOUS_RECOVERY_DIAGNOSTIC.json' if row else 'BASELINE_UNCHANGED',
                'failed_gates':json.dumps(row['failed_gates'] if row else ['NOT_AUDITED_THIS_RUN'])})
    with (OUT/'SOURCE_OBJECT_LINK_LEDGER.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    forward=defaultdict(list);reverse=defaultdict(list)
    for r in ledger:
        forward[r['observation_id']].append({'building_object_id':r['building_object_id'],'relationship_type':r['relationship_type'],'source_feature_id':r['source_feature_id'],'source_path':r['source_path']})
        if r['building_object_id']:reverse[r['building_object_id']].append({'observation_id':r['observation_id'],'relationship_type':r['relationship_type']})
    write('FORWARD_REVERSE_QUERY_INDEX.json',{'observation_to_objects':dict(forward),'object_to_observations':dict(reverse),
        'status':'DERIVED_RELATIONSHIP_VIEW_NOT_OBJECT_DATABASE'})
    gpkg=OUT/'IDENTITY_GRAPH_MINIMAL.gpkg'
    with fiona.open(PREV/'ROOF_BUILDING_IDENTITY_GRAPH.gpkg',layer='roof_instance') as source:
        schema=source.schema.copy();schema['properties']=dict(schema['properties']);schema['properties'].update({'single_date_status':'str','physical_id':'str'})
        with fiona.open(gpkg,'w',driver='GPKG',layer='roof_instance_reference',crs=source.crs,schema=schema) as dst:
            for f in source:
                props=dict(f['properties']);props['physical_id']=None
                props['single_date_status']=decided.get(props['node_id'],{}).get('adjudication',old_status[props['node_id']])
                dst.write({'geometry':f['geometry'],'properties':props})
    for layer in ['building_object_reference','manual_roof_reference']:
        with fiona.open(PREV/'ROOF_BUILDING_IDENTITY_GRAPH.gpkg',layer=layer) as source:
            with fiona.open(gpkg,'w',driver='GPKG',layer=layer,crs=source.crs,schema=source.schema) as dst:
                for f in source:
                    props=dict(f['properties']);props['physical_id']=None;dst.write({'geometry':f['geometry'],'properties':props})
    with sqlite3.connect(gpkg) as conn:
        keys=list(ledger[0]);conn.execute('CREATE TABLE source_object_links ('+','.join('"'+k+'" TEXT' for k in keys)+')')
        conn.executemany('INSERT INTO source_object_links VALUES ('+','.join('?' for k in keys)+')',[[r[k] for k in keys] for r in ledger])
        conn.execute('CREATE INDEX links_by_observation ON source_object_links(observation_id)')
        conn.execute('CREATE INDEX links_by_object ON source_object_links(building_object_id)')
        conn.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,description) VALUES ('source_object_links','attributes','source_object_links','Single-date provenance only; physical_id NULL')")
    states=Counter()
    for oid in observations:
        r=decided.get(oid)
        if r and r['link_level']=='TRACEABLE_SINGLE_DATE_LINK':states['TRACEABLE_SINGLE_DATE_LINK_COUNT']+=1
        elif r and r['adjudication']=='PLAUSIBLE_GEOMETRIC_LINK':states['PLAUSIBLE_LINK_COUNT']+=1
        elif (r and r['adjudication']=='COMPETING_LINKS') or (not r and old_status[oid]=='AMBIGUOUS'):states['AMBIGUOUS_LINK_COUNT']+=1
        else:states['UNRESOLVED_LINK_COUNT']+=1
    for k in ['TRACEABLE_SINGLE_DATE_LINK_COUNT','PLAUSIBLE_LINK_COUNT','AMBIGUOUS_LINK_COUNT','UNRESOLVED_LINK_COUNT']:states.setdefault(k,0)
    summary={**states,'CROSS_DATE_PHYSICAL_ID_CONFIRMED_COUNT':0,
        'GEOMETRY_SOURCE_CONFLICT_COUNT':sum(r['adjudication']=='SOURCE_CONFLICT' for r in results),
        'ORIGINAL_IDS_PRESERVED':True,'INDEPENDENT_VALIDATION_AVAILABLE':False,
        'count_unit':'UNIQUE_OBSERVATIONS_PARTITION_OF_1662','total_observations':1662,
        'audited_observations':len(scope),'link_ledger_rows':len(ledger),'physical_id_non_null_count':0,
        'status':'SINGLE_DATE_OBJECT_LINKAGE_PARTIALLY_CLOSED' if states['TRACEABLE_SINGLE_DATE_LINK_COUNT'] else 'NO_TRACEABLE_SINGLE_DATE_LINK_ESTABLISHED',
        'IDENTITY_GRAPH_READY':'NO','READY_FOR_GT10M_TRANSFER':'NO','ABSOLUTE_POSITION_ACCURACY':'UNKNOWN'}
    assert sum(states.values())==1662
    write('MINIMAL_CLOSURE_METRICS.json',summary)
    # Select contrasts by source-outline readability within predetermined outcome strata.
    def clarity(oid):
        g=observations[oid][1];return g.area/g.minimum_rotated_rectangle.area if 150<g.area<5000 and max(g.bounds[2]-g.bounds[0],g.bounds[3]-g.bounds[1])<180 else -1
    def choose(ids):return max(ids,key=lambda o:(clarity(o),o)) if ids else None
    chosen_cases=[('清楚来源关联',choose([r['observation_id'] for r in results if r['link_level']=='TRACEABLE_SINGLE_DATE_LINK'] or audit['plausible_ids'])),
        ('多对象贡献 / 合并风险',choose([r['observation_id'] for r in results if r['evidence']['multiple_contributors']])),
        ('竞争几何关联',choose([r['observation_id'] for r in results if len(r['previous_geometric_candidates'])>1 and r['link_level']!='TRACEABLE_SINGLE_DATE_LINK'])),
        ('未解决对象',choose([o for o,s in old_status.items() if s=='UNRESOLVED']))]
    cases=[]
    for title,oid in chosen_cases:
        if oid is None:continue
        r=decided.get(oid);g=observations[oid][1];x0,y0,x1,y1=g.bounds
        pad=25;side=max(90,x1-x0+2*pad,y1-y0+2*pad);cx=(x0+x1)/2;cy=(y0+y1)/2
        ids=[v['building_object_id'] for v in r['source_object_records']] if r else [v['reference_id'] for v in candidates[oid]]
        ids=ids or ([r['building_object_id']] if r and r['building_object_id'] else [])
        refids=sorted({x['roof_gt_id'] for x in old_table if x['roof_obs_id']==oid and x['roof_gt_id']})
        cases.append({'title':title,'observation_id':oid,'building_object_ids':ids,'manual_reference_fids':refids,
            'bounds':[cx-side/2,cy-side/2,cx+side/2,cy+side/2],'observed_geometry':mapping(g),
            'object_geometries':{uid:mapping(objects[uid][1]) for uid in ids if uid in objects},
            'manual_geometries':{fid:mapping(gts[fid][1]) for fid in refids if fid in gts},
            'result':r['adjudication'] if r else 'INSUFFICIENT_EVIDENCE','evidence_level':r['link_level'] if r else 'BASELINE_UNRESOLVED',
            'physical_id':None,'GT_semantics':'2019_MANUAL_ROOF_REFERENCE_NOT_2018_PRESENCE_PROOF'})
    write('PANEL_CASES.json',{'cases':cases,'selection':'SOURCE_BOUNDARY_READABILITY_WITHIN_FOUR_CONTRAST_STRATA',
        'GE_actual_date':'2018-11-01','manual_year':2019,'geometries_shifted':False})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
