"""Finalize actual-data source-link evidence after regression and read-back."""
import csv
import hashlib
import io
import json
import sqlite3
import sys
import unittest
from collections import Counter
from pathlib import Path
import fiona
from shapely.geometry import shape
from xian_r5.single_date_identity import adjudicate
from xian_identity_minimal import OUT,PREV,ROOT,OLD,read,write,sha,unreadable_check


def main():
    sys.path.insert(0,str(ROOT/'tests'))
    names=sorted(p.stem for p in (ROOT/'tests').glob('test_xian_*.py') if p.stem!='test_xian_additive_post_inference')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    assert result.wasSuccessful(),log.getvalue()
    frozen=read(OUT/'FROZEN_INPUTS.json');changed=[p for p,v in frozen['hashes'].items() if sha(p)!=v]
    assert not changed,changed
    baseline_frozen=read(PREV/'FROZEN_INPUTS.json')['hashes']
    gt_checks={p:sha(p)==h for p,h in baseline_frozen.items() if '西安建筑轮廓数据' in p}
    assert gt_checks and all(gt_checks.values())
    for path in ['F:/xian/亚米市区底图/date_audit/tables/tile_actual_date.csv',
                 'F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment/XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_20261013/SOURCE_COMPONENT_LEDGER.json']:
        old_hash=baseline_frozen.get(str(Path(path)))
        assert old_hash and sha(path)==old_hash
    damaged=read(OUT/'UNREADABLE_GPKG_AUDIT.json')['checks']
    assert all(sha(r['path'])==r['sha256'] for r in damaged)
    backup=OLD/'_partial_transfer_quarantine/tracking_buildings.gpkg.partial'
    backup_result=unreadable_check(backup)
    write('UNREADABLE_GPKG_BACKUP_AUDIT.json',{'backup':backup_result,'source_gpkg_retried':False,
        'usable_export_records':'CANONICAL_CSV_AND_TRACKING_CSV_READ_AND_MATCHED_BY_EXISTING_OBJECT_UID'})
    records=read(OUT/'PLAUSIBLE_41_ADJUDICATION.json')['records']+read(OUT/'AMBIGUOUS_RECOVERY_DIAGNOSTIC.json')['records']
    assert len({r['observation_id'] for r in records})==62
    for r in records:
        assert adjudicate(r['evidence'])['adjudication']==r['adjudication']
        assert r['physical_id'] is None
    graph=OUT/'IDENTITY_GRAPH_MINIMAL.gpkg'; geometry=[]
    for old_layer,new_layer in [('roof_instance','roof_instance_reference'),('building_object_reference','building_object_reference'),('manual_roof_reference','manual_roof_reference')]:
        with fiona.open(PREV/'ROOF_BUILDING_IDENTITY_GRAPH.gpkg',layer=old_layer) as old:
            source={f['properties']['node_id']:(shape(f['geometry']),dict(f['properties'])) for f in old}
        with fiona.open(graph,layer=new_layer) as new:
            assert len(new)==len(source) and new.crs.to_epsg()==32649
            for f in new:
                props=dict(f['properties']); g,original=source[props['node_id']]
                assert shape(f['geometry']).equals_exact(g,0)
                assert props['original_id']==original['original_id'] and props['source_fid']==original['source_fid']
                assert props['physical_id'] is None
        geometry.append({'layer':new_layer,'count':len(source),'original_ID_and_geometry_unchanged':True})
    examples=[]
    with sqlite3.connect(f'file:{graph.as_posix()}?mode=ro',uri=True) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        physical=conn.execute('SELECT count(*) FROM source_object_links WHERE physical_id IS NOT NULL').fetchone()[0]
        assert physical==0
        traces=conn.execute("SELECT observation_id,building_object_id,source_feature_id,source_path FROM source_object_links WHERE link_level='TRACEABLE_SINGLE_DATE_LINK'").fetchall()
        assert len(traces)==53
        for oid,uid,fid,path in traces:
            assert conn.execute('SELECT count(*) FROM source_object_links WHERE building_object_id=? AND observation_id=?',(uid,oid)).fetchone()[0]==1
            assert conn.execute('SELECT count(*) FROM building_object_reference WHERE node_id=? AND source_fid=? AND source_path=?',(uid,fid,path)).fetchone()[0]==1
            assert conn.execute('SELECT count(*) FROM roof_instance_reference WHERE node_id=?',(oid,)).fetchone()[0]==1
        examples=[{'observation_id':o,'building_object_id':b,'source_feature_id':f,'source_path':p,'physical_id':None} for o,b,f,p in traces[:3]]
        missing=conn.execute('''SELECT count(*) FROM source_object_links s LEFT JOIN building_object_reference b ON s.building_object_id=b.node_id
                               WHERE s.building_object_id IS NOT NULL AND b.node_id IS NULL''').fetchone()[0]
        assert missing==0
        manual=conn.execute('SELECT count(*),count(DISTINCT node_id),count(DISTINCT original_id) FROM manual_roof_reference').fetchone()
        assert manual==(4821,4821,1)
        forward_multiplicity=conn.execute('SELECT count(*) FROM (SELECT observation_id FROM source_object_links GROUP BY observation_id HAVING count(DISTINCT building_object_id)>1)').fetchone()[0]
        reverse_multiplicity=conn.execute('SELECT count(*) FROM (SELECT building_object_id FROM source_object_links WHERE building_object_id IS NOT NULL GROUP BY building_object_id HAVING count(DISTINCT observation_id)>1)').fetchone()[0]
        assert forward_multiplicity>0 and reverse_multiplicity>0
    dates=read(OUT/'SOURCE_DATE_HASH_VERIFICATION.json')
    assert all(r['hash_matches'] and r['date_matches'] for r in dates['tiles'].values())
    assert all(sha(r['source_JPG'])==r['actual_sha256'] for r in dates['tiles'].values())
    cases=read(OUT/'PANEL_CASES_FIXED_02.json')['cases']
    assert len(cases)==len({c['observation_id'] for c in cases})==4
    assert (OUT/'figure_fixed_02/IDENTITY_LINK_EVIDENCE_PANEL.png').exists()
    summary=read(OUT/'MINIMAL_CLOSURE_METRICS.json')
    checks={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'changed_protected_inputs':changed,'protected_input_count':len(frozen['hashes']),
        'manual_source_file_hashes_unchanged':gt_checks,'damaged_sources_unchanged':True,
        'native_source_tiles_sha_and_dates_verified':dates['checked_tile_count'],'geometry_readback':geometry,
        'physical_id_non_null_rows':physical,'traceable_forward_reverse_queries_passed':len(traces),
        'manual_reference_FIDs_distinct':manual[1],'manual_Id_distinct':manual[2],
        'one_to_many_relationships_preserved':forward_multiplicity,'many_to_one_relationships_preserved':reverse_multiplicity,
        'dangling_object_links':missing,'distinct_figure_cases':4,'independent_validation':'VALIDATION_NOT_AVAILABLE'}
    write('QUALITY_TESTS.json',checks)
    write('END_TO_END_QUERY_EXAMPLES.json',{'examples':examples,'all_53_links_checked':True,
        'forward_sql':'SELECT * FROM source_object_links WHERE observation_id = ?',
        'reverse_sql':'SELECT * FROM source_object_links WHERE building_object_id = ?',
        'history_track_field':'historical_candidate_track_id','physical_id_policy':'NULL'})
    with (OUT/'FINAL_REPORT.md').open('x',encoding='utf-8') as stream:
        stream.write('# XIAN-R5-I 既有建筑对象与屋顶实例关联最小闭环\n\n'
            '本轮状态：`SINGLE_DATE_OBJECT_LINKAGE_PARTIALLY_CLOSED`。实际核查62个既有组件，闭合53条单期来源关联。\n\n'
            '|指标（按唯一 observed component 计数）|数量|\n|---|---:|\n'+
            ''.join(f'|{k}|{summary[k]}|\n' for k in ['TRACEABLE_SINGLE_DATE_LINK_COUNT','PLAUSIBLE_LINK_COUNT','AMBIGUOUS_LINK_COUNT','UNRESOLVED_LINK_COUNT','CROSS_DATE_PHYSICAL_ID_CONFIRMED_COUNT','GEOMETRY_SOURCE_CONFLICT_COUNT'])+
            '\n原41条 PLAUSIBLE 全部通过。21条歧义对象作有界核查，其中12条恢复单期来源关联，9条保留多个对象贡献关系；其余1026条歧义及574条未解决对象保留原状。上述四类计数合计1662，不以5332条候选关系行作为对象分母。\n\n'
            '来源证据链：原始教师 TIF 的 geometry_source 元数据及既有生成记录 → 完整 canonical CSV 中的原始 object_uid / source_object_id / geometry_hash → 可读 building_objects 的原始 FID 与逐字节几何哈希 → tracking CSV 的同一对象、日期与几何记录 → 原始格网上逐像元复演。53条关联均只有一个来源对象贡献该组件的全部像元，且无额外来源对象共同贡献。没有重新生成组件 ID 或平移几何。\n\n'
            '这里闭合的是已有模型对象记录到已有栅格组件的来源关联，不认证一个组件必然对应独立、纯净的物理屋顶。教师栅格的旧 footprint 标签未被用作地面足迹证据。原始 canonical GPKG 不可读，CSV 记录、可读对象几何与原始像元复演是本轮采用的证据组合；其范围是记录层来源关联。\n\n'
            f'实际核验{dates["checked_tile_count"]}张相关 GE 瓦片的 SHA256 和日期记录，实际日期均为2018-11-01。人工 GT 只保留2019年份语义，未作为2018存在证明。原始 manual Id 重复，4821个原始 FID 保持区分；GT 在前轮候选生成及本轮展示中使用，没有替代或修复 observed geometry。\n\n'
            '三份损坏 GPKG 各做一次有界只读检查，仍为 GEOPACKAGE_SOURCE_UNREADABLE；现有 partial 备份检查见单独审计。使用可读 building_objects 和 tracking CSV 完成关联，所有原文件保持不变。\n\n'
            f'{result.testsRun}项回归测试通过；GPKG 回读验证所有原始 ID、几何、NULL physical_id 及53条正反查询。候选一对多/多对一关系保留，历史 track ID 仅写入 historical_candidate_track_id。\n\n'
            '展示图以 [修订后的四组案例](figure_fixed_02/IDENTITY_LINK_EVIDENCE_PANEL.png) 为权威。根目录初稿保留供审计，其图例拥挤和重复案例问题已修正；数值裁决不变。\n\n'
            'ORIGINAL_IDS_PRESERVED = YES\n\nINDEPENDENT_VALIDATION_AVAILABLE = NO\n\nIDENTITY_GRAPH_READY = NO\n\nREADY_FOR_GT10M_TRANSFER = NO\n\nABSOLUTE_POSITION_ACCURACY = UNKNOWN\n\n'
            '本轮属于已有训练窗口的 development 来源审计，没有独立验证，也没有新增跨期 physical ID、位移、模型训练、三态GT或10m标签。\n')
    write('AUTHORITATIVE_DELIVERY.json',{'run':str(OUT),'baseline':str(PREV),'report':'FINAL_REPORT.md',
        'graph':'IDENTITY_GRAPH_MINIMAL.gpkg','figure':'figure_fixed_02/IDENTITY_LINK_EVIDENCE_PANEL.png',
        'receipt':'FINAL_RECEIPT.json','initial_figure_status':'SUPERSEDED_KEEP_FOR_AUDIT'})
    paths=[p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts]
    write('FINAL_RECEIPT.json',{'status':summary['status'],'metrics':summary,'verification':checks,
        'original_input_hashes':frozen['hashes'],'source_code_sha256':{str(p):sha(p) for p in
            [ROOT/'src/xian_r5/single_date_identity.py',ROOT/'tests/test_xian_single_date_identity.py',
             *[ROOT/'tools'/name for name in ['xian_identity_minimal.py','xian_identity_minimal_links.py','xian_identity_panel_data.py',
                'xian_identity_panel_revision.py','plot_xian_identity_panel.py','verify_xian_identity_minimal.py']]]},
        'artifact_sha256':{str(p.relative_to(OUT)):sha(p) for p in paths},
        'excluded_non_delivery_directory':'plot_cache','receipt_sha256_file':'FINAL_RECEIPT.sha256'})
    with (OUT/'FINAL_RECEIPT.sha256').open('x',encoding='ascii') as stream:stream.write(sha(OUT/'FINAL_RECEIPT.json')+'  FINAL_RECEIPT.json\n')
    print(json.dumps({'metrics':summary,'tests':checks},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
