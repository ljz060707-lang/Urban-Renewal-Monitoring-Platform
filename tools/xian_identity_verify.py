"""Regression and read-back verification, append-only to completed run."""
import hashlib
import io
import json
import sqlite3
import sys
import unittest
from pathlib import Path
import fiona

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/23_building_object_identity/ROOF_BUILDING_IDENTITY_CLOSURE_20261013_FIXED_02')
sys.path.insert(0,str(ROOT/'tests'))
names=sorted(p.stem for p in (ROOT/'tests').glob('test_xian_*.py') if p.stem!='test_xian_additive_post_inference')
result=unittest.TextTestRunner(stream=io.StringIO(),verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
frozen=json.loads((OUT/'FROZEN_INPUTS.json').read_text(encoding='utf-8'))
changed=[p for p,v in frozen['hashes'].items() if sha(p)!=v]
graph=OUT/'ROOF_BUILDING_IDENTITY_GRAPH.gpkg'
with sqlite3.connect(f'file:{graph.as_posix()}?mode=ro',uri=True) as conn:
    assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    dangling=conn.execute('''SELECT count(*) FROM ROOF_IDENTITY_MATCH_TABLE m
        LEFT JOIN roof_instance r ON m.roof_obs_id=r.node_id
        LEFT JOIN manual_roof_reference g ON m.roof_gt_id=g.node_id
        LEFT JOIN building_object_reference b ON m.building_object_id=b.node_id
        LEFT JOIN physical_identity_reference p ON m.physical_id=p.physical_id
        WHERE r.node_id IS NULL OR (m.roof_gt_id IS NOT NULL AND g.node_id IS NULL)
        OR (m.building_object_id IS NOT NULL AND b.node_id IS NULL)
        OR (m.physical_id IS NOT NULL AND p.physical_id IS NULL)''').fetchone()[0]
    unique_roofs=conn.execute('SELECT count(DISTINCT roof_obs_id) FROM ROOF_IDENTITY_MATCH_TABLE').fetchone()[0]
    n=conn.execute('SELECT count(*) FROM roof_instance').fetchone()[0]
    gt_changes=conn.execute('SELECT count(*) FROM ROOF_IDENTITY_MATCH_TABLE WHERE GT_USED_FOR_GEOMETRY<>0').fetchone()[0]
    confirmed=conn.execute("SELECT count(*) FROM ROOF_IDENTITY_MATCH_TABLE WHERE match_status='CONFIRMED'").fetchone()[0]
assert not changed and dangling==0 and unique_roofs==n and gt_changes==0 and confirmed==0
checks={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'skipped':len(result.skipped),'details':result.failures+result.errors,'protected_inputs_checked':len(frozen['hashes']),
    'changed_inputs':changed,'dangling_ids':dangling,'complete_roof_denominator':n,'GT_geometry_replacements':gt_changes,
    'confirmed_candidate_promotions':confirmed,'gpkg_layers':fiona.listlayers(graph),
    'independent_validation':'VALIDATION_NOT_AVAILABLE'}
data=json.dumps(checks,ensure_ascii=False,indent=2)
with (OUT/'REGRESSION_AND_GRAPH_VERIFICATION.json').open('x',encoding='utf-8') as stream:stream.write(data+'\n')
assert result.wasSuccessful(),data
receipt={'status':'VERIFIED_RELATIONSHIP_VIEW_PHYSICAL_IDENTITY_UNRESOLVED',
    'baseline_receipt':str(OUT/'FINAL_RECEIPT.json'),'verification':checks,
    'IDENTITY_GRAPH_READY':'NO','INDEPENDENT_VALIDATION_AVAILABLE':'NO','READY_FOR_GT10M_TRANSFER':'NO',
    'RELATIONSHIP_VIEW_READY':'YES','ABSOLUTE_POSITION_ACCURACY':'UNKNOWN',
    'artifact_hashes':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
with (OUT/'VERIFIED_FINAL_RECEIPT.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
with (OUT/'AUTHORITATIVE_DELIVERY.json').open('x',encoding='utf-8') as stream:
    json.dump({'authoritative_run':str(OUT),'receipt':'VERIFIED_FINAL_RECEIPT.json','report':'FINAL_REPORT.md',
        'graph':'ROOF_BUILDING_IDENTITY_GRAPH.gpkg','prior_attempt':'ROOF_BUILDING_IDENTITY_CLOSURE_20261013',
        'prior_attempt_status':'INCOMPLETE_EXPORT_KEYERROR_ORIGINAL_SOURCE_FID_NO_RECEIPT'},stream,ensure_ascii=False,indent=2)
print(data,flush=True)
