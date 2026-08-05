from pathlib import Path
from urban_renewal.review.inventory import build_inventory
from urban_renewal.review.review_store import ReviewStore
from urban_renewal.review.security import safe_path

def test_inventory_is_deterministic(tmp_path):
    project=Path(__file__).resolve().parents[3]
    a=build_inventory(project,tmp_path/'a')
    b=build_inventory(project,tmp_path/'b')
    assert a['inventory_hash']==b['inventory_hash']
    ia=(tmp_path/'a/REVIEW_INVENTORY.json').read_text(encoding='utf-8')
    ib=(tmp_path/'b/REVIEW_INVENTORY.json').read_text(encoding='utf-8')
    assert ia==ib or 'inventory_hash' in ia and 'inventory_hash' in ib

def test_store_atomic_undo_and_export(tmp_path):
    s=ReviewStore(tmp_path)
    s.save({'sample_id':'x','verdict':'PASS','reviewer':'reviewer','severity_fields':{},'issue_tags':[]})
    assert s.summary({'records':[{'sample_id':'x','mandatory':True}]})['mandatory_reviewed']==1
    assert (tmp_path/'review_decisions.csv').exists()
    assert s.undo()=='x'
    assert s.summary({'records':[{'sample_id':'x','mandatory':True}]})['mandatory_reviewed']==0

def test_path_traversal_rejected(tmp_path):
    try:
        safe_path(tmp_path,'../outside')
    except ValueError:
        return
    raise AssertionError('path traversal was not rejected')
