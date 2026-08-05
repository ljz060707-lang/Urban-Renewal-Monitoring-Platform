import json
from .inventory import build_inventory
from .review_store import ReviewStore
def prepare(project,review_dir):
    inv_path=review_dir/'REVIEW_INVENTORY.json'
    if not inv_path.exists():
        build_inventory(project,review_dir)
    inventory=json.loads(inv_path.read_text(encoding='utf-8'))
    store=ReviewStore(review_dir)
    store.export()
    return {'inventory':inventory,'summary':store.summary(inventory),'review_dir':str(review_dir)}
