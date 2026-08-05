import json
from datetime import datetime, timezone
from .review_store import ReviewStore

def finalize_review(project, review_dir, confirmation):
    inventory = json.loads((review_dir / 'REVIEW_INVENTORY.json').read_text(encoding='utf-8'))
    store = ReviewStore(review_dir)
    summary = store.summary(inventory)
    required = ['all_mandatory_reviewed','fragmentation_checked','transitions_checked','surface_state_checked','candidate_not_ground_truth','not_used_for_checkpoint_threshold','reviewer_ack']
    missing = [k for k in required if not confirmation.get(k)]
    rows = list(store._read().get('decisions', {}).values())
    note_missing = [r['sample_id'] for r in rows if r.get('verdict') in ('MAJOR_ISSUE','BLOCKER') and not str(r.get('notes','')).strip()]
    if missing or note_missing or summary['mandatory_reviewed'] < summary['mandatory_count']:
        return {'status':'blocked','missing_confirmations':missing,'major_blocker_without_notes':note_missing,'summary':summary}
    overall = str(confirmation.get('overall_verdict','')).upper()
    if overall not in ('PASS','CONDITIONAL_PASS','FAIL'):
        return {'status':'blocked','error':'overall_verdict must be PASS, CONDITIONAL_PASS or FAIL'}
    now = datetime.now(timezone.utc).isoformat()
    result = {'review_version':'URRS-Tongzhou-v1.0-candidate','status':'success','overall_verdict':overall,'reviewer':confirmation.get('reviewer','\u674e\u91d1\u6cd3'),'reviewed_at':now,'summary':summary,'scientific_status':'candidate_not_ground_truth','inference_ready': overall == 'PASS','confirmation':confirmation}
    (review_dir/'HUMAN_VISUAL_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (review_dir/'HUMAN_VISUAL_REVIEW.md').write_text('# Human Visual Review\\n\\nOverall verdict: '+overall+'\\n\\nCandidate evidence only; not Ground Truth.\\n',encoding='utf-8')
    (review_dir/'HUMAN_REVIEW_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


