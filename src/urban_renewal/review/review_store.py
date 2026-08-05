from __future__ import annotations
import csv,json,os,tempfile
from datetime import datetime,timezone
from pathlib import Path
from .review_schema import normalize_decision,SEVERITY_FIELDS
class ReviewStore:
    def __init__(self,review_dir):
        self.review_dir=Path(review_dir).resolve()
        self.review_dir.mkdir(parents=True,exist_ok=True)
        self.state_path=self.review_dir/'review_state.json'
        self.events_path=self.review_dir/'review_events.jsonl'
        self.csv_path=self.review_dir/'review_decisions.csv'
    def _read(self):
        return json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {'version':'1.0','reviewer':'\u674e\u91d1\u6cd3','decisions':{},'last_event':None}
    def _atomic(self,path,text):
        fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
        os.close(fd)
        Path(tmp).write_text(text,encoding='utf-8')
        os.replace(tmp,path)
    def save(self,payload):
        state=self._read()
        now=datetime.now(timezone.utc).isoformat()
        decision=normalize_decision(payload,state.get('reviewer','\u674e\u91d1\u6cd3'))
        sid=decision['sample_id']; old=state['decisions'].get(sid)
        decision['reviewed_at']=old.get('reviewed_at',now) if old else now
        decision['updated_at']=now
        state['reviewer']=decision['reviewer']; state['decisions'][sid]=decision; state['last_event']=now
        self._atomic(self.state_path,json.dumps(state,ensure_ascii=False,indent=2))
        with self.events_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'event':'update','at':now,'sample_id':sid,'before':old,'after':decision},ensure_ascii=False)+'\n')
        self.export(state)
        return decision
    def undo(self):
        lines=self.events_path.read_text(encoding='utf-8').splitlines() if self.events_path.exists() else []
        if not lines:return None
        last=json.loads(lines[-1]); state=self._read(); sid=last['sample_id']; before=last.get('before')
        if before is None: state['decisions'].pop(sid,None)
        else: state['decisions'][sid]=before
        self._atomic(self.state_path,json.dumps(state,ensure_ascii=False,indent=2)); self.export(state); return sid
    def export(self,state=None):
        state=state or self._read(); rows=list(state.get('decisions',{}).values())
        fields=['sample_id','verdict']+list(SEVERITY_FIELDS)+['issue_tags','notes','reviewer','reviewed_at','updated_at','inventory_hash','source_hashes']
        with self.csv_path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for r in rows:
                w.writerow({'sample_id':r.get('sample_id'),'verdict':r.get('verdict'),**{k:r.get('severity_fields',{}).get(k,0) for k in SEVERITY_FIELDS},'issue_tags':json.dumps(r.get('issue_tags',[]),ensure_ascii=False),'notes':r.get('notes',''),'reviewer':r.get('reviewer',''),'reviewed_at':r.get('reviewed_at'),'updated_at':r.get('updated_at'),'inventory_hash':r.get('inventory_hash'),'source_hashes':json.dumps(r.get('source_hashes',{}),ensure_ascii=False)})
        (self.review_dir/'review_decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        return rows
    def summary(self,inventory):
        rows=list(self._read().get('decisions',{}).values()); mandatory=[r for r in inventory.get('records',[]) if r.get('mandatory')]; reviewed={r['sample_id'] for r in rows}
        return {'reviewed_count':len(rows),'mandatory_count':len(mandatory),'mandatory_reviewed':sum(x['sample_id'] in reviewed for x in mandatory),'counts':{v:sum(r.get('verdict')==v for r in rows) for v in ('PASS','MINOR_ISSUE','MAJOR_ISSUE','BLOCKER')},'total_count':len(inventory.get('records',[]))}

