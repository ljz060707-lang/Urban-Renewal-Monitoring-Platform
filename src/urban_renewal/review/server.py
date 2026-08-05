import json, mimetypes, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse
from .app import prepare
from .review_store import ReviewStore
from .security import safe_path
from .construction_inventory import build_construction_inventory
from .construction_store import ConstructionReviewStore
from .construction_corrections import export_qgis_corrections
from .seed_temporal import SeedTemporalLabelStore, build_seed_temporal_inventory

class ReviewServer:
    def __init__(self, project, review_dir, web_root):
        self.project=Path(project).resolve(); self.review_dir=Path(review_dir).resolve(); self.web_root=Path(web_root).resolve(); self.data=prepare(self.project,self.review_dir); self.store=ReviewStore(self.review_dir); self.construction_inventory=build_construction_inventory(self.review_dir,self.data['inventory']); self.construction_store=ConstructionReviewStore(self.review_dir, self.construction_inventory); self.seed_temporal_inventory=build_seed_temporal_inventory(self.review_dir); self.seed_temporal_store=SeedTemporalLabelStore(self.review_dir, self.seed_temporal_inventory) if self.seed_temporal_inventory.get('seed_dir') else None
        self.construction_asset_roots = [self.review_dir]
        migration_dir = self.construction_inventory.get("source_migration_dir")
        if migration_dir:
            self.construction_asset_roots.append((Path(migration_dir) / "transition_previews").resolve())
        self.construction_asset_files = set()
        for record in self.construction_inventory.get("records", []):
            for evidence in record.get("evidence_by_transition", {}).values():
                for item in evidence.values():
                    for path in item.get("paths", []) if isinstance(item, dict) else []:
                        if isinstance(path, str):
                            candidate = Path(path).resolve()
                            if candidate.is_file():
                                self.construction_asset_files.add(str(candidate))
    def handler(self):
        parent=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def send_json(self,obj,status=200):
                body=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
            def serve_file(self,p):
                if not p.is_file(): raise FileNotFoundError(str(p))
                data=p.read_bytes(); self.send_response(200); self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
            def serve_construction_asset(self, raw_path):
                candidate = Path(unquote(raw_path))
                if not candidate.is_absolute():
                    candidate = parent.review_dir / candidate
                candidate = candidate.resolve()
                if str(candidate) in parent.construction_asset_files:
                    return self.serve_file(candidate)
                for root in parent.construction_asset_roots:
                    try:
                        candidate.relative_to(root.resolve())
                    except ValueError:
                        continue
                    return self.serve_file(candidate)
                raise PermissionError("construction asset is outside the approved preview roots")
            def do_GET(self):
                try:
                    path=urlparse(self.path).path
                    if path=='/api/inventory': return self.send_json(parent.data['inventory'])
                    if path=='/api/state': return self.send_json({'summary':parent.store.summary(parent.data['inventory']),'state':parent.store._read()})
                    if path=='/api/construction/inventory': return self.send_json(parent.construction_inventory)
                    if path=='/api/construction/state': return self.send_json({'summary':parent.construction_store.summary(parent.construction_inventory),'state':parent.construction_store._read()})
                    if path=='/api/seed-temporal/inventory': return self.send_json(parent.seed_temporal_inventory)
                    if path=='/api/seed-temporal/state':
                        if parent.seed_temporal_store is None: return self.send_json({'status':'not_ready','summary':{},'state':{}})
                        return self.send_json({'summary':parent.seed_temporal_store.summary(),'state':parent.seed_temporal_store.state()})
                    if path.startswith('/construction-preview/'): return self.serve_construction_asset(path[len('/construction-preview/'):])
                    if path.startswith('/preview/'): return self.serve_file(safe_path(parent.review_dir,path[len('/preview/'):]))
                    if path=='/': path='/index_v2.html'
                    return self.serve_file(safe_path(parent.web_root,path.lstrip('/')))
                except Exception as exc: self.send_json({'error':str(exc)},404)
            def do_POST(self):
                try:
                    n=int(self.headers.get('Content-Length','0')); payload=json.loads(self.rfile.read(n)) if n else {}
                    if self.path=='/api/review': return self.send_json({'decision':parent.store.save(payload),'summary':parent.store.summary(parent.data['inventory'])})
                    if self.path=='/api/construction/review': return self.send_json({'decision':parent.construction_store.save(payload),'summary':parent.construction_store.summary(parent.construction_inventory)})
                    if self.path=='/api/seed-temporal/review':
                        if parent.seed_temporal_store is None: return self.send_json({'error':'seed temporal inventory not ready'},409)
                        decision=parent.seed_temporal_store.save(payload)
                        return self.send_json({'decision':decision,'summary':parent.seed_temporal_store.summary()})
                    if self.path=='/api/seed-temporal/export':
                        if parent.seed_temporal_store is None: return self.send_json({'error':'seed temporal inventory not ready'},409)
                        return self.send_json({'rows':len(parent.seed_temporal_store.export()),'csv':str(parent.seed_temporal_store.csv_path),'jsonl':str(parent.seed_temporal_store.events_path)})
                    if self.path=='/api/construction/export': return self.send_json({'rows':len(parent.construction_store.export()),'csv':str(parent.construction_store.csv_path),'jsonl':str(parent.construction_store.jsonl_path)})
                    if self.path=='/api/construction/corrections': return self.send_json(export_qgis_corrections(parent.review_dir,parent.construction_inventory))
                    if self.path=='/api/undo': return self.send_json({'undone':parent.store.undo(),'summary':parent.store.summary(parent.data['inventory'])})
                    if self.path=='/api/export': return self.send_json({'rows':len(parent.store.export())})
                    if self.path=='/api/finalize':
                        from .finalization import finalize_review
                        return self.send_json(finalize_review(parent.project,parent.review_dir,payload))
                    return self.send_json({'error':'not found'},404)
                except Exception as exc: self.send_json({'error':str(exc)},400)
        return Handler

def serve(project, review_dir, port=8765, open_browser=False):
    web_root=Path(__file__).resolve().parents[3]/'web/urrs_review'; server=ReviewServer(project,review_dir,web_root); httpd=ThreadingHTTPServer(('127.0.0.1',port),server.handler()); print(f'URRS Review Console\\nURL: http://127.0.0.1:{port}\\nReview directory: {review_dir}\\nMandatory samples: {server.data["summary"]["mandatory_count"]}\\nNo GPU/model tasks started.')
    if open_browser: webbrowser.open(f'http://127.0.0.1:{port}')
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: httpd.server_close()
