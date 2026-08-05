from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))

from urrs_lite_review.core import (  # noqa: E402
    APP_VERSION,
    DATA_ROOT,
    ReviewDB,
    EVENTS_PATH,
    safe_data_path,
    now_iso,
    atomic_write_json,
    project_id,
)

STATIC_ROOT = HERE / "static"
SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]+$")
_DB = ReviewDB()
_DB_LOCK = threading.RLock()


def _json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length > 1_048_576: raise ValueError("request body too large")
    raw = handler.rfile.read(length) if length else b"{}"
    value = json.loads(raw.decode("utf-8"))
    return value if isinstance(value, dict) else {}


def _revision() -> int:
    return int(_DB.setting("database_revision", "0") or 0)


def _write_response(success: bool, **extra) -> dict:
    return {"success": bool(success), "saved_at": now_iso(), "event_id": extra.pop("event_id", None), "database_revision": _revision(), **extra}


class Handler(BaseHTTPRequestHandler):
    server_version = "URRS-Lite-Review/0.1"

    def log_message(self, fmt, *args):
        # Local server logs are deliberately terse and contain no image payloads.
        sys.stderr.write("[urrs-lite] " + (fmt % args) + "\n")

    def _send_json(self, value: dict, status: int = 200):
        data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def _send_file(self, path: Path):
        try: path = safe_data_path(path)
        except PermissionError: self._send_json({"error": "path denied"}, 403); return
        if not path.is_file(): self._send_json({"error": "not found"}, 404); return
        data = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"); self.send_header("Cache-Control", "public, max-age=3600"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path); path = parsed.path; q = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/api/health": return self._send_json({"ok": True, "service": "URRS Lite Review", "app_version": APP_VERSION, "local_only": True, "bind_address": "127.0.0.1", "external_network": "disabled"})
            if path == "/api/projects": return self._send_json({"projects": [dict(r) for r in _DB.conn.execute("SELECT * FROM projects ORDER BY project_id")]})
            if path == "/api/queue":
                return self._send_json(_DB.queue(project=(q.get("project") or [None])[0], module=(q.get("module") or [None])[0], split=(q.get("split") or [None])[0], status=(q.get("status") or [None])[0], error_type=(q.get("error_type") or [None])[0], page=int((q.get("page") or [1])[0]), page_size=int((q.get("page_size") or [50])[0])))
            if path.startswith("/api/sample/"):
                sid = path.split("/", 3)[3]
                if not SAMPLE_ID_RE.match(sid): return self._send_json({"error": "invalid sample id"}, 400)
                sample = _DB.sample(sid)
                return self._send_json(sample or {"error": "sample not found"}, 200 if sample else 404)
            if path == "/api/progress": return self._send_json(_DB.progress((q.get("project") or [None])[0]))
            if path == "/api/training-gate":
                rows = [dict(r) for r in _DB.conn.execute("SELECT * FROM training_gates ORDER BY gate_id")]
                for r in rows: r["details"] = json.loads(r.pop("details_json"))
                return self._send_json({"gates": rows, "holdout_sealed": _DB.setting("holdout_sealed", "1") == "1"})
            if path == "/api/training/status":
                p = DATA_ROOT / "TRAINING_START_RECEIPT.json"; return self._send_json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "not_started", "holdout_reads": 0})
            if path.startswith("/api/chip/"):
                parts = path.split("/")
                sid = parts[3] if len(parts) > 3 else ""
                if not SAMPLE_ID_RE.match(sid): return self._send_json({"error": "invalid sample id"}, 400)
                sample = _DB.sample(sid)
                if not sample: return self._send_json({"error": "sample not found"}, 404)
                date = (q.get("date") or [None])[0]
                candidate = sample.get("chip_path")
                if date and sample.get("metadata", {}).get("date_chips", {}).get(date): candidate = sample["metadata"]["date_chips"][date].get("path")
                if not candidate: return self._send_json({"error": "chip not generated"}, 404)
                return self._send_file(Path(candidate))
            if path == "/": return self._send_file(STATIC_ROOT / "index.html")
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                if ".." in Path(rel).parts: return self._send_json({"error": "path denied"}, 403)
                return self._send_file(STATIC_ROOT / rel)
            return self._send_json({"error": "not found"}, 404)
        except Exception as exc:
            return self._send_json({"error": str(exc), "local_only": True}, 500)

    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        try:
            body = _json_body(self)
            with _DB_LOCK:
                if path in {"/api/review/save", "/api/block/complete"}:
                    sid = body.get("sample_id"); action = "complete" if path.endswith("complete") else body.get("action", "save")
                    result = _DB.record_review(sid, action, body, body.get("reviewer", "local_reviewer")); return self._send_json(result)
                if path in {"/api/annotation/create", "/api/annotation/update", "/api/annotation/delete"}:
                    action = "add" if path.endswith("create") else ("edit" if path.endswith("update") else "delete")
                    result = _DB.record_review(body.get("sample_id"), action, body, body.get("reviewer", "local_reviewer")); return self._send_json(result)
                if path == "/api/backup":
                    receipt = _DB.backup(); return self._send_json(_write_response(True, **receipt))
                if path == "/api/export":
                    from urrs_lite_review.export_verified_labels import export_verified
                    receipt = export_verified(body.get("project")); return self._send_json(_write_response(True, **receipt))
                if path == "/api/training/prepare":
                    from urrs_lite_review.check_training_gate import evaluate
                    requested_project = body.get("project") or "nanjing_building"
                    current_gate_id = f"{project_id(requested_project)}:pilot"
                    current_gate = _DB.conn.execute("SELECT status,details_json FROM training_gates WHERE gate_id=?", (current_gate_id,)).fetchone()
                    if current_gate:
                        current_details = json.loads(current_gate[1])
                        gates = {"project": requested_project, "pilot": {"gate_id": current_gate_id, "status": current_gate[0], "details": current_details}}
                        passed = current_gate[0] == "pass"
                    else:
                        gates = evaluate(); passed = gates["building"]["pilot"]["status"] == "pass" or gates["construction"]["classifier"]["status"] == "pass"
                    package_receipt = {"created_at": now_iso(), "status": "ready" if passed else "blocked", "project": requested_project, "command_id": "TR1_HEAD_ONLY" if passed else None, "gates": gates, "training_started": False, "holdout_reads": 0}
                    atomic_write_json(DATA_ROOT / "TRAINING_PACKAGE_RECEIPT.json", package_receipt)
                    return self._send_json(_write_response(passed, gates=gates, package=package_receipt, reason=None if passed else "training gate is blocked; no TR1 package generated"), 200)
                if path == "/api/training/start":
                    from urrs_lite_review.launch_controlled_training import launch
                    requested_project = body.get("project", "nanjing_building")
                    default_gate = f"{project_id(requested_project)}:pilot" if requested_project.startswith("nanjing_building_domain_") else "building_pilot"
                    receipt = launch(requested_project, body.get("gate") or default_gate, body.get("confirm", ""), body.get("command_id", "TR1_HEAD_ONLY"), execute=False)
                    return self._send_json(_write_response(receipt.get("status") != "blocked", **receipt), 200)
            return self._send_json({"error": "not found"}, 404)
        except (KeyError, ValueError) as exc:
            return self._send_json(_write_response(False, error=str(exc)), 400)
        except Exception as exc:
            return self._send_json(_write_response(False, error=str(exc)), 500)


def run(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False):
    if host != "127.0.0.1": raise ValueError("URRS Lite Review only binds to 127.0.0.1")
    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    print(f"URRS Lite Review LOCAL ONLY http://{host}:{port}", flush=True)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: httpd.server_close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765); p.add_argument("--open-browser", action="store_true")
    args = p.parse_args(argv); run(args.host, args.port, args.open_browser); return 0


if __name__ == "__main__": raise SystemExit(main())
