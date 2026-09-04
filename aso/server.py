"""A local HTTP server, so the model is loaded once instead of once per call.

Roughly 16 seconds of every CLI invocation was import and warm-up - 13.7s of it
the sentence encoder - against a fraction of a second of actual work. Moving
that behind a resident process is the whole point; nothing here changes what is
computed, only how often it is paid for.

Stdlib only, to match the rest of the project. `ThreadingHTTPServer` gives
concurrent reads; writes take a lock because SQLite allows one writer at a time
and two analyses racing to train would deadlock rather than queue.

    aso serve
    curl -s localhost:8765/analyze -d '{"keyword":"habit tracker"}' | jq
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, db

HOST = "127.0.0.1"
PORT = 8765

_write = threading.Lock()      # SQLite takes one writer; queue rather than fail
_state: dict = {"started": None, "requests": 0, "model": None}


def _warm():
    """Pay the load cost once, at startup, where it is visible."""
    from . import embed, train
    t0 = time.time()
    embed.encoder()                                   # the 13.7s one
    con = db.connect()
    _, _, ver = train.load_active(con)
    con.close()
    _state.update(started=time.time(), model=ver, warm_seconds=time.time() - t0)
    return _state["warm_seconds"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt, *args):
        pass                                          # the routes log themselves

    def _send(self, obj, status=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            up = time.time() - (_state["started"] or time.time())
            return self._send({"ok": True, "model": _state["model"],
                               "uptime_seconds": round(up, 1),
                               "requests": _state["requests"],
                               "warm_seconds": round(_state.get("warm_seconds", 0), 1)})
        if path == "/config":
            return self._send(config.load(reload=True))
        if path == "/status":
            con = db.connect()
            try:
                q = lambda s: con.execute(s).fetchone()[0]
                return self._send({
                    "apps": q("SELECT COUNT(*) FROM apps"),
                    "keywords": q("SELECT COUNT(DISTINCT keyword) FROM observations"),
                    "ranked": q("SELECT COUNT(*) FROM observations "
                                "WHERE position IS NOT NULL"),
                    "model": _state["model"]})
            finally:
                con.close()
        return self._send({"error": "not found",
                           "routes": ["/health", "/status", "/config",
                                      "POST /analyze", "POST /why",
                                      "POST /correct", "POST /train"]}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._body()
        _state["requests"] += 1
        t0 = time.time()
        try:
            out = self._route(path, body)
        except SystemExit as e:
            return self._send({"error": str(e)}, 400)
        except Exception as e:                        # noqa: BLE001
            return self._send({"error": f"{type(e).__name__}: {e}"}, 500)
        if out is None:
            return self._send({"error": "not found"}, 404)
        print(f"  {path} {body.get('keyword', '')!r} {time.time() - t0:.1f}s", flush=True)
        return self._send(out)

    def _route(self, path, body):
        country = body.get("country") or config.get("country", "us")
        kw = (body.get("keyword") or "").strip()
        pkg = body.get("pkg")
        if pkg:
            from . import scrape
            pkg = scrape.parse_package(pkg)

        if path == "/analyze":
            if not kw:
                raise SystemExit("keyword is required")
            from .analyze import analyze, recommend
            con = db.connect()
            try:
                # learn=False by default: a resident server should answer fast,
                # and a 30s fit inside a request blocks every other caller behind
                # the write lock. Training is its own endpoint.
                o = analyze(con, kw, country=country, pkg=pkg, verbose=False,
                            learn=bool(body.get("learn", False)))
                # Thresholds may come per request. An agent deciding its own bar
                # should not have to write to the config file to ask a question.
                o["recommendation"] = recommend(o, {
                    k: body[k] for k in
                    ("min_downloads", "min_downloads_unit", "max_rank",
                     "min_build_score", "min_model_confidence") if k in body})
                o.pop("_your_group", None)
                return o
            finally:
                con.close()

        if path == "/why":
            if not (kw and pkg):
                raise SystemExit("keyword and pkg are required")
            from . import diagnose
            con = db.connect()
            try:
                return diagnose.why(con, kw, pkg, country=country)
            finally:
                con.close()

        if path == "/correct":
            from . import correct
            with _write:
                con = db.connect()
                try:
                    done = correct.apply(
                        con, kw, pkg=pkg, rank=body.get("rank"),
                        demand=body.get("demand"), crowding=body.get("crowding"),
                        reviewer=body.get("reviewer", "api"), country=country)
                    return {"keyword": kw, "applied": done}
                finally:
                    con.close()

        if path == "/train":
            from . import train
            with _write:
                con = db.connect()
                try:
                    ver, meta, gated = train.train(
                        con, country=country,
                        epochs=int(body.get("epochs", 400)), verbose=False)
                    _state["model"] = train.load_active(con)[2]
                    return {"version": ver, "promoted": not gated,
                            "active": _state["model"], **(meta or {})}
                finally:
                    con.close()
        return None


def serve(host: str = HOST, port: int = PORT) -> None:
    took = _warm()
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"  model {_state['model']} warm in {took:.1f}s")
    print(f"  listening on http://{host}:{port}")
    print(f"  GET  /health /status /config")
    print(f"  POST /analyze /why /correct /train\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        srv.server_close()
