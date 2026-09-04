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

import errno
import hmac
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pathlib import Path

from . import config, db
from . import env as _env

HOST = "127.0.0.1"
PORT = 8765

# Where a running server records itself, so a client can find one that was
# started on a non-default port without being told which.
RUNFILE = Path(__file__).resolve().parent.parent / "data" / "server.json"

# Set ASO_API_TOKEN to require a bearer token on every route except /health.
# Binding anywhere but loopback WITHOUT one is refused: this server scrapes,
# trains, and writes corrections into the database, so an open port is a stranger
# with write access, not just a read-only annoyance.
_env.load()
# Optional. Unset means no auth, which is the right default for a loopback
# server on a laptop. Set it in .env when the port is reachable from anywhere.
API_TOKEN = os.environ.get("ASO_API_TOKEN", "").strip()
PUBLIC_ROUTES = {"/health"}

# Scraping a cold keyword takes ~40s and holds a slot the whole time, so the
# ceiling is small by default: unbounded threads here means dozens of concurrent
# scrapes, which Play rate-limits and which starve each other on the write lock.
MAX_CONCURRENT = 4
TIMEOUT = 180

HARD_MAX = 32                  # ceiling the live limit can be raised to

_write = threading.Lock()      # SQLite takes one writer; queue rather than fail
_pool: ThreadPoolExecutor | None = None
_limits = {"max_concurrent": MAX_CONCURRENT, "timeout": TIMEOUT}


class Gate:
    """An admission limit that can be changed while requests are in flight.

    threading.Semaphore cannot be resized, so raising the limit used to mean a
    restart - and a restart costs the 16 second warm-up plus every queued
    caller. A counter behind a Condition can be retuned live: lowering it lets
    running work finish and simply admits fewer afterwards, rather than
    cancelling anything.
    """

    def __init__(self, limit: int):
        self._cv = threading.Condition()
        self._limit = limit
        self._held = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def held(self) -> int:
        return self._held

    def resize(self, limit: int) -> int:
        limit = max(1, min(int(limit), HARD_MAX))
        with self._cv:
            self._limit = limit
            self._cv.notify_all()          # a raise may unblock waiters at once
        return limit

    def acquire(self) -> bool:
        with self._cv:
            if self._held >= self._limit:
                return False
            self._held += 1
            return True

    def release(self) -> None:
        with self._cv:
            self._held = max(0, self._held - 1)
            self._cv.notify()


_gate: Gate | None = None
_state: dict = {"started": None, "requests": 0, "model": None,
                "in_flight": 0, "rejected": 0, "timed_out": 0, "dropped": 0}


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

    def handle_one_request(self):
        """A client that hangs up mid-answer is routine, not a crash.

        curl with --max-time, an agent that gave up, a closed tab: the socket
        goes away while we are still writing and socketserver prints a full
        traceback for it. Nothing is wrong with the server, so count it and
        carry on."""
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            _state["dropped"] += 1
            self.close_connection = True
        except socket.timeout:
            self.close_connection = True

    def _send(self, obj, status=200):
        body = json.dumps(obj, default=str).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            _state["dropped"] += 1        # answered into a socket nobody holds

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- routes -----------------------------------------------------------
    def _authorised(self) -> bool:
        if not API_TOKEN:
            return True
        sent = (self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                or self.headers.get("X-API-Key", "").strip())
        # compare_digest so a wrong token cannot be found a character at a time.
        return bool(sent) and hmac.compare_digest(sent, API_TOKEN)

    def _guard(self, path: str) -> bool:
        if path in PUBLIC_ROUTES or self._authorised():
            return True
        self._send({"error": "unauthorised",
                    "detail": "send Authorization: Bearer <token> or X-API-Key"}, 401)
        return False

    def do_GET(self):
        path = self.path.split("?")[0]
        if not self._guard(path):
            return
        if path == "/health":
            up = time.time() - (_state["started"] or time.time())
            return self._send({"ok": True, "model": _state["model"],
                               "uptime_seconds": round(up, 1),
                               "requests": _state["requests"],
                               "in_flight": _gate.held if _gate else 0,
                               "max_concurrent": _gate.limit if _gate else 0,
                               "timeout_seconds": _limits["timeout"],
                               "hard_max": HARD_MAX,
                               "rejected_busy": _state["rejected"],
                               "timed_out": _state["timed_out"],
                               "client_disconnects": _state["dropped"],
                               "warm_seconds": round(_state.get("warm_seconds", 0), 1)})
        if path == "/config":
            _refresh_limits()
            return self._send({**config.load(reload=True),
                               "server_max_concurrent": _gate.limit if _gate else 0,
                               "server_timeout": _limits["timeout"]})
        if path == "/status":
            con = db.connect()
            try:
                q = lambda s: db.scalar(con, s)
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
        if not self._guard(path):
            return
        body = self._body()
        _state["requests"] += 1
        t0 = time.time()

        _refresh_limits()
        timeout = float(body.get("timeout") or _limits["timeout"])
        if not _gate.acquire():
            _state["rejected"] += 1
            return self._send({
                "error": "server busy",
                "detail": f"{_gate.limit} requests already running. "
                          f"A cold keyword takes ~40s to scrape.",
                "in_flight": _gate.held,
                "max_concurrent": _gate.limit,
                "retry_after_seconds": 30}, 503)

        _state["in_flight"] += 1
        try:
            fut = _pool.submit(self._route, path, body)
            try:
                out = fut.result(timeout=timeout)
            except FutureTimeout:
                _state["timed_out"] += 1
                # The work keeps going - a thread cannot be killed safely, and a
                # half-finished scrape would leave the row partly written. Say so
                # plainly rather than implying it was cancelled.
                return self._send({
                    "error": "timeout",
                    "detail": f"still running after {timeout:.0f}s. It continues in "
                              f"the background, so the same request will be fast "
                              f"once the scrape lands.",
                    "timeout_seconds": timeout,
                    "keyword": body.get("keyword")}, 504)
        except SystemExit as e:
            return self._send({"error": "bad request", "detail": str(e)}, 400)
        except Exception as e:                        # noqa: BLE001
            return self._send({"error": type(e).__name__, "detail": str(e)}, 500)
        finally:
            _state["in_flight"] -= 1
            _gate.release()

        if out is None:
            return self._send({"error": "not found", "detail": f"no route {path}"}, 404)
        print(f"  {path} {body.get('keyword', '')!r} {time.time() - t0:.1f}s", flush=True)
        return self._send(out)

    def _route(self, path, body):
        if path == "/config":
            # Retune a live server. `persist` writes it to the config file too,
            # so the next start comes up the same way; without it the change
            # lasts until this process exits.
            changed = {}
            if "server_max_concurrent" in body:
                changed["server_max_concurrent"] = _gate.resize(
                    body["server_max_concurrent"])
            if "server_timeout" in body:
                _limits["timeout"] = max(1.0, float(body["server_timeout"]))
                changed["server_timeout"] = _limits["timeout"]
            if not changed:
                raise SystemExit("send server_max_concurrent and/or server_timeout")
            if body.get("persist"):
                config.save(changed)
                _limits["config_seen"] = None      # do not undo what we just set
            print(f"  /config {changed}"
                  f"{' (persisted)' if body.get('persist') else ''}", flush=True)
            return {"applied": changed, "persisted": bool(body.get("persist")),
                    "in_flight": _gate.held, "hard_max": HARD_MAX}

        country = body.get("country") or config.get("country", "us")
        kw = (body.get("keyword") or "").strip()
        pkg = body.get("pkg")
        if pkg:
            from . import scrape
            pkg = scrape.parse_package(pkg)

        # --- raw Play passthroughs. No model, no database. ---------------
        if path in ("/suggest", "/search", "/details", "/publisher"):
            from . import play
            if path == "/suggest":
                if not kw:
                    raise SystemExit("keyword is required")
                no_cache = bool(body.get("no_cache") or body.get("fresh"))
                out = play.suggest(kw, games=bool(body.get("games")),
                                   no_cache=no_cache)
                from . import suggest as _s
                c = db.connect()
                try:
                    age = _s.age_hours(c, kw)
                finally:
                    c.close()
                return {"query": kw, "suggestions": out, "count": len(out),
                        "cached": not no_cache,
                        "age_hours": None if age is None else round(age, 2)}
            if path == "/search":
                if not kw:
                    raise SystemExit("keyword is required")
                out = play.search(kw, with_details=bool(body.get("details")),
                                  limit=body.get("limit"))
                return {"query": kw, "results": out,
                        "organic": sum(1 for r in out if r["position"])}
            if path == "/details":
                if not pkg:
                    raise SystemExit("pkg is required (package name or Play URL)")
                return play.details(pkg)
            dev = body.get("developer") or body.get("name")
            if not dev:
                raise SystemExit("developer is required")
            out = play.publisher(dev)
            return {"developer": dev, "apps": out, "count": len(out),
                    "note": "Play search caps near 50; scripts/fetch_publisher.py "
                            "scrolls the developer page and reaches 100"}

        if path == "/analyze":
            if not kw:
                raise SystemExit("keyword is required")
            from .analyze import analyze, recommend
            # Opened here and closed in the finally below. analyze() itself no
            # longer holds the handle across scraping, so this is short even on
            # a cold keyword.
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
            # Serialised because SQLite takes one writer. busy_timeout would
            # cover it, but queueing here is cheaper than every caller spinning
            # on the lock.
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


def _refresh_limits() -> None:
    """Pick up `aso config server_max_concurrent N` without a restart.

    The config file is only re-read when its mtime moves, so this costs a stat
    per request rather than a parse. A flag passed to `aso serve` pins the value
    and is not overridden by later file edits.
    """
    if _limits.get("pinned"):
        return
    try:
        mtime = config.path().stat().st_mtime
    except OSError:
        return
    if mtime == _limits.get("config_seen"):
        return
    _limits["config_seen"] = mtime
    cfg = config.load(reload=True)
    want = int(cfg.get("server_max_concurrent", MAX_CONCURRENT))
    if _gate and want != _gate.limit:
        print(f"  config changed: max_concurrent {_gate.limit} -> "
              f"{_gate.resize(want)}", flush=True)
    _limits["timeout"] = float(cfg.get("server_timeout", TIMEOUT))


def serve(host: str = HOST, port: int = PORT,
          max_concurrent: int | None = None, timeout: float | None = None,
          strict_port: bool = False) -> None:
    global _gate, _pool
    _limits["pinned"] = max_concurrent is not None or timeout is not None
    _limits["max_concurrent"] = int(max_concurrent or config.get(
        "server_max_concurrent", MAX_CONCURRENT))
    _limits["timeout"] = float(timeout or config.get("server_timeout", TIMEOUT))
    _gate = Gate(_limits["max_concurrent"])
    # Sized to the ceiling once: the gate decides how many run, so the pool never
    # needs resizing and idle threads cost nothing.
    _pool = ThreadPoolExecutor(max_workers=HARD_MAX, thread_name_prefix="aso")

    # Bind before paying the 16s warm-up: failing after loading the model wastes
    # a quarter of a minute to tell you a port was busy.
    #
    # A taken port moves to the next free one by default. Refusing to start over
    # something so easily resolved, and making the caller re-run with a flag to
    # say "yes, obviously", is not a decision worth asking about. `--strict-port`
    # is there for a supervisor that needs a fixed address.
    # No token, no server, wherever it binds. The loopback exemption was wrong
    # in a container: every other service on the same network reaches it, and
    # an unset variable is exactly how a token goes missing.
    if not API_TOKEN:
        raise SystemExit(
            "refusing to start without ASO_API_TOKEN.\n"
            "  This server scrapes, trains and writes to the database, so a\n"
            "  reachable port hands a stranger write access.\n\n"
            "  Set it in .env:   ASO_API_TOKEN=$(openssl rand -hex 24)")

    requested = port
    try:
        srv = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        if strict_port:
            who = "another aso server" if _probe(host, port) else "another process"
            raise SystemExit(
                f"port {port} is held by {who}.\n"
                f"  You asked for this port specifically, so it was not moved.\n"
                f"  aso serve --port {port} --auto-port   move up if busy\n"
                f"  aso serve                            use the default, {PORT}")
        found = _free_port(host, port + 1)
        if found is None:
            raise SystemExit(f"no free port between {port + 1} and {port + 20}")
        port = found
        srv = ThreadingHTTPServer((host, port), Handler)

    took = _warm()
    srv.daemon_threads = True
    write_runfile(host, port)
    print(f"  model {_state['model']} warm in {took:.1f}s")
    if port != requested:
        other = _probe(host, requested)
        held = (f"another aso server (model {other.get('model')})" if other
                else "another process")
        print(f"  port {requested} is held by {held}, using {port} instead")
    print(f"  listening on http://{host}:{port}")
    print(f"  auth: {'bearer token required' if API_TOKEN else 'NONE (loopback only)'}")
    print(f"  {_gate.limit} concurrent requests, {_limits['timeout']:.0f}s timeout"
          f"{' (pinned by flag)' if _limits['pinned'] else ', follows aso config'}")
    print(f"  GET  /health /status /config")
    print(f"  POST /analyze /why /correct /train /config")
    print(f"  POST /suggest /search /details /publisher   (raw Play, no model)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        srv.server_close()
        RUNFILE.unlink(missing_ok=True)


def _probe(host: str, port: int, timeout: float = 0.5) -> dict | None:
    """Is an aso server already answering here?"""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health",
                                    timeout=timeout) as r:
            got = json.loads(r.read())
        return got if got.get("ok") else None
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def _free_port(host: str, start: int, tries: int = 20) -> int | None:
    for port in range(start, start + tries):
        with socket.socket() as sk:
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sk.bind((host, port))
                return port
            except OSError:
                continue
    return None


def write_runfile(host: str, port: int) -> None:
    import os
    RUNFILE.parent.mkdir(parents=True, exist_ok=True)
    RUNFILE.write_text(json.dumps(
        {"host": host, "port": port, "pid": os.getpid(), "started": db.now()}))


def read_runfile() -> dict | None:
    """Whatever a server last recorded, if that process is still alive."""
    try:
        got = json.loads(RUNFILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    try:
        import os
        os.kill(int(got["pid"]), 0)          # signal 0: does the process exist
    except (OSError, KeyError, ValueError):
        return None
    return got
