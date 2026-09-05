"""Talk to a running `aso serve`, or say plainly that none is running.

Deliberately stdlib and tiny. The CLI uses this when a server is up so a lookup
costs a request instead of 16 seconds of model loading, and falls back to
in-process work when it is not.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from . import env as _env
from .runfile import HOST, PORT

_env.load()

def base() -> str:
    """Where the server is.

    ASO_SERVER wins. Otherwise the default port, and failing that whatever a
    running server recorded in its runfile - so `aso serve --port 9000` is found
    without every command needing to be told.
    """
    if os.environ.get("ASO_SERVER"):
        return os.environ["ASO_SERVER"]
    from .runfile import read_runfile
    got = read_runfile()
    if got:
        return f"http://{got['host']}:{got['port']}"
    return f"http://{HOST}:{PORT}"


BASE = base()


def up(timeout: float = 0.4) -> dict | None:
    """Is a server listening? Short timeout: this runs before every command."""
    global BASE
    BASE = base()
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def call(route: str, payload: dict | None = None, timeout: float = 900) -> dict:
    data = json.dumps(payload or {}).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("ASO_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{base()}{route}", data=data, headers=headers,
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = json.loads(e.read() or b"{}")
        raise SystemExit(body.get("error", f"server returned {e.code}"))
    except urllib.error.URLError as e:
        raise SystemExit(f"no server at {base()} ({e.reason}). Start one: aso serve")
