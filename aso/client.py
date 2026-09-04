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

from .server import HOST, PORT

BASE = os.environ.get("ASO_SERVER", f"http://{HOST}:{PORT}")


def up(timeout: float = 0.4) -> dict | None:
    """Is a server listening? Short timeout: this runs before every command."""
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def call(route: str, payload: dict | None = None, timeout: float = 900) -> dict:
    data = json.dumps(payload or {}).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{BASE}{route}", data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = json.loads(e.read() or b"{}")
        raise SystemExit(body.get("error", f"server returned {e.code}"))
    except urllib.error.URLError as e:
        raise SystemExit(f"no server at {BASE} ({e.reason}). Start one: aso serve")
