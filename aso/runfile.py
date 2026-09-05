"""Where a running server records itself, so a client can find it.

Extracted from a second, complete HTTP server implementation that sat beside
`aso.api` for months. Nothing ran it: the container starts uvicorn on
`aso.api:app` and the CLI's serve command does the same, and only these four
names were ever imported from it. Keeping five hundred lines of unreachable
request handling meant two versions of every route, where a fix to one silently
missed the other.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import db

HOST = "127.0.0.1"
PORT = 8765

RUNFILE = Path(__file__).resolve().parent.parent / "data" / "server.json"


def write_runfile(host: str, port: int) -> None:
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
        os.kill(int(got["pid"]), 0)          # signal 0: does the process exist
    except (OSError, KeyError, ValueError):
        return None
    return got
