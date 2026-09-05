"""Which address a fetch leaves from.

The scraper library calls `urllib.request.urlopen` directly, and a proxy is
per-request while that call is process-wide. So the proxy is held in a
thread-local and the library's `fetch` is wrapped once, at import, to read it.

Thread-local rather than an argument threaded through five call sites because
the library's own functions sit between us and the fetch, and passing it would
mean forking them.
"""
from __future__ import annotations

import threading
import urllib.request
from contextlib import contextmanager

_local = threading.local()


def current() -> str | None:
    return getattr(_local, "proxy", None)


@contextmanager
def through(proxy: str | None):
    """Route every fetch on this thread through `proxy` for the duration."""
    was = current()
    _local.proxy = proxy
    try:
        yield
    finally:
        _local.proxy = was


def key(proxy: str | None) -> str:
    """A stable name for an egress, safe to log and to use as a dict key.

    The proxy URL carries credentials, so the key is a digest of it rather than
    the thing itself. Two users who configure the same proxy land on the same
    key, which is correct: Play sees one address.
    """
    if not proxy:
        return "shared"
    import hashlib
    return "proxy:" + hashlib.sha256(proxy.encode()).hexdigest()[:16]


def install() -> None:
    """Teach the scraper library's fetch to honour `through(...)`.

    Wrapped rather than reimplemented: the library owns the headers, the
    encoding and the retries, and copying those here would leave two versions to
    keep in step. Only the opener changes.
    """
    try:
        from google_play_api_unofficial import http as _http
    except Exception:                                    # noqa: BLE001
        return
    if getattr(_http, "_aso_proxy_aware", False):
        return

    original = _http.fetch

    def fetch(url: str, headers: dict | None = None, timeout: int = 15) -> str:
        proxy = current()
        if not proxy:
            return original(url, headers, timeout)
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        req = urllib.request.Request(url, headers=headers or _http.HEADERS)
        with opener.open(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")

    _http.fetch = fetch
    _http._aso_proxy_aware = True
