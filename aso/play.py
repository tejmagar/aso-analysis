"""A thin wrapper over the Play Store reader.

Everything the underlying package can do, reachable from the CLI and the HTTP
API without going through the model. Useful for an agent that wants the raw
page, and for checking what the analyser was actually looking at when a result
seems wrong.

These are passthroughs: they fetch, normalise the numbers, and return. Nothing
here writes to the database or touches the model.
"""
from __future__ import annotations

import time

from .scrape import _api, organic_positions, parse_package, to_app_row


def suggest(query: str, games: bool = False, no_cache: bool = False,
            con=None) -> list[str]:
    """Play autocomplete, in the order Play returned it. The order is the signal.

    Cached for a day by default (`suggest_ttl_hours`).

    `no_cache` skips the READ but still performs the write: you get a fresh list
    and the cache is brought up to date by the same call. Bypassing storage as
    well would mean the next caller pays for the same fetch again, which is the
    opposite of what asking for fresh data should cost everyone else.

    Games use a different filter and are never served from the app cache.
    """
    if games:
        g = _api()
        return [s.strip() for s in g.fetch_suggestions(
            query, filter=g.Filter.GAMES) if s.strip()]

    from . import db as _db
    from . import suggest as _s
    own = con is None
    con = con or _db.connect()
    try:
        return _s.refresh(con, query) if no_cache else _s.ensure(con, query)
    finally:
        if own:
            con.close()


def search(query: str, with_details: bool = False, sleep: float = 0.6,
           limit: int | None = None) -> list[dict]:
    """Top results for a query, with organic rank assigned correctly.

    `position` is the ORGANIC slot. Play prepends a promoted hero card that is
    not part of the ranked list, so numbering straight off the array records a
    paid placement as #1 and shifts every real position by one. A promoted card
    comes back with `position: null` and `featured: true`.
    """
    g = _api()
    results = g.fetch_apps(query)
    out = []
    for pos, r in organic_positions(results):
        row = {**r, "position": pos, "featured": bool(r.get("featured"))}
        if with_details and r.get("package"):
            try:
                d = g.fetch_app_details(r["package"])
                if d:
                    row.update({k: v for k, v in d.items() if v is not None})
            except Exception as e:                   # noqa: BLE001
                row["detail_error"] = type(e).__name__
            time.sleep(sleep)
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def details(package: str) -> dict:
    """Full listing for one app. Accepts a package name or any Play Store URL."""
    g = _api()
    pkg = parse_package(package)
    d = g.fetch_app_details(pkg)
    if not d:
        raise SystemExit(f"could not read a listing for {pkg!r}")
    # Both shapes: what Play gave us, and the normalised row the model uses, so
    # a caller can see how "500,000,000+" and "May 29, 2013" were interpreted.
    return {"raw": d, "normalised": to_app_row(d)}


def publisher(developer: str) -> list[dict]:
    """Apps by a developer, via Play search.

    Caps at ~50 however many they have. `scripts/fetch_publisher.py` scrolls the
    developer page instead and reaches 100, which is Play's own ceiling.
    """
    g = _api()
    return g.fetch_publisher_apps(developer)
