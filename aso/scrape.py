"""Adapter over the local google-play-api-unofficial package.

Lazy scrape: nothing is crawled up front. You ask for a keyword, it fetches that
one SERP, upserts the apps, and records positions. Re-running is cheap and
appends a fresh snapshot rather than overwriting the old one.
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from . import db, ui

# Normally installed as a dependency. ASO_GPAPI_SRC points at a working tree
# instead, which is only useful when hacking on the scraper itself.
GPAPI_SRC = os.environ.get("ASO_GPAPI_SRC")


def _api():
    """The Play Store reader. Installed copy first, working tree if pointed at one."""
    try:
        import google_play_api_unofficial as g
    except ImportError:
        if GPAPI_SRC and Path(GPAPI_SRC).is_dir():
            sys.path.insert(0, str(GPAPI_SRC))
            import google_play_api_unofficial as g
        else:
            raise SystemExit(
                "the Play Store reader is missing. Install it with:\n"
                "  pip install 'google-play-api-unofficial @ "
                "git+https://github.com/tejmagar/google-play-api-unofficial'")
    return g


_MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def parse_int(v) -> int:
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    digits = re.sub(r"[^\d]", "", str(v))
    return int(digits) if digits else 0


def parse_float(v) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return 0.0


def parse_date(v) -> str | None:
    """'May 29, 2013' -> '2013-05-29'. Returns None rather than guessing."""
    if not v:
        return None
    m = re.match(r"([A-Za-z]{3})[a-z]*\s+(\d{1,2}),\s*(\d{4})", str(v).strip())
    if not m:
        return None
    mon, day, year = m.groups()
    if mon.capitalize() not in _MONTHS:
        return None
    return f"{int(year):04d}-{_MONTHS[mon.capitalize()]:02d}-{int(day):02d}"


def to_app_row(d: dict, country="us", lang="en") -> dict:
    """Map a scraper payload onto the apps table. Tolerates both the thin
    fetch_apps shape and the rich fetch_app_details shape."""
    installs = d.get("installs_real") or d.get("installs_min") or d.get("installs")
    return {
        "pkg": d.get("package"),
        "title": d.get("title") or "",
        "short_desc": d.get("short_description") or "",
        "description": d.get("description") or "",
        "developer": d.get("developer") or "",
        "category": d.get("category") or "",
        "installs": parse_int(installs),
        "rating": parse_float(d.get("score") or d.get("rating")),
        "reviews": parse_int(d.get("ratings_count") or d.get("reviews_count")),
        "icon": d.get("icon"),
        "released_at": parse_date(d.get("released")),
        "updated_at": parse_date(d.get("updated")),
        "country": country, "lang": lang, "raw_json": None, "scraped_at": None,
    }


def organic_positions(results: list[dict]) -> list[tuple[int | None, dict]]:
    """Assign organic ranks, skipping Play's promoted hero card.

    fetch_apps prepends featured cards that are absent from the organic block, so
    a naive enumerate() records a paid placement as organic #1 and shifts every
    real position by one. Only the LEADING run of featured entries is prepended;
    a featured app that also ranks organically keeps its slot and its flag.

    The residual ambiguity (organic #1 being itself featured) is why `featured`
    is stored on the observation: the labels stay recoverable either way.
    """
    lead = 0
    while lead < len(results) and results[lead].get("featured"):
        lead += 1
    out: list[tuple[int | None, dict]] = [(None, r) for r in results[:lead]]
    out += [(i, r) for i, r in enumerate(results[lead:], start=1)]
    return out


# A candidate list has to be mostly apps, and big enough that "mostly" means
# something. Both numbers are deliberately loose: the point is to accept a real
# result list carrying one entry we cannot read, not to guess at what a list is.
_MIN_ENTRIES = 3
_MIN_SHARE = 0.6

# Below this many results, try the relaxed pass as well. A real Play page for a
# phrase worth analysing does not hold two apps; that number means something was
# dropped on the way in.
_THIN = 5


def _relaxed_fetch(query: str, timeout: int = 20) -> list[dict]:
    """Workaround for two upstream bugs, both of which throw away real results.

    The first is a size floor: `_find_apps_block` only accepts a block of ten or
    more entries, so a query Play answers with fewer comes back empty rather than
    short. "shake flashlight" returns six real apps and yields nothing.

    The second is stricter and worse. The walker requires EVERY child of a list
    to parse as an app, so a single entry it cannot read discards the entire
    page. "all social media app" returns nineteen apps of which eighteen parse
    cleanly, and the one that does not costs all nineteen: the tool reported no
    organic field at all for a phrase whose page opens with Facebook, Instagram
    and TikTok.

    This reuses the package's own DS4_RE and _extract, so nothing about the
    parsing is duplicated. Only the two thresholds move: any number of entries,
    and a strong majority rather than all of them.
    """
    try:
        from google_play_api_unofficial.http import fetch
        from google_play_api_unofficial.search import (
            DS4_RE, PLAY_BASE, _extract, _looks_like_app_entry)
    except Exception:                                    # noqa: BLE001
        return []

    import json
    import urllib.parse

    try:
        html = fetch(f"{PLAY_BASE}/store/search?"
                     f"q={urllib.parse.quote_plus(query)}&c=apps&hl=en-US", timeout=timeout)
        m = DS4_RE.search(html)
        if not m:
            return []
        data = json.loads(m.group(1))
    except Exception:                                    # noqa: BLE001
        return []

    best: list = []
    best_score = 0

    def entries(node) -> int:
        """How many children of this list read as apps."""
        return sum(1 for x in node
                   if isinstance(x, list) and len(x) == 1 and _looks_like_app_entry(x[0]))

    def walk(node):
        nonlocal best, best_score
        if isinstance(node, list):
            if len(node) >= _MIN_ENTRIES:
                good = entries(node)
                if good >= _MIN_ENTRIES and good >= _MIN_SHARE * len(node) \
                        and good > best_score:
                    best, best_score = node, good
            for x in node:
                walk(x)

    walk(data)
    out, seen = [], set()
    for wrapper in best:
        # Skip the entries that did not parse rather than dropping the list they
        # were found in; their neighbours are still the result page.
        if not (isinstance(wrapper, list) and len(wrapper) == 1):
            continue
        a = _extract(wrapper[0])
        if a and a["package"] not in seen:
            seen.add(a["package"])
            a.setdefault("featured", False)
            out.append(a)
    return out


def scrape_keyword(con, keyword: str, country="us", lang="en",
                   with_details=True, sleep=1.0, verbose=True, progress=None) -> int:
    """Fetch one SERP and record it. Returns the number of ranked observations.

    `progress(line)` is called as each stage completes. This work takes about a
    minute and a caller watching a spinner cannot tell it apart from a hang, so
    the stages say what is actually happening rather than being estimated.
    """
    g = _api()
    keyword = keyword.strip().lower()
    quiet = not verbose
    say = progress or (lambda _line: None)
    say(f"Searching Play for {keyword!r}")

    with ui.Task(f"searching Play for {keyword!r}", quiet=quiet) as t:
        results = g.fetch_apps(keyword) or []

        # A handful of results is as suspicious as none. On "all social media
        # app" the reader returned exactly one, the featured card, because the
        # organic list beneath it carried a single entry it could not parse and
        # the whole list was discarded. Falling back only on zero missed that
        # entirely, and the tool reported no field for a page that opens with
        # Facebook and Instagram.
        if len(results) < _THIN:
            relaxed = _relaxed_fetch(keyword)
            if len(relaxed) > len(results):
                # Keep whatever the reader found, in front: the featured cards
                # live there, and the relaxed pass only sees the organic list.
                have = {r.get("package") for r in results}
                results = results + [r for r in relaxed if r.get("package") not in have]
                t.done(f"{len(results)} results "
                       f"({len(relaxed)} recovered that upstream would drop)")
            else:
                t.done(f"{len(results)} results")
        else:
            t.done(f"{len(results)} results")
    say(f"Found {len(results)} results")
    if not results:
        return 0

    ranked = organic_positions(results)
    n_featured = sum(1 for pos, _ in ranked if pos is None)
    if n_featured:
        ui.say(f"skipping {n_featured} promoted card(s) when numbering ranks", quiet)

    observed_at = db.now()
    n = 0
    failed = 0
    # Buffer the rows and write them in ONE short transaction at the end. Writing
    # inside the fetch loop held the database open across every network call -
    # roughly 40 seconds per keyword - and that is what made concurrent commands
    # fail with "database is locked".
    pending: list[tuple[dict, int | None, bool]] = []
    task = ui.Task("fetching app details", total=len(ranked), quiet=quiet or not with_details)
    task.__enter__()
    for pos, r in ranked:
        pkg = r.get("package")
        if not pkg:
            continue
        payload = r
        if with_details:
            task.step(pkg)
            # Every app would be a line a second for thirty seconds, which is
            # noise. Every fifth is enough to show it is still moving.
            if task.n % 5 == 0 or task.n == len(ranked):
                say(f"Reading listings, {task.n} of {len(ranked)}")
            try:
                detail = g.fetch_app_details(pkg)
                if detail:
                    payload = {**r, **{k: v for k, v in detail.items() if v is not None}}
            except Exception:                           # noqa: BLE001 - keep the crawl going
                failed += 1
            time.sleep(sleep)
        pending.append((to_app_row(payload, country, lang), pos,
                        bool(r.get("featured"))))
        n += pos is not None

    with con.transaction():                                  # one transaction, milliseconds
        for row, pos, is_featured in pending:
            # A dated point for this app, alongside the row that overwrites its
            # current state. The overwrite is what loses history; this keeps it.
            db.add_snapshot(con, {**row, "scraped_at": observed_at})
            db.upsert_app(con, row)
            con.execute(
                "INSERT INTO observations "
                "(keyword, country, pkg, position, source, featured, observed_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (keyword, country, pkg, observed_at) DO NOTHING",
                (keyword, country, row["pkg"], pos, "serp", int(is_featured),
                 observed_at))

    # Record the field's shape at this moment. Written HERE, not in analyze(),
    # because every path that scrapes must extend the series: ingest and track
    # never call analyze, and their snapshots are the ones a future comparison
    # depends on. A scrape whose state is not recorded is a before-state lost.
    try:
        from . import features as _F
        from . import history as _H
        _rows = db.ranked_field(con, keyword, country)
        if _rows:
            _H.record(con, keyword, country,
                      _F.compute_field(_rows, keyword), observed_at=observed_at)
    except Exception:                                    # noqa: BLE001
        pass                                             # never fail a scrape over history

    task.done(f"{len(ranked) - failed} of {len(ranked)} enriched"
              + (f", {failed} failed" if failed else ""))
    ui.say(f"stored {n} ranked apps for {keyword!r}", quiet)
    say(f"Stored {n} ranked apps")
    return n


def parse_package(value: str) -> str:
    """Accept a package name or any Play Store link and return the package.

    People copy the URL from the browser far more often than they type the
    package, and pasting one used to fail with an unhelpful "unknown app".
    Handles the details URL, a shortened one, and a bare package unchanged.
    """
    v = (value or "").strip().strip("<>\"'")
    if not v:
        return v
    if "://" in v or "play.google.com" in v or "/" in v:
        import urllib.parse
        try:
            q = urllib.parse.urlparse(v if "://" in v else "https://" + v)
            pkg = urllib.parse.parse_qs(q.query).get("id", [None])[0]
            if pkg:
                return pkg.strip()
            # Some share links carry the package as the last path segment.
            tail = [x for x in q.path.split("/") if x]
            if tail and "." in tail[-1]:
                return tail[-1]
        except ValueError:
            pass
    return v


def suggest(prefix: str, games=False) -> list[str]:
    """Play autocomplete. Used as the demand proxy, not as a ranking signal.

    `filter` takes the package's Filter enum, not a string; passing a string
    fails deep inside with an unhelpful AttributeError on `.value`.
    """
    g = _api()
    return g.fetch_suggestions(prefix, filter=(g.Filter.GAMES if games else g.Filter.APPS))
