"""The page archive: every historical result page, app by app, at its position.

Separate from the SQLite store on purpose. SQLite holds the working set the
model trains against and must stay fast and local; this holds the raw history,
which is append-only, wanted from more than one machine, and worth keeping even
when a derived feature is later found to be wrong. Losing a summary is cheap
because it can be recomputed from here. Losing the page is not.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path

import psycopg

SCHEMA = Path(__file__).resolve().parent.parent / "pg_schema.sql"

DSN_ENV = "ASO_PG_DSN"


def dsn() -> str:
    v = os.environ.get(DSN_ENV)
    if not v:
        raise RuntimeError(
            f"{DSN_ENV} is not set. Expected postgresql://user:pw@host:port/db")
    return v


def connect(url: str | None = None):
    return psycopg.connect(url or dsn(), connect_timeout=20)


def init(con) -> None:
    con.execute(SCHEMA.read_text())
    con.commit()


def _int(v) -> int | None:
    """Shares the importer's reader so the shorthand '5M' survives into the
    archive rather than being stripped to the digit 5."""
    from .oldpages import _installs
    return _installs(v)


def _released(v) -> tuple[date | None, str | None]:
    """A release date plus how precisely it was recorded.

    Three shapes appear across the sessions: the store's 'May 29, 2013', a plain
    ISO date, and a bare year. The bare year is kept rather than dropped - being
    released in 2017 is still worth years of age against a 2026 page - but it is
    labelled, so a mid-year placeholder is never read as an observed date.
    """
    raw = str(v or "").strip()
    if not raw:
        return None, None
    from . import scrape
    iso = scrape.parse_date(raw)
    if iso:
        return date.fromisoformat(iso), "day"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        try:
            return date.fromisoformat(raw), "day"
        except ValueError:
            return None, None
    if re.fullmatch(r"(19|20)\d{2}", raw):
        return date(int(raw), 7, 1), "year"
    return None, None


def _order_is_rank(apps: list[dict]) -> bool | None:
    """Is this list's order the page's order, or was it sorted before saving?

    A list that runs strictly descending by installs is consistent with both, so
    it answers None rather than guessing. Anything out of install order can only
    be page order, because nobody sorts a list into that.
    """
    inst = [i for i in (_int(a.get("installs_real") or a.get("installs"))
                        for a in apps) if i]
    if len(inst) < 3:
        return None
    return None if all(a >= b for a, b in zip(inst, inst[1:])) else True


def _at(v) -> datetime:
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        d = datetime.now()
    if d.tzinfo is None:
        from datetime import timezone
        d = d.replace(tzinfo=timezone.utc)
    return d


def write_page(con, keyword: str, observed_at, source: str, apps: list[dict],
               country: str = "us", ordered: bool = True) -> int:
    """Insert one page. Re-running is idempotent: the same page on the same date
    replaces its apps rather than appending a second copy of them."""
    row = con.execute(
        "INSERT INTO snapshots (keyword, country, observed_at, source, n_apps, "
        "order_is_rank) VALUES (%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (keyword, country, observed_at, source) DO UPDATE SET "
        "n_apps = EXCLUDED.n_apps, order_is_rank = EXCLUDED.order_is_rank "
        "RETURNING id",
        (" ".join(str(keyword).lower().split()), country, _at(observed_at),
         source, len(apps),
         _order_is_rank(apps) if ordered else False)).fetchone()
    sid = row[0]
    con.execute("DELETE FROM snapshot_apps WHERE snapshot_id = %s", (sid,))
    with con.cursor() as cur:
        cur.executemany(
            "INSERT INTO snapshot_apps (snapshot_id, rank, title, pkg, developer,"
            " installs, installs_real, rating, reviews, released, "
            "released_precision) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [_app_row(sid, i, a) for i, a in enumerate(apps, start=1)])
    return sid


def _app_row(sid: int, rank: int, a: dict) -> tuple:
    rel, prec = _released(a.get("released") or a.get("released_at"))
    rating = a.get("score") if a.get("score") is not None else a.get("rating")
    try:
        rating = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating = None
    return (sid, rank,
            a.get("title") or a.get("app") or a.get("name"),
            a.get("package") or a.get("appId") or a.get("pkg"),
            a.get("dev") or a.get("developer"),
            _int(a.get("installs")) or _int(a.get("installs_display")),
            _int(a.get("installs_real")),
            rating,
            _int(a.get("reviews")),
            rel, prec)


def load_research(con, workspace=None) -> dict:
    """Every page recorded in the research workspace, at its recorded order.

    The JSON sessions nest the same page under several keys, so the walk emits
    it more than once; the widest copy wins, since a page that lists eight apps
    and one that lists three are the same page seen through different filters.
    """
    from . import oldpages
    best: dict[tuple, dict] = {}
    for s in oldpages.from_db(workspace) + oldpages.from_json(workspace):
        key = (s["keyword"], str(s["observed_at"])[:10])
        if key not in best or len(s["apps"]) > len(best[key]["apps"]):
            best[key] = s
    for s in best.values():
        write_page(con, s["keyword"], s["observed_at"], "research", s["apps"],
                   ordered=s.get("ordered", True))
    con.commit()
    return {"pages": len(best)}


def load_scrapes(con, sq, country: str = "us") -> dict:
    """Our own scraped pages, which carry real observed positions.

    Unlike the research rows these need no order guessing: the position was
    recorded at scrape time, so it is written straight through and the page is
    marked as ranked.
    """
    pages = sq.execute(
        "SELECT DISTINCT keyword, observed_at FROM observations "
        "WHERE country=? AND position IS NOT NULL", (country,)).fetchall()
    n = 0
    for kw, at in pages:
        apps = sq.execute(
            "SELECT o.position, a.title, a.pkg, a.developer, a.installs, "
            "a.rating, a.reviews, a.released_at FROM observations o "
            "JOIN apps a USING (pkg) WHERE o.keyword=? AND o.country=? "
            "AND o.observed_at=? AND o.position IS NOT NULL "
            "ORDER BY o.position", (kw, country, at)).fetchall()
        if not apps:
            continue
        write_page(con, kw, at, "scrape", [
            {"title": r[1], "package": r[2], "developer": r[3],
             "installs_real": r[4], "rating": r[5], "reviews": r[6],
             "released": r[7]} for r in apps], country=country)
        # Position was measured, not inferred, so the ambiguity flag that
        # applies to a written-down list does not apply here.
        con.execute("UPDATE snapshots SET order_is_rank = TRUE WHERE keyword=%s "
                    "AND country=%s AND observed_at=%s AND source='scrape'",
                    (" ".join(kw.lower().split()), country, _at(at)))
        n += 1
    con.commit()
    return {"pages": n}
