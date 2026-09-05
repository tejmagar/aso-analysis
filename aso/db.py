"""Postgres access. One schema file, no ORM, no migrations framework.

Ported from SQLite. The table and column names did not change, so most SQL is
the same statement with a different placeholder; the handful of SQLite-only
constructs are rewritten and each is marked where it appears.

Connections run with autocommit on. Under SQLite a bare `con.execute("INSERT
...")` committed by itself, and roughly fifty call sites rely on that; leaving
autocommit off would make every one of them a write that vanishes when the
connection closes. Where several statements have to land together they say so
explicitly with `con.transaction()`, which is atomic whether or not autocommit
is set.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema_pg.sql"

DSN_ENV = "ASO_PG_DSN"

ABSENT = 251  # checked and not in the top 250
CONNECT_TIMEOUT = 20   # seconds to wait for the server before giving up

_READY = False


def dsn() -> str:
    v = os.environ.get(DSN_ENV)
    if not v:
        raise SystemExit(
            f"{DSN_ENV} is not set.\n"
            f"  export {DSN_ENV}=postgresql://user:password@host:5470/aso")
    return v


def now() -> str:
    """Microsecond precision, deliberately. observations is UNIQUE on
    (keyword, country, pkg, observed_at) and inserts ignore conflicts, so a
    second-resolution stamp lets a reviewed rank filed in the same second as a
    scrape be silently dropped, and the correction vanishes without a trace."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def scalar(con, sql: str, params=()):
    """The first column of the first row, or None.

    Rows come back as dicts, so `fetchone()[0]` no longer works. Wrapping it
    here keeps the call sites reading as the counting queries they are, rather
    than each inventing its own column alias.
    """
    row = con.execute(sql, params).fetchone()
    return None if row is None else next(iter(row.values()))


def connect(url: str | None = None):
    """Open a connection, creating the schema the first time in this process.

    The schema is applied once per process rather than once per connection: it
    is thirteen CREATE TABLE IF NOT EXISTS statements, and running them on every
    connect is pure round trips against a server that is not local.
    """
    global _READY
    con = psycopg.connect(url or dsn(), row_factory=dict_row,
                          autocommit=True, connect_timeout=CONNECT_TIMEOUT)
    if not _READY:
        exists = con.execute("SELECT to_regclass('public.observations') IS NOT NULL AS ok"
                             ).fetchone()["ok"]
        if not exists:
            con.execute(SCHEMA.read_text())
        _READY = True
    return con


def init(url: str | None = None) -> str:
    """Apply the schema explicitly. Every statement is IF NOT EXISTS, so this is
    safe to re-run and there is no migration step to remember."""
    con = connect(url)
    con.execute(SCHEMA.read_text())
    con.close()
    return url or dsn()


def upsert_app(con, app: dict) -> None:
    cols = ("pkg title short_desc description developer category installs rating "
            "reviews icon released_at updated_at country lang raw_json scraped_at").split()
    row = {c: app.get(c) for c in cols}
    row["scraped_at"] = row["scraped_at"] or now()
    row["raw_json"] = row["raw_json"] or json.dumps(app, default=str)
    placeholders = ",".join(f"%({c})s" for c in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "pkg")
    con.execute(
        f"INSERT INTO apps ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(pkg) DO UPDATE SET {updates}", row)


def add_snapshot(con, app: dict, observed_at: str | None = None,
                 source: str = "scrape") -> None:
    """Record what an app looked like at this moment.

    Called on every scrape, so growth becomes measurable rather than inferred.
    Everywhere else stores a lifetime total and a release date, and dividing one
    by the other averages across the app's whole life: an app that took a million
    installs in year one and nothing since reads exactly like one earning
    steadily now. Two of these a month apart give the real rate.
    """
    con.execute(
        "INSERT INTO app_snapshots (pkg, observed_at, installs, rating, reviews, "
        "updated_at, source) VALUES (%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (pkg, observed_at) DO NOTHING",
        (app.get("pkg"), observed_at or app.get("scraped_at") or now(),
         app.get("installs"), app.get("rating"), app.get("reviews"),
         app.get("updated_at"), source))


def add_observation(con, keyword, pkg, position, country="us",
                    source="serp", observed_at=None) -> None:
    con.execute(
        "INSERT INTO observations "
        "(keyword, country, pkg, position, source, observed_at) VALUES (%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT (keyword, country, pkg, observed_at) DO NOTHING",
        (keyword.strip().lower(), country, pkg, position, source, observed_at or now()))


def latest_observations(con, country="us", keyword: str | None = None):
    """Most recent observation per (keyword, pkg). Older snapshots stay for history.

    `keyword` narrows the scan inside the query. That is not a micro-optimisation:
    the one-keyword caller below used to pull the whole join and discard 99% of it
    in Python, so building every field for training dragged the entire corpus
    across the network once per keyword.
    """
    # Newest row per (keyword, pkg), ties broken by insertion order so "latest"
    # is deterministic. A reviewed rank filed after a scrape supersedes it.
    # The derived table carries an alias because Postgres requires one.
    where, params = "WHERE country = %s AND position IS NOT NULL", [country]
    if keyword is not None:
        where += " AND keyword = %s"
        params.append(keyword.strip().lower())
    return con.execute(f"""
        SELECT o.keyword, o.pkg, o.position, o.source, o.featured, o.observed_at, a.*
        FROM observations o
        JOIN apps a USING (pkg)
        WHERE o.id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY keyword, pkg
                    ORDER BY observed_at DESC, id DESC) AS rn
                FROM observations
                {where}
            ) ranked WHERE rn = 1
        )
    """, tuple(params)).fetchall()


def set_override(con, keyword, field, value, country="us", reviewer="you") -> None:
    con.execute(
        "INSERT INTO overrides (keyword, country, field, value, reviewer, ts) "
        "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(keyword, country, field) DO UPDATE SET "
        "value=excluded.value, reviewer=excluded.reviewer, ts=excluded.ts",
        (keyword.strip().lower(), country, field, float(value), reviewer, now()))


def get_override(con, keyword, field, country="us"):
    r = con.execute(
        "SELECT value FROM overrides WHERE keyword=%s AND country=%s AND field=%s",
        (keyword.strip().lower(), country, field)).fetchone()
    return None if r is None else r["value"]


def featured_apps(con, keyword: str, country="us") -> list[dict]:
    """Play's promoted hero card(s) for a keyword.

    These carry position NULL because they are not organic, so
    latest_observations() filters them out. They still matter enormously: the
    card sits above every organic result and takes the taps, and when it is
    on-intent it is a direct competitor occupying the most visible slot on the
    page. Dropping it silently made that signal invisible to the model, and it
    is a strong one: Play shows a featured card when it is confident the phrase
    belongs to one app.

    DISTINCT ON, not GROUP BY: the SQLite original selected whole rows while
    grouping by pkg alone, which Postgres rejects outright and which SQLite only
    allowed by picking an arbitrary row per group.
    """
    rows = con.execute(
        "SELECT DISTINCT ON (o.pkg) o.pkg, o.observed_at, a.* "
        "FROM observations o JOIN apps a USING (pkg) "
        "WHERE o.keyword=%s AND o.country=%s AND o.featured=1 "
        "ORDER BY o.pkg, o.observed_at DESC",
        (keyword.strip().lower(), country)).fetchall()
    return [dict(r) for r in rows]


def measured_growth(con, pkgs: list[str]) -> dict[str, float]:
    """Installs per day between each app's oldest and newest snapshot.

    Measured over a real window rather than divided out of a lifetime, so an app
    that earned its million a decade ago and one earning steadily now do not
    read the same. Apps with a single snapshot are absent rather than zero: not
    yet measured is not the same as not growing.
    """
    if not pkgs:
        return {}
    rows = con.execute("""
        SELECT pkg,
               MIN(observed_at) AS first_at, MAX(observed_at) AS last_at,
               MIN(installs)    AS first_n,  MAX(installs)    AS last_n
          FROM app_snapshots WHERE pkg = ANY(%s)
         GROUP BY pkg HAVING COUNT(*) > 1""", (list(pkgs),)).fetchall()
    out: dict[str, float] = {}
    for r in rows:
        days = _days_between(r["first_at"], r["last_at"])
        # Under a week the difference is mostly rounding in Play's own reporting.
        if days < 7 or r["first_n"] is None or r["last_n"] is None:
            continue
        gained = (r["last_n"] or 0) - (r["first_n"] or 0)
        if gained >= 0:
            out[r["pkg"]] = gained / days
    return out


def _days_between(a: str, b: str) -> float:
    from datetime import datetime
    try:
        pa = datetime.fromisoformat(str(a).replace("Z", "+00:00")).replace(tzinfo=None)
        pb = datetime.fromisoformat(str(b).replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return 0.0
    return abs((pb - pa).total_seconds()) / 86400.0


def ranked_field(con, keyword: str, country="us") -> list[dict]:
    """The field for one keyword, with positions made internally consistent.

    Observed positions come from different moments and different sources, so
    they collide: recording "our app really ranks #2" leaves the app that was
    already at #2 still claiming it, and a top-10 window then holds eleven apps
    or double-counts a slot.

    Renumbering the others would be inventing observations we never made, so the
    stored rows are left alone and the ORDER is re-derived here: sort by observed
    position, break ties in favour of the reviewed row, and hand out dense slots.
    `position` stays the raw observation; `slot` is the consistent one.
    """
    rows = [dict(r) for r in latest_observations(con, country, keyword)]
    return _slot(rows, measured_growth(con, [r["pkg"] for r in rows]))


def _slot(rows: list[dict], growth: dict[str, float]) -> list[dict]:
    """Shared body of the two readers above and below: order, then dense slots."""
    rows.sort(key=lambda r: (r["position"],
                             0 if r.get("source") == "review" else 1,
                             -(r.get("installs") or 0)))
    for i, r in enumerate(rows, start=1):
        r["slot"] = i
        r["position"] = i          # what every downstream reader uses
        # Absent when the app has only ever been seen once, which is different
        # from measured at zero and has to stay distinguishable downstream.
        r["measured_per_day"] = growth.get(r["pkg"])
    return rows


def ranked_fields(con, country="us") -> dict[str, list[dict]]:
    """Every keyword's field, from one pass over the table.

    Identical to calling ranked_field for each keyword, minus the round trips.
    Training reads every field there is, and asking for them one at a time meant
    258 full fetches of the same 6,000 rows.
    """
    by_kw: dict[str, list[dict]] = defaultdict(list)
    for r in latest_observations(con, country):
        by_kw[r["keyword"]].append(dict(r))
    growth = measured_growth(
        con, list({r["pkg"] for rows in by_kw.values() for r in rows}))
    return {kw: _slot(rows, growth) for kw, rows in by_kw.items()}


@contextmanager
def session(url=None):
    """A connection held for exactly as long as it is used.

    Postgres readers do not block writers, so this is no longer about lock
    contention. It is about not leaking connections: the server allows a fixed
    number, and a handle kept across a 40-second scrape is one nobody else can
    use.
    """
    con = connect(url)
    try:
        yield con
    finally:
        con.close()
