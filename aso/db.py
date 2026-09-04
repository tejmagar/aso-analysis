"""SQLite access. One file, no ORM, no migrations framework."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("ASO_DB", ROOT / "data" / "aso.db"))
SCHEMA = ROOT / "schema.sql"

ABSENT = 251  # checked and not in the top 250


def now() -> str:
    """Microsecond precision, deliberately. observations is UNIQUE on
    (keyword, country, pkg, observed_at) with INSERT OR IGNORE, so a
    second-resolution stamp lets a reviewed rank filed in the same second as a
    scrape be silently dropped, and the correction vanishes without a trace."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Opens, and creates the schema on first use. There is no `init` step to
    forget: any command works against a fresh checkout."""
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA.read_text())          # every statement is IF NOT EXISTS
    return con


def init(path: Path | None = None) -> Path:
    con = connect(path)
    with con:
        con.executescript(SCHEMA.read_text())
    con.close()
    return Path(path or DB_PATH)


def upsert_app(con, app: dict) -> None:
    cols = ("pkg title short_desc description developer category installs rating "
            "reviews released_at updated_at country lang raw_json scraped_at").split()
    row = {c: app.get(c) for c in cols}
    row["scraped_at"] = row["scraped_at"] or now()
    row["raw_json"] = row["raw_json"] or json.dumps(app, default=str)
    placeholders = ",".join(f":{c}" for c in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "pkg")
    con.execute(
        f"INSERT INTO apps ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(pkg) DO UPDATE SET {updates}", row)


def add_observation(con, keyword, pkg, position, country="us",
                    source="serp", observed_at=None) -> None:
    con.execute(
        "INSERT OR IGNORE INTO observations "
        "(keyword, country, pkg, position, source, observed_at) VALUES (?,?,?,?,?,?)",
        (keyword.strip().lower(), country, pkg, position, source, observed_at or now()))


def latest_observations(con, country="us"):
    """Most recent observation per (keyword, pkg). Older snapshots stay for history."""
    # Newest row per (keyword, pkg), ties broken by insertion order so "latest"
    # is deterministic. A reviewed rank filed after a scrape supersedes it.
    return con.execute("""
        SELECT o.keyword, o.pkg, o.position, o.source, o.featured, o.observed_at, a.*
        FROM observations o
        JOIN apps a USING (pkg)
        WHERE o.id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY keyword, pkg
                    ORDER BY observed_at DESC, id DESC) AS rn
                FROM observations
                WHERE country = ? AND position IS NOT NULL
            ) WHERE rn = 1
        )
    """, (country,)).fetchall()


def set_override(con, keyword, field, value, country="us", reviewer="you") -> None:
    with con:
        con.execute(
            "INSERT INTO overrides (keyword, country, field, value, reviewer, ts) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(keyword, country, field) DO UPDATE SET "
            "value=excluded.value, reviewer=excluded.reviewer, ts=excluded.ts",
            (keyword.strip().lower(), country, field, float(value), reviewer, now()))


def get_override(con, keyword, field, country="us"):
    r = con.execute("SELECT value FROM overrides WHERE keyword=? AND country=? AND field=?",
                    (keyword.strip().lower(), country, field)).fetchone()
    return None if r is None else r["value"]


def featured_apps(con, keyword: str, country="us") -> list[dict]:
    """Play's promoted hero card(s) for a keyword.

    These carry position NULL because they are not organic, so
    latest_observations() filters them out. They still matter enormously: the
    card sits above every organic result and takes the taps, and when it is
    on-intent it is a direct competitor occupying the most visible slot on the
    page. Dropping it silently made a bought top slot invisible to the model.
    """
    rows = con.execute(
        "SELECT o.pkg, o.observed_at, a.* FROM observations o JOIN apps a USING (pkg) "
        "WHERE o.keyword=? AND o.country=? AND o.featured=1 "
        "GROUP BY o.pkg HAVING MAX(o.observed_at)",
        (keyword.strip().lower(), country)).fetchall()
    return [dict(r) for r in rows]


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
    rows = [dict(r) for r in latest_observations(con, country)
            if r["keyword"] == keyword.strip().lower()]
    rows.sort(key=lambda r: (r["position"],
                             0 if r.get("source") == "review" else 1,
                             -(r.get("installs") or 0)))
    for i, r in enumerate(rows, start=1):
        r["slot"] = i
        r["position"] = i          # what every downstream reader uses
    return rows
