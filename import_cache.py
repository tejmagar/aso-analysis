#!/usr/bin/env python3
"""Import the research workspace's app cache as dated snapshots.

~/app-idea/db.sqlite3 holds a `play_store_cache` table: 52 apps captured in June
2026 with their install counts at that moment. Nothing else in the project has
an install count from a known earlier date, so this is the only place a real
growth rate can come from rather than a lifetime total divided by an age.

Today's numbers are written alongside them, so the pair exists immediately.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from aso import db, scrape

SOURCE = Path(os.environ.get("ASO_RESEARCH_DB",
                             Path.home() / "app-idea" / "db.sqlite3"))


def main() -> int:
    if not SOURCE.exists():
        print(f"no research database at {SOURCE}")
        return 1

    sq = sqlite3.connect(SOURCE)
    sq.row_factory = sqlite3.Row
    con = db.connect()

    rows = sq.execute("""
        SELECT package, installs_real, score, ratings_count, updated, cached_at
        FROM play_store_cache WHERE package IS NOT NULL""").fetchall()

    past = now = 0
    for r in rows:
        # The cache's own timestamp, not today: the whole value of the row is
        # that it says what was true in June.
        con.execute(
            "INSERT INTO app_snapshots (pkg, observed_at, installs, rating, "
            "reviews, updated_at, source) VALUES (%s,%s,%s,%s,%s,%s,'research') "
            "ON CONFLICT (pkg, observed_at) DO NOTHING",
            (r["package"], r["cached_at"], _int(r["installs_real"]),
             _float(r["score"]), _int(r["ratings_count"]),
             scrape.parse_date(r["updated"])))
        past += 1

        # And where we hold the app now, the matching end of the pair.
        cur = con.execute(
            "SELECT installs, rating, reviews, updated_at, scraped_at "
            "FROM apps WHERE pkg = %s", (r["package"],)).fetchone()
        if cur:
            con.execute(
                "INSERT INTO app_snapshots (pkg, observed_at, installs, rating, "
                "reviews, updated_at, source) VALUES (%s,%s,%s,%s,%s,%s,'scrape') "
                "ON CONFLICT (pkg, observed_at) DO NOTHING",
                (r["package"], cur["scraped_at"], cur["installs"], cur["rating"],
                 cur["reviews"], cur["updated_at"]))
            now += 1

    pairs = db.scalar(con, """
        SELECT COUNT(*) FROM (SELECT pkg FROM app_snapshots
        GROUP BY pkg HAVING COUNT(*) > 1) t""")
    print(f"  {past} June snapshots, {now} matched to today")
    print(f"  {pairs} apps now have two points in time")
    sq.close()
    con.close()
    return 0


def _int(v):
    try:
        return int(str(v).replace(",", "")) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
