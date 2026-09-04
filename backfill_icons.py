#!/usr/bin/env python3
"""Fill in app icons from search results.

The icon arrives on both the search result and the detail payload, and was
simply not being stored. Backfilling by re-fetching each app's detail page would
be one request per app; a search returns thirty apps with their icons in one, and
the apps that matter are exactly the ones that appear on a result page. So this
walks the keywords instead, which is roughly twenty times fewer requests.
"""
from __future__ import annotations

import sys
import time

from aso import db, scrape


def main() -> int:
    con = db.connect()
    keywords = [r["keyword"] for r in con.execute(
        "SELECT DISTINCT keyword FROM observations ORDER BY keyword")]
    missing = db.scalar(con, "SELECT COUNT(*) FROM apps WHERE icon IS NULL")
    print(f"  {missing} apps without an icon, across {len(keywords)} keywords")

    g = scrape._api()
    filled = 0
    for i, kw in enumerate(keywords, 1):
        try:
            hits = g.fetch_apps(kw) or scrape._relaxed_fetch(kw)
        except Exception:                       # noqa: BLE001 - keep going
            continue
        rows = [(h.get("icon"), h.get("package")) for h in (hits or [])
                if h.get("icon") and h.get("package")]
        if rows:
            with con.transaction():
                for icon, pkg in rows:
                    con.execute(
                        "UPDATE apps SET icon = %s WHERE pkg = %s AND icon IS NULL",
                        (icon, pkg))
            filled += len(rows)
        if i % 20 == 0:
            left = db.scalar(con, "SELECT COUNT(*) FROM apps WHERE icon IS NULL")
            print(f"  {i}/{len(keywords)} keywords, {left} apps still without one",
                  flush=True)
        time.sleep(0.4)

    left = db.scalar(con, "SELECT COUNT(*) FROM apps WHERE icon IS NULL")
    have = db.scalar(con, "SELECT COUNT(icon) FROM apps")
    print(f"\n  done: {have} apps have an icon, {left} still do not")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
