#!/usr/bin/env python3
"""Fill the icons the keyword sweep could not reach.

Searching by keyword covers whatever Play returns for that keyword today, which
is not every app already on file: result sets move, and an app recorded weeks
ago may sit below the window now. Those are fetched one at a time from their own
detail page, which is slower per app but only applies to the remainder.
"""
from __future__ import annotations

import sys
import time

from aso import db, scrape


def main() -> int:
    con = db.connect()
    pkgs = [r["pkg"] for r in con.execute(
        "SELECT pkg FROM apps WHERE icon IS NULL ORDER BY pkg")]
    print(f"  {len(pkgs)} apps still without an icon", flush=True)

    g = scrape._api()
    got = missing = 0
    for i, pkg in enumerate(pkgs, 1):
        try:
            d = g.fetch_app_details(pkg)
        except Exception:                       # noqa: BLE001 - delisted, renamed, rate limited
            missing += 1
            d = None
        icon = (d or {}).get("icon")
        if icon:
            con.execute("UPDATE apps SET icon = %s WHERE pkg = %s", (icon, pkg))
            got += 1
        else:
            missing += 1
        if i % 100 == 0:
            print(f"  {i}/{len(pkgs)}: {got} filled, {missing} unavailable", flush=True)
        time.sleep(0.3)

    left = db.scalar(con, "SELECT COUNT(*) FROM apps WHERE icon IS NULL")
    print(f"\n  done: {got} filled, {left} still without one "
          f"(delisted or no longer served)")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
