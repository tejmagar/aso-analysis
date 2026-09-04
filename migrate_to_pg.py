#!/usr/bin/env python3
"""Copy the SQLite database into Postgres, table by table.

One-way and re-runnable: every table is truncated before it is filled, so a run
that fails halfway can simply be run again. It reads column names from the
SQLite side and writes only the ones Postgres also has, which is what lets the
two schemas drift by a column without this script needing to know.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import psycopg

# Order matters: apps before observations, predictions before corrections,
# corrections before residuals. A foreign key cannot point at a row that has
# not been copied yet.
TABLES = ["apps", "observations", "fields", "predictions", "corrections",
          "residuals", "registry", "demand", "overrides", "suggestions",
          "families", "field_history", "holdout"]

# SQLite hands back whatever was written; these columns are declared INTEGER in
# Postgres and will refuse a float or a bool.
BATCH = 1000


def columns(pg, table: str) -> list[str]:
    return [r[0] for r in pg.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=%s", (table,)).fetchall()]


def main() -> int:
    sqlite_path = Path(os.environ.get("ASO_SQLITE", "data/aso.db"))
    if not sqlite_path.exists():
        print(f"no sqlite database at {sqlite_path}")
        return 1
    dsn = os.environ.get("ASO_PG_DSN")
    if not dsn:
        print("ASO_PG_DSN is not set")
        return 1

    sq = sqlite3.connect(sqlite_path)
    sq.row_factory = sqlite3.Row
    pg = psycopg.connect(dsn, autocommit=False)

    from aso import db
    pg.execute(db.SCHEMA.read_text())
    pg.commit()

    total = 0
    for table in TABLES:
        have = set(columns(pg, table))
        src = [r[1] for r in sq.execute(f"PRAGMA table_info({table})")]
        cols = [c for c in src if c in have]
        skipped = [c for c in src if c not in have]
        rows = sq.execute(f"SELECT {','.join(cols)} FROM {table}").fetchall()

        # Truncate rather than upsert: this is a one-way copy of a source of
        # truth, and leaving older rows behind would quietly merge two states.
        pg.execute(f"TRUNCATE {table} CASCADE")
        marks = ",".join(["%s"] * len(cols))
        with pg.cursor() as cur:
            for i in range(0, len(rows), BATCH):
                cur.executemany(
                    f"INSERT INTO {table} ({','.join(cols)}) VALUES ({marks})",
                    [tuple(r) for r in rows[i:i + BATCH]])
        pg.commit()

        # A table whose primary key came from a sequence needs that sequence
        # moved past the copied ids, or the next insert collides with row one.
        # Not every `id` is one: predictions keys on a text id, which has no
        # sequence and no maximum to advance past.
        if "id" in cols:
            seq = pg.execute(
                f"SELECT pg_get_serial_sequence('{table}', 'id') AS s").fetchone()[0]
            if seq:
                pg.execute(f"SELECT setval(%s, COALESCE((SELECT MAX(id) FROM {table}), 1))",
                           (seq,))
            pg.commit()

        total += len(rows)
        note = f"   (skipped {', '.join(skipped)})" if skipped else ""
        print(f"  {table:15s} {len(rows):6d} rows{note}")

    print(f"\n  {total} rows copied")
    sq.close()
    pg.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
