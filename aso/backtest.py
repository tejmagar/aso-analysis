"""Did the past predict the present?

The only honest test of a forecaster is a forward one: take what was known
BEFORE the app existed, and check it against what happened after. Everything
else measures how well the model fits data it has already seen.

Two sources of "before":

  research  what was recorded at research time in ~/app-idea, months before the
            app shipped. Genuinely prior, but thin: a handful of competitor
            titles and install buckets, no descriptions, ranks or dates, so only
            the coarsest features can be reconstructed.

  snapshot  our own observation history. Every scrape appends rows stamped with
            `observed_at` rather than overwriting, so once two scrapes of a
            keyword exist, the earlier one is a full 46-feature before-state.
            This is the one that will matter; it needs time to accumulate.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from . import db
from .ingest import RESEARCH_DB

TOP_K = 10


def _installs(v) -> int | None:
    d = re.sub(r"[^\d]", "", str(v or ""))
    return int(d) if d else None


def research_before(path: Path | None = None) -> dict[str, dict]:
    """Field state as recorded at research time, keyed by keyword."""
    p = Path(path or RESEARCH_DB)
    if not p.exists():
        return {}
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    out = {}
    for r in con.execute("SELECT keyword, date_utc, top_apps_sample, title_hits "
                         "FROM research WHERE built=1 AND keyword IS NOT NULL"):
        kw = " ".join((r["keyword"] or "").lower().split())
        try:
            samp = json.loads(r["top_apps_sample"] or "[]")
        except json.JSONDecodeError:
            samp = []
        inst = [i for i in (_installs(a.get("installs")) for a in samp) if i]
        hits = None
        if r["title_hits"] and "/" in str(r["title_hits"]):
            a_, b_ = str(r["title_hits"]).split("/")[:2]
            try:
                hits = int(a_) / max(int(b_), 1)
            except ValueError:
                hits = None
        out[kw] = {"observed_at": r["date_utc"], "n_apps": len(samp),
                   "installs_median": float(np.median(inst)) if inst else None,
                   "title_hit_rate": hits}
    con.close()
    return out


def outcomes(con, country="us") -> dict[str, int | None]:
    """Where our own app ended up, per keyword. None means it never appeared."""
    rows = con.execute("""
        SELECT o.keyword, MIN(o.position) AS pos
        FROM observations o
        WHERE o.country = ?
          AND (o.source IN ('owned', 'review')
               OR o.pkg IN (SELECT pkg FROM apps
                            WHERE developer LIKE 'Flash%' OR developer LIKE 'Monova%'))
        GROUP BY o.keyword""", (country,)).fetchall()
    return {r["keyword"]: (None if r["pos"] is None or r["pos"] >= db.ABSENT
                           else r["pos"]) for r in rows}


def snapshot_pairs(con, country="us", min_gap_days=7) -> list[dict]:
    """Keywords scraped at least twice, far enough apart to be a real before/after.

    This is the pipeline that eventually replaces the research source: the early
    scrape reconstructs every feature the model uses, not just an install median.
    """
    rows = con.execute("""
        SELECT keyword, COUNT(DISTINCT substr(observed_at, 1, 10)) AS days,
               MIN(observed_at) AS first, MAX(observed_at) AS last
        FROM observations WHERE country = ? GROUP BY keyword HAVING days > 1""",
        (country,)).fetchall()
    out = []
    for r in rows:
        gap = (np.datetime64(r["last"][:10]) - np.datetime64(r["first"][:10])
               ).astype("timedelta64[D]").astype(int)
        if gap >= min_gap_days:
            out.append({"keyword": r["keyword"], "first": r["first"],
                        "last": r["last"], "gap_days": int(gap)})
    return out


def run(con, country="us") -> dict:
    before = research_before()
    after = outcomes(con, country)
    pairs = [(kw, before[kw], after[kw]) for kw in before if kw in after]

    ranked = [(k, b) for k, b, p in pairs if p and p <= TOP_K]
    missed = [(k, b) for k, b, p in pairs if not p or p > TOP_K]

    def med(rows, key):
        v = [b[key] for _, b in rows if b.get(key) is not None]
        return float(np.median(v)) if v else None

    return {
        "pairs": len(pairs),
        "ranked": len(ranked), "missed": len(missed),
        "installs_median_when_ranked": med(ranked, "installs_median"),
        "installs_median_when_missed": med(missed, "installs_median"),
        "examples_ranked": [k for k, _ in ranked[:6]],
        "examples_missed": [k for k, _ in missed[:6]],
        "snapshot_pairs": snapshot_pairs(con, country),
    }
