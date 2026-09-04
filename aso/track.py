"""Watching keywords over time, and learning from what changes.

The insight this implements: we do not need to publish an app to observe one
entering. Re-scrape a keyword a week or a month apart and other people's new
apps appear in the gap, each one a natural experiment we did not have to run. A
rank is a rank, so an entrant is worth the same whoever shipped it, and the
supply is limited only by how many keywords we watch rather than by how many
apps we build.

What a snapshot pair yields, per keyword:

  entered   absent before, present now. Its age at entry, its metadata and the
            field it walked into are all known, which is exactly the question
            `score_entry` asks - answered here by something that actually happened.
  left      present before, gone now. The negative half, and unobservable from a
            single scrape.
  moved     same app, different slot. What a rank costs and pays over time.

Entrants are the valuable rows. Everything else is context that explains them.
"""
from __future__ import annotations

from . import db

NEW_APP_YEARS = 1.5      # released this recently: a genuine newcomer, not a re-entry


def snapshots(con, keyword: str, country: str = "us") -> list[str]:
    """Distinct scrape times for a keyword, oldest first."""
    return [r["observed_at"] for r in con.execute(
        "SELECT DISTINCT observed_at FROM observations "
        "WHERE keyword=%s AND country=%s ORDER BY observed_at",
        (keyword.strip().lower(), country))]


def at(con, keyword: str, observed_at: str, country: str = "us") -> dict[str, dict]:
    """The field as it stood at one moment, keyed by package."""
    rows = con.execute(
        "SELECT o.pkg, o.position, o.featured, a.* FROM observations o "
        "JOIN apps a USING (pkg) WHERE o.keyword=%s AND o.country=%s "
        "AND o.observed_at=? AND o.position IS NOT NULL",
        (keyword.strip().lower(), country, observed_at)).fetchall()
    return {r["pkg"]: dict(r) for r in rows}


def transitions(con, keyword: str, country: str = "us",
                min_gap_days: int = 3) -> list[dict]:
    """Every entry, exit and move between consecutive snapshots of a keyword.

    Snapshots closer together than `min_gap_days` are skipped: a re-scrape
    minutes later is the same observation twice, and treating its noise as
    movement would teach the model that ranks churn when they did not.
    """
    snaps = snapshots(con, keyword, country)
    out = []
    for older, newer in zip(snaps, snaps[1:]):
        gap = _days_between(older, newer)
        if gap < min_gap_days:
            continue
        a, b = at(con, keyword, older, country), at(con, keyword, newer, country)
        for pkg in b.keys() - a.keys():
            row = b[pkg]
            out.append({"keyword": keyword, "pkg": pkg, "event": "entered",
                        "from": None, "to": row["position"], "gap_days": gap,
                        "observed_at": newer, "title": row.get("title"),
                        "installs": row.get("installs"),
                        "released_at": row.get("released_at"),
                        # A brand-new app entering is the clean experiment; an
                        # older app appearing may just be re-ranking, so the two
                        # are labelled rather than lumped together.
                        "is_new_app": _is_new(row, newer)})
        for pkg in a.keys() - b.keys():
            out.append({"keyword": keyword, "pkg": pkg, "event": "left",
                        "from": a[pkg]["position"], "to": None, "gap_days": gap,
                        "observed_at": newer, "title": a[pkg].get("title")})
        for pkg in a.keys() & b.keys():
            if a[pkg]["position"] != b[pkg]["position"]:
                out.append({"keyword": keyword, "pkg": pkg, "event": "moved",
                            "from": a[pkg]["position"], "to": b[pkg]["position"],
                            "gap_days": gap, "observed_at": newer,
                            "title": b[pkg].get("title")})
    return out


def all_transitions(con, country: str = "us", min_gap_days: int = 3) -> list[dict]:
    kws = [r["keyword"] for r in con.execute(
        "SELECT DISTINCT keyword FROM observations WHERE country=%s", (country,))]
    out = []
    for kw in kws:
        out += transitions(con, kw, country, min_gap_days)
    return out


def summary(con, country: str = "us", min_gap_days: int = 3) -> dict:
    ts = all_transitions(con, country, min_gap_days)
    entered = [t for t in ts if t["event"] == "entered"]
    newcomers = [t for t in entered if t.get("is_new_app")]
    top10 = [t for t in newcomers if t["to"] and t["to"] <= 10]
    return {
        "transitions": len(ts),
        "entered": len(entered), "left": len([t for t in ts if t["event"] == "left"]),
        "moved": len([t for t in ts if t["event"] == "moved"]),
        "new_app_entries": len(newcomers),
        "new_app_reached_top10": len(top10),
        "examples": newcomers[:6],
    }


def due(con, country: str = "us", every_days: int = 7) -> list[str]:
    """Keywords whose last scrape is older than the tracking interval."""
    rows = con.execute(
        "SELECT keyword, MAX(observed_at) last FROM observations "
        "WHERE country=%s GROUP BY keyword", (country,)).fetchall()
    now = db.now()
    return [r["keyword"] for r in rows
            if _days_between(r["last"], now) >= every_days]


def _days_between(a: str, b: str) -> int:
    from .history import _days
    return abs(int(_days(a, b)))


def _is_new(row: dict, seen_at: str) -> bool:
    from datetime import datetime
    rel = row.get("released_at")
    if not rel:
        return False
    try:
        age = (datetime.fromisoformat(seen_at).date()
               - datetime.fromisoformat(rel).date()).days / 365.0
    except ValueError:
        return False
    return 0 <= age <= NEW_APP_YEARS
