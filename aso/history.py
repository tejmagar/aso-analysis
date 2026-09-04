"""Field state over time, and the change between then and now.

A single scrape says how hard a keyword is today. Two scrapes say which way it
is going, and that is a different and better question: an entrenched field whose
bar has been flat for a year is a worse bet than an identical one that has
doubled in three months, because the second is being actively contested.

Rows come from two places. `research` backfills what was recorded in ~/app-idea
months ago, which is thin but genuinely prior. `scrape` is written on every
scrape from now on, and is complete. Both feed the same delta features, so the
model improves as history accumulates without any change to its inputs.
"""
from __future__ import annotations

import math

from . import db

DELTA_DEFAULTS = {
    "hist_known": 0.0, "hist_age_days": 0.0,
    "hist_installs_delta": 0.0, "hist_size_delta": 0.0,
    "hist_match_delta": 0.0, "hist_hardening": 0.0,
}


def record(con, keyword: str, country: str, field: dict,
           observed_at: str | None = None, source: str = "scrape") -> None:
    """Append the current field state. Never overwrites: the series is the point."""
    with con:
        con.execute(
            "INSERT OR REPLACE INTO field_history (keyword, country, observed_at, "
            "source, n_apps, installs_p50, installs_p90, rating_p50, exact_match, "
            "newcomers) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (keyword.strip().lower(), country, observed_at or db.now(), source,
             field.get("n"), field.get("installs_p50"), field.get("installs_p90"),
             field.get("rating_p50"), field.get("exact_match_count"),
             field.get("newcomers")))


def series(con, keyword: str, country: str = "us") -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM field_history WHERE keyword=? AND country=? "
        "ORDER BY observed_at", (keyword.strip().lower(), country))]


def deltas(con, keyword: str, country: str, current: dict) -> dict:
    """How the field has moved since the earliest state we hold.

    Compared against the OLDEST row rather than the previous one, so the signal
    is the whole arc rather than the last wobble. All left unconstrained in the
    registry: a hardening field is worse for a newcomer but often means a market
    worth entering, and asserting either direction would be a guess.
    """
    rows = [r for r in series(con, keyword, country)
            if r["installs_p50"] is not None]
    if not rows:
        return dict(DELTA_DEFAULTS)
    first = rows[0]
    age = _days(first["observed_at"], db.now())
    if age <= 0:
        return dict(DELTA_DEFAULTS)

    def lg(x):
        return math.log10((x or 0) + 1)

    inst_d = lg(current.get("installs_p50")) - lg(first["installs_p50"])
    size_d = ((current.get("n") or 0) - (first["n_apps"] or 0)) / max(first["n_apps"] or 1, 1)
    match_d = ((current.get("exact_match_count") or 0)
               - (first["exact_match"] or 0)) / max(first["exact_match"] or 1, 1)
    return {
        "hist_known": 1.0,
        "hist_age_days": min(age / 365.0, 3.0),
        "hist_installs_delta": inst_d,
        "hist_size_delta": size_d,
        "hist_match_delta": match_d,
        # Per-year rate, so a change over three months and the same change over
        # two years are not read as the same event.
        "hist_hardening": inst_d / max(age / 365.0, 0.08),
    }


def backfill_research(con, country: str = "us") -> int:
    """Seed history from the research notes: the only pre-existing before-state."""
    from .backtest import research_before
    n = 0
    for kw, b in research_before().items():
        if not b.get("observed_at") or b.get("installs_median") is None:
            continue
        record(con, kw, country,
               {"n": b.get("n_apps"), "installs_p50": b.get("installs_median"),
                "installs_p90": None, "rating_p50": None,
                "exact_match_count": None, "newcomers": None},
               observed_at=b["observed_at"], source="research")
        n += 1
    return n


def _days(a: str, b: str) -> float:
    """Timestamps arrive in two shapes: research rows end in 'Z', our own scrapes
    carry an explicit offset. Comparing one of each raises, so both are reduced
    to naive UTC before subtracting."""
    from datetime import datetime

    def parse(x):
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        return d.replace(tzinfo=None) if d.tzinfo is None else d.astimezone(
            __import__("datetime").timezone.utc).replace(tzinfo=None)

    try:
        return (parse(b) - parse(a)).days
    except (ValueError, TypeError):
        return 0.0
