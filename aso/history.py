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
    "hist_dated": 0.0, "hist_renewal": 0.0, "hist_velocity_delta": 0.0,
}

# Below this share of a past page carrying a release date, its median age is a
# median over a handful of apps rather than the field's, so the renewal features
# report unknown instead of a number the model would read as measured.
DATED_ENOUGH = 0.5

# A pair closer together than this is not a trend. Without the floor a same-day
# research row divides by the 0.08-year guard and reports a 30x rate from what is
# really two readings of one moment.
MIN_ELAPSED_DAYS = 60


def record(con, keyword: str, country: str, field: dict,
           observed_at: str | None = None, source: str = "scrape") -> None:
    """Append the current field state. Never overwrites: the series is the point."""
    with con.transaction():
        con.execute(
            "INSERT INTO field_history (keyword, country, observed_at, "
            "source, n_apps, installs_p50, installs_p90, rating_p50, exact_match, "
            "newcomers, age_p50, velocity_p50, age_known_frac) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (keyword, country, observed_at) DO UPDATE SET "
            "source=excluded.source, n_apps=excluded.n_apps, "
            "installs_p50=excluded.installs_p50, installs_p90=excluded.installs_p90, "
            "rating_p50=excluded.rating_p50, exact_match=excluded.exact_match, "
            "newcomers=excluded.newcomers, age_p50=excluded.age_p50, "
            "velocity_p50=excluded.velocity_p50, "
            "age_known_frac=excluded.age_known_frac",
            (keyword.strip().lower(), country, observed_at or db.now(), source,
             field.get("n"), field.get("installs_p50"), field.get("installs_p90"),
             field.get("rating_p50"), field.get("exact_match_count"),
             field.get("newcomers"), field.get("age_p50"),
             field.get("velocity_p50"), field.get("age_known_frac")))


def series(con, keyword: str, country: str = "us") -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM field_history WHERE keyword=%s AND country=%s "
        "ORDER BY observed_at", (keyword.strip().lower(), country))]


def deltas(con, keyword: str, country: str, current: dict,
           rows_today: list[dict] | None = None) -> dict:
    """How the field has moved since the earliest state we hold.

    Compared against the OLDEST row rather than the previous one, so the signal
    is the whole arc rather than the last wobble. All left unconstrained in the
    registry: a hardening field is worse for a newcomer but often means a market
    worth entering, and asserting either direction would be a guess.

    `rows_today` is today's raw page. It is needed because the earlier
    states are not all full pages: a research row holds the three to eight apps
    that session happened to write down, and their median install count runs
    about 2x a full page's simply because the weak tail was never recorded.
    Differencing that against today's top ten reports a collapse on every
    keyword that has any history. So when the earlier state is smaller, today is
    cut to the same depth before subtracting. It is not a perfect match - we
    cannot know the earlier sample was exactly slots 1..n - but comparing eight
    against eight is a different order of error from eight against thirty.
    """
    rows = [r for r in series(con, keyword, country)
            if r["installs_p50"] is not None]
    if not rows:
        return dict(DELTA_DEFAULTS)
    first = rows[0]
    age = _days(first["observed_at"], db.now())
    if age < MIN_ELAPSED_DAYS:
        return dict(DELTA_DEFAULTS)
    current = _matched(rows_today, first["n_apps"], current)

    def lg(x):
        return math.log10((x or 0) + 1)

    inst_d = lg(current.get("installs_p50")) - lg(first["installs_p50"])
    size_d = ((current.get("n") or 0) - (first["n_apps"] or 0)) / max(first["n_apps"] or 1, 1)
    match_d = ((current.get("exact_match_count") or 0)
               - (first["exact_match"] or 0)) / max(first["exact_match"] or 1, 1)
    out = {
        "hist_known": 1.0,
        "hist_age_days": min(age / 365.0, 3.0),
        "hist_installs_delta": inst_d,
        "hist_size_delta": size_d,
        "hist_match_delta": match_d,
        # Per-year rate, so a change over three months and the same change over
        # two years are not read as the same event.
        "hist_hardening": inst_d / max(age / 365.0, 0.08),
        "hist_dated": 0.0, "hist_renewal": 0.0, "hist_velocity_delta": 0.0,
    }
    out.update(_renewal(first, current, age / 365.0))
    return out


def _matched(rows: list[dict] | None, n: int | None, current: dict) -> dict:
    """Today's field re-measured over its top `n` slots, to match an older
    shallower snapshot. Returns `current` untouched when there is nothing to
    match or today is already the shallower of the two."""
    if not rows or not n or n >= (current.get("n") or 0):
        return current
    from .features import compute_field
    cut = compute_field(rows, "", top_n=n)
    # Relevance and intent stats in `cut` are meaningless without a keyword
    # vector, so only the count-and-distribution fields are taken from it.
    return {**current, **{k: cut[k] for k in
                          ("n", "installs_p50", "installs_p90", "age_p50",
                           "velocity_p50", "age_known_frac", "newcomers")}}


def _renewal(first: dict, current: dict, years: float) -> dict:
    """Whether the field took on new apps, or just got older in place.

    A page with nobody new is a page nobody is winning. The test is arithmetic:
    if every app from the earlier snapshot were still there, the median age would
    have risen by exactly the elapsed time. Rising by less means younger apps
    displaced older ones, and rising by more is impossible without churn the
    other way. So the shortfall - elapsed minus observed ageing - is churn
    measured directly, in years of age the field shed.
    """
    old_age, new_age = first.get("age_p50"), current.get("age_p50")
    if (old_age is None or new_age is None
            or (first.get("age_known_frac") or 0) < DATED_ENOUGH
            or (current.get("age_known_frac") or 0) < DATED_ENOUGH):
        return {}
    out = {"hist_dated": 1.0,
           "hist_renewal": (years - (new_age - old_age)) / max(years, 0.08)}
    old_v, new_v = first.get("velocity_p50"), current.get("velocity_p50")
    if old_v and new_v:
        out["hist_velocity_delta"] = math.log10(new_v + 1) - math.log10(old_v + 1)
    return out


def backfill_ages(con, country: str = "us") -> int:
    """Fill age columns on scrape rows written before those columns existed.

    Without this every past scrape is undated, so the first dated pair would be
    two future scrapes away for every keyword. The observations are still on
    disk with their release dates, so the state is recoverable rather than lost.
    """
    from datetime import datetime

    import numpy as np

    from .features import _days_since

    rows = con.execute(
        "SELECT keyword, observed_at FROM field_history WHERE source='scrape' "
        "AND country=? AND age_p50 IS NULL", (country,)).fetchall()
    n = 0
    for kw, at in rows:
        try:
            when = datetime.fromisoformat(str(at).replace("Z", "+00:00")).date()
        except (ValueError, TypeError):
            continue
        apps = con.execute(
            "SELECT a.released_at, a.installs FROM observations o JOIN apps a "
            "USING (pkg) WHERE o.keyword=%s AND o.country=%s AND o.observed_at=%s",
            (kw, country, at)).fetchall()
        if not apps:
            continue
        ages, vel = [], []
        for rel, inst in apps:
            days = _days_since(rel, when)
            if not rel or days <= 0:
                continue
            ages.append(days / 365.0)
            if inst:
                vel.append(inst / max(days / 365.0, 0.25))
        if not ages:
            continue
        with con.transaction():
            con.execute(
                "UPDATE field_history SET age_p50=%s, velocity_p50=%s, newcomers=%s, "
                "age_known_frac=%s WHERE keyword=%s AND country=%s AND observed_at=%s",
                (float(np.percentile(ages, 50)),
                 float(np.percentile(vel, 50)) if vel else None,
                 sum(1 for a in ages if a <= 1.0), len(ages) / len(apps),
                 kw, country, at))
        n += 1
    return n


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
