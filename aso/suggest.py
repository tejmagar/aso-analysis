"""Play autocomplete, stored exactly as returned.

`fetch_suggestions("shake flashlight")` hands back the real keyword family:

    1. shake flashlight
    2. shake flashlight and camera
    3. shake flashlight free
    4. shake flashlight on off app
    5. shake flashlight fast

That is the data. Nothing here slices the phrase into invented prefixes or
splits it into guessed tokens, and nothing is collapsed into a score: the model
receives the structure of the real list and learns what it is worth.
"""
from __future__ import annotations

from . import db, scrape


def fetch(query: str, games: bool = False) -> list[str]:
    return [s.strip().lower() for s in scrape.suggest(query, games=games) if s.strip()]


def refresh(con, query: str, country: str = "us", games: bool = False) -> list[str]:
    q = " ".join(query.strip().lower().split())
    sugg = fetch(q, games=games)
    with con:
        con.execute("DELETE FROM suggestions WHERE query=? AND country=?", (q, country))
        for i, s in enumerate(sugg):
            con.execute("INSERT INTO suggestions (query, country, position, suggestion, "
                        "fetched_at) VALUES (?,?,?,?,?)", (q, country, i, s, db.now()))
    return sugg


def get(con, query: str, country: str = "us") -> list[str]:
    return [r["suggestion"] for r in con.execute(
        "SELECT suggestion FROM suggestions WHERE query=? AND country=? ORDER BY position",
        (" ".join(query.strip().lower().split()), country))]


def age_hours(con, query: str, country: str = "us") -> float | None:
    """How long since this query's suggestions were fetched. None if never."""
    r = con.execute(
        "SELECT MAX(fetched_at) AS at FROM suggestions WHERE query=? AND country=?",
        (" ".join(query.strip().lower().split()), country)).fetchone()
    if not r or not r["at"]:
        return None
    from .history import _days
    return _days(r["at"], db.now()) * 24.0


def is_fresh(con, query: str, country: str = "us", ttl_hours: float | None = None) -> bool:
    if ttl_hours is None:
        from .config import get as cfg
        ttl_hours = float(cfg("suggest_ttl_hours", 24))
    age = age_hours(con, query, country)
    return age is not None and age < ttl_hours


def ensure(con, query: str, country: str = "us",
           ttl_hours: float | None = None) -> list[str]:
    """Cached suggestions, refetched once they age past the TTL.

    Autocomplete moves, but slowly: a day-old list is almost always the same
    list, and refetching on every analyze spent a network round trip to learn
    nothing. Stale rows are overwritten rather than accumulated, since only the
    current ordering is of any use.
    """
    if is_fresh(con, query, country, ttl_hours):
        return get(con, query, country)
    try:
        return refresh(con, query, country)
    except Exception:                                # noqa: BLE001
        # Network trouble should not erase a usable answer. A stale list beats
        # no list, and the next call tries again.
        return get(con, query, country)


def signals(con, keyword: str, country: str = "us") -> dict:
    """Raw model inputs read off the real list. Every value is a count or a
    proportion of what Play returned, never a weighting of them."""
    kw = " ".join(keyword.strip().lower().split())
    sugg = get(con, kw, country)
    n = len(sugg)
    if not n:
        return dict(DEFAULTS)

    self_rank = sugg.index(kw) if kw in sugg else None
    extends = [s for s in sugg if s != kw and kw in s]
    unrelated = [s for s in sugg if kw not in s]
    extra = [len(s.split()) - len(kw.split()) for s in extends]

    return {
        # Play returning the phrase itself, first, means it is a query people
        # actually type rather than a string we made up.
        "sugg_returned": n / 10.0,
        "sugg_self_listed": 1.0 if self_rank is not None else 0.0,
        "sugg_self_rank": (self_rank / n) if self_rank is not None else 1.0,
        "sugg_is_canonical": 1.0 if self_rank == 0 else 0.0,
        # A phrase with a family of longer variants beneath it is a head term.
        "sugg_extends": len(extends) / n,
        "sugg_extra_words": (sum(extra) / len(extra)) if extra else 0.0,
        # Suggestions that do NOT contain the phrase mean Play is reinterpreting
        # the query, so the intent is not locked to these words.
        "sugg_unrelated": len(unrelated) / n,
    }


DEFAULTS = {"sugg_returned": 0.0, "sugg_self_listed": 0.0, "sugg_self_rank": 1.0,
            "sugg_is_canonical": 0.0, "sugg_extends": 0.0, "sugg_extra_words": 0.0,
            "sugg_unrelated": 0.0}


def detail(con, keyword: str, country: str = "us") -> dict:
    """What to show a person: the real list, and where their phrase sits in it."""
    kw = " ".join(keyword.strip().lower().split())
    sugg = get(con, kw, country)
    return {
        "query": kw,
        "suggestions": sugg,
        "self_rank": sugg.index(kw) if kw in sugg else None,
        "extends": [s for s in sugg if s != kw and kw in s],
        "unrelated": [s for s in sugg if kw not in s],
    }
