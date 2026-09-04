"""Promote historical pages into training observations.

The archive holds pages the research sessions wrote down; the model trains from
`observations`. This is the bridge, and it is deliberately conservative about
which pages cross it.

A page is only promoted when EVERY app on it is already in the app table. That
sounds strict until you look at what a partial promotion does: the apps a
research session named that we have never scraped are the obscure ones - small,
renamed, delisted - so dropping them and keeping the rest leaves a page whose
survivors are its biggest apps. The field stats computed from that describe a
harder page than the one that existed, which is the same install-inflation that
made `hist_installs_delta` read -0.33 before it was depth-matched. A page missing
its weak half is not a smaller sample of the page, it is a biased one.
"""
from __future__ import annotations

import re

from . import db, oldpages

SOURCE = "research"


def norm(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _known(con) -> dict[str, str]:
    """Normalised title -> package, for everything already in the app table."""
    out: dict[str, str] = {}
    for r in con.execute("SELECT pkg, title FROM apps"):
        out.setdefault(norm(r["title"]), r["pkg"])
    return out


def survey(con) -> dict:
    """What could be promoted, and what is holding the rest back."""
    known = _known(con)
    full, partial, slots, missing = 0, 0, 0, set()
    for page in _pages():
        titles = [t for t in (_title(a) for a in page["apps"]) if t]
        if not titles:
            continue
        absent = [t for t in titles if norm(t) not in known]
        if absent:
            partial += 1
            missing.update(absent)
        else:
            full += 1
            slots += len(titles)
    return {"pages_ready": full, "observations_ready": slots,
            "pages_blocked": partial, "apps_missing": len(missing)}


def promote(con, country: str = "us") -> dict:
    """Write one observation per app per fully-covered page.

    Idempotent: observations are unique on (keyword, country, pkg, observed_at),
    so re-running an already-promoted page changes nothing.
    """
    known = _known(con)
    pages = kept = rows = 0
    for page in _pages():
        pages += 1
        titles = [_title(a) for a in page["apps"]]
        if not all(t and norm(t) in known for t in titles):
            continue
        at = _iso(page["observed_at"])
        # One transaction per page, so an interrupted run leaves whole pages
        # rather than a page with half its ranks.
        with con.transaction():
            for rank, title in enumerate(titles, start=1):
                con.execute(
                    "INSERT INTO observations "
                    "(keyword, country, pkg, position, source, featured, observed_at) "
                    "VALUES (%s,%s,%s,%s,%s,0,%s) "
                    "ON CONFLICT (keyword, country, pkg, observed_at) DO NOTHING",
                    (page["keyword"], country, known[norm(title)], rank, SOURCE, at))
                rows += 1
        kept += 1
    return {"pages_seen": pages, "pages_promoted": kept, "observations": rows}


def _pages() -> list[dict]:
    """Positional historical pages, one per keyword and date, widest copy kept.

    Order-ambiguous pages are excluded upstream by `ordered`: a list that runs
    strictly descending by installs may be a ranking or may have been sorted
    before it was saved, and training on a sorted list would teach the model that
    installs determine rank, which is the thing it is supposed to be predicting.
    """
    best: dict[tuple, dict] = {}
    for s in oldpages.from_db() + oldpages.from_json():
        if not s.get("ordered"):
            continue
        key = (s["keyword"], str(s["observed_at"])[:10])
        if key not in best or len(s["apps"]) > len(best[key]["apps"]):
            best[key] = s
    return list(best.values())


def _title(a: dict) -> str | None:
    return a.get("title") or a.get("app") or a.get("name")


def _iso(v) -> str:
    from datetime import datetime, timezone
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return db.now()
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.isoformat()
