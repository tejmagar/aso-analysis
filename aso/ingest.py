"""Import the research workspace: keywords we built for, and what happened.

This is the only causal data in the project. Everything scraped from a SERP is a
survivor, so absence there means nothing. Here we know we published FOR a
keyword, which makes "and it never ranked" a real, observed failure - the label
a scrape can never produce and the one the model learns most from.

Reads ~/app-idea: `research.built=1` gives the keywords, `play_store_cache`
gives our own packages, and the two are matched on title tokens.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from . import db, scrape, ui

RESEARCH_DB = Path(os.environ.get("ASO_RESEARCH_DB",
                                  Path.home() / "app-idea" / "db.sqlite3"))
MATCH = 0.6          # share of keyword tokens that must appear in our app's title


def _toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def targets(path: Path | None = None) -> list[dict]:
    """Keyword -> the app we published for it. Unmatched keywords are kept with
    pkg=None: we still built something, we just cannot name it, and the keyword
    is worth scraping either way."""
    p = Path(path or RESEARCH_DB)
    if not p.exists():
        raise SystemExit(
            f"no research database at {p}.\n"
            f"`aso ingest` reads a private keyword-research workspace and is not\n"
            f"much use without one. To import your own published apps instead:\n"
            f"  python scripts/fetch_publisher.py \"Your Developer Name\" --json apps.json\n"
            f"  aso ingest --from-publisher apps.json\n"
            f"Or point ASO_RESEARCH_DB at a sqlite file with a `research` table.")
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    built = [dict(r) for r in con.execute(
        "SELECT keyword, date_utc, is_game FROM research WHERE built=1 "
        "AND keyword IS NOT NULL AND keyword != ''")]
    apps = [dict(r) for r in con.execute(
        "SELECT package, title, developer FROM play_store_cache")]
    con.close()

    out, seen = [], set()
    for b in built:
        kw = " ".join(b["keyword"].strip().lower().split())
        if kw in seen:
            continue
        seen.add(kw)
        kt = _toks(kw)
        best, score = None, 0.0
        for a in apps:
            ov = len(kt & _toks(a["title"])) / max(len(kt), 1)
            if ov > score:
                best, score = a, ov
        out.append({"keyword": kw, "is_game": bool(b["is_game"]),
                    "researched_at": b["date_utc"] or None,
                    "pkg": best["package"] if score >= MATCH else None,
                    "title": best["title"] if score >= MATCH else None,
                    "developer": best["developer"] if score >= MATCH else None,
                    "match": round(score, 2)})
    return out


def keyword_from_title(title: str) -> str | None:
    """The phrase an app was published FOR, read off its own title.

    House ASO practice puts the target keyword at the front of the title and
    hangs qualifiers behind a dash or colon, so "Shake Flashlight - Fast Torch"
    was built for "shake flashlight". Take the head, drop the marketing tail.
    """
    if not title:
        return None
    head = re.split(r"[-:|–—(]", title, maxsplit=1)[0]
    head = re.sub(r"[^A-Za-z0-9 ]+", " ", head).lower()
    words = [w for w in head.split() if w not in {"app", "free", "pro", "the"}]
    if not words or len(words) > 5:
        # A one-word or essay-length title is not a usable keyword: the first is
        # too broad to attribute a rank to, the second is not a query anyone types.
        return None if not words else " ".join(words[:4])
    return " ".join(words)


def publisher_targets(path: str | Path) -> list[dict]:
    """Every published app as a (keyword, our package) pair.

    Unlike the research table these are not guesses about what we built: the app
    exists, we shipped it, and its title states the phrase it was aimed at. Where
    it does not rank for its own title, that is the strongest negative available.
    """
    import json
    apps = json.loads(Path(path).read_text())
    out, seen = [], set()
    for a in apps:
        kw = keyword_from_title(a.get("title") or "")
        if not kw or kw in seen or not a.get("package"):
            continue
        seen.add(kw)
        out.append({"keyword": kw, "pkg": a["package"], "title": a.get("title"),
                    "developer": a.get("developer"), "match": 1.0,
                    "is_game": False, "researched_at": None})
    return out


def pending(con, todo: list[dict], country: str = "us",
            fresh_days: int = 7) -> list[dict]:
    """Drop keywords already scraped recently.

    This is what makes a long ingest interruptible: stop it whenever, re-run it
    whenever, and it picks up where it left off instead of re-fetching thousands
    of app pages it already has. `fresh_days` is the line between "already have
    this" and "old enough to be worth re-observing".
    """
    seen = {r["keyword"]: r["last"] for r in con.execute(
        "SELECT keyword, MAX(observed_at) AS last FROM observations "
        "WHERE country=%s GROUP BY keyword", (country,))}
    from .history import _days
    now = db.now()
    return [t for t in todo
            if t["keyword"] not in seen
            or _days(seen[t["keyword"]], now) >= fresh_days]


def run(con, limit: int | None = None, sleep: float = 0.8,
        with_details: bool = True, verbose: bool = True,
        source: str | Path | None = None, country: str = "us",
        resume: bool = True, fresh_days: int = 7) -> dict:
    """Scrape each keyword we built for and record where our own app landed."""
    todo = publisher_targets(source) if source else targets()
    total = len(todo)
    if resume:
        todo = pending(con, todo, country, fresh_days)
        if verbose and len(todo) < total:
            ui.say(f"{total - len(todo)} of {total} keywords already scraped "
                   f"within {fresh_days} days, {len(todo)} to go")
    if limit:
        todo = todo[:limit]
    stats = {"keywords": 0, "ranked": 0, "ours_ranked": 0, "ours_absent": 0,
             "unmatched": 0, "failed": 0}

    for i, t in enumerate(todo, 1):
        kw = t["keyword"]
        if verbose:
            ui.say(f"[{i}/{len(todo)}] {kw!r}"
                   + (f"  ours: {t['pkg']}" if t["pkg"] else "  (no app matched)"))
        try:
            n = scrape.scrape_keyword(con, kw, with_details=with_details,
                                      sleep=sleep, verbose=verbose)
        except Exception as e:                              # noqa: BLE001
            stats["failed"] += 1
            ui.say(f"  ! {type(e).__name__}, skipping")
            continue
        stats["keywords"] += 1
        stats["ranked"] += n

        if not t["pkg"]:
            stats["unmatched"] += 1
            continue

        found = con.execute(
            "SELECT position FROM observations WHERE keyword=%s AND pkg=%s "
            "AND position IS NOT NULL ORDER BY observed_at DESC LIMIT 1",
            (kw, t["pkg"])).fetchone()
        if found:
            stats["ours_ranked"] += 1
            if verbose:
                ui.say(f"  ours ranks #{found['position']}")
        else:
            # We published for this keyword and it is not on the page. That is a
            # real negative, and scraped SERPs structurally cannot produce one.
            #
            # Fetch its REAL listing rather than storing a zeroed stub. The whole
            # value of a negative is being able to ask why it failed, and a row
            # with no title, description or install count cannot answer that -
            # worse, the stub overwrote real metadata if the app was already known.
            detail = None
            try:
                detail = g.fetch_app_details(t["pkg"])
            except Exception:                            # noqa: BLE001
                pass
            if detail:
                db.upsert_app(con, to_app_row(detail, country, lang))
            else:
                existing = con.execute("SELECT pkg FROM apps WHERE pkg=%s",
                                       (t["pkg"],)).fetchone()
                if not existing:
                    db.upsert_app(con, {"pkg": t["pkg"], "title": t["title"] or "",
                                        "developer": t["developer"] or "", "installs": 0,
                                        "rating": 0.0, "reviews": 0, "country": "us",
                                        "lang": "en", "scraped_at": None,
                                        "short_desc": "", "description": "",
                                        "category": "", "released_at": None,
                                        "updated_at": None, "raw_json": None})
            db.add_observation(con, kw, t["pkg"], db.ABSENT, source="owned")
            con.commit()
            stats["ours_absent"] += 1
            if verbose:
                ui.say("  ours does NOT rank (recorded as a real negative)")
    return stats
