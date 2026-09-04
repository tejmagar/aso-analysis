"""Import historical result pages from the research workspace.

The point is the pair: what a keyword's page looked like months ago, and what it
looks like today. One snapshot says how hard a keyword is; two say which way it
is moving, and that is what `hist_installs_delta` and `hist_hardening` are built
from.

Two sources, both dated:
  db.sqlite3   the `research` table's top_apps_sample column
  *.json       session files, whose schema changed over time - so this walks the
               structure looking for a keyword next to a list of apps with
               installs, rather than assuming any one layout
"""
from __future__ import annotations

import glob
import json
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np

from . import db, history, scrape

WORKSPACE = Path("~/app-idea").expanduser()


_SUFFIX = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def _installs(v) -> int | None:
    """A count from any of the shapes the sessions wrote: 100000, '100,000+',
    or the shorthand '5M' / '500k' used in the free-text samples."""
    raw = str(v or "").strip()
    m = re.fullmatch(r"([\d.,]+)\s*([kKmMbB])\+?", raw)
    if m:
        try:
            return int(float(m.group(1).replace(",", "")) * _SUFFIX[m.group(2).lower()])
        except ValueError:
            return None
    d = re.sub(r"[^\d]", "", raw)
    return int(d) if d else None


def _installs_of(app: dict) -> int | None:
    """Prefer the measured count over the displayed bucket. Reading only
    `installs` dropped sixty apps whose session recorded `installs_real` and
    nothing else, taking their release dates and their page position with them.
    """
    for key in ("installs_real", "installs", "installs_display"):
        got = _installs(app.get(key))
        if got:
            return got
    return None


def _text_sample(raw: str) -> list[dict]:
    """Parse a hand-typed sample: 'Plant Nanny 5M, Planta 1M, Gardenize 100k'.

    Two sessions recorded the page as prose rather than JSON. The order is still
    the page order and the counts are still counts, so the page is recoverable;
    refusing it only because of its serialisation would throw away real history.
    """
    out = []
    for part in str(raw or "").split(","):
        m = re.fullmatch(r"\s*(.+?)\s+([\d.,]+\s*[kKmMbB]\+?)\s*", part)
        if not m:
            continue
        out.append({"title": m.group(1).strip(), "installs": m.group(2).strip()})
    return out


def _released(v) -> date | None:
    """A historical release date, in whatever shape that session happened to save.

    Three shapes appear across the sessions: the store's own 'May 29, 2013', a
    plain ISO date, and a bare year. The bare year is kept rather than dropped -
    'released in 2017' is still worth years of age against a 2026 field, and
    resolving it to mid-year costs at most six months on a multi-year number.
    """
    raw = str(v or "").strip()
    if not raw:
        return None
    iso = scrape.parse_date(raw)
    if iso:
        return date.fromisoformat(iso)
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None
    m = re.fullmatch(r"(19|20)\d{2}", raw)
    if m:
        return date(int(raw), 7, 1)
    return None


def _at(observed_at: str) -> date:
    try:
        return datetime.fromisoformat(str(observed_at).replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return date.today()


def _age_stats(apps: list[dict], observed_at: str) -> dict:
    """Age, churn and growth of a field AS IT STOOD, not as it reads today.

    Measuring against the snapshot's own date is the whole point: every app in a
    June page is three months older now, so dating them from today would report
    a field that had aged without anything happening in it.
    """
    when = _at(observed_at)
    ages, vel = [], []
    for a in apps:
        rel = _released(a.get("released") or a.get("released_at"))
        if not rel or rel > when:
            continue
        years = max((when - rel).days / 365.0, 0.0)
        ages.append(years)
        inst = _installs_of(a)
        if inst:
            vel.append(inst / max(years, 0.25))
    if not ages:
        return {"age_p50": None, "velocity_p50": None,
                "newcomers": None, "age_known_frac": 0.0}
    return {
        "age_p50": float(np.percentile(ages, 50)),
        "velocity_p50": float(np.percentile(vel, 50)) if vel else None,
        "newcomers": sum(1 for a in ages if a <= 1.0),
        # How much of the page we could actually date. A median over two of
        # thirty apps is not the field's age, so the reader is told the weight.
        "age_known_frac": len(ages) / max(len(apps), 1),
    }


def _apps_in(node: dict) -> list[dict]:
    """Every app-shaped list hanging off this node, whatever it is called."""
    out = []
    for key in ("top_apps_sample", "semantic_matches", "top_10_semantic_match",
                "ranked_app", "apps"):
        v = node.get(key)
        if isinstance(v, list):
            out += [x for x in v if isinstance(x, dict)]
    mp = node.get("market_proof")
    if isinstance(mp, dict):
        out += [x for x in (mp.get("semantic_matches") or []) if isinstance(x, dict)]
    return _dedupe(out)


def _grouped_in(node: dict) -> list[dict]:
    """App lists a session split by a verdict rather than by position.

    `qualifying_apps` and `non_qualifying` hold the same rich per-app record as
    a sample - title, package, real installs, release date - but their order is
    the grouping, not the page. They are worth importing for what the apps say
    about the field and worthless as a ranking, so they come back separately and
    are stored with the ranking explicitly denied.
    """
    out = []
    for key in ("qualifying_apps", "non_qualifying"):
        v = node.get(key)
        if isinstance(v, list):
            out += [x for x in v if isinstance(x, dict)]
    return _dedupe(out)


def _dedupe(apps: list[dict]) -> list[dict]:
    """One entry per app, in the order first seen.

    A session often saved the same page under two keys - the full sample and the
    subset that matched semantically - and concatenating them stacked the page on
    top of itself. Left alone that turns a ten-app page into a twenty-app one
    whose second half repeats the first, which doubles the apparent field size
    and makes list position meaningless past the join.
    """
    seen, out = set(), []
    for a in apps:
        key = (a.get("package") or a.get("appId")
               or str(a.get("title") or a.get("app") or a.get("name") or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _walk(node, date: str, found: list) -> None:
    if isinstance(node, dict):
        kw = node.get("keyword") or node.get("seed_keyword")
        # A positional list wins over a grouped one: both describe the same
        # page, but only the first records where each app sat on it.
        apps, ordered = _apps_in(node), True
        if not apps:
            apps, ordered = _grouped_in(node), False
        if kw and apps:
            inst = [i for i in (_installs_of(a) for a in apps) if i]
            if inst:
                found.append({"keyword": " ".join(str(kw).lower().split()),
                              "observed_at": date, "apps": apps,
                              "installs": inst, "ordered": ordered})
        for v in node.values():
            _walk(v, date, found)
    elif isinstance(node, list):
        for v in node:
            _walk(v, date, found)


def from_json(workspace: Path | None = None) -> list[dict]:
    ws = Path(workspace or WORKSPACE)
    found: list[dict] = []
    for f in sorted(glob.glob(str(ws / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        date = None
        if isinstance(d, dict):
            date = d.get("research_date_utc") or d.get("date_utc")
        # Fall back to the timestamp in the filename: the session's own date is
        # the only thing that makes a snapshot a BEFORE rather than a duplicate.
        date = date or (Path(f).stem[:8] + "T00:00:00Z")
        _walk(d, date, found)
    return found


def from_db(workspace: Path | None = None) -> list[dict]:
    path = Path(workspace or WORKSPACE) / "db.sqlite3"
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    out = []
    for r in con.execute("SELECT keyword, date_utc, top_apps_sample FROM research "
                         "WHERE top_apps_sample NOT IN ('', '[]') "
                         "AND top_apps_sample IS NOT NULL"):
        try:
            apps = json.loads(r["top_apps_sample"])
        except json.JSONDecodeError:
            apps = _text_sample(r["top_apps_sample"])
        apps = [a for a in apps if isinstance(a, dict)]
        inst = [i for i in (_installs_of(a) for a in apps) if i]
        if not (r["keyword"] and inst):
            continue
        out.append({"keyword": " ".join(r["keyword"].lower().split()),
                    "observed_at": r["date_utc"] or "2026-06-01T00:00:00Z",
                    "apps": apps, "installs": inst, "ordered": True})
    con.close()
    return out


def load(con, country: str = "us", workspace: Path | None = None) -> dict:
    """Write every historical page into field_history, oldest kept.

    Snapshots are keyed by (keyword, country, observed_at), so re-running is
    idempotent and a keyword researched twice keeps both dates.
    """
    snaps = from_db(workspace) + from_json(workspace)
    written, keywords = 0, set()
    for s in snaps:
        inst = np.array(s["installs"], dtype="float64")
        history.record(con, s["keyword"], country, {
            "n": len(s["apps"]),
            "installs_p50": float(np.percentile(inst, 50)),
            "installs_p90": float(np.percentile(inst, 90)),
            "rating_p50": None,
            "exact_match_count": None,
            **_age_stats(s["apps"], s["observed_at"]),
        }, observed_at=s["observed_at"], source="research")
        written += 1
        keywords.add(s["keyword"])

    have = {r["keyword"] for r in con.execute(
        "SELECT DISTINCT keyword FROM observations WHERE country=%s", (country,))}
    dated = db.scalar(
        con, "SELECT COUNT(*) FROM field_history WHERE source='research' "
             "AND age_p50 IS NOT NULL")
    return {"snapshots": written, "keywords": len(keywords),
            "dated_snapshots": dated,
            "with_current_page": len(keywords & have),
            "needs_scraping": sorted(keywords - have)}
