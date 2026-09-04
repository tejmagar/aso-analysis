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
from pathlib import Path

import numpy as np

from . import db, history

WORKSPACE = Path("~/app-idea").expanduser()


def _installs(v) -> int | None:
    d = re.sub(r"[^\d]", "", str(v or ""))
    return int(d) if d else None


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
    return out


def _walk(node, date: str, found: list) -> None:
    if isinstance(node, dict):
        kw = node.get("keyword") or node.get("seed_keyword")
        apps = _apps_in(node)
        if kw and apps:
            inst = [i for i in (_installs(a.get("installs")) for a in apps) if i]
            if inst:
                found.append({"keyword": " ".join(str(kw).lower().split()),
                              "observed_at": date, "apps": apps, "installs": inst})
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
            continue
        inst = [i for i in (_installs(a.get("installs")) for a in apps) if i]
        if not (r["keyword"] and inst):
            continue
        out.append({"keyword": " ".join(r["keyword"].lower().split()),
                    "observed_at": r["date_utc"] or "2026-06-01T00:00:00Z",
                    "apps": apps, "installs": inst})
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
            "newcomers": None,
        }, observed_at=s["observed_at"], source="research")
        written += 1
        keywords.add(s["keyword"])

    have = {r["keyword"] for r in con.execute(
        "SELECT DISTINCT keyword FROM observations WHERE country=?", (country,))}
    return {"snapshots": written, "keywords": len(keywords),
            "with_current_page": len(keywords & have),
            "needs_scraping": sorted(keywords - have)}
