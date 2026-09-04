"""Your thresholds, kept out of the model.

The network predicts where an app would rank and what it would earn. Whether
those numbers are good ENOUGH is not a modelling question, it is your call: a
keyword worth building for at 2,000 downloads a year is worthless to someone who
needs 50,000. So thresholds live here, in a file you own, and can be changed
without retraining anything.

Resolution order, first hit wins:
    $ASO_CONFIG                 an explicit path
    ./aso.config.json           per project
    ~/.config/aso/config.json   per user
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULTS: dict = {
    # --- what counts as ranking -------------------------------------------
    "top_k": 10,               # a slot at or above this is "ranked"
    "country": "us",

    # --- your bar ---------------------------------------------------------
    # 0 disables a threshold. Set them to what a build actually has to earn.
    #
    # One download threshold, stated in whatever period you actually think in.
    # Holding a separate number per period invites them to disagree; this way
    # "50 a day" and "18,000 a year" cannot both be set and mean different things.
    "min_downloads": 0,
    "min_downloads_unit": "year",      # day | month | year
    "max_rank": 0,             # a landing slot worse than this is not worth it
    "min_build_score": 0,      # 0-100, the model's chance of reaching top_k
    "min_model_confidence": 0, # 0-100, ignore calls the model is unsure of

    # --- how the verdict is worded ----------------------------------------
    "build_score_strong": 55,  # at or above this, a clear yes
    "build_score_maybe": 25,   # between the two, worth a look

    # --- collection -------------------------------------------------------
    "fresh_days": 7,           # a keyword scraped this recently is not re-fetched
    "track_every_days": 7,     # `aso track` re-scrapes anything older than this
    "new_app_years": 1.5,      # released within this is an entry experiment
    "scrape_sleep": 0.7,
    # Autocomplete changes slowly, so a day-old list is almost always current.
    "suggest_ttl_hours": 24,

    # --- server -----------------------------------------------------------
    # Small on purpose: a cold keyword holds its slot for ~40s of scraping, and
    # unbounded threads means concurrent scrapes that Play rate-limits.
    "server_max_concurrent": 4,
    "server_timeout": 180,
}

PATHS = [
    os.environ.get("ASO_CONFIG"),
    "aso.config.json",
    str(Path.home() / ".config" / "aso" / "config.json"),
]

_CACHE: dict | None = None


def path() -> Path:
    """Where a write goes: the first existing file, else the per-user one."""
    for p in PATHS:
        if p and Path(p).exists():
            return Path(p)
    return Path(PATHS[-1])


def load(reload: bool = False) -> dict:
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE
    cfg = dict(DEFAULTS)
    for p in PATHS:
        if p and Path(p).exists():
            try:
                user = json.loads(Path(p).read_text())
            except (json.JSONDecodeError, OSError):
                break
            # Unknown keys are kept rather than dropped, so a config written by a
            # newer version is not silently erased by an older one.
            cfg.update(user)
            break
    _CACHE = cfg
    return cfg


def get(key: str, default=None):
    return load().get(key, DEFAULTS.get(key, default))


def save(updates: dict) -> Path:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if p.exists():
        try:
            current = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            current = {}
    current.update(updates)
    p.write_text(json.dumps(current, indent=2) + "\n")
    load(reload=True)
    return p


def coerce(key: str, raw: str):
    """Match the type of the default, so '5000' does not become a string."""
    d = DEFAULTS.get(key)
    if isinstance(d, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(d, int) and not isinstance(d, bool):
        return int(float(raw))
    if isinstance(d, float):
        return float(raw)
    return raw


def effective(overrides: dict | None = None) -> dict:
    """Config with per-call overrides applied.

    A caller - a CLI flag, an API body, an agent deciding its own bar - can
    supply thresholds for one question without writing them to disk. Anything
    not supplied falls back to the saved config, and that to the defaults, so a
    partial override is exactly a partial override.
    """
    c = dict(load())
    for k, v in (overrides or {}).items():
        if v is not None and k in DEFAULTS:
            c[k] = v
    return c


def shortfalls(build: int, confidence: int, rank: int, downloads: dict | None,
               overrides: dict | None = None) -> list[str]:
    """Which thresholds this keyword fails, in plain words.

    Reported rather than folded into the score: a keyword that ranks easily but
    earns nothing and one that earns well but cannot be won are both rejections,
    and collapsing them into a single number hides which is which.
    """
    c = effective(overrides)
    out = []
    if c["min_build_score"] and build < c["min_build_score"]:
        out.append(f"build score {build} is under your minimum of {c['min_build_score']}")
    if c["min_model_confidence"] and confidence < c["min_model_confidence"]:
        out.append(f"model certainty {confidence} is under your minimum of "
                   f"{c['min_model_confidence']}")
    if c["max_rank"] and rank > c["max_rank"]:
        out.append(f"it would land at #{rank}, past your limit of #{c['max_rank']}")
    if downloads and c["min_downloads"]:
        unit = c.get("min_downloads_unit", "year")
        got = downloads.get(f"per_{unit}", 0)
        if got < c["min_downloads"]:
            out.append(f"{got:,.0f} downloads a {unit} is under your minimum of "
                       f"{c['min_downloads']:,.0f} a {unit}")
    return out


UNITS = ("day", "month", "year")


def set_min_download(unit: str, amount: float) -> Path:
    if unit not in UNITS:
        raise SystemExit(f"unit must be one of {', '.join(UNITS)}")
    return save({"min_downloads": float(amount), "min_downloads_unit": unit})
