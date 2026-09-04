"""`aso analyze <keyword>` - one command, everything about a keyword.

Scrapes if the keyword is new, refreshes demand if it is stale, and reports a
predicted rank when you name an app. Degrades rather than failing: with no model
trained yet, crowding falls back to a documented field heuristic.
"""
from __future__ import annotations

import math

from . import db, features, predict, train
from .dataset import TOP_K

class NoResults(Exception):
    """Play returned nothing for this keyword. Not a crash: usually a typo, a
    keyword nobody has built for, or a storefront where it does not exist."""

    def __init__(self, keyword):
        self.keyword = keyword
        super().__init__(keyword)


DEMAND_BANDS = ((70, "more demand"), (40, "moderate demand"), (0, "less demand"))
CROWD_BANDS = ((80, "too crowded"), (60, "crowded"),
               (35, "moderately crowded"), (0, "less crowded"))


def band(score, bands):
    return None if score is None else next(l for cut, l in bands if score >= cut)


def fmt_n(n) -> str:
    if not n:
        return "0"
    n = float(n)
    for cut, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= cut:
            return f"{n / cut:.1f}".rstrip("0").rstrip(".") + suf
    return str(int(n))



def analyze(con, keyword, country="us", pkg=None, refresh_demand=True, verbose=True,
            learn=True, progress=None):
    """`progress(line)` reports each stage as it completes, for a caller that is
    watching rather than waiting."""
    say = progress or (lambda _line: None)
    keyword = keyword.strip().lower()
    rows = predict.field_rows(con, keyword, country)

    if not rows:
        from . import scrape, ui
        ui.say(f"{keyword!r} is new, scraping it now", quiet=not verbose)
        say(f"{keyword!r} has not been looked at before, fetching its page")
        scrape.scrape_keyword(con, keyword, country=country, verbose=verbose,
                              progress=progress)
        rows = predict.field_rows(con, keyword, country)
    else:
        say(f"Reading the stored page, {len(rows)} apps")
    if not rows:
        raise NoResults(keyword)

    featured = db.featured_apps(con, keyword, country)
    from . import embed
    say("Reading what each listing is about")
    kv = embed.keyword_vec(keyword)
    vecs = {r["pkg"]: embed.app_vec(r.get("title") or "", r.get("short_desc") or "",
                              r.get("description") or "")
            for r in rows + featured}
    fld = features.compute_field(rows, keyword, top_n=TOP_K, featured=featured,
                                 kw_vec=kv, app_vecs=vecs)
    # No history write here: scrape_keyword already recorded this field when it
    # fetched it. Recording again on every read would fill the series with reads
    # rather than observations, and fake movement between two views of one scrape.

    # Split the page by meaning before measuring anything. A page answering two
    # questions is two competitions, and averaging them is how a wide-open
    # intent gets reported as crowded.
    from . import intent as intent_mod
    intents = intent_mod.split(rows, vecs, kv)
    # Strip the raw centroids before the result leaves: they are numpy arrays
    # used for assignment, not something a JSON consumer should receive.
    intents = {**intents,
               "groups": [{k: v for k, v in g.items() if k != "centroid"}
                          for g in intents["groups"]]}
    ranked = sorted((r for r in rows if r.get("position")), key=lambda r: r["position"])

    from . import suggest as sugg_mod
    if refresh_demand and not sugg_mod.is_fresh(con, keyword, country):
        from . import ui
        with ui.Task("reading Play autocomplete", quiet=not verbose) as t:
            got = sugg_mod.ensure(con, keyword, country)
            t.done(f"{len(got)} suggestions returned")
    sg = sugg_mod.signals(con, keyword, country)

    # Demand is REPORTED, not computed. An override is a human saying otherwise.
    d = db.get_override(con, keyword, "demand", country)
    d_src = "override" if d is not None else "suggestions"

    model, _, version = train.load_active(con)
    candidate, crowd, c_src = None, None, "untrained"

    from . import ui
    scoring = ui.Task("scoring the field", quiet=not verbose or model is None)
    scoring.__enter__()
    if pkg and model is not None:
        candidate = predict.score(con, pkg, keyword, country=country)
        crowd = candidate["crowding"]
        c_src = "override" if candidate["crowding_overridden"] else "model"
    elif model is not None and len(ranked) > TOP_K:
        gate = ranked[TOP_K - 1]
        probe = predict.score(con, gate["pkg"], keyword, country=country, record=False)
        crowd = probe["crowding"]
        c_src = "override" if probe["crowding_overridden"] else "model"
    if crowd is None:
        ov = db.get_override(con, keyword, "crowding", country)
        # No formula fallback. Without a model there is no difficulty number,
        # and inventing one from hand-picked weights would be a guess wearing a
        # score's clothing.
        crowd, c_src = (ov, "override") if ov is not None else (None, "untrained")
    scoring.done(f"{len(ranked)} apps scored")

    out = {
        "keyword": keyword, "country": country, "model_version": version,
        "suggestions": {**sugg_mod.detail(con, keyword, country), "features": sg},
        "demand": {"score": None if d is None else round(d, 1),
                   "band": band(d, DEMAND_BANDS), "source": d_src},
        "crowding": {"score": None if crowd is None else round(crowd, 1),
                     "band": band(crowd, CROWD_BANDS), "source": c_src},
        "field": {
            "scraped": len(ranked), "top_n": fld["n"],
            "installs_p10": fld["installs_p10"], "installs_p50": fld["installs_p50"],
            "installs_p90": fld["installs_p90"], "rating_p50": fld["rating_p50"],
            "exact_match_count": fld["exact_match_count"],
            "age_p50_years": round(fld.get("age_p50") or 0.0, 1),
            "newcomers": fld.get("newcomers") or 0,
            "newcomer_installs": fld.get("newcomer_installs") or 0,
            "newest_entrant_age": round(fld.get("newest_entrant_age") or 0.0, 2),
            "age_known_frac": round(fld.get("age_known_frac") or 0.0, 2),
            "velocity_p50": fld.get("velocity_p50") or 0.0,
            "relevant_count": fld.get("relevant_count") or 0,
            "relevance_p50": round(fld.get("relevance_p50") or 0.0, 2),
            "installs_p50_relevant": fld.get("installs_p50_relevant") or 0,
            "weakest_installs": fld.get("weakest_installs") or 0,
            "weakest_rank": fld.get("weakest_rank") or 99,
            "weakest_title": fld.get("weakest_title"),
            "has_featured": bool(fld.get("has_featured")),
            "featured_relevance": round(fld.get("featured_relevance") or 0.0, 2),
            "featured_installs": fld.get("featured_installs") or 0,
            "age_spread": round(fld.get("age_spread") or 0.0, 1),
            "spread": round(math.log10((fld["installs_p90"] or 0) + 1)
                            - math.log10((fld["installs_p10"] or 0) + 1), 2),
        },
        "bar": {"installs": fld["installs_p50"], "rating": fld["rating_p50"],
                "keyword_in_title": fld["exact_match_count"] >= max(fld["n"] // 2, 1)},
        "intents": intents,
        "top": [{"position": r["position"], "pkg": r["pkg"], "title": r.get("title"),
                 "installs": r.get("installs"), "rating": r.get("rating"),
                 "exact_match": features.exact_match(r.get("title") or "", keyword)}
                # The whole page, not a preview. A caller wanting five can take
                # five; one wanting to see what actually ranks cannot invent the
                # rest.
                for r in ranked],
        "candidate": None,
    }

    # Every analysis is a training step. The rows this keyword just contributed
    # are folded in before the answer is produced, so each lookup leaves the
    # model slightly better than it found it.
    if learn:
        from . import ui
        with ui.Task("learning from this keyword", quiet=not verbose) as t:
            _, tmeta, _ = train.train_quick(con, country=country)
            t.done(f"trained on {tmeta['n_rows']} rows across "
                   f"{tmeta['n_keywords']} keywords" if tmeta else "not enough data yet")

    # "Would a well-built new app rank here?" asked of the model directly.
    say("Placing a new app on the page")
    out["entry"] = predict.score_entry(con, keyword, country)
    # The meaning a new app would be entering, so the narrative and the numbers
    # describe the same set of apps.
    if intents.get("groups"):
        out["_your_group"] = max(intents["groups"], key=lambda g: g["affinity"])

    kw_total = db.scalar(
        con, "SELECT COUNT(DISTINCT keyword) FROM observations")
    reg = con.execute("SELECT n_rows FROM registry WHERE active=1").fetchone()
    out["trained_on"] = {"keywords": kw_total,
                         "rows": (reg["n_rows"] if reg else 0) or 0,
                         "validated": kw_total >= 25}

    # A field smaller than the top-N cut still has a last place, and that is the
    # slot a newcomer has to beat. Falling through to None left the difficulty
    # blank on exactly the narrow keywords most worth looking at.
    if crowd is None and out["entry"] is not None:
        crowd, c_src = out["entry"]["crowding"], "model"
        out["crowding"] = {"score": None if crowd is None else round(crowd, 1),
                           "band": band(crowd, CROWD_BANDS), "source": c_src}

    if candidate:
        out["candidate"] = {
            "pkg": pkg,
            "predicted_rank": candidate["predicted_rank"],
            "field_size": candidate["field_size"],
            "chance": round(candidate["chance"], 3),
            "fit": candidate["fit"],
            "verdict": predict.verdict(candidate["predicted_rank"]),
            "corrected_by": round(candidate["residual"], 3) or None,
            "prediction_id": candidate["prediction_id"],
        }
    elif pkg and model is None:
        out["candidate"] = {"pkg": pkg, "error": "no model trained yet. Run: aso train"}
    return out


# --------------------------------------------------------------------------
# The plain answer: is this keyword worth building for?

# Bands are LABELS on the model's own predicted rank, not a calculation. The
# number is the network's; these words only name it.
VERDICTS = ((3, "BUILD IT"), (TOP_K, "WORTH A TRY"), (10 ** 9, "SKIP"))
# While the model is unvalidated the same finding is a lean, not a call. Printing
# "BUILD IT" and "low confidence" on consecutive lines states two opposite things
# with equal force, so the label itself carries the hedge instead.
HEDGED = {"BUILD IT": "LEANS BUILD", "WORTH A TRY": "LEANS TRY", "SKIP": "LEANS SKIP"}


def recommend(o: dict, thresholds: dict | None = None) -> dict:
    """No score is synthesised here. The headline is the model's predicted rank
    for a well-optimized new app, and demand is reported beside it as a separate
    measurement rather than multiplied into a single made-up number."""
    entry = o.get("entry")
    cand = o["candidate"] if o["candidate"] and "error" not in o["candidate"] else None

    rank = (cand["predicted_rank"] if cand else
            entry["entry_rank"] if entry else None)
    if rank is None:                      # no SERP at all, nothing to rank against
        return {"rank": o["field"]["scraped"] + 1,
                "field_size": o["field"]["scraped"],
                "verdict": "SKIP", "reason": _reason(o, cand, None)}

    conf = (entry or {}).get("confidence", 0.0)
    # The model's probability of reaching the top K. Calling this a "build
    # score" put a decision word on a probability: 53/100 was the chance of
    # ranking, while whether that is worth building is decided by the
    # thresholds in config, which are yours and not the model's to guess.
    chance = (cand or entry or {}).get("chance")
    rank_chance = round((chance if chance is not None else 0.0) * 100)
    # Publishing adds one app to the page, so the denominator grows by one.
    size = (cand["field_size"] if cand else entry["field_size"] + 1)
    from . import config as _cfg
    dl = (entry or {}).get("downloads")
    misses = _cfg.shortfalls(rank_chance, round(conf * 100), rank, dl, thresholds)
    eff = _cfg.effective(thresholds)
    thresholds_set = any(v for k, v in eff.items()
                         if k.startswith(("min_", "max_")) and v)

    return {"rank": rank,
            "field_size": size,
            "rank_chance": rank_chance,               # model P(top K), 0-100
            "build_score": rank_chance,               # deprecated alias
            "shortfalls": misses,                     # thresholds it fails
            "meets_thresholds": not misses,
            "thresholds_set": thresholds_set,
            "thresholds_used": {k: eff[k] for k in
                                ("min_downloads", "min_downloads_unit", "max_rank",
                                 "min_build_score", "min_model_confidence")},
            "model_confidence": round(conf * 100),    # shown 0-100, held 0-1
            "confidence": round(conf * 100),          # kept for the json consumers
            "downloads": (entry or {}).get("downloads"),
            "for_your_app": bool(cand),
            "verdict": next(v for cut, v in VERDICTS if rank <= cut),
            "reason": _reason(o, cand, o["demand"]["score"])}



def _ago(years: float) -> str:
    if years < 1.0:
        months = max(1, round(years * 12))
        return f"{months} month{'s' if months != 1 else ''}"
    return f"{years:.0f} year{'s' if round(years) != 1 else ''}"


def _reason(o, cand, dem) -> str:
    """Two to four sentences, strongest signal first."""
    from .features import RELEVANT
    f = o["field"]
    n = f["top_n"] or 1
    parts = []
    if o.get("entry") is None and not cand:
        parts.append("No model has been trained yet, so there is no prediction. "
                     "What follows is only what was scraped.")

    # 1. the promoted card
    if (f.get("featured_relevance") or 0) >= RELEVANT:
        parts.append(
            f"Play runs a promoted card above the results and it is a direct rival, "
            f"so the most visible slot on the page is already taken and organic "
            f"rank 1 is worth much less than it looks.")
    elif f.get("has_featured"):
        parts.append(
            f"Play runs a promoted card above the organic results, which takes a "
            f"share of the taps before anyone scrolls.")

    # 2. intent, defined ONE way: the group of apps answering the same question.
    # Lexical relevance said "5 of 10 are about this" while the clustering said
    # one, because sharing the word "mirror" is not answering the same question.
    # Two definitions of on-intent in one paragraph is a contradiction, so the
    # clustering wins and the word-overlap count is not narrated.
    grp = o.get("_your_group")
    scraped = f.get("scraped") or n
    if grp is None:
        parts.append(f"The page could not be split by meaning.")
    elif grp["size"] == 1:
        parts.append(
            f"Only one app on this page answers the same question, and it sits "
            f"at {fmt_n(grp['weakest_installs'])} installs at rank "
            f"{grp['best_rank']}. The other {scraped - 1} are answering something "
            f"else.")
    elif grp["size"] <= max(scraped // 3, 2):
        parts.append(
            f"{grp['size']} of the {scraped} apps here answer the same question, "
            f"sitting around {fmt_n(grp['median_installs'])} installs. The rest "
            f"are a different product.")
    else:
        parts.append(
            f"{grp['size']} of the {scraped} apps answer the same question, "
            f"around {fmt_n(grp['median_installs'])} installs, so this meaning "
            f"is well served already.")

    # 2b. the opening, page-wide: a tiny app holding a high slot says the top of
    # the page is not defended regardless of which meaning it belongs to.
    wi, wr = f.get("weakest_installs"), f.get("weakest_rank")
    if wi is not None and wr and wr <= n and wi < 10_000 and (
            not grp or grp["weakest_installs"] != wi):
        parts.append(f"A {fmt_n(wi)}-install app is holding rank {wr}, so the top "
                     f"slots are not defended.")

    # 3. the door, scoped to the meaning being discussed so the numbers in this
    # paragraph all describe the same set of apps
    grp = o.get("_your_group")
    newest = f.get("newest_entrant_age") or 0.0
    if (f.get("age_known_frac") or 0) < 0.5:
        pass
    elif grp and grp.get("newcomers"):
        got = grp.get("newcomer_installs") or 0
        reach = f", the strongest at {fmt_n(got)} installs" if got >= 10_000 else ""
        if grp["size"] == 1:
            # The single sentence above already named this app; do not
            # re-announce it as though it were a second finding.
            parts.append(f"It launched within the last year, so the door is open.")
        else:
            parts.append(f"{grp['newcomers']} of them launched within the last "
                         f"year{reach}, so the door is open.")
    elif grp is not None and grp.get("newcomers") == 0 and newest >= 4:
        parts.append(f"Nothing new has entered this meaning in {_ago(newest)}.")
    elif (f.get("newcomers") or 0) >= 2:
        parts.append(f"{f['newcomers']} apps on the page launched within the last "
                     f"year, though answering a different question.")

    # 4. demand, and the call
    dem = dem if dem is not None else 50.0
    if dem >= 70:
        parts.append("Plenty of people search it.")
    elif dem < 40 and o["demand"]["score"] is not None:
        parts.append("Almost nobody searches it, which caps the upside whatever "
                     "the competition looks like.")

    if cand:
        r = cand["predicted_rank"]
        parts.append(f"Your app would land around #{r}"
                     + (f", inside the top {TOP_K}." if r <= TOP_K
                        else f", outside the top {TOP_K} where the installs go."))
    elif o.get("entry"):
        r = o["entry"]["entry_rank"]
        parts.append(f"A well-built new app would land around #{r}"
                     + (f", inside the top {TOP_K} where the installs go."
                        if r <= TOP_K
                        else f", outside the top {TOP_K} where the installs go."))
    return " ".join(parts)
