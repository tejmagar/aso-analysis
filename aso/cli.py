"""Five commands. `analyze` is the one you use.

    aso analyze "habit tracker"                    # should I build for this?
    aso analyze "habit tracker" --depth            # the full breakdown
    aso analyze "habit tracker" --pkg com.you.habits
    aso correct "habit tracker" --rank 4 --pkg com.you.habits
    aso correct "habit tracker" --demand 60 --crowding 45
    aso train
    aso status

Add --json to any read command for machine output. `analyze` scrapes the keyword
itself if it has never seen it, so there is no separate setup step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import db, env

env.load()          # .env first, so ASO_API_TOKEN and friends are visible

W = 22


def bar(v, width=W):
    if v is None:
        return "." * width
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "#" * n + "." * (width - n)


def row(label, value, frac, note=""):
    val = "  n/a" if value is None else f"{value:>5.0f}"
    print(f"  {label:<13} {bar(frac)} {val}   {note}")


def render_deep(o):
    from .analyze import fmt_n
    f, c, d = o["field"], o["crowding"], o["demand"]
    head = o.get("recommendation", {}).get("verdict") or (c["band"] or "untrained")
    print(f'\n  {o["keyword"]}  ·  {o["country"]}'
          f'{head.upper():>{max(1, 58 - len(o["keyword"]) - len(o["country"]))}}\n')

    row("demand", d["score"], None if d["score"] is None else d["score"] / 100,
        f'{d["band"] or "unknown"}  ({d["source"]})')
    row("crowding", c["score"], None if c["score"] is None else c["score"] / 100,
        f'{c["band"] or "needs a trained model"}  ({c["source"]})')

    cand = o["candidate"]
    if cand and "error" in cand:
        print(f'\n  {cand["pkg"]}: {cand["error"]}')
    elif cand:
        print(f'\n  app           {cand["pkg"]}')
        print(f'  predicted     rank #{cand["predicted_rank"]} of '
              f'{cand["field_size"]}      {cand["verdict"]}')
        row("fit", cand["fit"], cand["fit"] / 100)
        row("chance top10", cand["chance"] * 100, cand["chance"])
        if cand["corrected_by"]:
            print(f'  (includes a reviewer correction of '
                  f'{cand["corrected_by"]:+.2f} logits)')

    print(f'\n  Field, top {f["top_n"]} of {f["scraped"]} scraped')
    print(f'    installs      p10 {fmt_n(f["installs_p10"]):<7} '
          f'p50 {fmt_n(f["installs_p50"]):<7} p90 {fmt_n(f["installs_p90"]):<7} '
          f'spread {f["spread"]}')
    print(f'    rating        {f["rating_p50"]:.1f} median')
    print(f'    exact match   {f["exact_match_count"]} of {f["top_n"]} '
          f'carry the full keyword in the title')
    fresh = f.get("newcomers", 0)
    known = f.get("age_known_frac", 0)
    print(f'    age           {f.get("age_p50_years", 0)}y median, '
          f'{f.get("age_spread", 0)}y spread')
    if known < 0.5:
        print(f'    newest entry  unknown, release dates missing for '
              f'{(1 - known) * 100:.0f}% of the field')
    else:
        newest = f.get("newest_entrant_age", 0)
        note = (f'{fresh} within the last year' if fresh
                else "nobody new has broken in")
        if fresh and f.get("newcomer_installs"):
            note += f', best at {fmt_n(f["newcomer_installs"])} installs'
        print(f'    newest entry  {newest}y ago, {note}')
    print(f'    growth bar    {fmt_n(f.get("velocity_p50", 0))} installs/year '
          f'at the median')

    b = o["bar"]
    print(f'\n  Bar to reach the top {f["top_n"]}')
    print(f'    installs      {fmt_n(b["installs"])}')
    print(f'    rating        {b["rating"]:.1f}')
    print(f'    title         {"put the keyword in it" if b["keyword_in_title"] else "not decisive here"}')

    if o["top"]:
        print("\n  Top of the field")
        for t in o["top"]:
            title = (t["title"] or "")[:34]
            print(f'  {t["position"]:>4}  {title:<34} {fmt_n(t["installs"]):>7}  '
                  f'{(t["rating"] or 0):.1f}  {"exact" if t["exact_match"] else ""}')

    pkg = (cand or {}).get("pkg")
    hint = f' --pkg {pkg}' if pkg else ""
    print(f'\n  correct any value:'
          f'\n    aso correct "{o["keyword"]}" --rank 4{hint}'
          f'\n    aso correct "{o["keyword"]}" --demand 60 --crowding 45\n')


def render_simple(o):
    """Model output only. Nothing on this screen is computed by a formula."""
    import textwrap

    from . import tui
    from .analyze import fmt_n

    rec = o["recommendation"]
    cand = o["candidate"] if o["candidate"] and "error" not in o["candidate"] else None
    d, c, f = o["demand"], o["crowding"], o["field"]

    title = f'{tui.BOLD}{tui.WHITE}{o["keyword"]}{tui.RESET}'
    if cand:
        title += f'  {tui.TRACK}·{tui.RESET}  {tui.GRAY}{cand["pkg"]}{tui.RESET}'
    print(f'\n  {title}  {tui.FAINT}{o["country"]}{tui.RESET}')
    print(f'  {tui.rule()}\n')


    dscore, n = d["score"], f["top_n"] or 1

    def stat(label, frac, tint, text):
        print(f'  {tui.GRAY}{label:<14}{tui.RESET}{tui.bar(frac, 20, tint)}  '
              f'{tui.WHITE}{text}{tui.RESET}\n')

    # Two blocks, because they are two different kinds of claim. Everything
    # under PREDICTED is a forward pass through the network. Everything under
    # OBSERVED is a fact read off the store. Mixing them made the model's
    # guesses look like measurements.
    print(f'  {tui.FAINT}PREDICTED BY THE MODEL{tui.RESET}\n')

    who = "your app" if rec.get("for_your_app") else "a new app"
    print(f'  {tui.GRAY}{"rank":<14}{tui.RESET}{tui.WHITE}#{rec["rank"]}{tui.RESET}'
          f'{tui.FAINT} of {rec["field_size"]}, for {who} published here{tui.RESET}\n')

    dl = rec.get("downloads")
    if dl:
        print(f'  {tui.GRAY}{"downloads":<14}{tui.RESET}'
              f'{tui.WHITE}{fmt_n(dl["per_day"])}{tui.RESET}{tui.FAINT}/day   '
              f'{tui.RESET}{tui.WHITE}{fmt_n(dl["per_month"])}{tui.RESET}'
              f'{tui.FAINT}/month   {tui.RESET}'
              f'{tui.WHITE}{fmt_n(dl["per_year"])}{tui.RESET}{tui.FAINT}/year{tui.RESET}')
        print(f'  {tui.FAINT}{"":<14}range {fmt_n(dl["low_per_year"])} to '
              f'{fmt_n(dl["high_per_year"])} a year{tui.RESET}\n')

    if c["score"] is None:
        stat("competition", None, tui.TRACK, "no field to score against")
    else:
        stat("competition", c["score"] / 100,
             tui.color_for(c["score"], invert=True),
             f'{c["band"]}  (score needed to hold slot {n})')

    print(f'  {tui.FAINT}OBSERVED IN THE STORE{tui.RESET}\n')

    sg = o.get("suggestions") or {}
    if dscore is not None:
        stat("demand", dscore / 100, tui.color_for(dscore), f'{d["band"]}  (set by hand)')
    else:
        sugg, rank = sg.get("suggestions") or [], sg.get("self_rank")
        if not sugg:
            stat("autocomplete", None, tui.TRACK, "not checked")
        elif rank is None:
            stat("autocomplete", 0.0, tui.RED,
                 f'Play does not list it, it suggests "{sugg[0]}" instead')
        else:
            ext = len(sg.get("extends") or [])
            stat("autocomplete", 1 - rank / len(sugg), tui.color_for(100 - rank * 25),
                 f'listed {rank + 1} of {len(sugg)}'
                 + (f', {ext} longer variant{"s" if ext != 1 else ""}' if ext else ''))
            for term in sugg[:5]:
                mark = f'{tui.CYAN}▸{tui.RESET}' if term == sg.get("query") else ' '
                print(f'    {mark} {tui.GRAY}{term}{tui.RESET}')
            print()

    it_pre = o.get("intents") or {}
    grp_pre = (max(it_pre["groups"], key=lambda g: g["affinity"])
               if it_pre.get("groups") else None)
    wi, wr = f.get("weakest_installs"), f.get("weakest_rank")
    # Skip when the intent row below is about to report the same app.
    if grp_pre and grp_pre["weakest_installs"] == wi and grp_pre["weakest_rank"] == wr:
        wi = None
    if wi is not None and wr and wr <= n:
        # A ceiling of 1M is just the drawing scale for the bar; the number and
        # the rank beside it are the raw facts.
        import math as _m
        frac = 1 - min(_m.log10(wi + 1) / 6.0, 1.0)
        stat("weakest held", frac, tui.color_for(frac * 100),
             f'{fmt_n(wi)} installs at rank {wr}'
             + (f'  ({(f.get("weakest_title") or "")[:26]})' if f.get("weakest_title") else ''))

    it0 = o.get("intents") or {}
    groups0 = it0.get("groups") or []
    scraped = f.get("scraped") or n
    if groups0:
        yours = max(groups0, key=lambda g: g["affinity"])
        # "On-intent" means one thing: apps answering the same question. The
        # word-overlap count said 5 while the clustering said 1; narrating both
        # was the contradiction.
        stat("same question", yours["size"] / max(scraped, 1),
             tui.color_for(100 - yours["size"] * 100 / max(scraped, 1), invert=True),
             f'{yours["size"]} of {scraped} apps, '
             f'weakest {fmt_n(yours["weakest_installs"])} at #{yours["weakest_rank"]}')

    # The page split by meaning. A keyword answering two questions is two
    # competitions, and the one you would enter is the only one that matters.
    it = o.get("intents") or {}
    groups = it.get("groups") or []
    if groups:
        if it.get("ambiguous"):
            print(f'  {tui.AMBER}intent split{tui.RESET}{tui.FAINT}: Play is answering '
                  f'{len(groups)} different questions on this page{tui.RESET}\n')
        closest = max(groups, key=lambda g: g["affinity"])
        for g in groups[:4]:
            lead = (f'{tui.CYAN}▸{tui.RESET}' if g is closest else ' ')
            weak = tui.GREEN if g["weakest_installs"] < 50_000 else tui.WHITE
            print(f'    {lead} {tui.WHITE}{g["label"][:26]:<26}{tui.RESET}'
                  f'{tui.FAINT}{g["size"]:>3} app{"s" if g["size"] != 1 else " "}'
                  f'  weakest {tui.RESET}{weak}{fmt_n(g["weakest_installs"]):>7}'
                  f'{tui.RESET}{tui.FAINT} at #{g["weakest_rank"]}{tui.RESET}')
            print(f'      {tui.GRAY}{", ".join(t or "" for t in g["titles"][:2])[:58]}{tui.RESET}')
        print()

    if f.get("age_known_frac", 0) >= 0.5:
        newest = f.get("newest_entrant_age", 0)
        stat("newest entry", max(0.0, min(1.0, (4.0 - newest) / 4.0)),
             tui.color_for(max(0.0, min(100.0, (4.0 - newest) / 4.0 * 100))),
             f'{newest}y ago, growth bar {fmt_n(f.get("velocity_p50", 0))}/yr')
    else:
        stat("newest entry", None, tui.TRACK, "unknown, release dates missing")

    if f.get("has_featured"):
        on_intent = f.get("featured_relevance", 0) >= 0.40
        tint = tui.RED if on_intent else tui.AMBER
        note = "a direct rival holds it" if on_intent else "unrelated to your intent"
        print(f'  {tint}▲ promoted card above the organic results{tui.RESET}'
              f'{tui.FAINT}, {note}{tui.RESET}')

    import textwrap

    print(f'  {tui.rule()}')
    print(f'  {tui.FAINT}CONCLUSION{tui.RESET}\n')

    topk = (o.get("field") or {}).get("top_n") or 10
    ch = rec.get("rank_chance", rec.get("build_score", 0))
    ct = tui.GREEN if ch >= 55 else tui.AMBER if ch >= 25 else tui.RED
    print(f'  {tui.GRAY}{"chance of top " + str(topk):<17}{tui.RESET}'
          f'{tui.bar(ch / 100, 20, ct)}  {ct}{tui.BOLD}{ch}/100{tui.RESET}'
          f'{tui.FAINT}  the model\'s odds of ranking{tui.RESET}\n')

    mc = rec.get("model_confidence", 0)
    mt = tui.GREEN if mc >= 60 else tui.AMBER if mc >= 30 else tui.RED
    print(f'  {tui.GRAY}{"model certainty":<17}{tui.RESET}'
          f'{tui.bar(mc / 100, 20, mt)}  {mt}{mc}/100{tui.RESET}'
          f'{tui.FAINT}  how sure it is of that{tui.RESET}\n')

    # Worth is decided by YOUR thresholds, not by the model. Saying so plainly
    # keeps the two apart: the model reports odds, the config reports whether
    # those odds are good enough for you.
    miss = rec.get("shortfalls") or []
    if miss:
        # Amber, not red: falling short of your own bar is a caution, not an
        # error. Red is for a model that cannot answer.
        print(f'  {tui.AMBER}{tui.BOLD}⚠ not worth building{tui.RESET}'
              f'{tui.FAINT}, by your thresholds{tui.RESET}')
        for m in miss:
            print(f'    {tui.AMBER}·{tui.RESET} {tui.GRAY}{m}{tui.RESET}')
        print()
    elif rec.get("thresholds_set"):
        print(f'  {tui.GREEN}{tui.BOLD}worth building{tui.RESET}'
              f'{tui.FAINT}, it clears every threshold you set{tui.RESET}\n')
    else:
        print(f'  {tui.FAINT}no thresholds set, so worth is your call: '
              f'aso config min download day 100{tui.RESET}\n')

    for line in textwrap.wrap(rec["reason"], 62):
        print(f"  {tui.GRAY}{line}{tui.RESET}")
    print()

    if o["candidate"] and "error" in o["candidate"]:
        print(f'  {tui.AMBER}{o["candidate"]["error"]}{tui.RESET}')
    t = o.get("trained_on") or {}
    kws = t.get("keywords", 0)
    if t.get("validated"):
        print(f'  {tui.FAINT}model trained on {t.get("rows", 0)} rows across '
              f'{kws} keywords{tui.RESET}')
    else:
        print(f'  {tui.AMBER}model trained on only {kws} keywords{tui.RESET}'
              f'{tui.FAINT}, {25 - kws} more before its score can be '
              f'validated{tui.RESET}')
    print()



# --------------------------------------------------------------- interactive

def ask(msg, default=None):
    """None means the user wants out: EOF, Ctrl-C, or a piped-in end of input."""
    try:
        v = input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return v or default


def _run(con, keyword, country, pkg):
    from . import scrape
    from .analyze import analyze, recommend
    o = analyze(con, keyword, country=country, pkg=scrape.parse_package(pkg) if pkg else None)
    o["recommendation"] = recommend(o)
    return o


def _correct_menu(con, o, country):
    """Correct one value, showing what it currently is so the choice is informed."""
    from . import correct
    cand = o["candidate"] if o["candidate"] and "error" not in o["candidate"] else None
    d, c = o["demand"], o["crowding"]

    from . import tui
    print(f'\n  {tui.FAINT}which value is wrong?{tui.RESET}')
    rank_lbl = (f'Rank         currently #{cand["predicted_rank"]}' if cand
                else 'Rank         needs an app, pick "Compare an app" first')
    which = tui.select([
        ("r", rank_lbl),
        ("d", f'Demand       currently '
              f'{"n/a" if d["score"] is None else round(d["score"])}'),
        ("c", f'Crowding     currently {round(c["score"])}'),
        ("b", "Back"),
    ])
    if which in (None, "b"):
        return o

    if which == "r":
        if not cand:
            print(f"  {tui.AMBER}pick an app first{tui.RESET}")
            return o
        v = tui.prompt("what rank does it actually hold?")
        if not v:
            return o
        correct.apply(con, o["keyword"], pkg=cand["pkg"], rank=int(v), country=country)
    elif which in ("d", "c"):
        label = "demand" if which == "d" else "crowding"
        v = tui.prompt(f"new {label} (0-100)")
        if not v:
            return o
        correct.apply(con, o["keyword"], country=country,
                      **{label: float(v)})
    else:
        print("  not an option")
        return o

    print(f"  {tui.GREEN}corrected{tui.RESET}{tui.FAINT}, re-scoring{tui.RESET}")
    return _run(con, o["keyword"], country, cand["pkg"] if cand else None)


def interactive(con, o, country):
    """After a result, offer the next move. Skipped when stdout is piped."""
    from . import tui
    pkg = (o["candidate"] or {}).get("pkg")
    while True:
        choice = tui.select(
            [("c", "Correct a value"), ("d", "Full breakdown"),
             ("p", "Compare an app"), ("w", "Why is an app not ranking"),
             ("n", "Another keyword"), ("q", "Quit")],
            hint="up/down to move, enter to choose")

        if choice in (None, "q"):
            print()
            return
        if choice == "d":
            render_deep(o)
        elif choice == "c":
            o = _correct_menu(con, o, country)
            pkg = (o["candidate"] or {}).get("pkg")
            render_simple(o)
        elif choice == "p":
            v = tui.prompt("package name or Play Store link")
            if v:
                from . import scrape
                pkg = scrape.parse_package(v)
                o = _run(con, o["keyword"], country, pkg)
                render_simple(o)
        elif choice == "w":
            from . import scrape
            v = tui.prompt(f'package or link (blank for {pkg})' if pkg
                           else "package name or Play Store link")
            target = scrape.parse_package(v) if v else pkg
            if not target:
                print(f"  {tui.AMBER}need an app to explain{tui.RESET}\n")
            else:
                try:
                    _why(con, o["keyword"], target, country)
                except SystemExit as e:
                    print(f"  {tui.AMBER}{e}{tui.RESET}\n")
        elif choice == "n":
            v = tui.prompt("keyword")
            if v:
                try:
                    o = _run(con, v, country, pkg)
                    render_simple(o)
                except Exception as e:                       # noqa: BLE001
                    print(f"  {v!r}: {e}\n")                 # stay in the loop



def _thresholds_from_args(a) -> dict:
    """Per-call overrides. Absent flags stay absent so the saved config wins."""
    out = {}
    if getattr(a, "min_download", None) is not None:
        out["min_downloads"] = a.min_download
    if getattr(a, "min_download_unit", None):
        out["min_downloads_unit"] = a.min_download_unit
    if getattr(a, "max_rank", None) is not None:
        out["max_rank"] = a.max_rank
    if getattr(a, "min_chance", None) is not None:
        out["min_build_score"] = a.min_chance
    if getattr(a, "min_certainty", None) is not None:
        out["min_model_confidence"] = a.min_certainty
    return out


def cmd_analyze(a):
    from .analyze import NoResults, analyze
    from . import client
    from . import scrape as _scrape
    pkg = _scrape.parse_package(a.pkg) if a.pkg else None

    # A resident server has already paid the 16s load cost, so use it when it is
    # there. Falling back in-process keeps the CLI usable with no server at all.
    thresholds = _thresholds_from_args(a)
    if not a.no_server and client.up():
        o = client.call("/analyze", {"keyword": a.keyword, "pkg": pkg,
                                     "country": a.country, **thresholds})
        if a.json:
            print(json.dumps(o, indent=2))
        else:
            render_deep(o) if a.depth else render_simple(o)
        return

    con = db.connect()
    try:
        o = analyze(con, a.keyword, country=a.country, pkg=pkg, verbose=not a.json)
    except NoResults as e:
        print(f'\n  Play returned no apps for {e.keyword!r}.\n'
              f'  Check the spelling, try a broader phrase, or the term may have\n'
              f'  no apps built for it at all, which is its own kind of answer.\n')
        con.close()
        return 1
    from .analyze import recommend
    o["recommendation"] = recommend(o, thresholds)
    if a.json:
        print(json.dumps(o, indent=2))
        con.close()
        return
    render_deep(o) if a.depth else render_simple(o)
    # Only prompt a real terminal. Piped or redirected output must stay a
    # one-shot command, or every script that calls this hangs forever.
    if not a.once and sys.stdin.isatty() and sys.stdout.isatty():
        interactive(con, o, a.country)
    con.close()


def cmd_correct(a):
    from . import correct
    con = db.connect()
    done = correct.apply(con, a.keyword, pkg=a.pkg, rank=a.rank, demand=a.demand,
                         crowding=a.crowding, reviewer=a.reviewer, country=a.country)
    if a.json:
        print(json.dumps({"keyword": a.keyword, "applied": done}, indent=2))
    else:
        for line in done:
            print(f"  {line}")
        print("  applied. it takes effect on the next analyze")
    con.close()


def cmd_scrape(a):
    from . import scrape
    con = db.connect()
    for kw in a.keywords:
        print(f"scraping {kw!r}")
        scrape.scrape_keyword(con, kw, country=a.country, with_details=not a.fast,
                              sleep=a.sleep)
    con.close()


def cmd_ingest(a):
    from . import ingest
    con = db.connect()
    if a.dry_run:
        t = (ingest.publisher_targets(a.from_publisher) if a.from_publisher
             else ingest.targets())
        left = t if a.all else ingest.pending(con, t, a.country, a.fresh_days)
        linked = [x for x in left if x["pkg"]]
        for x in left[:12]:
            print(f'  {x["keyword"][:34]:<34} -> {x["pkg"] or "(unmatched)"}')
        print(f'\n  {len(t)} keywords total, {len(left)} still to scrape '
              f'({len(linked)} linked to a published app)')
        print(f'  roughly {max(len(left) * 43 // 60, 1)} minutes from here')
        con.close()
        return
    st = ingest.run(con, limit=a.limit, sleep=a.sleep, with_details=not a.fast,
                    source=a.from_publisher, country=a.country,
                    resume=not a.all, fresh_days=a.fresh_days)
    print(f'\n  keywords scraped      {st["keywords"]}')
    print(f'  ranked observations   {st["ranked"]}')
    print(f'  our app ranks         {st["ours_ranked"]}')
    print(f'  our app does NOT      {st["ours_absent"]}   <- real negatives')
    print(f'  keyword unmatched     {st["unmatched"]}')
    if st["failed"]:
        print(f'  failed                {st["failed"]}')
    con.close()


def cmd_track(a):
    from . import scrape, track, tui
    con = db.connect()

    if not a.show:
        todo = track.due(con, a.country, a.every)
        if a.limit:
            todo = todo[:a.limit]
        if not todo:
            print(f'  {tui.FAINT}nothing due: every tracked keyword was scraped '
                  f'within the last {a.every} days{tui.RESET}')
        for i, kw in enumerate(todo, 1):
            print(f'  [{i}/{len(todo)}] {kw!r}')
            try:
                scrape.scrape_keyword(con, kw, country=a.country, sleep=a.sleep)
            except Exception as e:                            # noqa: BLE001
                print(f'    ! {type(e).__name__}, skipping')

    r = track.summary(con, a.country, a.gap)
    if a.json:
        print(json.dumps(r, indent=2))
        con.close()
        return

    print(f'\n  {tui.BOLD}What changed on the keywords we watch{tui.RESET}')
    print(f'  {tui.rule()}\n')
    if not r["transitions"]:
        print(f'  {tui.FAINT}no snapshot pairs far enough apart yet. Re-run this '
              f'in a week{tui.RESET}')
        print(f'  {tui.FAINT}and every keyword becomes a before/after pair on '
              f'its own.{tui.RESET}\n')
        con.close()
        return

    print(f'    {tui.GREEN}entered{tui.RESET}  {r["entered"]:>4}'
          f'{tui.FAINT}   of which brand new apps: {r["new_app_entries"]}{tui.RESET}')
    print(f'    {tui.RED}left{tui.RESET}     {r["left"]:>4}')
    print(f'    {tui.GRAY}moved{tui.RESET}    {r["moved"]:>4}\n')
    if r["new_app_entries"]:
        print(f'  {r["new_app_reached_top10"]} of {r["new_app_entries"]} new apps '
              f'landed in the top 10:')
        for t in r["examples"]:
            slot = f'#{t["to"]}' if t["to"] else "-"
            print(f'    {tui.GRAY}{(t["title"] or "")[:30]:<30}{tui.RESET} '
                  f'{slot:>4} on {tui.FAINT}{t["keyword"][:24]}{tui.RESET}')
    print()
    con.close()


def cmd_backtest(a):
    from . import backtest, tui
    from .analyze import fmt_n
    con = db.connect()
    r = backtest.run(con, country=a.country)
    if a.json:
        print(json.dumps(r, indent=2))
        con.close()
        return

    print(f'\n  {tui.BOLD}Did the research call it right?{tui.RESET}')
    print(f'  {tui.rule()}\n')
    if not r["pairs"]:
        print(f'  {tui.FAINT}no keyword yet has both a research snapshot and a '
              f'known outcome{tui.RESET}\n')
    else:
        print(f'  {r["pairs"]} keywords researched before building, outcome now known')
        print(f'    {tui.GREEN}reached the top 10{tui.RESET}  {r["ranked"]}')
        print(f'    {tui.RED}did not{tui.RESET}              {r["missed"]}\n')
        a_, b_ = (r["installs_median_when_ranked"], r["installs_median_when_missed"])
        if a_ and b_:
            print(f'  field size we were up against, as recorded back then:')
            print(f'    {tui.GREEN}when we ranked {tui.RESET}{fmt_n(a_):>9} median installs')
            print(f'    {tui.RED}when we did not{tui.RESET}{fmt_n(b_):>9} median installs')
            verdict = ("the research signal held up" if a_ < b_
                       else "the research signal did NOT hold up")
            print(f'    {tui.FAINT}{verdict}{tui.RESET}\n')

    sp = r["snapshot_pairs"]
    print(f'  {tui.FAINT}full-feature before/after pairs from our own scrape '
          f'history: {len(sp)}{tui.RESET}')
    if not sp:
        print(f'  {tui.FAINT}none yet. Every scrape is stamped and kept, so '
              f're-scraping{tui.RESET}')
        print(f'  {tui.FAINT}these keywords in a few weeks builds them '
              f'automatically.{tui.RESET}')
    else:
        for x in sp[:5]:
            print(f'    {x["keyword"][:30]:<30} {x["gap_days"]}d apart')
    print()
    con.close()


def cmd_config(a):
    from . import config, tui
    if a.path:
        print(config.path())
        return
    cfg = config.load(reload=True)
    args = list(a.args)

    # aso config min download <day|month|year> <number>
    if len(args) >= 2 and args[0] == "min" and args[1] in ("download", "downloads"):
        rest = args[2:]
        if len(rest) == 2:
            unit, amount = rest
            p = config.set_min_download(unit, config.coerce("min_downloads", amount))
            print(f'  minimum download = {config.get("min_downloads"):,.0f} '
                  f'a {config.get("min_downloads_unit")}   {tui.FAINT}({p}){tui.RESET}')
        elif not rest:
            v, u = config.get("min_downloads"), config.get("min_downloads_unit")
            print(f'{v:,.0f} a {u}' if v else "off")
        else:
            raise SystemExit("usage: aso config min download <day|month|year> <number>")
        return

    if len(args) == 2:
        key, value = args
        if key not in config.DEFAULTS:
            raise SystemExit(f"unknown setting {key!r}. "
                             f"Try: aso config   to list them all")
        p = config.save({key: config.coerce(key, value)})
        print(f'  {key} = {config.get(key)}   {tui.FAINT}({p}){tui.RESET}')
        return
    if len(args) == 1:
        print(cfg.get(args[0], config.DEFAULTS.get(args[0])))
        return
    if args:
        raise SystemExit("usage: aso config [key [value]] | "
                         "aso config min download <day|month|year> <number>")
    if a.json:
        print(json.dumps(cfg, indent=2))
        return

    print(f'\n  {tui.FAINT}{config.path()}{tui.RESET}\n')
    groups = [("your bar", ("min_downloads", "min_downloads_unit",
                            "max_rank", "min_build_score", "min_model_confidence")),
              ("wording", ("build_score_strong", "build_score_maybe")),
              ("what counts as ranking", ("top_k", "country")),
              ("collection", ("fresh_days", "track_every_days", "new_app_years",
                              "scrape_sleep")),
              ("server", ("server_max_concurrent", "server_timeout",
                          "suggest_ttl_hours"))]
    for title, keys in groups:
        print(f'  {tui.GRAY}{title}{tui.RESET}')
        for k in keys:
            v = cfg.get(k, config.DEFAULTS[k])
            same = v == config.DEFAULTS[k]
            mark = f'{tui.FAINT}(default){tui.RESET}' if same else f'{tui.GREEN}(yours){tui.RESET}'
            off = f'{tui.FAINT} off{tui.RESET}' if (v == 0 and k.startswith(("min_", "max_"))) else ""
            print(f'    {k:<26} {tui.WHITE}{v}{tui.RESET}{off}  {mark}')
        print()
    print(f'  {tui.FAINT}aso config min download year 50000{tui.RESET}')
    print(f'  {tui.FAINT}aso config max_rank 5{tui.RESET}\n')


def _fmt_feature_value(name, v):
    from .analyze import fmt_n
    if name.startswith(("log_", "field_installs", "intent_median", "intent_weakest",
                        "field_weakest_installs", "field_installs_relevant")):
        import math
        return fmt_n(math.expm1(v)) if v < 25 else f"{v:.1f}"
    if isinstance(v, float) and abs(v) < 100:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return fmt_n(v)


def _why(con, keyword, pkg, country):
    from . import diagnose, tui
    from .analyze import fmt_n
    r = diagnose.why(con, keyword, pkg, country=country)

    print(f'\n  {tui.BOLD}{tui.WHITE}{r["title"] or pkg}{tui.RESET}'
          f'{tui.FAINT}  vs  {tui.RESET}{tui.WHITE}{r["keyword"]}{tui.RESET}')
    print(f'  {tui.rule()}\n')
    if r["ranked_at"]:
        print(f'  {tui.GREEN}it does rank, at #{r["ranked_at"]}{tui.RESET}'
              f'{tui.FAINT} - this shows what still separates it from the '
              f'last qualifying slot{tui.RESET}\n')
    else:
        print(f'  {tui.RED}not on the page at all{tui.RESET}\n')

    g = r["gate"]
    print(f'  {tui.GRAY}{"to displace":<16}{tui.RESET}{tui.WHITE}'
          f'{(g["title"] or "")[:32]}{tui.RESET}{tui.FAINT} at #{g["position"]}, '
          f'{fmt_n(g["installs"])} installs{tui.RESET}')
    print(f'  {tui.GRAY}{"score gap":<16}{tui.RESET}{tui.WHITE}{r["gap"]:+.2f}{tui.RESET}'
          f'{tui.FAINT} logits to close{tui.RESET}\n')

    print(f'  {tui.FAINT}WHAT IS HOLDING IT BACK{tui.RESET}')
    print(f'  {tui.FAINT}each line: move that one thing to the level of the app '
          f'at #{g["position"]}{tui.RESET}\n')
    if not r["holding_back"]:
        print(f'    {tui.GRAY}nothing single-handedly. The gap is spread across '
              f'many small differences.{tui.RESET}\n')
    for c in r["holding_back"]:
        share = c["gain"] / r["gap"] if r["gap"] > 0 else 0
        bar = tui.bar(min(max(share, 0), 1), 14, tui.color_for(share * 100))
        print(f'    {tui.WHITE}{c["feature"]:<26}{tui.RESET}{bar} '
              f'{tui.WHITE}{c["gain"]:+.2f}{tui.RESET}')
        print(f'      {tui.FAINT}{c["doc"][:58]}{tui.RESET}')
        print(f'      {tui.GRAY}yours {_fmt_feature_value(c["feature"], c["yours"])}'
              f'   theirs {_fmt_feature_value(c["feature"], c["theirs"])}{tui.RESET}\n')

    if r["in_your_favour"]:
        print(f'  {tui.FAINT}ALREADY IN ITS FAVOUR{tui.RESET}\n')
        for c in sorted(r["in_your_favour"], key=lambda x: x["gain"]):
            print(f'    {tui.GREEN}{c["feature"]:<26}{tui.RESET}'
                  f'{tui.GRAY}yours {_fmt_feature_value(c["feature"], c["yours"])}'
                  f'   theirs {_fmt_feature_value(c["feature"], c["theirs"])}{tui.RESET}')
        print()

    covered = r["closable"] / r["gap"] if r["gap"] > 0 else 0
    print(f'  {tui.FAINT}fixing every listed lever closes {covered:.0%} of the '
          f'gap. The rest is field strength you do not control.{tui.RESET}\n')
    return r


def cmd_why(a):
    from . import diagnose, scrape, tui
    from .analyze import fmt_n
    con = db.connect()
    r = diagnose.why(con, a.keyword, scrape.parse_package(a.pkg), country=a.country)
    if a.json:
        print(json.dumps(r, indent=2, default=float))
        con.close()
        return

    print(f'\n  {tui.BOLD}{tui.WHITE}{r["title"] or a.pkg}{tui.RESET}'
          f'{tui.FAINT}  vs  {tui.RESET}{tui.WHITE}{r["keyword"]}{tui.RESET}')
    print(f'  {tui.rule()}\n')

    if r["ranked_at"]:
        print(f'  {tui.GREEN}it does rank, at #{r["ranked_at"]}{tui.RESET}'
              f'{tui.FAINT} - this shows what still separates it from the '
              f'last qualifying slot{tui.RESET}\n')
    else:
        print(f'  {tui.RED}not on the page at all{tui.RESET}\n')

    g = r["gate"]
    print(f'  {tui.GRAY}{"to displace":<16}{tui.RESET}{tui.WHITE}'
          f'{(g["title"] or "")[:32]}{tui.RESET}{tui.FAINT} at #{g["position"]}, '
          f'{fmt_n(g["installs"])} installs{tui.RESET}')
    print(f'  {tui.GRAY}{"score gap":<16}{tui.RESET}{tui.WHITE}{r["gap"]:+.2f}{tui.RESET}'
          f'{tui.FAINT} logits to close{tui.RESET}\n')

    print(f'  {tui.FAINT}WHAT IS HOLDING IT BACK{tui.RESET}')
    print(f'  {tui.FAINT}each line: move that one thing to the level of the app '
          f'at #{g["position"]}{tui.RESET}\n')
    if not r["holding_back"]:
        print(f'    {tui.GRAY}nothing single-handedly. The gap is spread across '
              f'many small differences.{tui.RESET}\n')
    for c in r["holding_back"]:
        share = c["gain"] / r["gap"] if r["gap"] > 0 else 0
        bar = tui.bar(min(max(share, 0), 1), 14, tui.color_for(share * 100))
        print(f'    {tui.WHITE}{c["feature"]:<26}{tui.RESET}{bar} '
              f'{tui.WHITE}{c["gain"]:+.2f}{tui.RESET}')
        print(f'      {tui.FAINT}{c["doc"][:58]}{tui.RESET}')
        print(f'      {tui.GRAY}yours {_fmt_feature_value(c["feature"], c["yours"])}'
              f'   theirs {_fmt_feature_value(c["feature"], c["theirs"])}{tui.RESET}')

    if r["in_your_favour"]:
        print(f'\n  {tui.FAINT}ALREADY IN ITS FAVOUR{tui.RESET}\n')
        for c in sorted(r["in_your_favour"], key=lambda x: x["gain"]):
            print(f'    {tui.GREEN}{c["feature"]:<26}{tui.RESET}'
                  f'{tui.GRAY}yours {_fmt_feature_value(c["feature"], c["yours"])}'
                  f'   theirs {_fmt_feature_value(c["feature"], c["theirs"])}{tui.RESET}')

    covered = r["closable"] / r["gap"] if r["gap"] > 0 else 0
    print(f'\n  {tui.FAINT}fixing every listed lever closes {covered:.0%} of the '
          f'gap. The rest is field strength you do not control.{tui.RESET}\n')
    con.close()


def cmd_suggest(a):
    from . import play, tui
    out = play.suggest(a.query, games=a.games, no_cache=a.no_cache)
    if a.json:
        print(json.dumps(out, indent=2))
        return
    if not out:
        print(f'  {tui.AMBER}Play suggests nothing for {a.query!r}{tui.RESET}')
        return
    from . import db as _db
    from . import suggest as _s
    _c = _db.connect()
    age = _s.age_hours(_c, a.query)
    _c.close()
    when = ("just fetched" if a.no_cache or age is None or age < 0.02
            else f"cached {age:.0f}h ago" if age >= 1 else "cached moments ago")
    print(f'\n  {tui.FAINT}{len(out)} suggestions for {a.query!r}, {when}{tui.RESET}\n')
    q = " ".join(a.query.strip().lower().split())
    for i, t in enumerate(out, 1):
        mark = f'{tui.CYAN}▸{tui.RESET}' if t.lower() == q else " "
        print(f'  {mark} {tui.FAINT}{i}{tui.RESET}  {tui.WHITE}{t}{tui.RESET}')
    print()


def cmd_search(a):
    from . import play, tui
    from .analyze import fmt_n
    out = play.search(a.query, with_details=a.details, limit=a.limit)
    if a.json:
        print(json.dumps(out, indent=2))
        return
    ranked = sum(1 for r in out if r["position"])
    print(f'\n  {tui.FAINT}{ranked} organic results for {a.query!r}'
          f'{", plus a promoted card" if ranked < len(out) else ""}{tui.RESET}\n')
    for r in out:
        slot = f'{r["position"]:>3}' if r["position"] else f'{tui.AMBER}ad{tui.RESET} '
        inst = r.get("installs_real") or r.get("installs") or ""
        print(f'  {slot}  {tui.WHITE}{(r.get("title") or "")[:34]:<34}{tui.RESET}'
              f'{tui.GRAY}{str(inst)[:12]:>12}{tui.RESET}  '
              f'{tui.FAINT}{r.get("package", "")[:30]}{tui.RESET}')
    print()


def cmd_details(a):
    from . import play, tui
    d = play.details(a.package)
    if a.json:
        print(json.dumps(d["raw"] if a.raw else d, indent=2, default=str))
        return
    raw, norm = d["raw"], d["normalised"]
    print(f'\n  {tui.BOLD}{tui.WHITE}{raw.get("title")}{tui.RESET}'
          f'{tui.FAINT}  {norm["pkg"]}{tui.RESET}')
    print(f'  {tui.rule()}\n')
    for label, shown, parsed in (
            ("installs", raw.get("installs"), f'{norm["installs"]:,}'),
            ("rating", raw.get("score"), f'{norm["rating"]}'),
            ("ratings", raw.get("ratings_count"), f'{norm["reviews"]:,}'),
            ("released", raw.get("released"), norm["released_at"] or "unparsed"),
            ("updated", raw.get("updated"), norm["updated_at"] or "unparsed")):
        print(f'  {tui.GRAY}{label:<10}{tui.RESET}{tui.WHITE}{str(shown or "-"):<22}'
              f'{tui.RESET}{tui.FAINT}read as {parsed}{tui.RESET}')
    print(f'\n  {tui.GRAY}{"developer":<10}{tui.RESET}{raw.get("developer") or "-"}')
    if raw.get("short_description"):
        print(f'  {tui.GRAY}{"tagline":<10}{tui.RESET}{raw["short_description"][:56]}')
    print()


def cmd_publisher(a):
    from . import play, tui
    out = play.publisher(a.developer)
    if a.json:
        print(json.dumps(out, indent=2))
        return
    print(f'\n  {tui.FAINT}{len(out)} apps for {a.developer!r} '
          f'(Play search caps near 50){tui.RESET}\n')
    for r in out:
        print(f'  {tui.WHITE}{(r.get("title") or "")[:36]:<36}{tui.RESET}'
              f'{tui.FAINT}{r.get("package", "")}{tui.RESET}')
    print(f'\n  {tui.FAINT}for the full catalogue: '
          f'python scripts/fetch_publisher.py "{a.developer}" --json apps.json{tui.RESET}\n')


def cmd_serve(a):
    """Run the API under uvicorn."""
    import socket

    import uvicorn

    from . import api, config, server

    if a.max_concurrent:
        api._gate.resize(a.max_concurrent)
    if a.timeout:
        api._limits["timeout"] = a.timeout

    explicit = a.port is not None
    port = a.port if explicit else server.PORT
    strict = explicit and not a.auto_port

    def free(p):
        with socket.socket() as sk:
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sk.bind((a.host, p))
                return True
            except OSError:
                return False

    if not free(port):
        if strict:
            raise SystemExit(
                f"port {port} is taken.\n"
                f"  You asked for this port specifically, so it was not moved.\n"
                f"  aso serve --port {port} --auto-port   move up if busy\n"
                f"  aso serve                            use the default, "
                f"{server.PORT}")
        found = next((p for p in range(port + 1, port + 21) if free(p)), None)
        if found is None:
            raise SystemExit(f"no free port between {port + 1} and {port + 20}")
        print(f"  port {port} taken, using {found}")
        port = found

    if (a.host not in ("127.0.0.1", "localhost", "::1") and not api.API_TOKEN
            and not os.environ.get("ASO_ALLOW_INSECURE")):
        raise SystemExit(
            f"refusing to bind {a.host} without ASO_API_TOKEN.\n"
            f"  This server scrapes, trains and writes to the database, so a\n"
            f"  reachable port hands a stranger write access.\n\n"
            f"  Set it in .env:   ASO_API_TOKEN=$(openssl rand -hex 24)\n"
            f"  Or, if the port is genuinely private (a container network that\n"
            f"  nothing outside can reach):   ASO_ALLOW_INSECURE=1")

    server.write_runfile(a.host, port)
    print(f"  http://{a.host}:{port}   docs at /docs")
    try:
        uvicorn.run(api.app, host=a.host, port=port, log_level="warning",
                    timeout_keep_alive=65)
    finally:
        server.RUNFILE.unlink(missing_ok=True)


def cmd_train(a):
    from . import train
    con = db.connect()
    train.train(con, country=a.country, k=a.members, epochs=a.epochs, seed=a.seed)
    con.close()


def cmd_status(a):
    con = db.connect()
    q = lambda s: con.execute(s).fetchone()[0]
    stats = {
        "apps": q("SELECT COUNT(*) FROM apps"),
        "keywords": q("SELECT COUNT(DISTINCT keyword) FROM observations"),
        "ranked_observations": q("SELECT COUNT(*) FROM observations WHERE position IS NOT NULL"),
        "corrections": q("SELECT COUNT(*) FROM corrections"),
        "live_residuals": q("SELECT COUNT(*) FROM residuals WHERE retired_at IS NULL"),
        "overrides": q("SELECT COUNT(*) FROM overrides"),
        "models": [dict(r) for r in con.execute(
            "SELECT version, n_rows, golden_auc, golden_ece, active FROM registry "
            "ORDER BY created_at")],
    }
    if a.json:
        print(json.dumps(stats, indent=2))
    else:
        for k in ("apps", "keywords", "ranked_observations", "corrections",
                  "live_residuals", "overrides"):
            print(f"  {k:<20} {stats[k]}")
        for m in stats["models"]:
            print(f'  {"*" if m["active"] else " "} {m["version"]:<5} '
                  f'rows {m["n_rows"]:<6} AUC {m["golden_auc"]:.3f}  '
                  f'ECE {m["golden_ece"]:.3f}')
    con.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="aso", description="Can this app rank for this keyword")
    p.add_argument("--country", default="us")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("analyze", help="everything about one keyword")
    s.add_argument("keyword")
    s.add_argument("--pkg", help="also predict this app's rank for it")
    s.add_argument("--depth", action="store_true",
                   help="full breakdown: field quantiles, the bar, the top of the SERP")
    s.add_argument("--once", action="store_true",
                   help="print and exit, no interactive prompt")
    s.add_argument("--no-server", action="store_true",
                   help="do the work in this process, ignoring any running server")
    s.add_argument("--min-download", type=float, metavar="N",
                   help="override the saved minimum downloads for this call only")
    s.add_argument("--min-download-unit", choices=("day", "month", "year"),
                   help="unit for --min-download (default: your saved unit)")
    s.add_argument("--max-rank", type=int, help="override the saved rank limit")
    s.add_argument("--min-chance", type=int, metavar="0-100",
                   help="override the minimum chance of ranking")
    s.add_argument("--min-certainty", type=int, metavar="0-100",
                   help="override the minimum model certainty")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("correct", help="correct any value the model got wrong")
    s.add_argument("keyword")
    s.add_argument("--pkg")
    s.add_argument("--rank", type=int, help="the rank it actually holds")
    s.add_argument("--demand", type=float, help="0-100")
    s.add_argument("--crowding", type=float, help="0-100")
    s.add_argument("--reviewer", default="you")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_correct)

    s = sub.add_parser("scrape", help="pre-warm several keywords at once")
    s.add_argument("keywords", nargs="+")
    s.add_argument("--fast", action="store_true", help="skip per-app details")
    s.add_argument("--sleep", type=float, default=1.0)
    s.set_defaults(fn=cmd_scrape)

    s = sub.add_parser("ingest", help="import ~/app-idea: keywords we built for")
    s.add_argument("--limit", type=int, help="only the first N keywords")
    s.add_argument("--fast", action="store_true", help="skip per-app details")
    s.add_argument("--sleep", type=float, default=0.8)
    s.add_argument("--from-publisher", metavar="JSON",
                   help="a publisher app list from scripts/fetch_publisher.py; "
                        "each app's title is the keyword it was built for")
    s.add_argument("--dry-run", action="store_true", help="show the plan, scrape nothing")
    s.add_argument("--all", action="store_true",
                   help="re-scrape everything, even keywords already fetched")
    s.add_argument("--fresh-days", type=int, default=7,
                   help="treat a keyword scraped within this many days as done")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("backtest",
                       help="did what we knew BEFORE predict what happened after")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("track", help="re-scrape watched keywords and diff the snapshots")
    s.add_argument("--every", type=int, default=7,
                   help="re-scrape a keyword when its last scrape is this old")
    s.add_argument("--limit", type=int)
    s.add_argument("--gap", type=int, default=3,
                   help="ignore snapshot pairs closer together than this")
    s.add_argument("--show", action="store_true",
                   help="only report what changed, scrape nothing")
    s.add_argument("--sleep", type=float, default=0.7)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_track)

    s = sub.add_parser("config", help="your thresholds: minimum downloads, rank, score")
    s.add_argument("args", nargs="*", metavar="...",
                   help='a setting to read or write, or: min download year 50000')
    s.add_argument("--path", action="store_true", help="print the config file location")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_config)

    s = sub.add_parser("why", help="why an app is not ranking for a keyword")
    s.add_argument("keyword")
    s.add_argument("--pkg", required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_why)

    s = sub.add_parser("suggest", help="Play autocomplete for a query")
    s.add_argument("query")
    s.add_argument("--games", action="store_true")
    s.add_argument("--no-cache", action="store_true",
                   help="fetch fresh and update the cache, ignoring the TTL")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_suggest)

    s = sub.add_parser("search", help="top results for a query, with organic ranks")
    s.add_argument("query")
    s.add_argument("--details", action="store_true", help="also fetch each listing")
    s.add_argument("--limit", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("details", help="full listing for one app")
    s.add_argument("package", help="package name or Play Store URL")
    s.add_argument("--raw", action="store_true", help="Play's fields, unnormalised")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_details)

    s = sub.add_parser("publisher", help="apps by a developer (Play search, caps at ~50)")
    s.add_argument("developer")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_publisher)

    s = sub.add_parser("serve", help="load the model once and answer over HTTP")
    s.add_argument("--host", default="127.0.0.1")
    # No default: whether --port was PASSED is the signal. Naming a port means
    # you want that port, so it is used or the start fails. Saying nothing means
    # you do not care, so a busy default quietly moves up.
    s.add_argument("--port", type=int, metavar="N",
                   help="use this exact port (fails if taken). "
                        "Omit for 8765, which moves up if busy")
    s.add_argument("--max-concurrent", type=int,
                   help="requests served at once (default: config, or 4)")
    s.add_argument("--timeout", type=float,
                   help="seconds before a request returns 504 (default: 180)")
    s.add_argument("--auto-port", action="store_true",
                   help="with --port, allow moving to the next free one anyway")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("train")
    s.add_argument("--members", type=int, default=7)
    s.add_argument("--epochs", type=int, default=400)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
