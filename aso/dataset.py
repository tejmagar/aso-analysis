"""Turn the observation table into a feature matrix.

A rank is a rank: rows from scraped SERPs and rows from apps we published share
one label space and one model. `source` is kept only because it records whether
we can observe an app's FAILURES, which scraped rows never can.
"""
from __future__ import annotations

import math

import numpy as np

from . import db, embed, features, history, intent, suggest

TOP_K = 10          # "ranks" means reaching the top 10
NEW_APP_YEARS = 1.5  # only these carry a meaningful installs-per-year label


def build(con, country: str = "us", top_n: int = TOP_K):
    # One pass for every keyword. Per-keyword reads here fetched the whole
    # observations join once each, which is minutes of network for one epoch.
    by_kw = {kw: rs for kw, rs in db.ranked_fields(con, country).items() if rs}

    feats, labels, vel, groups, meta = [], [], [], [], []
    pages, blind, masks, ranks = [], [], [], []
    for kw, rs in by_kw.items():
        kv = embed.keyword_vec(kw)
        feat_rows = db.featured_apps(con, kw, country)
        vecs = {x["pkg"]: embed.app_vec(x.get("title") or "", x.get("short_desc") or "",
                              x.get("description") or "")
                for x in rs + feat_rows}
        sg = suggest.signals(con, kw, country)     # reads stored rows only
        sg = {**sg, **history.deltas(con, kw, country,
                                     features.compute_field(rs, kw, top_n=top_n),
                                     rows_today=rs)}
        split = intent.split(rs, vecs, kv)
        for r in rs:
            fld = features.compute_field(
                rs, kw, top_n=top_n, exclude_pkg=r["pkg"], featured=feat_rows,
                kw_vec=kv, app_vecs=vecs,
                intent_group=intent.group_for(split, r["pkg"], vecs.get(r["pkg"])))
            if fld["n"] == 0:
                continue                       # nothing to compare against
            av = embed.app_vec(r.get("title") or "", r.get("short_desc") or "",
                              r.get("description") or "")
            # The page this row was scored against, one row per app. Same
            # leave-one-out and same rank as the aggregates above, so the two
            # views of the field can never disagree about what was on it.
            grp = intent.group_for(split, r["pkg"], vecs.get(r["pkg"]))
            px, pm = features.page_matrix(
                rs, kw, kw_vec=kv, app_vecs=vecs, intent_group=grp,
                exclude_pkg=r["pkg"], at_rank=r["position"])
            # The same page with the rank withheld.
            #
            # Serving asks the two questions in two passes: the rank head runs
            # first, when the app has no rank yet, and only then can the
            # downloads head be asked "at THAT rank". So the rank task is
            # trained on the page without rank_gap and the downloads task on
            # the page with it, which is exactly the pair serving produces. Feed
            # both heads the ranked page and every serving rank prediction is
            # made on an input shape that never occurred in training.
            bx, _ = features.page_matrix(
                rs, kw, kw_vec=kv, app_vecs=vecs, intent_group=grp,
                exclude_pkg=r["pkg"], at_rank=None)
            pages.append(px)
            blind.append(bx)
            masks.append(pm)
            # The rank this row actually held, on the same scale serving uses
            # for the rank a new app would enter at.
            ranks.append((r["position"] - 1) / 10.0)
            feats.append(features.extract(r, kw, fld, kv, av, sg))
            labels.append(1.0 if r["position"] <= top_n else 0.0)
            # The downloads label: what this app achieved per year since release.
            #
            # Only NEW apps carry it. For an eight-year-old app the same
            # arithmetic gives a LIFETIME average earned over years of already
            # ranking - median 168,848/yr against 4,947/yr for a newcomer, a 34x
            # gap. Training on both taught the head to answer "what do apps in
            # this slot have" when the question is "what would a new app get",
            # and it forecast 57,000 downloads a day for an unlaunched app.
            #
            # NaN is masked out of the loss, so old rows still train the RANK
            # head and simply contribute nothing to the downloads head.
            age = features._days_since(r.get("released_at")) / 365.0
            # The same floor the features use. At a quarter year this label told
            # the head that a one-month-old app earning ten a day earned three,
            # so the thing it was fitted to predict was wrong for every app in
            # the range that matters most.
            vel.append(math.log1p(features._rate_of(r))
                       if 0 < age <= NEW_APP_YEARS else float("nan"))
            groups.append(kw)
            meta.append({"pkg": r["pkg"], "keyword": kw, "position": r["position"],
                         "source": r["source"],
                         # Age is what makes a row an experiment, not ownership.
                         "age_years": age})
    if not feats:
        return None
    xm, xf = features.vectorize(feats)
    return {"xm": xm, "xf": xf, "y": np.array(labels, "float32"),
            "v": np.array(vel, "float32"),
            "xs": np.stack(pages), "xs_blind": np.stack(blind),
            "mask": np.stack(masks), "rank": np.array(ranks, "float32"),
            "groups": np.array(groups), "meta": meta, "feats": feats}


def group_split(groups: np.ndarray, holdout: float = 0.25, seed: int = 0):
    """Split by KEYWORD, never by row. A random row split puts the same SERP on
    both sides and reports an accuracy that will not survive a new keyword."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_hold = max(1, int(len(uniq) * holdout))
    hold = set(uniq[:n_hold])
    mask = np.array([g in hold for g in groups])
    return ~mask, mask


def fixed_holdout(con, groups, country="us", frac=0.25, seed=0):
    """A held-out keyword set chosen ONCE and reused for every model.

    Re-drawing the split each run made the gate meaningless: v15 scored 0.731 on
    one set of keywords and v28 scored 0.594 on a different set, and comparing
    them decided promotion on which keywords happened to be held out. Fixing the
    set makes successive numbers comparable, which is the whole point of a gate.

    Keywords added later join TRAIN, never the evaluation set, so the test stays
    the same test as the dataset grows.
    """
    have = {r["keyword"] for r in con.execute(
        "SELECT keyword FROM holdout WHERE country=%s", (country,))}
    uniq = list(np.unique(groups))
    if not have:
        rng = np.random.default_rng(seed)
        pick = list(rng.permutation(uniq)[:max(1, int(len(uniq) * frac))])
        with con.transaction():
            for kw in pick:
                con.execute("INSERT INTO holdout (keyword, country, chosen_at) "
                            "VALUES (%s,%s,%s) ON CONFLICT (keyword, country) "
                            "DO NOTHING", (str(kw), country, db.now()))
        have = set(map(str, pick))
    mask = np.array([g in have for g in groups])
    return ~mask, mask
