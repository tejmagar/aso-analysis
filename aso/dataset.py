"""Turn the observation table into a feature matrix.

A rank is a rank: rows from scraped SERPs and rows from apps we published share
one label space and one model. `source` is kept only because it records whether
we can observe an app's FAILURES, which scraped rows never can.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import db, embed, features, history, intent, suggest

TOP_K = 10          # "ranks" means reaching the top 10


def build(con, country: str = "us", top_n: int = TOP_K):
    kws = [r["keyword"] for r in con.execute(
        "SELECT DISTINCT keyword FROM observations WHERE country=?", (country,))]
    by_kw: dict[str, list[dict]] = {}
    for kw in kws:
        rs = db.ranked_field(con, kw, country)   # consistent slots, no collisions
        if rs:
            by_kw[kw] = rs

    feats, labels, vel, groups, meta = [], [], [], [], []
    for kw, rs in by_kw.items():
        kv = embed.keyword_vec(kw)
        feat_rows = db.featured_apps(con, kw, country)
        vecs = {x["pkg"]: embed.app_vec(x.get("title") or "", x.get("short_desc") or "",
                              x.get("description") or "")
                for x in rs + feat_rows}
        sg = suggest.signals(con, kw, country)     # reads stored rows only
        sg = {**sg, **history.deltas(con, kw, country,
                                     features.compute_field(rs, kw, top_n=top_n))}
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
            feats.append(features.extract(r, kw, fld, kv, av, sg))
            labels.append(1.0 if r["position"] <= top_n else 0.0)
            # The downloads label, free from data already scraped: what this app
            # actually achieved per year since release. NaN when the release
            # date is unknown, and masked out of the loss rather than guessed.
            age = features._days_since(r.get("released_at")) / 365.0
            vel.append(math.log1p((r.get("installs") or 0) / max(age, 0.25))
                       if age > 0 else float("nan"))
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
        "SELECT keyword FROM holdout WHERE country=?", (country,))}
    uniq = list(np.unique(groups))
    if not have:
        rng = np.random.default_rng(seed)
        pick = list(rng.permutation(uniq)[:max(1, int(len(uniq) * frac))])
        with con:
            for kw in pick:
                con.execute("INSERT OR IGNORE INTO holdout (keyword, country, "
                            "chosen_at) VALUES (?,?,?)", (str(kw), country, db.now()))
        have = set(map(str, pick))
    mask = np.array([g in have for g in groups])
    return ~mask, mask
