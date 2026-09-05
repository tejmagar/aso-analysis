"""Scoring, and turning one logit into a predicted RANK.

The model emits a single score per (app, keyword). To get a rank out of it, score
every app currently ranking for the keyword with the same function, insert the
candidate, and read off where it lands. That is more useful than a probability
and it is what makes crowding and fit comparable: they are the same scale.
"""
from __future__ import annotations

import json
import math
import uuid

import numpy as np
import torch

from . import (db, downloads, embed, features, history, intent, memory, suggest,
               train)
from .dataset import TOP_K


def field_rows(con, keyword: str, country="us") -> list[dict]:
    return db.ranked_field(con, keyword, country)


def scale_all(sm, sf, V: np.ndarray) -> np.ndarray:
    """Scale a raw [mono | free] matrix with the CURRENT model's scaler.
    Residuals are stored raw precisely so this can be reapplied after the
    scaler moves in a retrain."""
    k = len(features.MONO)
    return np.concatenate([sm.transform(V[:, :k]), sf.transform(V[:, k:])], axis=1)


def _logits(model, sm, sf, feats, ss=None, pages=None, masks=None) -> np.ndarray:
    xm, xf = features.vectorize(feats)
    xs, mk = _pages(ss, pages, masks)
    with torch.no_grad():
        return model.mean_logit(torch.from_numpy(sm.transform(xm)),
                                torch.from_numpy(sf.transform(xf)), xs, mk).numpy()


def _pages(ss, pages, masks):
    """Scale a batch of page matrices the way training did, padding kept inert.

    Returns (None, None) for a model saved before the page encoder existed, so
    an older checkpoint in the registry still loads and still answers.
    """
    if ss is None or not pages:
        return None, None
    xs = np.stack(pages)
    mk = np.stack(masks).astype("float32")
    xs = (ss.transform(xs.reshape(-1, xs.shape[-1]))
          .reshape(xs.shape) * mk[..., None])
    return torch.from_numpy(xs), torch.from_numpy(mk)


def _page_for(row, keyword, rows, kv, vecs=None, split=None, at_rank=None):
    grp = (intent.group_for(split, row.get("pkg"), vecs.get(row.get("pkg")) if vecs else None)
           if split else None)
    return features.page_matrix(rows, keyword, kw_vec=kv, app_vecs=vecs,
                                intent_group=grp, exclude_pkg=row.get("pkg"),
                                at_rank=at_rank)


def _set_scaler(blob):
    st = blob.get("scaler_set")
    return features.Scaler.load(st) if st else None


def _feat_for(row, keyword, rows, kv, featured=None, vecs=None, sugg=None, split=None,
              at_rank=None):
    grp = (intent.group_for(split, row.get("pkg"), vecs.get(row.get("pkg")) if vecs else None)
           if split else None)
    fld = features.compute_field(rows, keyword, top_n=TOP_K, exclude_pkg=row.get("pkg"),
                                 featured=featured, kw_vec=kv, app_vecs=vecs,
                                 intent_group=grp, at_rank=at_rank)
    av = embed.app_vec(row.get("title") or "", row.get("short_desc") or "",
                      row.get("description") or "")
    return features.extract(row, keyword, fld, kv, av, sugg)


def score(con, pkg: str, keyword: str, country="us", record=True):
    model, blob, version = train.load_active(con)
    if model is None:
        raise SystemExit("no model yet. Run: aso train")

    keyword = keyword.strip().lower()
    app = con.execute("SELECT * FROM apps WHERE pkg=%s", (pkg,)).fetchone()
    if app is None:
        raise SystemExit(f"unknown app {pkg!r}. Scrape a keyword it ranks for first.")
    app = dict(app)

    rows = field_rows(con, keyword, country)
    if not rows:
        raise SystemExit(f"no SERP for {keyword!r}")
    ranked = sorted((r for r in rows if r.get("position")), key=lambda r: r["position"])

    sm = features.Scaler.load(blob["scaler_mono"])
    sf = features.Scaler.load(blob["scaler_free"])
    kv = embed.keyword_vec(keyword)

    # Score the ENTIRE ranked field, then the candidate, in one batched pass.
    # The field is scored candidate-independently on purpose: crowding describes
    # the keyword, not the matchup, so it stays stable across candidates and can
    # be cached per keyword. Only predicted_rank drops the candidate's own row.
    featured = db.featured_apps(con, keyword, country)
    sg = suggest.signals(con, keyword, country)
    vecs = {r["pkg"]: embed.app_vec(r.get("title") or "", r.get("short_desc") or "",
                              r.get("description") or "")
            for r in rows + featured + [app]}
    split = intent.split(rows, vecs, kv)
    ss = _set_scaler(blob)
    feats = [_feat_for(r, keyword, rows, kv, featured, vecs, sg, split) for r in ranked]
    # Rank-blind, for the field and the candidate alike.
    #
    # The rank head trains on the page with rank_gap withheld, because the app
    # being asked about has no rank yet at serving time. Handing the field its
    # own positions did two things at once: it fed the head an input pairing it
    # never saw in training, and it leaked each app's own rank into its own rank
    # prediction. The field's logits came out inflated - a real app on 17
    # installs scoring 33 against a ghost on the same relevance scoring 10 - so
    # the comparison the entire answer rests on was between two numbers computed
    # different ways. It shows up worst on small pages, where a handful of
    # inflated rivals is the whole field.
    pages = [_page_for(r, keyword, rows, kv, vecs, split) for r in ranked]
    cand_feat = _feat_for(app, keyword, rows, kv, featured, vecs, sg, split)
    cand_page = _page_for(app, keyword, rows, kv, vecs, split)
    all_logits = _logits(model, sm, sf, feats + [cand_feat], ss,
                         [x for x, _ in pages] + [cand_page[0]],
                         [m for _, m in pages] + [cand_page[1]])
    field_logits, base_logit = all_logits[:-1], float(all_logits[-1])
    inc_logits = np.array([lg for lg, r in zip(field_logits, ranked)
                           if r["pkg"] != pkg], dtype="float32")

    # T0 residual memory: reviewer corrections, applied before anything is ranked.
    xm, xf = features.vectorize([cand_feat])
    raw = np.concatenate([xm[0], xf[0]])
    X, r, cw, keys = memory.load_active(con, len(features.REGISTRY))
    resid = (memory.read(scale_all(sm, sf, raw[None, :])[0], scale_all(sm, sf, X),
                         r, cw, keys, exact=(keyword, pkg)) if X.size else 0.0)
    logit = base_logit + resid

    predicted_rank = 1 + int((inc_logits > logit).sum())
    chance = float(torch.sigmoid(torch.tensor(logit)))

    # Crowding: the score needed to hold the last qualifying slot of the field
    # as it actually stands, computed over every ranked app including this one.
    field_order = np.sort(field_logits)[::-1]
    gate_logit = (float(field_order[min(TOP_K, len(field_order)) - 1])
                  if len(field_order) else None)
    q = blob["meta"]["logit_q"]
    to_pct = lambda v: None if v is None else int(np.searchsorted(np.asarray(q), v))

    override = db.get_override(con, keyword, "crowding", country)
    out = {
        "prediction_id": "p_" + uuid.uuid4().hex[:8],
        "model_version": version, "pkg": pkg, "keyword": keyword, "country": country,
        "predicted_rank": predicted_rank,
        "field_size": len(ranked),
        "chance": chance,
        "fit": to_pct(logit),
        "crowding": override if override is not None else to_pct(gate_logit),
        "crowding_overridden": override is not None,
        "residual": resid, "logit": logit, "base_logit": base_logit,
        "features": cand_feat, "raw": raw.tolist(),
        "inc_logits": sorted(inc_logits.tolist(), reverse=True),
    }
    if record:
        with con.transaction():
            con.execute(
                "INSERT INTO predictions (id, ts, model_version, pkg, keyword, country, "
                "chance, uncertainty, crowding, fit, logit, features_json) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (out["prediction_id"], db.now(), version, pkg, keyword, country,
                 chance, 0.0, out["crowding"], out["fit"], logit,
                 json.dumps(out["raw"])))
    return out


def verdict(rank: int, top_k: int = TOP_K) -> str:
    if rank <= 3:
        return "strong"
    if rank <= top_k:
        return "winnable"
    if rank <= top_k * 3:
        return "borderline"
    return "unlikely"


def hypothetical_app(keyword: str) -> dict:
    """A well-optimized brand new app: keyword in the title, nothing else.

    Scoring THIS against the live field is how the tool answers "is this worth
    building for". No formula combines difficulty into a score; the network is
    asked the question directly and answers with a rank.
    """
    return {
        "pkg": "__hypothetical__",
        "title": keyword.title(),
        "short_desc": keyword,
        "description": f"{keyword}. " * 8,
        "installs": 0, "rating": 0.0, "reviews": 0,
        "released_at": db.date.today().isoformat(),
        "updated_at": db.date.today().isoformat(),
    }


def _downloads_caveats(rows, keyword: str, points, at_rank: int) -> list[str]:
    """Why the DOWNLOADS figure may not be worth much on this page.

    Downloads only. The rank answer is not qualified here and keeps its own
    certainty, because rank is the question this page can answer: it shows who
    is already there, which is exactly the evidence a position needs. What it
    cannot show is whether anyone SEARCHES the phrase, and that is what a
    download figure rests on.

    These are FACTS about the evidence, checked by inspection, not a confidence
    score. That distinction is deliberate. A percentage is a claim about how
    often the answer is wrong, and the held-out set cannot support one for the
    case that matters: every row in it is an app that already exists, with a
    position and an install count. A brand new app entering at the top of a page
    where nothing matches the phrase never occurs in training, so any number
    attached to it would be extrapolated from apps in a different situation. An
    ensemble-spread confidence was built and dropped for exactly this reason: it
    correlated +0.02 with actual error.

    A sentence the reader can check against the page in front of them is worth
    more than a number they cannot.
    """
    why = []
    if points and not any(rank < at_rank for _, rank, _ in points):
        why.append("nothing here ranks above your slot, so this rate is "
                   "extended past the apps we can read rather than taken from "
                   "between them")

    # Play returns something for every query. When none of it carries the
    # phrase, what came back is a fallback - apps answering a NEIGHBOURING
    # question - and their download rates describe that question, not this one.
    ranked = [r for r in rows if r.get("position")]
    if ranked and not any(features.exact_match(r.get("title") or "", keyword)
                          for r in ranked):
        why.append(f"none of the {len(ranked)} apps here carry the phrase in "
                   "their title, so the page shows what Play falls back to "
                   "rather than anyone competing for this search")
    return why


def score_entry(con, keyword: str, country="us"):
    """Where a well-built new app would land. The model's answer, unmediated."""
    model, blob, version = train.load_active(con)
    if model is None:
        return None

    keyword = keyword.strip().lower()
    rows = field_rows(con, keyword, country)
    ranked = sorted((r for r in rows if r.get("position")), key=lambda r: r["position"])
    if not ranked:
        return None

    sm = features.Scaler.load(blob["scaler_mono"])
    sf = features.Scaler.load(blob["scaler_free"])
    kv = embed.keyword_vec(keyword)
    featured = db.featured_apps(con, keyword, country)
    ghost = hypothetical_app(keyword)
    vecs = {r["pkg"]: embed.app_vec(r.get("title") or "", r.get("short_desc") or "",
                              r.get("description") or "")
            for r in rows + featured + [ghost]}

    sg = suggest.signals(con, keyword, country)
    sg = {**sg, **history.deltas(con, keyword, country,
                                 features.compute_field(rows, keyword, top_n=TOP_K),
                                 rows_today=rows)}
    split = intent.split(rows, vecs, kv)
    feats = [_feat_for(r, keyword, rows, kv, featured, vecs, sg, split) for r in ranked]
    ss = _set_scaler(blob)
    # Rank-blind for the field, exactly as the ghost is. See score().
    pages = [_page_for(r, keyword, rows, kv, vecs, split) for r in ranked]
    ghost_feat = _feat_for(ghost, keyword, rows, kv, featured, vecs, sg, split)
    ghost_page = _page_for(ghost, keyword, rows, kv, vecs, split)
    lg = _logits(model, sm, sf, feats + [ghost_feat], ss,
                 [x for x, _ in pages] + [ghost_page[0]],
                 [m for _, m in pages] + [ghost_page[1]])
    field_logits, ghost_logit = lg[:-1], float(lg[-1])

    order = np.sort(field_logits)[::-1]
    gate = float(order[min(TOP_K, len(order)) - 1]) if len(order) else None
    q = blob["meta"]["logit_q"]
    gm = torch.from_numpy(sm.transform(features.vectorize([ghost_feat])[0]))
    gf = torch.from_numpy(sf.transform(features.vectorize([ghost_feat])[1]))
    gs, gk = _pages(ss, [ghost_page[0]], [ghost_page[1]])

    # Second pass, for the downloads question only.
    #
    # "How many would I get" is really "how many would I get AT THAT RANK", and
    # the rank was not known when the field features were first built. Every app
    # already in the SERP is scored against its own rank, so scoring the ghost
    # against no rank at all was the one place train and serve disagreed. The
    # rank head is NOT re-run on these: it answered already, and feeding its own
    # answer back to it would be circular.
    entry_rank = 1 + int((field_logits > ghost_logit).sum())
    ranked_feat = _feat_for(ghost, keyword, rows, kv, featured, vecs, sg, split,
                            at_rank=entry_rank)
    ranked_page = _page_for(ghost, keyword, rows, kv, vecs, split, at_rank=entry_rank)
    rm = torch.from_numpy(sm.transform(features.vectorize([ranked_feat])[0]))
    rf = torch.from_numpy(sf.transform(features.vectorize([ranked_feat])[1]))
    rs, rk = _pages(ss, [ranked_page[0]], [ranked_page[1]])
    entry = torch.tensor([(entry_rank - 1) / 10.0], dtype=torch.float32)
    with torch.no_grad():
        v_mean, v_std = model.velocity(rm, rf, rs, rk, entry)
        agree = float(model.agreement(gm, gf, gs, gk))
    # Confidence is agreement TIMES evidence. Ensemble agreement alone reads
    # 1.00 on a model fitted to four keywords, because seven members bootstrapped
    # from the same handful of rows converge on the same answer while knowing
    # almost nothing. Reporting that as certainty would be the most misleading
    # number in the tool.
    n_kw = blob["meta"].get("n_keywords", 0)
    evidence = min(n_kw / 25.0, 1.0)
    conf = agree * evidence
    # The head predicts log1p(installs/year) for a NEW app. Cap at the largest
    # first-year figure the training data actually contains: beyond that the
    # model is extrapolating past anything it has seen, and expm1 overflows to
    # inf above ~709 regardless.
    LOG_CAP = blob["meta"].get("velocity_max", math.log1p(5e7))
    SPREAD_CAP = 2.0        # the ensemble's std, bounded so one wild member
                            # cannot turn the range into "3.8K to 10 billion"

    def undo(x):
        return float(np.expm1(min(max(x, 0.0), LOG_CAP)))

    spread = min(float(v_std), SPREAD_CAP)
    per_year = undo(float(v_mean))
    lo = undo(float(v_mean) - spread)
    hi = undo(float(v_mean) + spread)

    # If this checkpoint carries the page-reading downloads model, it answers
    # instead. It is asked about an app shipping TODAY at the rank the model
    # just predicted, against the page as it was scraped - the same shape it was
    # trained on, with the release date as the coordinate rather than an age, so
    # "today" is an ordinary point on the line and not the edge of one.
    answered_by = "head"
    dl_why: list[str] = []
    dl_states = blob.get("downloads")
    if dl_states:
        today = downloads.stamp(db.date.today().isoformat())
        pts = downloads.page_points(rows, near_rank=entry_rank)
        if pts:
            dx, dmask = downloads.describe(pts, today, entry_rank, today)
            dmean, dspread = downloads.Ensemble.load(dl_states)(dx[None], dmask[None])
            ds = min(float(dspread[0]), SPREAD_CAP)
            per_year = undo(float(dmean[0]))
            lo = undo(float(dmean[0]) - ds)
            hi = undo(float(dmean[0]) + ds)
            spread = ds
            answered_by = "page"
            dl_why = _downloads_caveats(rows, keyword, pts, entry_rank)
    return {
        "entry_rank": entry_rank,
        "downloads": {
            "per_year": per_year, "per_month": per_year / 12.0,
            "per_day": per_year / 365.0,
            "low_per_year": lo, "high_per_year": hi,
            "answered_by": answered_by,
            # Why this figure may not be worth much, in terms the reader can
            # check. Downloads only - the rank answer above is not qualified,
            # and keeps its own certainty.
            "uncertain_because": dl_why,
        },
        "confidence": conf, "agreement": agree, "evidence": evidence,
        "field_size": len(ranked),
        "chance": float(torch.sigmoid(torch.tensor(ghost_logit))),
        "crowding": int(np.searchsorted(np.asarray(q), gate)) if gate is not None else None,
        "model_version": version,
    }
