"""Why an app did not rank, asked of the model rather than asserted.

The method is counterfactual, not a weighting: take the app as it is, replace
ONE feature with the value the rank-10 holder has, and re-score through the real
network. The change in logit is that feature's contribution to the gap. Repeat
for every feature and sort.

That works because the model is monotone in the features that matter: raising
match quality or installs can only raise the score, so a positive delta means
"this is genuinely holding you back" rather than an artefact of a linear
approximation around a point.

Nothing here decides what to fix. It reports how far each lever would move the
score if pulled all the way to the level of the app currently holding the last
qualifying slot.
"""
from __future__ import annotations

import numpy as np
import torch

from . import db, embed, features, intent, predict, suggest, train
from .config import get as cfg


def _feature_vectors(con, keyword: str, pkg: str, country: str = "us"):
    """Feature rows for the candidate and for every ranked app, built the same way."""
    rows = predict.field_rows(con, keyword, country)
    if not rows:
        raise SystemExit(f"no SERP on file for {keyword!r}")
    app = con.execute("SELECT * FROM apps WHERE pkg=?", (pkg,)).fetchone()
    if app is None:
        raise SystemExit(f"unknown app {pkg!r}")
    app = dict(app)

    top_k = int(cfg("top_k", 10))
    ranked = sorted((r for r in rows if r.get("position")), key=lambda r: r["position"])
    kv = embed.keyword_vec(keyword)
    featured = db.featured_apps(con, keyword, country)
    vecs = {r["pkg"]: embed.app_vec(r.get("title") or "", r.get("short_desc") or "",
                                    r.get("description") or "")
            for r in rows + featured + [app]}
    sg = suggest.signals(con, keyword, country)
    split = intent.split(rows, vecs, kv)

    cand = predict._feat_for(app, keyword, rows, kv, featured, vecs, sg, split)
    field = [(r, predict._feat_for(r, keyword, rows, kv, featured, vecs, sg, split))
             for r in ranked]
    # The app you actually have to displace: whoever holds the last slot that counts.
    gate = next((x for x in field if x[0]["position"] >= top_k), field[-1] if field else None)
    return app, cand, field, gate, ranked, top_k


def why(con, keyword: str, pkg: str, country: str = "us", top_n: int = 6) -> dict:
    model, blob, version = train.load_active(con)
    app, cand, field, gate, ranked, top_k = _feature_vectors(con, keyword, pkg, country)
    if gate is None:
        raise SystemExit("nothing ranked to compare against")

    sm = features.Scaler.load(blob["scaler_mono"])
    sf = features.Scaler.load(blob["scaler_free"])

    def score(feat: dict) -> float:
        xm, xf = features.vectorize([feat])
        with torch.no_grad():
            return float(model.mean_logit(torch.from_numpy(sm.transform(xm)),
                                          torch.from_numpy(sf.transform(xf))))

    base = score(cand)
    target = score(gate[1])

    # One feature at a time, moved to the gatekeeper's value.
    contributions = []
    for f in features.REGISTRY:
        if cand[f.name] == gate[1][f.name]:
            continue
        probe = dict(cand)
        probe[f.name] = gate[1][f.name]
        contributions.append({
            "feature": f.name, "doc": f.doc, "direction": f.direction,
            "yours": cand[f.name], "theirs": gate[1][f.name],
            "gain": score(probe) - base,
        })
    contributions.sort(key=lambda c: -c["gain"])

    # And everything at once, to show how much of the gap the levers actually explain.
    all_probe = dict(cand)
    for c in contributions:
        if c["gain"] > 0:
            all_probe[c["feature"]] = c["theirs"]

    return {
        "keyword": keyword, "pkg": pkg, "title": app.get("title"),
        "model_version": version,
        "your_score": base, "needed": target, "gap": target - base,
        "ranked_at": next((r["position"] for r in ranked if r["pkg"] == pkg), None),
        "gate": {"pkg": gate[0]["pkg"], "title": gate[0].get("title"),
                 "position": gate[0]["position"],
                 "installs": gate[0].get("installs")},
        "holding_back": [c for c in contributions if c["gain"] > 0][:top_n],
        "in_your_favour": [c for c in contributions if c["gain"] < 0][-3:],
        "closable": score(all_probe) - base,
        "top_k": top_k,
    }
