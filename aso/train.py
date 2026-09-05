"""Train the ensemble, gate it on a held-out keyword split, register it.

Nothing is promoted that regresses the golden split, however many corrections it
satisfies. That gate is the only thing standing between a helpful feedback loop
and a model that agrees with whoever used the CLI most.
"""
from __future__ import annotations

import json as _json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import dataset, db, features, memory, metrics
from .model import Ensemble, n_params

# Where trained checkpoints live. Overridable because in a container the repo
# directory is part of the image: writing there means every rebuild silently
# discards every model ever trained, and leaves the registry pointing at paths
# that no longer exist. Point this at a volume.
MODELS = Path(os.environ.get(
    "ASO_MODELS", Path(__file__).resolve().parent.parent / "data" / "models"))

# The checkpoint that ships with the repo. Versioned models live under data/
# (gitignored working history); this one is committed so a fresh clone can
# predict immediately instead of needing 150 keywords scraped first.
SHIPPED = Path(__file__).resolve().parent.parent / "models" / "latest.pt"

# A held-out split needs this many keywords before its score means anything.
# Below it the model still trains and still predicts; there is simply nothing to
# validate against, so the regression gate is not applied yet.
MIN_KEYWORDS = 25
MIN_AUC = 0.55        # only enforced once the split is measurable


def _fit_member(model, xm, xf, y, v, w, xs=None, xb=None, mask=None, rank=None,
                epochs=400, lr=0.05, seed=0, on_epoch=None):
    """Two heads, one trunk. The rank task is the objective; the downloads task
    is a second signal over the same representation, which regularises it."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    known = ~torch.isnan(v)              # release date missing: no label, no gradient
    vt = torch.nan_to_num(v)
    for i in range(epochs):
        opt.zero_grad()
        # Rank on the rank-blind page, downloads on the ranked one: the two
        # inputs serving actually builds, in the same order.
        loss = (F.binary_cross_entropy_with_logits(model(xm, xf, xb, mask), y,
                                                   reduction="none") * w).mean()
        if known.any():
            loss = loss + 0.3 * F.mse_loss(
                model.velocity(xm, xf, xs, mask, rank)[known], vt[known])
        loss.backward()
        opt.step()
        # Report every 50 epochs rather than every one: the callback writes to
        # disk, and a run should not spend its time telling you about itself.
        if on_epoch and i % 50 == 0:
            on_epoch(i, float(loss.detach()))
    return float(loss.detach())


def train(con, country="us", k=7, hidden=24, epochs=400, seed=0, verbose=True,
          progress=None):
    """`progress(phase, done, total, note)` is called as the run advances.

    Training takes ~45 seconds and is started from a chat where "working on it"
    with no further word is indistinguishable from "hung". The callback is what
    makes the run observable while it happens rather than only when it lands.
    """
    def say(phase, done=0, total=1, note=""):
        if progress:
            progress(phase, done, total, note)

    say("reading", 0, 1, "building the feature matrix")
    data = dataset.build(con, country)
    if data is None:
        raise SystemExit("no training rows yet. Scrape a few keywords first.")

    # A FIXED evaluation set, not a fresh draw each run: see fixed_holdout().
    tr, ho = dataset.fixed_holdout(con, data["groups"], country=country, seed=seed)
    if tr.sum() == 0 or ho.sum() == 0:
        tr = ho = np.ones(len(data["y"]), dtype=bool)     # too few keywords to split
        if verbose:
            print("! only one keyword group: golden split is not meaningful yet")

    sm = features.Scaler().fit(data["xm"][tr])
    sf = features.Scaler().fit(data["xf"][tr])
    XM, XF = sm.transform(data["xm"]), sf.transform(data["xf"])
    # The page rows are scaled per column over the REAL apps only. Fitting over
    # the zero padding too would drag every mean toward zero by however many
    # slots happened to be empty, which is a property of short pages and not of
    # apps. Padding is re-zeroed after transform so it stays inert.
    XS, XB, MK = data["xs"], data["xs_blind"], data["mask"]
    real = MK[tr].reshape(-1) > 0.5
    ss = features.Scaler().fit(XS[tr].reshape(-1, XS.shape[-1])[real])

    def _scale_pages(A):
        return (ss.transform(A.reshape(-1, A.shape[-1]))
                .reshape(A.shape) * MK[..., None])

    XS, XB = _scale_pages(XS), _scale_pages(XB)
    y = data["y"]

    # Weight by what a row actually demonstrates, not by who owns the app.
    #
    # The question this model answers is "would a NEW app rank here", so the
    # rows that answer it are recently released apps - whoever built them. A
    # ten-year-old incumbent at rank 3 says little about entering today; an app
    # three months old at rank 3 says everything, and there are 25x more of
    # those in the data than there are rows where ownership was known at all.
    #
    # Ownership survives only for what it uniquely proves: an app that never
    # appears at all. A scraped page cannot show an app that is not on it, so
    # "we published for this and it is absent" stays the one label scraping
    # cannot produce.
    # Recency counts continuously, not as a step. A step at eighteen months
    # weighted an app published last quarter exactly like one published a year
    # and a half ago, when the first is far better evidence about what entering
    # this page pays now: it earned its installs under the demand that exists
    # today, and the second earned them under whatever held a year ago.
    #
    # Halving every year, floored at 1. The newest rows carry about four times
    # the weight of an established app, a two-year-old entrant about 1.5, and
    # nothing is discarded, because an old app at rank 3 is still evidence about
    # what holds rank 3.
    HALF_LIFE = 1.0
    w = np.ones(len(data["y"]), dtype="float32")
    for i, m in enumerate(data["meta"]):
        age = m.get("age_years") or 0.0
        if age > 0:
            w[i] = 1.0 + 3.0 * float(0.5 ** (age / HALF_LIFE))
        if m["source"] == "review":
            w[i] = max(w[i], 5.0)            # a human confirmed this rank
        elif m["source"] == "owned":
            w[i] = max(w[i], 3.0)            # observed absence, unobtainable otherwise

    say("fitting", 0, k, f"{len(y)} rows across "
        f"{len(np.unique(data['groups']))} keywords")
    model = Ensemble(XM.shape[1], XF.shape[1], k=k, hidden=hidden,
                     n_app=XS.shape[-1])
    rng = np.random.default_rng(seed)
    idx_tr = np.flatnonzero(tr)
    losses = []
    for i, member in enumerate(model.members):
        say("fitting", i, k, f"member {i + 1} of {k}")
        boot = rng.choice(idx_tr, size=len(idx_tr), replace=True)   # bootstrap resample
        losses.append(_fit_member(member,
                    torch.from_numpy(XM[boot]), torch.from_numpy(XF[boot]),
                    torch.from_numpy(y[boot]), torch.from_numpy(data["v"][boot]),
                    torch.from_numpy(w[boot].astype("float32")),
                    xs=torch.from_numpy(XS[boot]), xb=torch.from_numpy(XB[boot]),
                    mask=torch.from_numpy(MK[boot]),
                    rank=torch.from_numpy(data["rank"][boot]),
                    epochs=epochs, lr=0.05, seed=seed + i,
                    on_epoch=lambda e, l, _i=i: say(
                        "fitting", _i, k, f"member {_i + 1} of {k}, epoch {e}")))

    say("scoring", k, k, "measuring against the held-out keywords")
    model.eval()
    with torch.no_grad():
        p_ho, _ = model(torch.from_numpy(XM[ho]), torch.from_numpy(XF[ho]),
                        torch.from_numpy(XB[ho]), torch.from_numpy(MK[ho]))
        logits_all = model.mean_logit(
            torch.from_numpy(XM), torch.from_numpy(XF),
            torch.from_numpy(XB), torch.from_numpy(MK)).numpy()
    p_ho = p_ho.numpy()
    g_auc, g_ece = metrics.auc(y[ho], p_ho), metrics.ece(y[ho], p_ho)

    # The downloads head, measured on the same held-out keywords.
    #
    # Nothing scored this before, so a run could be reported as good on the
    # strength of its rank AUC while the downloads number was wandering by
    # multiples between retrains. It wanders because the head regresses
    # log1p(installs per year) and the answer is exponentiated back: an error of
    # 1.4 in log space is a four-fold error on screen, which is invisible in any
    # loss printed in log units. So the honest figure is a RATIO - typically how
    # many times out the answer is - and that is what gets recorded.
    with torch.no_grad():
        v_ho, v_spread = model.velocity(
            torch.from_numpy(XM[ho]), torch.from_numpy(XF[ho]),
            torch.from_numpy(XS[ho]), torch.from_numpy(MK[ho]),
            torch.from_numpy(data["rank"][ho]))
    v_true = data["v"][ho]
    lab = ~np.isnan(v_true)
    if lab.sum():
        err = np.abs(v_ho.numpy()[lab] - v_true[lab])
        pred_n = np.expm1(np.clip(v_ho.numpy()[lab], 0, 30))
        true_n = np.expm1(np.clip(v_true[lab], 0, 30))
        ratio = np.maximum(pred_n, 1.0) / np.maximum(true_n, 1.0)
        ratio = np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-9))   # always >= 1x
        vel = {"n": int(lab.sum()),
               "mae_log": float(err.mean()),
               "factor_p50": float(np.percentile(ratio, 50)),
               "factor_p90": float(np.percentile(ratio, 90)),
               "spread_log_p50": float(np.percentile(v_spread.numpy()[lab], 50))}
    else:
        vel = {"n": 0, "mae_log": None, "factor_p50": None,
               "factor_p90": None, "spread_log_p50": None}

    meta = {
        "features": [f.name for f in features.REGISTRY],
        "n_rows": int(len(y)), "n_keywords": int(len(np.unique(data["groups"]))),
        "golden_auc": g_auc, "golden_ece": g_ece,
        # The ceiling for the downloads head: the highest first-year rate the
        # data contains, so predictions cannot be uncapped extrapolation.
        "velocity_max": float(np.nanmax(data["v"])) if np.isfinite(
            np.nanmax(data["v"])) else 0.0,
        # reference quantiles: turn a raw logit into a 0-100 fit/crowding score
        "logit_q": np.percentile(logits_all, np.arange(0, 101)).tolist(),
        "page_features": features.PAGE_FEATS,
        # What this run actually fitted, kept so two runs can be compared
        # instead of only the newest being visible.
        "train_loss": {"per_member": [round(x, 4) for x in losses],
                       "mean": float(np.mean(losses)) if losses else None,
                       "spread": float(np.std(losses)) if losses else None},
        "downloads": vel,
        "labelled_rows": int((~np.isnan(data["v"])).sum()),
        "epochs": int(epochs), "seed": int(seed), "members": int(k),
    }

    # Two independent gates. The relative one stops a regression; the absolute
    # one stops a FIRST model that never had anything to regress from. Without
    # it, a model with no signal at all promotes simply because it is the first,
    # which is exactly what a handful of keywords produces.
    # The gate only exists once it can measure something. With a handful of
    # keywords the golden split is one keyword and its AUC is noise, so blocking
    # on it would freeze the model at its initial weights forever.
    measurable = meta["n_keywords"] >= MIN_KEYWORDS
    prev = con.execute("SELECT golden_auc FROM registry WHERE active=1").fetchone()
    regressed = measurable and prev is not None and g_auc < prev["golden_auc"] - 0.01
    no_signal = measurable and g_auc < MIN_AUC
    gated = regressed or no_signal
    version = f"v{db.scalar(con, 'SELECT COUNT(*) FROM registry') + 1}"
    path = MODELS / f"{version}.pt"
    model.save(path, sm.state(), sf.state(), meta, scaler_set=ss.state())

    with con.transaction():
        con.execute("INSERT INTO registry (version, path, created_at, n_rows, "
                    "golden_auc, golden_ece, metrics, active) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,0)",
                    (version, str(path), db.now(), len(y), g_auc, g_ece,
                     _json.dumps({"train_loss": meta["train_loss"],
                                  "downloads": vel,
                                  "labelled_rows": meta["labelled_rows"],
                                  "epochs": epochs, "seed": seed,
                                  "members": k})))
        if not gated:
            con.execute("UPDATE registry SET active=0")
            con.execute("UPDATE registry SET active=1 WHERE version=%s", (version,))
            # Keep the shipped checkpoint pointing at whatever is actually
            # serving, so committing it never publishes a model the gate
            # rejected. Only from a run whose held-out split means something:
            # a handful of keywords promotes unconditionally, because there is
            # nothing to regress against, and the test suite's synthetic fixture
            # was quietly overwriting the checkpoint a fresh clone would get.
            if measurable:
                model.save(SHIPPED, sm.state(), sf.state(),
                           {**meta, "version": version, "promoted_at": db.now()},
                           scaler_set=ss.state())
            con.execute("UPDATE corrections SET status='absorbed' WHERE status='queued'")
            _retire_learned(con, model, sm, sf)

    if verbose:
        print(f"  rows {len(y)} across {meta['n_keywords']} keywords "
              f"| params {n_params(model)}")
        if meta["n_keywords"] < MIN_KEYWORDS:
            print(f"  ! only {meta['n_keywords']} keywords: the golden split is a "
                  f"single keyword and its score is noise, not a measurement")
        print(f"  golden AUC {g_auc:.3f}  ECE {g_ece:.3f}")
        tl = meta["train_loss"]
        print(f"  train loss {tl['mean']:.4f} +/- {tl['spread']:.4f} across "
              f"{len(tl['per_member'])} members")
        if vel["n"]:
            # Reported as a factor, not in log units, because that is the error
            # the reader sees: 6x means an app really earning 100 a day is shown
            # somewhere between 17 and 600.
            print(f"  downloads head: typically {vel['factor_p50']:.1f}x out, "
                  f"{vel['factor_p90']:.1f}x at the 90th percentile "
                  f"({vel['n']} held-out labels)")
        else:
            print("  downloads head: NOT MEASURED - no held-out row carries a "
                  "downloads label, so the number it produces is unvalidated")
        if not measurable:
            print(f"  promoted: {version} is now active "
                  f"({meta['n_keywords']}/{MIN_KEYWORDS} keywords, not yet validated)")
        elif no_signal:
            print(f"  NOT PROMOTED: AUC {g_auc:.3f} is below the {MIN_AUC} floor, "
                  f"so this model has no usable signal.")
            if meta["n_keywords"] < MIN_KEYWORDS:
                print(f"  You have {meta['n_keywords']} keywords. A held-out split "
                      f"needs at least {MIN_KEYWORDS} to mean anything.")
                print(f"  Scrape more keywords, then train again.")
        elif regressed:
            print("  NOT PROMOTED: regresses the golden split, active model unchanged")
        else:
            print(f"  promoted: {version} is now active")
    return version, meta, gated


def bootstrap(con, k=7, hidden=24):
    """Create the initial model from random weights.

    An untrained network still predicts. Refusing to answer until a gate is
    satisfied would mean the tool has no model on day one and no way to show it
    improving; this way there is always a prediction, and the registry records
    how much data stood behind it.
    """

    from . import features as F
    model = Ensemble(len(F.MONO), len(F.FREE), k=k, hidden=hidden,
                     n_app=len(F.PAGE_FEATS))
    model.eval()
    n_feat = len(F.REGISTRY)
    ident = {"mean": [0.0] * len(F.MONO), "std": [1.0] * len(F.MONO)}
    identf = {"mean": [0.0] * len(F.FREE), "std": [1.0] * len(F.FREE)}
    meta = {"features": [f.name for f in F.REGISTRY],
            "page_features": F.PAGE_FEATS, "n_rows": 0, "n_keywords": 0,
            "golden_auc": 0.5, "golden_ece": 0.5, "untrained": True,
            "logit_q": np.linspace(-4, 4, 101).tolist()}
    version = "v0"
    path = MODELS / "v0.pt"
    idents = {"mean": [0.0] * len(F.PAGE_FEATS), "std": [1.0] * len(F.PAGE_FEATS)}
    model.save(path, ident, identf, meta, scaler_set=idents)
    with con.transaction():
        # Clear the flag first. Without this the bootstrap adds a second active
        # row beside a real trained one, and "the active model" becomes whichever
        # the query happens to return.
        con.execute("UPDATE registry SET active=0")
        con.execute("INSERT INTO registry (version, path, created_at, "
                    "n_rows, golden_auc, golden_ece, active) VALUES (%s,%s,%s,0,0.5,0.5,1) "
                    "ON CONFLICT (version) DO UPDATE SET path=excluded.path, "
                    "created_at=excluded.created_at, active=excluded.active",
                    (version, str(path), db.now()))
    return model, {"cfg": model.cfg, "scaler_mono": ident, "scaler_free": identf,
                   "meta": meta}, version


def _usable(path):
    """Load a checkpoint, or None if it is missing or predates the feature set."""
    from . import features as F
    try:
        model, blob = Ensemble.load(Path(path))
    except (FileNotFoundError, RuntimeError, EOFError):
        return None
    if blob["meta"].get("features") != [f.name for f in F.REGISTRY]:
        return None
    # The page rows are an input too, and changing them invalidates a checkpoint
    # exactly as changing the scalar features does. Left unchecked, a model
    # trained on a different set of per-app columns loads without complaint and
    # answers confidently from weights that mean something else.
    if blob["meta"].get("page_features") != F.PAGE_FEATS:
        return None
    return model, blob


def load_active(con, create=True):
    """Always returns a model.

    Order: whatever this machine trained, then the checkpoint shipped with the
    repo, then random weights. The middle step is what lets someone clone and
    predict without scraping anything first.
    """
    # Newest first: a stray second active row is a bug, but it should not make
    # the served model depend on row order while it is being fixed.
    row = con.execute("SELECT version, path FROM registry WHERE active=1 "
                      "ORDER BY created_at DESC LIMIT 1").fetchone()
    if row:
        got = _usable(row["path"])
        if got:
            return got[0], got[1], row["version"]

    got = _usable(SHIPPED)
    if got:
        return got[0], got[1], got[1]["meta"].get("version", "shipped")

    return bootstrap(con) if create else (None, None, None)


def train_quick(con, country="us", epochs=400, verbose=False):
    """The fit that runs after each analyze.

    Full epochs, because throttling them saved nothing: 120 epochs takes 9s and
    400 takes 13s, and almost all of both is rebuilding the dataset rather than
    descending. Underfitting to save four seconds just produced models the gate
    rejected."""
    try:
        return train(con, country=country, epochs=epochs, verbose=verbose)
    except SystemExit:
        return None, None, True


def _retire_learned(con, model, sm, sf, tol=0.5) -> int:
    """Retire only the residuals this model demonstrably learned.

    Retiring blindly on every retrain was silent data loss: a correction the
    model could not absorb, because monotonicity or the weight of the other data
    argued against it, vanished anyway and the answer reverted on the next score.
    Keeping it alive is the honest behaviour. It also makes the disagreement
    visible: a residual that never retires is the model telling you it does not
    believe the correction.
    """

    n_feat = len(features.REGISTRY)
    dropped = memory.retire_stale(con, n_feat)
    if dropped:
        print(f"  {dropped} correction(s) retired: the feature set changed since "
              f"they were filed, so they can no longer be applied")
    rows = memory.active_rows(con, n_feat)
    if not rows:
        return 0
    from .predict import scale_all
    V = np.array([_json.loads(r["features_json"]) for r in rows], dtype="float32")
    S = scale_all(sm, sf, V)
    with torch.no_grad():
        k = len(features.MONO)
        lg = model.mean_logit(torch.from_numpy(S[:, :k].copy()),
                              torch.from_numpy(S[:, k:].copy())).numpy()
    learned = [r["id"] for r, v in zip(rows, lg) if v >= r["target_logit"] - tol]
    return memory.retire(con, learned) if learned else 0
