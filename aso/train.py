"""Train the ensemble, gate it on a held-out keyword split, register it.

Nothing is promoted that regresses the golden split, however many corrections it
satisfies. That gate is the only thing standing between a helpful feedback loop
and a model that agrees with whoever used the CLI most.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import dataset, db, features, memory, metrics
from .model import Ensemble, n_params

MODELS = Path(__file__).resolve().parent.parent / "data" / "models"

# The checkpoint that ships with the repo. Versioned models live under data/
# (gitignored working history); this one is committed so a fresh clone can
# predict immediately instead of needing 150 keywords scraped first.
SHIPPED = Path(__file__).resolve().parent.parent / "models" / "latest.pt"

# A held-out split needs this many keywords before its score means anything.
# Below it the model still trains and still predicts; there is simply nothing to
# validate against, so the regression gate is not applied yet.
MIN_KEYWORDS = 25
MIN_AUC = 0.55        # only enforced once the split is measurable


def _fit_member(model, xm, xf, y, v, w, epochs=400, lr=0.05, seed=0,
                on_epoch=None):
    """Two heads, one trunk. The rank task is the objective; the downloads task
    is a second signal over the same representation, which regularises it."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    known = ~torch.isnan(v)              # release date missing: no label, no gradient
    vt = torch.nan_to_num(v)
    for i in range(epochs):
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(model(xm, xf), y,
                                                   reduction="none") * w).mean()
        if known.any():
            loss = loss + 0.3 * F.mse_loss(model.velocity(xm, xf)[known], vt[known])
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
    NEW_YEARS = 1.5
    w = np.ones(len(data["y"]), dtype="float32")
    for i, m in enumerate(data["meta"]):
        age = m.get("age_years") or 0.0
        if 0 < age <= NEW_YEARS:
            w[i] = 3.0                       # a real entry experiment
        if m["source"] == "review":
            w[i] = max(w[i], 5.0)            # a human confirmed this rank
        elif m["source"] == "owned":
            w[i] = max(w[i], 3.0)            # observed absence, unobtainable otherwise

    say("fitting", 0, k, f"{len(y)} rows across "
        f"{len(np.unique(data['groups']))} keywords")
    model = Ensemble(XM.shape[1], XF.shape[1], k=k, hidden=hidden)
    rng = np.random.default_rng(seed)
    idx_tr = np.flatnonzero(tr)
    for i, member in enumerate(model.members):
        say("fitting", i, k, f"member {i + 1} of {k}")
        boot = rng.choice(idx_tr, size=len(idx_tr), replace=True)   # bootstrap resample
        _fit_member(member,
                    torch.from_numpy(XM[boot]), torch.from_numpy(XF[boot]),
                    torch.from_numpy(y[boot]), torch.from_numpy(data["v"][boot]),
                    torch.from_numpy(w[boot].astype("float32")),
                    epochs=epochs, lr=0.05, seed=seed + i,
                    on_epoch=lambda e, l, _i=i: say(
                        "fitting", _i, k, f"member {_i + 1} of {k}, epoch {e}"))

    say("scoring", k, k, "measuring against the held-out keywords")
    model.eval()
    with torch.no_grad():
        p_ho, _ = model(torch.from_numpy(XM[ho]), torch.from_numpy(XF[ho]))
        logits_all = model.mean_logit(torch.from_numpy(XM), torch.from_numpy(XF)).numpy()
    p_ho = p_ho.numpy()
    g_auc, g_ece = metrics.auc(y[ho], p_ho), metrics.ece(y[ho], p_ho)

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
    model.save(path, sm.state(), sf.state(), meta)

    with con.transaction():
        con.execute("INSERT INTO registry (version, path, created_at, n_rows, "
                    "golden_auc, golden_ece, active) VALUES (%s,%s,%s,%s,%s,%s,0)",
                    (version, str(path), db.now(), len(y), g_auc, g_ece))
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
                           {**meta, "version": version, "promoted_at": db.now()})
            con.execute("UPDATE corrections SET status='absorbed' WHERE status='queued'")
            _retire_learned(con, model, sm, sf)

    if verbose:
        print(f"  rows {len(y)} across {meta['n_keywords']} keywords "
              f"| params {n_params(model)}")
        if meta["n_keywords"] < MIN_KEYWORDS:
            print(f"  ! only {meta['n_keywords']} keywords: the golden split is a "
                  f"single keyword and its score is noise, not a measurement")
        print(f"  golden AUC {g_auc:.3f}  ECE {g_ece:.3f}")
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
    import json as _json

    from . import features as F
    model = Ensemble(len(F.MONO), len(F.FREE), k=k, hidden=hidden)
    model.eval()
    n_feat = len(F.REGISTRY)
    ident = {"mean": [0.0] * len(F.MONO), "std": [1.0] * len(F.MONO)}
    identf = {"mean": [0.0] * len(F.FREE), "std": [1.0] * len(F.FREE)}
    meta = {"features": [f.name for f in F.REGISTRY], "n_rows": 0, "n_keywords": 0,
            "golden_auc": 0.5, "golden_ece": 0.5, "untrained": True,
            "logit_q": np.linspace(-4, 4, 101).tolist()}
    version = "v0"
    path = MODELS / "v0.pt"
    model.save(path, ident, identf, meta)
    with con.transaction():
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
    return model, blob


def load_active(con, create=True):
    """Always returns a model.

    Order: whatever this machine trained, then the checkpoint shipped with the
    repo, then random weights. The middle step is what lets someone clone and
    predict without scraping anything first.
    """
    row = con.execute("SELECT version, path FROM registry WHERE active=1").fetchone()
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
    import json as _json

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
