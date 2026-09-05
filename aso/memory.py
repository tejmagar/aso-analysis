"""T0 residual memory: instant correction that never touches a weight.

Read on every prediction, written the moment a reviewer disagrees. Because it is
an additive layer keyed by the feature vector, correcting one keyword also nudges
similar keywords in proportion to how similar they are, which a per-keyword bias
table would not do.

It is a write-ahead buffer for the weights. Corrections land here in under a
second and get absorbed on the scheduled retrain, at which point they are retired
so nothing is ever counted twice.
"""
from __future__ import annotations

import json

import numpy as np

TAU = 0.75      # kernel width in scaled-feature space
TOP_K = 16
SHRINK = 1.0    # pulls the residual toward 0 when nothing similar was corrected


def load_active(con, n_features: int | None = None):
    rows = con.execute(
        "SELECT id, keyword, pkg, features_json, residual, weight, target_logit "
        "FROM residuals WHERE retired_at IS NULL").fetchall()
    if not rows:
        return (np.zeros((0, 0), "float32"), np.zeros((0,), "float32"),
                np.zeros((0,), "float32"), [])
    vecs = [json.loads(r["features_json"]) for r in rows]
    # Adding a feature changes the vector's length, and a residual stored under
    # the old shape can no longer be located in the new space. Silently
    # reshaping it would move the correction somewhere nobody asked for, so
    # stale-shaped rows are dropped from the read instead. They stay in the
    # table, and `retire_stale` is what finally clears them.
    if n_features is not None:
        keep = [i for i, v in enumerate(vecs) if len(v) == n_features]
        rows = [rows[i] for i in keep]
        vecs = [vecs[i] for i in keep]
    if not rows:
        return (np.zeros((0, 0), "float32"), np.zeros((0,), "float32"),
                np.zeros((0,), "float32"), [])
    X = np.array(vecs, dtype="float32")
    # Weight is CONFIDENCE, not amplitude. Folding it into the residual made a
    # weight-5 outcome overshoot the reviewer by 5x; it belongs in the kernel,
    # where it decides how far the memory overrides the model, not by how much.
    r = np.array([r["residual"] for r in rows], dtype="float32")
    cw = np.array([r_["weight"] for r_ in rows], dtype="float32")
    keys = [(r_["keyword"], r_["pkg"]) for r_ in rows]
    return X, r, cw, keys


def read(x: np.ndarray, X: np.ndarray, r: np.ndarray, cw: np.ndarray,
         keys=None, exact: tuple | None = None,
         tau: float = TAU, top_k: int = TOP_K) -> float:
    """Additive logit correction from the nearest past corrections.

    Kernel similarity times reviewer confidence decides the influence of each
    stored residual. An exact match filed with high confidence is applied almost
    in full; a distant one barely registers.
    """
    if X.size == 0:
        return 0.0

    # A correction filed against this exact (keyword, app) applies in full.
    # Kernel distance alone was not enough: recording the reviewed rank shifts
    # that app's position, which reshapes the field, which moves the features
    # the residual was keyed on. The correction would then be quietly damped by
    # the very act of recording it.
    if exact is not None and keys:
        hits = [i for i, k in enumerate(keys) if k == exact]
        if hits:
            return float(r[hits[-1]] * cw[hits[-1]] / (cw[hits[-1]] + SHRINK))

    d = np.linalg.norm(X - x[None, :], axis=1)
    w = np.exp(-((d / tau) ** 2)) * cw
    if len(w) > top_k:
        keep = np.argpartition(-w, top_k)[:top_k]
        w, r = w[keep], r[keep]
    # SHRINK in the denominator is the safety valve: when nothing similar has
    # ever been corrected the kernel weights are tiny, the residual collapses
    # toward zero, and the model's own opinion stands.
    return float((w * r).sum() / (w.sum() + SHRINK))


def write(con, correction_id, keyword, x: np.ndarray, pkg: str | None,
          predicted_logit: float, target_logit: float, weight: float = 1.0) -> int:
    """Store the residual pre-compensated for the kernel's own attenuation.

    read() divides by (sum(w) + SHRINK), so a correction re-queried at the exact
    point it was filed would otherwise come back at w/(w+SHRINK) of its value and
    land a rank or two short. Inverting that here makes an exact re-query honour
    the reviewer exactly, while distance still decays it normally: shrinkage
    keeps protecting neighbours, it just stops taxing the filed point itself.
    """
    from .db import now
    compensated = (target_logit - predicted_logit) * (weight + SHRINK) / weight
    row = con.execute(
        "INSERT INTO residuals (ts, correction_id, keyword, pkg, features_json, "
        "residual, target_logit, weight) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
        "RETURNING id",
        (now(), correction_id, keyword, pkg, json.dumps([float(v) for v in x]),
         float(compensated), float(target_logit), float(weight))).fetchone()
    return row["id"]


def active_rows(con, n_features: int | None = None):
    """Raw feature vectors plus what each correction was asking for."""
    rows = con.execute(
        "SELECT id, features_json, target_logit FROM residuals "
        "WHERE retired_at IS NULL AND target_logit IS NOT NULL").fetchall()
    if n_features is None:
        return rows
    return [r for r in rows if len(json.loads(r["features_json"])) == n_features]


def retire_stale(con, n_features: int) -> int:
    """Retire residuals whose feature vector predates the current registry.

    They cannot be applied or verified any more. Leaving them live would mean a
    correction that silently stopped working with no way to tell; retiring says
    plainly that it needs filing again."""
    from .db import now
    stale = [r["id"] for r in con.execute(
        "SELECT id, features_json FROM residuals WHERE retired_at IS NULL")
        if len(json.loads(r["features_json"])) != n_features]
    if stale:
        q = ",".join("?" * len(stale))
        con.execute(f"UPDATE residuals SET retired_at=%s WHERE id IN ({q})",
                    (now(), *stale))
        con.commit()
    return len(stale)


def retire(con, ids: list[int] | None = None) -> int:
    """Retire ONLY residuals the weights have demonstrably learned.

    Blind retirement on every retrain was a silent data-loss bug: the residual
    was cleared whether or not training had absorbed anything, so a correction
    that the model could not or would not learn simply disappeared on the next
    train and the answer reverted."""
    from .db import now
    if ids is None:
        cur = con.execute("UPDATE residuals SET retired_at=%s WHERE retired_at IS NULL", (now(),))
    else:
        q = ",".join("?" * len(ids))
        cur = con.execute(
            f"UPDATE residuals SET retired_at=%s WHERE id IN ({q}) AND retired_at IS NULL",
            (now(), *ids))
    con.commit()
    return cur.rowcount
