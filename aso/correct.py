"""Correcting a value. Addressed by keyword, so there are no ids to carry.

Each value is corrected through the mechanism that actually owns it:

  --rank      the learned one. It lands in TWO places: residual memory for an
              instant effect, and the observations table as a reviewed rank, so
              the next retrain learns it permanently and the residual can retire.
              A corrected rank IS a rank observation, and belongs with the rest.
  --demand    not learned at all. A human override simply wins.
  --crowding  a property of the field. A human override simply wins.

All three take effect on the very next `aso analyze`, in well under a second.
"""
from __future__ import annotations

import numpy as np

from . import db, memory, predict
from .dataset import TOP_K

RANK_WEIGHT = 5.0        # an observed rank is a fact, not an opinion
MARGIN = 0.25
MIN_BATCH = 50           # governs WEIGHT updates only; memory responds immediately


def target_logit_for_rank(inc_desc: list[float], rank: int) -> float:
    """The score that lands the app at exactly this rank.

    To sit at rank r the candidate needs precisely r-1 incumbents above it, so
    the target is the midpoint of the gap it must fall into. This is exact,
    unlike mapping a rank onto some assumed probability.
    """
    if not inc_desc:
        return MARGIN
    if rank <= 1:
        return inc_desc[0] + MARGIN
    if rank > len(inc_desc):
        return inc_desc[-1] - MARGIN
    return (inc_desc[rank - 2] + inc_desc[rank - 1]) / 2.0


def apply(con, keyword: str, pkg: str | None = None, rank: int | None = None,
          demand: float | None = None, crowding: float | None = None,
          reviewer: str = "you", country: str = "us") -> list[str]:
    keyword = keyword.strip().lower()
    done = []

    if demand is not None:
        db.set_override(con, keyword, "demand", demand, country, reviewer)
        done.append(f"demand set to {demand:.0f}")

    if crowding is not None:
        db.set_override(con, keyword, "crowding", crowding, country, reviewer)
        done.append(f"crowding set to {crowding:.0f}")

    if rank is not None:
        if not pkg:
            raise SystemExit("--rank needs --pkg: a rank belongs to an app, not a keyword")
        before = predict.score(con, pkg, keyword, country=country, record=False)

        # Record the reviewed rank FIRST. latest_observations() takes the newest
        # row per (keyword, pkg), so this supersedes what the scrape saw, and it
        # is what makes the correction survive the next retrain.
        #
        # Order matters: moving the app to its true rank reshuffles the top slots
        # and changes every feature on the page. A residual aimed at the old
        # field lands a rank or two short, so both the aim and the key are taken
        # AFTER the field has settled.
        db.add_observation(con, keyword, pkg, int(rank), country=country,
                           source="review")
        con.commit()

        p = predict.score(con, pkg, keyword, country=country)
        target = target_logit_for_rank(p["inc_logits"], int(rank))
        with con:
            cur = con.execute(
                "INSERT INTO corrections (ts, prediction_id, reviewer, kind, value, "
                "weight, status) VALUES (?,?,?,?,?,?,'queued')",
                (db.now(), p["prediction_id"], reviewer, "rank", float(rank), RANK_WEIGHT))
        memory.write(con, cur.lastrowid, keyword,
                     np.array(p["raw"], dtype="float32"), pkg,
                     p["logit"], target, RANK_WEIGHT)
        queued = con.execute(
            "SELECT COUNT(*) FROM corrections WHERE status='queued'").fetchone()[0]
        done.append(f"rank {before['predicted_rank']} corrected to {rank} "
                    f"({target - p['logit']:+.2f} logits, {queued}/{MIN_BATCH} queued)")

    if not done:
        raise SystemExit("give at least one of --rank, --demand, --crowding")
    return done
