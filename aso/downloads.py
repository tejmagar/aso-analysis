"""Estimating downloads by reading the page, not by knowing the world.

The rank model answers "where would this app land" from a large feature vector,
and that is the right shape for a question about competition. Downloads is a
different question and was being answered the same way, badly: the head
predicted an absolute rate, so before it could be right about anything it had to
rebuild the page's LEVEL out of features. Across the corpus that level runs from
0.4 a day to 251,000 - about thirteen log units - and it was being reconstructed
from scratch every time, from 172 labelled examples.

This model never learns the level. It is handed the page as a set of points -
when each app was published, where it ranks, and what it earns - and asked what
an app published at a given moment, landing at a given rank, would earn. The
level arrives with the points, so the only thing left to learn is the part that
generalises: how far above or below its neighbours such an app sits. That
relation is the same on a page paying 3 a day and a page paying 300,000.

Which is also how a person reads the page. Nobody looks at a search result and
recalls what apps earn globally; they see a two-month-old at 2 a day sitting
above the slot they would take, and conclude they would earn about one.

Apps are placed by RELEASE DATE, not by age. Age is measured backwards from the
present, so it bottoms out at zero: an app shipping today and one shipping next
month are the same number, and there is no way to say "a week from now" at all.
It also drifts, because the same scraped page reports a larger age every day it
sits in the database. A release date is a fixed point on a line that grows with
every new app, so a future release is simply a larger value and the question
"what would an app published next week earn" has an answer the model can reach.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DAY = 86_400.0
# Any fixed origin works - it cancels out of every difference below. Play
# predates it, but nothing in the corpus does, and a smaller number keeps the
# float honest.
EPOCH = datetime(2010, 1, 1, tzinfo=timezone.utc)

# Points are described RELATIVE to the app being asked about. Absolute dates and
# ranks would make the model learn each page's coordinates; the differences are
# what carry the relation, and they mean the same thing on every page.
POINT_FEATS = 7


def stamp(when) -> float | None:
    """A date as days since a fixed origin. Bigger means newer, always.

    Dates reach here as ISO strings from the database and as datetimes from
    callers that already parsed one, so both are accepted; an unparseable date
    is None rather than a zero, which would place the app in 2010.
    """
    if when is None:
        return None
    if isinstance(when, (int, float)):
        return float(when)
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when)
        except ValueError:
            return None
    if not isinstance(when, datetime):
        when = datetime(when.year, when.month, when.day)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (when - EPOCH).total_seconds() / DAY


def _slog(days: float) -> float:
    """Signed log of a span in days.

    Plain days would let a decade of separation drown out the difference
    between a week and a month, which is the resolution that matters at the
    young end. Plain log cannot represent a release in the future. Signed log
    keeps both: fine near zero, compressed far out, and negative on the side
    where the other app is older than the one being asked about.
    """
    return math.copysign(math.log1p(abs(days)), days)


def describe(points: list[tuple[float, float, float]],
             q_released: float, q_rank: float, observed: float,
             max_points: int = 16):
    """One page as (points, mask), each point relative to the asking app.

    `points` are (released, rank, rate_per_year) for the apps already there,
    `released` being `stamp()` of each one's release date. The query is the
    release date and rank of the app being asked about - for a new app that is
    the day it ships, which may be after `observed`, the day the page was read.
    """
    X = np.zeros((max_points, POINT_FEATS), dtype="float32")
    m = np.zeros(max_points, dtype="float32")
    for i, (released, rank, rate) in enumerate(points[:max_points]):
        newer = _slog(released - q_released)   # +: shipped after me. -: before.
        by_rank = (rank - q_rank) / 10.0       # +: below me on the page.
        X[i] = [
            newer,
            by_rank,
            math.log1p(max(rate, 0.0)),        # what it earns: the value read off
            abs(newer),                        # how far apart in time, either way
            abs(by_rank),                      # how far apart in rank, either way
            1.0 if rank < q_rank else 0.0,
            _slog(observed - released),        # its own age when the page was read,
                                               # since a young app's number says
                                               # more about a standing start
        ]
        m[i] = 1.0
    return X, m


class DownloadNet(nn.Module):
    """Learned kernel regression over the page.

    The answer is a weighted blend of what the apps on this page earn, plus a
    correction. The weights are learned from each app's distance in release date
    and rank from the app being asked about, so the model discovers for itself
    which neighbours are worth listening to - which is the thing that kept being
    hardcoded as a window of two ranks, or a year of age.

    Deliberately small. The relation it has to represent is two-dimensional and
    the corpus is a few hundred pages, so capacity here buys memorisation rather
    than accuracy.
    """

    def __init__(self, d: int = 24):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(POINT_FEATS, d), nn.Tanh(), nn.Linear(d, 1))
        self.adjust = nn.Sequential(
            nn.Linear(POINT_FEATS + 1, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # The page's level is subtracted before the network sees anything and
        # added back at the end, so being level-free is a property of the
        # ARCHITECTURE rather than something the fit is trusted to discover.
        # Left to itself the correction term can read absolute rates and quietly
        # learn a global level, which is the exact failure this model exists to
        # avoid; centring makes that unreachable. Every other column is already
        # a difference, so after this nothing inside the network can tell a page
        # paying 3 a day from one paying 300,000.
        n = mask.sum(-1, keepdim=True).clamp(min=1.0)
        rates = X[..., 2]
        base = (rates * mask).sum(-1, keepdim=True) / n
        Xr = torch.cat([X[..., :2], (rates - base.expand_as(rates)).unsqueeze(-1),
                        X[..., 3:]], dim=-1)

        # Attention over the points, masked so padding cannot win.
        s = self.score(Xr).squeeze(-1)
        s = s.masked_fill(mask < 0.5, float("-inf"))
        empty = mask.sum(-1, keepdim=True) < 0.5
        w = torch.softmax(torch.where(empty, torch.zeros_like(s), s), dim=-1)
        w = torch.where(empty, torch.zeros_like(w), w)

        blend = (w * Xr[..., 2]).sum(-1)        # where this slot sits, relative

        # A correction on top, from the same weighted view plus the blend. The
        # blend alone can only return something between the values already on
        # the page; a new app can sit below all of them.
        ctx = (w.unsqueeze(-1) * Xr).sum(1)
        delta = self.adjust(torch.cat([ctx, blend.unsqueeze(-1)], -1)).squeeze(-1)
        return base.squeeze(-1) + blend + delta


def page_points(rows, exclude_pkg: str | None = None, near_rank: float | None = None,
                max_points: int = 16):
    """The apps on a page as (released, rank, rate_per_year) triples.

    Anything without a release date is dropped: a point with no position on the
    time axis cannot be placed relative to the query, and guessing one would
    invent the very relation the model is here to learn. A zero-install app is
    kept, because earning nothing is an answer.
    """
    from . import features as _f
    pts = []
    for r in rows:
        if exclude_pkg and r.get("pkg") == exclude_pkg:
            continue
        rel = stamp(r.get("released_at"))
        if rel is None or not r.get("position"):
            continue
        pts.append((rel, float(r["position"]), _f._rate_of(r)))
    # Trimming keeps the apps NEAREST the slot being asked about, not the
    # highest-ranked ones. On a long page those are different sets, and the
    # neighbours are the whole point: what matters for an app entering at 30 is
    # what 28 and 32 earn, not what the top ten do.
    pts.sort(key=lambda p: abs(p[1] - near_rank) if near_rank is not None else p[1])
    return pts[:max_points]


def fit(X, mask, y, *, epochs: int = 600, lr: float = 0.01, seed: int = 0,
        d: int = 24, weights=None, on_epoch=None) -> tuple[DownloadNet, list[float]]:
    """Fit one member on (points, mask) -> log rate.

    Huber rather than squared error. The corpus holds pages paying 3 a day and
    pages paying 300,000, and under a squared loss the handful of enormous ones
    set the gradient for everything - which is how a model ends up 29x out at
    the tail while its median looks respectable.
    """
    torch.manual_seed(seed)
    net = DownloadNet(d=d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = torch.from_numpy(X)
    mt = torch.from_numpy(mask)
    yt = torch.from_numpy(y)
    wt = torch.from_numpy(weights) if weights is not None else torch.ones_like(yt)
    wt = wt / wt.mean()
    curve = []
    for i in range(epochs):
        opt.zero_grad()
        loss = (F.huber_loss(net(xt, mt), yt, reduction="none", delta=1.0) * wt).mean()
        loss.backward()
        opt.step()
        curve.append(loss.item())
        if on_epoch:
            on_epoch(i + 1, epochs, float(loss))
    return net, curve


def predict(net: DownloadNet, X, mask) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(X), torch.from_numpy(mask)).numpy()


class Ensemble:
    """Several members on bootstrap resamples; their disagreement is the range.

    Same reasoning as the rank model: a single fit gives a number with no way to
    say how much to trust it, and members that saw different subsets of the
    corpus agree closely where the evidence is thick and diverge where it is
    thin. That divergence is the honest interval, and it costs nothing to have -
    each member is 458 parameters.
    """

    def __init__(self, nets: list[DownloadNet]):
        self.nets = nets

    def __call__(self, X, mask):
        P = np.stack([predict(n, X, mask) for n in self.nets])
        return P.mean(0), P.std(0)

    def state(self):
        return [n.state_dict() for n in self.nets]

    @classmethod
    def load(cls, states, d: int = 24):
        nets = []
        for sd in states:
            n = DownloadNet(d=d)
            n.load_state_dict(sd)
            n.eval()
            nets.append(n)
        return cls(nets)


def fit_ensemble(X, mask, y, *, k: int = 7, epochs: int = 600, seed: int = 0,
                 weights=None, on_member=None, **kw):
    rng = np.random.default_rng(seed)
    nets, curves = [], []
    for i in range(k):
        pick = rng.choice(len(y), size=len(y), replace=True)
        net, curve = fit(X[pick], mask[pick], y[pick], epochs=epochs, seed=seed + i,
                         weights=None if weights is None else weights[pick], **kw)
        nets.append(net)
        curves.append(curve)
        if on_member:
            on_member(i + 1, k, curve[-1])
    return Ensemble(nets), curves
