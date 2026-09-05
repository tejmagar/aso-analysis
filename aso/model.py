"""The model. ~390 parameters per member, 7 members, ~2.7k trainable weights total.

Small on purpose. Google's ranker already ran, so every SERP is a sorted list and
we are only learning a one-dimensional projection of it: where does a new app slot
into an ordering that already exists%s That projection is far simpler than the
function behind it, which is why this fits in a few hundred parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """Attention over the ranked page, so the model picks its own comparisons.

    The hand-built field features are all aggregates, and the question they are
    meant to answer is not an aggregate question. "What would I earn at rank 6"
    depends on which apps sit near rank 6, how old they are, and whether they
    answer the same phrase at all - and a page routinely holds a ten-year-old
    giant at rank 10 on a lifetime average of thousands a day while genuinely
    new apps hold ranks 1 and 2 on twenty a day. A median over that page is a
    number with no referent.

    Choosing the comparison set by hand does not fix it either: Play fills a
    narrow phrase out with whatever is popular, and which apps are really
    competing is a judgement about a page assembled partly by taste. So no
    subset is selected here. Every app goes in with its rank, age, installs,
    realised rate and intent membership attached, and the attention weights are
    learned from which pages actually produced which downloads.

    The context vector joins at the hidden layer through the free path, exactly
    as the free block does, so it cannot touch the monotone guarantee: the model
    still can never claim that improving your own app hurt you.
    """

    def __init__(self, n_app: int, n_query: int, d: int = 16):
        super().__init__()
        self.key = nn.Linear(n_app, d)
        self.val = nn.Linear(n_app, d)
        self.query = nn.Linear(n_query, d)
        self.scale = d ** 0.5

    def forward(self, xs: torch.Tensor, mask: torch.Tensor,
                q: torch.Tensor) -> torch.Tensor:
        # xs (B, N, n_app), mask (B, N), q (B, n_query) -> (B, d)
        k = self.key(xs)                                    # (B, N, d)
        v = self.val(xs)
        a = (k @ self.query(q).unsqueeze(-1)).squeeze(-1) / self.scale   # (B, N)
        # Padding must not win attention. Masking before the softmax rather than
        # zeroing after keeps the weights a distribution over real apps only.
        a = a.masked_fill(mask < 0.5, float("-inf"))
        # A page with nothing on it would make every logit -inf and the softmax
        # NaN, which propagates through the whole batch.
        empty = mask.sum(-1, keepdim=True) < 0.5
        w = torch.softmax(torch.where(empty, torch.zeros_like(a), a), dim=-1)
        return torch.where(empty, torch.zeros_like(w), w).unsqueeze(1) @ v  # (B,1,d)


class MonotonicMLP(nn.Module):
    """Provably non-decreasing in every feature routed through x_mono.

    softplus keeps all weights on the monotone path positive; softplus is itself
    non-decreasing; a composition of non-decreasing maps is non-decreasing. The
    free block joins at the hidden layer and cannot break the guarantee, because
    d(out)/d(x_mono) never depends on its sign.

    That guarantee is most of why a few hundred rows are enough to fit this, and
    it means the model can never claim that improving your title hurt you.
    """

    def __init__(self, n_mono: int, n_free: int, hidden: int = 24,
                 n_app: int = 0, d_set: int = 16):
        super().__init__()
        self.wm = nn.Parameter(torch.randn(n_mono, hidden) * 0.1)
        self.free = nn.Linear(n_free, hidden) if n_free else None
        # The page encoder is queried with the asking app's own free features,
        # so "which apps should I compare myself to" is conditioned on who is
        # asking rather than being the same average for everyone.
        self.setenc = SetEncoder(n_app, n_free, d_set) if n_app else None
        self.set_in = nn.Linear(d_set, hidden) if n_app else None
        self.w2 = nn.Parameter(torch.randn(hidden) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.vel = nn.Linear(hidden, 1)          # downloads head
        # Downloads are non-increasing in your own rank, by construction.
        #
        # Entering a page at rank 4 cannot earn more than rank 1 on that same
        # page: the slot above takes the taps first. The model had no way to
        # know that. Its own entry rank reached the downloads head only through
        # rank_gap inside the page rows, where the network was free to fit any
        # shape it liked, and it fit an upward one - quoting a new app at rank 4
        # more than the app actually holding rank 1 was earning.
        #
        # softplus keeps this weight positive and it is SUBTRACTED, so a worse
        # rank can only ever lower the answer. That is the same construction the
        # rank head uses: direction guaranteed, magnitude learned. It is not a
        # formula for how much rank costs - only an assertion of the sign, which
        # is the part no amount of data should be asked to rediscover.
        self.w_rank = nn.Parameter(torch.tensor(0.1))

    def hidden(self, xm: torch.Tensor, xf: torch.Tensor,
               xs: torch.Tensor | None = None,
               mask: torch.Tensor | None = None) -> torch.Tensor:
        h = xm @ F.softplus(self.wm)
        if self.free is not None:
            h = h + self.free(xf)
        if self.setenc is not None and xs is not None:
            h = h + self.set_in(self.setenc(xs, mask, xf).squeeze(1))
        return F.softplus(h)

    def forward(self, xm: torch.Tensor, xf: torch.Tensor,
                xs: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.hidden(xm, xf, xs, mask) @ F.softplus(self.w2) + self.bias

    def velocity(self, xm: torch.Tensor, xf: torch.Tensor,
                 xs: torch.Tensor | None = None,
                 mask: torch.Tensor | None = None,
                 rank: torch.Tensor | None = None) -> torch.Tensor:
        """Predicted log1p(installs per year), with the answer masked out.

        A second head on the same representation, learned rather than derived.
        The label is free: every scraped app carries installs and a release
        date, so its realised rate is known and needs no annotation. Sharing the
        trunk is the point - whatever makes an app rank is most of what makes it
        get downloaded, so the two tasks reinforce each other.

        The app's own installs, reviews and realised rate are zeroed first: the
        rate IS the label, so leaving it in taught the head to echo its input and
        forecast zero for anything that has not launched yet.

        Deliberately NOT monotone-constrained: a stronger field means a harder
        rank but often a bigger market, and asserting either direction would be
        the hand-written assumption this is replacing.
        """
        from . import features as _F

        mm = torch.as_tensor(_F.VEL_MASK_MONO, device=xm.device, dtype=xm.dtype)
        mf = torch.as_tensor(_F.VEL_MASK_FREE, device=xf.device, dtype=xf.dtype)
        # The page rows are NOT masked: they are other apps' numbers, not this
        # app's, so they leak nothing about its own label. They are the entire
        # point - what comparable apps actually earned is the only evidence
        # there is for what an unlaunched app would earn.
        out = self.vel(self.hidden(xm * mm, xf * mf, xs, mask)).squeeze(-1)
        if rank is None:
            return out
        return out - F.softplus(self.w_rank) * rank


class Ensemble(nn.Module):
    """Members differ by seed and bootstrap resample. Their spread IS the
    uncertainty estimate, with no Bayesian machinery, and it is what drives
    keyword selection."""

    def __init__(self, n_mono: int, n_free: int, k: int = 7, hidden: int = 24,
                 n_app: int = 0, d_set: int = 16):
        super().__init__()
        self.cfg = dict(n_mono=n_mono, n_free=n_free, k=k, hidden=hidden,
                        n_app=n_app, d_set=d_set)
        self.members = nn.ModuleList(
            [MonotonicMLP(n_mono, n_free, hidden, n_app, d_set) for _ in range(k)])

    def logits(self, xm, xf, xs=None, mask=None) -> torch.Tensor:
        return torch.stack([m(xm, xf, xs, mask) for m in self.members])   # (K, B)

    def forward(self, xm, xf, xs=None, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        p = torch.sigmoid(self.logits(xm, xf, xs, mask))
        return p.mean(0), p.std(0)                                  # chance, uncertainty

    def mean_logit(self, xm, xf, xs=None, mask=None) -> torch.Tensor:
        return self.logits(xm, xf, xs, mask).mean(0)

    def velocity(self, xm, xf, xs=None, mask=None, rank=None):
        """Predicted installs/year, with the ensemble's disagreement as the
        error bar. The spread IS the uncertainty; nothing else estimates it."""
        v = torch.stack([m.velocity(xm, xf, xs, mask, rank) for m in self.members])
        return v.mean(0), v.std(0)

    def agreement(self, xm, xf, xs=None, mask=None) -> torch.Tensor:
        """0 to 1. How much the seven members agree about this app.

        This is necessary for confidence but nowhere near sufficient: members
        bootstrapped from the same 80 rows agree readily, and agreeing about
        almost nothing is not knowledge. `evidence` supplies the other half.
        """
        p = torch.sigmoid(self.logits(xm, xf, xs, mask))
        return 1.0 - (p.std(0) / 0.5).clamp(max=1.0)

    def thompson(self, xm, xf, xs=None, mask=None) -> torch.Tensor:
        """Draw one member at random and score with it. That single line is
        Thompson sampling, and it is the entire exploration strategy: it favours
        keywords that are either confidently winnable or informatively uncertain,
        which stops the portfolio collapsing onto shapes you already understand."""
        i = int(torch.randint(len(self.members), (1,)))
        return torch.sigmoid(self.members[i](xm, xf, xs, mask))

    # ---- persistence -----------------------------------------------------
    def save(self, path: Path, scaler_mono, scaler_free, meta: dict,
             scaler_set=None, downloads=None) -> None:
        # `downloads` is the separate page-reading downloads ensemble. It shares
        # nothing with this model but the checkpoint file, so that one version
        # number covers both and neither can be served against a mismatched
        # partner. A checkpoint predating it simply has None here.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "state": self.state_dict(),
                    "scaler_mono": scaler_mono, "scaler_free": scaler_free,
                    "scaler_set": scaler_set, "downloads": downloads,
                    "meta": meta}, path)

    @staticmethod
    def load(path: Path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = Ensemble(**blob["cfg"])
        m.load_state_dict(blob["state"])
        m.eval()
        return m, blob


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
