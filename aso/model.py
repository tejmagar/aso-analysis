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


class MonotonicMLP(nn.Module):
    """Provably non-decreasing in every feature routed through x_mono.

    softplus keeps all weights on the monotone path positive; softplus is itself
    non-decreasing; a composition of non-decreasing maps is non-decreasing. The
    free block joins at the hidden layer and cannot break the guarantee, because
    d(out)/d(x_mono) never depends on its sign.

    That guarantee is most of why a few hundred rows are enough to fit this, and
    it means the model can never claim that improving your title hurt you.
    """

    def __init__(self, n_mono: int, n_free: int, hidden: int = 24):
        super().__init__()
        self.wm = nn.Parameter(torch.randn(n_mono, hidden) * 0.1)
        self.free = nn.Linear(n_free, hidden) if n_free else None
        self.w2 = nn.Parameter(torch.randn(hidden) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.vel = nn.Linear(hidden, 1)          # downloads head, unconstrained

    def hidden(self, xm: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        h = xm @ F.softplus(self.wm)
        if self.free is not None:
            h = h + self.free(xf)
        return F.softplus(h)

    def forward(self, xm: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        return self.hidden(xm, xf) @ F.softplus(self.w2) + self.bias     # rank logit

    def velocity(self, xm: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
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
        return self.vel(self.hidden(xm * mm, xf * mf)).squeeze(-1)


class Ensemble(nn.Module):
    """Members differ by seed and bootstrap resample. Their spread IS the
    uncertainty estimate, with no Bayesian machinery, and it is what drives
    keyword selection."""

    def __init__(self, n_mono: int, n_free: int, k: int = 7, hidden: int = 24):
        super().__init__()
        self.cfg = dict(n_mono=n_mono, n_free=n_free, k=k, hidden=hidden)
        self.members = nn.ModuleList(
            [MonotonicMLP(n_mono, n_free, hidden) for _ in range(k)])

    def logits(self, xm, xf) -> torch.Tensor:
        return torch.stack([m(xm, xf) for m in self.members])       # (K, B)

    def forward(self, xm, xf) -> tuple[torch.Tensor, torch.Tensor]:
        p = torch.sigmoid(self.logits(xm, xf))
        return p.mean(0), p.std(0)                                  # chance, uncertainty

    def mean_logit(self, xm, xf) -> torch.Tensor:
        return self.logits(xm, xf).mean(0)

    def velocity(self, xm, xf):
        """Predicted installs/year, with the ensemble's disagreement as the
        error bar. The spread IS the uncertainty; nothing else estimates it."""
        v = torch.stack([m.velocity(xm, xf) for m in self.members])   # (K, B)
        return v.mean(0), v.std(0)

    def agreement(self, xm, xf) -> torch.Tensor:
        """0 to 1. How much the seven members agree about this app.

        This is necessary for confidence but nowhere near sufficient: members
        bootstrapped from the same 80 rows agree readily, and agreeing about
        almost nothing is not knowledge. `evidence` supplies the other half.
        """
        p = torch.sigmoid(self.logits(xm, xf))
        return 1.0 - (p.std(0) / 0.5).clamp(max=1.0)

    def thompson(self, xm, xf) -> torch.Tensor:
        """Draw one member at random and score with it. That single line is
        Thompson sampling, and it is the entire exploration strategy: it favours
        keywords that are either confidently winnable or informatively uncertain,
        which stops the portfolio collapsing onto shapes you already understand."""
        i = int(torch.randint(len(self.members), (1,)))
        return torch.sigmoid(self.members[i](xm, xf))

    # ---- persistence -----------------------------------------------------
    def save(self, path: Path, scaler_mono, scaler_free, meta: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "state": self.state_dict(),
                    "scaler_mono": scaler_mono, "scaler_free": scaler_free,
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
