"""AUC and calibration error, without pulling in scikit-learn."""
from __future__ import annotations

import numpy as np


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-statistic AUC. Returns 0.5 when one class is absent."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype="float64")
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: does 0.79 actually mean 79 percent?
    This is the metric the product's credibility rests on, not AUC."""
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return float(total)
