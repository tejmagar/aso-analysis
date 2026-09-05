"""Intent groups on a search result page.

A page is not one competition. "phone mirror" returns two different products
answering two different questions: cast my screen to a TV, and use the camera as
a mirror. Seven casting apps hold millions of installs; the mirror-camera app
holds rank 1 with 125 installs.

Averaging those together says "crowded" and is wrong. If one app can hold the
top slot for a meaning, a second app with that meaning faces almost no
competition. So the field is split by meaning first, and competition is measured
INSIDE the group you would be entering.

Grouping is done on the embeddings, not on word overlap, because the two
meanings share the word that matters.
"""
from __future__ import annotations

import numpy as np

MIN_GROUP = 1
MAX_GROUPS = 4


SIMILARITY = 0.50      # below this, two apps are answering different questions


def _agglomerate(V: np.ndarray, cutoff: float = SIMILARITY) -> list[list[int]]:
    """Average-linkage clustering on cosine, merging until nothing is close
    enough. The number of groups falls out of the data rather than being chosen:
    k-means with a fixed k fragments a single meaning into three and pairs the
    odd one out with whatever is nearest, which is exactly how the mirror-camera
    app kept landing beside the screen-casting apps.
    """
    S = V @ V.T
    groups = [[i] for i in range(len(V))]
    while len(groups) > 1:
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                sim = float(np.mean([[S[i][j] for j in groups[b]] for i in groups[a]]))
                if sim > best:
                    best, pair = sim, (a, b)
        if best < cutoff:
            break
        a, b = pair
        groups[a] += groups[b]
        groups.pop(b)
    return groups


def split(rows: list[dict], vecs: dict, kw_vec=None) -> dict:
    """Partition ranked apps by meaning, then describe each group."""
    ranked = sorted((r for r in rows if r.get("position")), key=lambda r: r["position"])
    ranked = [r for r in ranked if vecs.get(r["pkg"]) is not None]
    if len(ranked) < 3:
        return _one_group(ranked, kw_vec, vecs)

    V = np.array([vecs[r["pkg"]] for r in ranked], dtype="float32")
    idx_groups = _agglomerate(V, SIMILARITY)

    groups = []
    for idxs in idx_groups:
        members = [ranked[i] for i in idxs]
        c = V[idxs].mean(0)
        c /= np.linalg.norm(c) + 1e-9
        g = _describe(members, c, kw_vec)
        g["centroid"] = c
        groups.append(g)
    groups.sort(key=lambda g: g["best_rank"])
    return {"groups": groups, "ambiguous": len(groups) > 1,
            "by_pkg": {p: i for i, g in enumerate(groups) for p in g["packages"]}}


def _one_group(ranked, kw_vec, vecs):
    if not ranked:
        return {"groups": [], "ambiguous": False, "by_pkg": {}}
    V = np.array([vecs[r["pkg"]] for r in ranked], dtype="float32")
    c = V.mean(0)
    c /= np.linalg.norm(c) + 1e-9
    g = _describe(ranked, c, kw_vec)
    g["centroid"] = c
    return {"groups": [g], "ambiguous": False,
            "by_pkg": {p: 0 for p in g["packages"]}}


def _describe(members: list[dict], centroid, kw_vec) -> dict:
    from .features import _days_since

    inst = [m.get("installs") or 0 for m in members]
    weakest = min(members, key=lambda m: m.get("installs") or 0)
    # Newcomers WITHIN this meaning. Reporting page-wide newcomers next to
    # intent-scoped install figures mixed two frames in one sentence: "these sit
    # at 27.8K" beside "the best newcomer passed 574.3K", where the newcomer was
    # a different product entirely.
    recent = [m for m in members
              if 0 < _days_since(m.get("released_at")) / 365.0 <= 1.0]
    return {
        "size": len(members),
        "label": _label(members),
        "best_rank": min(m["position"] for m in members),
        "median_installs": int(np.median(inst)),
        "weakest_installs": weakest.get("installs") or 0,
        "weakest_rank": weakest["position"],
        "weakest_title": weakest.get("title"),
        "newcomers": len(recent),
        "newcomer_installs": max((m.get("installs") or 0 for m in recent), default=0),
        "titles": [m.get("title") for m in members[:4]],
        "packages": [m["pkg"] for m in members],
        # How close this group is to the query itself. With a real encoder this
        # is what tells you which meaning the keyword leans toward.
        "affinity": (float(np.dot(kw_vec, centroid)) if kw_vec is not None else 0.0),
    }


def _label(members: list[dict]) -> str:
    """A name for the group, taken from the words its members actually share."""
    import re
    from collections import Counter

    stop = {"the", "and", "for", "app", "apps", "free", "pro", "to", "my", "your",
            "with", "of", "in", "on", "a", "best", "new", "phone", "mobile"}
    counts = Counter()
    for m in members:
        seen = set()
        for w in re.findall(r"[a-z0-9]+", (m.get("title") or "").lower()):
            if w not in stop and len(w) > 2 and w not in seen:
                seen.add(w)
                counts[w] += 1
    need = 1 if len(members) <= 2 else max(2, len(members) // 2)
    top = [w for w, c in counts.most_common(3) if c >= need]
    return " / ".join(top) if top else "mixed"


def lead(split_result: dict) -> dict | None:
    """The meaning that holds the top of the page.

    Play decides what a phrase means, and it says so by what it ranks first.
    Groups arrive sorted by best rank, so this is the group containing rank 1.
    """
    groups = split_result.get("groups") or []
    return groups[0] if groups else None


def group_for(split_result: dict, pkg: str | None, vec=None) -> dict | None:
    """Which competition would this app actually be in.

    A ranked app is placed by its own membership. Anything not on the page yet -
    your app, or the hypothetical entrant - joins the meaning that holds the top
    of the page, because that is the one the phrase is understood to be about.

    It used to be placed by nearest centroid to the keyword's own embedding, and
    that is unreliable exactly where it matters. A short phrase embeds weakly:
    for "native cam" every app on the page scored under the relevance threshold,
    so the field looked empty of competitors, three separate lower-is-better
    features all read zero at once, and an unreadable page came back as the best
    opportunity on offer. Play had in fact answered clearly, ranking a camera app
    first; nothing was asking it what it had ranked.
    """
    groups = split_result.get("groups") or []
    if not groups:
        return None
    idx = split_result.get("by_pkg", {}).get(pkg)
    if idx is not None:
        return groups[idx]
    return groups[0]
