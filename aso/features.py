"""Feature registry, extraction, and the monotonicity contract.

Adding a feature is an entry here plus a re-featurization pass. It is never a
scraping project, which is why the raw listing JSON is kept in the apps table.

direction:
  inc   rank chance must not decrease as this rises   (match quality, your strength)
  dec   rank chance must not increase as this rises   (field strength)
  free  genuinely non-monotone, left unconstrained
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np


@dataclass(frozen=True)
class Feat:
    name: str
    direction: str
    doc: str = ""


REGISTRY: list[Feat] = [
    # --- your metadata vs the keyword -------------------------------------
    Feat("kw_in_title",       "inc", "all keyword tokens present in the title"),
    Feat("kw_title_pos",      "inc", "1.0 when the keyword opens the title, decaying to 0"),
    Feat("kw_density_title",  "inc", "share of title tokens that are keyword tokens"),
    Feat("kw_in_short",       "inc", "all keyword tokens present in the short description"),
    Feat("kw_desc_tf",        "inc", "saturating term frequency in the full description"),
    Feat("kw_cosine",         "inc", "cosine of cached keyword and metadata embeddings"),
    # --- your strength ----------------------------------------------------
    Feat("log_installs",      "inc", "log1p installs"),
    Feat("rating",            "inc", "mean star rating"),
    Feat("log_reviews",       "inc", "log1p rating count"),
    # Installs alone say how big an app is. Installs against AGE say how fast it
    # got there, which is the far stronger signal: 200k in one year and 200k
    # over eight years are not the same app in the same market.
    Feat("app_velocity",      "inc", "log1p installs per year since release"),
    # --- how hard the field is (higher = harder, so decreasing) -----------
    Feat("field_installs_p50", "dec", "median installs of the ranked field"),
    Feat("field_installs_p90", "dec", "the entrenched top of the field"),
    Feat("field_rating_p50",   "dec", "median rating of the field"),
    Feat("field_exact_match",  "dec", "how many ranked apps exact-match the keyword"),
    # Intent. A field of large apps that are not about the keyword is an
    # opportunity, not a wall, and install medians alone cannot tell them apart.
    Feat("field_relevance_p50", "dec", "median how-much-about-the-keyword of the field"),
    Feat("field_relevant_count", "dec", "how many ranked apps are genuinely on-intent"),
    Feat("field_installs_relevant", "dec", "install median among ON-INTENT apps: the real bar"),
    # The opening. A median says what a typical holder looks like; the WEAKEST
    # app holding a top slot says what you actually have to beat. "phone mirror"
    # has a 125-install app at rank 1, which a median of 394K completely buried.
    Feat("field_weakest_installs", "dec", "installs of the weakest app in the top slots"),
    Feat("field_weakest_rank",     "inc", "how high that weak app sits: higher is a bigger opening"),
    # The competition you would actually be in. A page answering four questions
    # is four competitions; being handed the whole page's install median is how
    # a keyword with one 125-install rival gets scored as if it had a 48M one.
    Feat("intent_rivals",        "dec", "apps sharing your meaning, not the whole page"),
    Feat("intent_median",        "dec", "install bar inside your meaning"),
    Feat("intent_weakest",       "dec", "weakest app holding a slot for your meaning"),
    Feat("intent_reach",         "inc", "how high your meaning already ranks: proof it can"),
    Feat("intent_share",         "free", "share of the page your meaning occupies"),
    # Play's promoted hero card sits above the organic list and takes the taps,
    # so organic slot 1 is worth materially less when one is present.
    Feat("field_has_featured", "dec", "a promoted card sits above the organic results"),
    Feat("field_featured_relevant", "dec",
         "the promoted card is ON-INTENT: a direct rival holding the most visible slot"),
    Feat("field_age_p50",      "dec", "median age of the field: entrenched veterans are harder to displace"),
    # The penetrability signal. If apps published in the last year already hold
    # top slots, the field demonstrably lets newcomers in, whatever its install
    # medians say. Two fields with identical installs behave nothing alike when
    # one has not admitted a new entrant since 2019.
    Feat("field_newcomers",    "inc", "how many of the ranked field launched within a year"),
    # The single most decision-relevant number in the whole feature set: how
    # long since ANYONE new made it into the top slots. A field whose youngest
    # member arrived five years ago has a shut door, whatever its installs say.
    Feat("newest_entrant_age", "dec", "years since the most recent arrival in the field"),
    Feat("field_velocity_p50", "dec", "median installs per year of the field: the growth bar"),
    # --- shape, genuinely non-monotone ------------------------------------
    Feat("field_installs_p10", "free", "the soft underbelly of the field"),
    Feat("field_spread",       "free", "log p90/p10: one giant vs ten equals"),
    # Deliberately unconstrained: a newcomer with huge installs can mean the
    # market rewards new entrants, or that someone well funded just arrived.
    # The direction is genuinely ambiguous, so it is not asserted.
    Feat("field_newcomer_installs", "free", "best installs achieved by a recent entrant"),
    Feat("field_staleness_p50", "free", "median days since the field last updated"),
    Feat("field_age_spread",    "free", "p90 minus p10 age: a multi-cohort field is a ladder"),
    Feat("field_age_known",     "free", "fraction of the field whose release date we actually have"),
    # Autocomplete, exactly as Play returned it. All unconstrained: how a
    # phrase's standing in its own suggestion list relates to organic rank is
    # genuinely unknown, and asserting a direction would reinvent the formula.
    Feat("sugg_returned",      "free", "how many suggestions came back"),
    Feat("sugg_self_listed",   "free", "the phrase appears in its own suggestion list"),
    Feat("sugg_self_rank",     "free", "its position in that list"),
    Feat("sugg_is_canonical",  "free", "Play returns it first: a query people really type"),
    Feat("sugg_extends",       "free", "share of suggestions that are longer forms of it"),
    Feat("sugg_extra_words",   "free", "how much longer those forms are"),
    Feat("sugg_unrelated",     "free", "share Play returned that do NOT contain the phrase"),
    # How the field has MOVED since we first looked. A single snapshot cannot
    # distinguish a bar that has been flat for a year from an identical one that
    # doubled in three months, and those are different propositions. All left
    # unconstrained: a hardening field is worse to enter but often proves the
    # market is worth entering, so the direction is genuinely unknown.
    Feat("hist_known",           "free", "we hold an earlier state of this field"),
    Feat("hist_age_days",        "free", "how far back that earlier state goes, in years"),
    Feat("hist_installs_delta",  "free", "log change in the field's install bar since then"),
    Feat("hist_size_delta",      "free", "change in how many apps rank"),
    Feat("hist_match_delta",     "free", "change in how many target the keyword in their title"),
    Feat("hist_hardening",       "free", "that install change expressed per year"),
    Feat("hist_dated",           "free", "both states carried enough release dates to compare age"),
    Feat("hist_renewal",         "free", "years of age the field shed per year - churn, measured"),
    Feat("hist_velocity_delta",  "free", "log change in how fast the field's apps were growing"),
    Feat("kw_tokens",          "free", "keyword length in tokens"),
    Feat("days_since_release", "free", "new apps behave differently, not monotonically"),
    Feat("days_since_update",  "free", "freshness"),
]

MONO = [f for f in REGISTRY if f.direction != "free"]
FREE = [f for f in REGISTRY if f.direction == "free"]
MONO_NAMES = [f.name for f in MONO]
FREE_NAMES = [f.name for f in FREE]
SIGN = np.array([-1.0 if f.direction == "dec" else 1.0 for f in MONO], dtype="float32")


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _days_since(iso: str | None, ref: date | None = None) -> float:
    if not iso:
        return 0.0
    try:
        d = datetime.fromisoformat(iso).date()
    except ValueError:
        return 0.0
    return max((ref or date.today()) - d, __import__("datetime").timedelta(0)).days


def app_relevance(row: dict, keyword: str, kw_vec=None, app_vec=None) -> float:
    """How much this app is actually ABOUT the keyword, 0 to 1.

    The point the install medians miss: a 1.5B-install "Find Hub" ranking for
    "spider tracker" is not competing for that intent, it is filler Play used to
    pad a thin result set. Counting it as competition reads a wide-open keyword
    as a wall.

    Every keyword token must land somewhere or the score collapses, which is
    what separates a real match from an app that merely shares the word
    "tracker".
    """
    kw = tokens(keyword)
    if not kw:
        return 0.0
    title = set(tokens(row.get("title") or ""))
    short = set(tokens(row.get("short_desc") or ""))
    desc = set(tokens(row.get("description") or ""))

    per = []
    for t in kw:
        per.append(1.0 if t in title else 0.6 if t in short else 0.35 if t in desc else 0.0)
    found = sum(1 for v in per if v > 0) / len(kw)
    lexical = (sum(per) / len(kw)) * found       # missing a token is punishing

    # Blend in meaning. Token matching alone is string comparison dressed up as
    # judgement: it scores "Arachnid Finder" at zero for "spider tracker" and
    # gives partial credit to any app with the word "tracker" in it. The cached
    # embeddings already exist, so this costs a dot product.
    #
    # Only worth much with a real sentence encoder installed. On the trigram
    # hash fallback the semantic half is itself mostly lexical, which is why it
    # is a blend and not a replacement.
    if kw_vec is None or app_vec is None:
        return lexical
    semantic = max(0.0, float(np.dot(kw_vec, app_vec)))
    return 0.6 * lexical + 0.4 * semantic


RELEVANT = 0.40


def exact_match(title: str, keyword: str) -> bool:
    return all(t in tokens(title) for t in tokens(keyword))


from .history import DELTA_DEFAULTS
from .suggest import DEFAULTS as SUGG_DEFAULTS

SUGG_DEFAULTS = {**SUGG_DEFAULTS, **DELTA_DEFAULTS}


def extract(app: dict, keyword: str, field: dict,
            kw_vec=None, app_vec=None, sugg: dict | None = None) -> dict[str, float]:
    kw_t = tokens(keyword)
    ti_t = tokens(app.get("title") or "")
    sd_t = tokens(app.get("short_desc") or "")
    de_t = tokens(app.get("description") or "")

    hits = [i for i, t in enumerate(ti_t) if t in kw_t]
    first = hits[0] if hits else None
    tf = sum(de_t.count(t) for t in kw_t)

    cos = 0.0
    if kw_vec is not None and app_vec is not None:
        cos = float(np.dot(kw_vec, app_vec))

    p10 = float(field.get("installs_p10") or 0.0)
    p90 = float(field.get("installs_p90") or 0.0)

    return {
        "kw_in_title":      1.0 if kw_t and all(t in ti_t for t in kw_t) else 0.0,
        "kw_title_pos":     0.0 if first is None else 1.0 - first / max(len(ti_t), 1),
        "kw_density_title": len(hits) / max(len(ti_t), 1),
        "kw_in_short":      1.0 if kw_t and all(t in sd_t for t in kw_t) else 0.0,
        "kw_desc_tf":       tf / (tf + 1.5),                 # saturating, BM25-style
        "kw_cosine":        cos,

        "log_installs":     math.log1p(app.get("installs") or 0),
        "app_velocity":     math.log1p((app.get("installs") or 0)
                                       / max(_days_since(app.get("released_at")) / 365.0,
                                             0.25)),
        "rating":           float(app.get("rating") or 0.0),
        "log_reviews":      math.log1p(app.get("reviews") or 0),

        "field_installs_p50": math.log1p(field.get("installs_p50") or 0.0),
        "field_installs_p90": math.log1p(p90),
        "field_rating_p50":   float(field.get("rating_p50") or 0.0),
        "field_exact_match":  float(field.get("exact_match_count") or 0),

        "field_age_p50":      float(field.get("age_p50") or 0.0),
        "field_newcomers":    float(field.get("newcomers") or 0),
        "newest_entrant_age": float(field.get("newest_entrant_age") or 0.0),
        "field_velocity_p50": math.log1p(field.get("velocity_p50") or 0.0),

        "field_relevance_p50":  float(field.get("relevance_p50") or 0.0),
        "field_relevant_count": float(field.get("relevant_count") or 0),
        "field_installs_relevant": math.log1p(field.get("installs_p50_relevant") or 0),
        "intent_rivals":   float(field.get("intent_rivals") or 0),
        "intent_median":   math.log1p(field.get("intent_median") or 0),
        "intent_weakest":  math.log1p(field.get("intent_weakest") or 0),
        "intent_reach":    float(field.get("intent_reach") or 0.0),
        "intent_share":    float(field.get("intent_share") or 0.0),

        "field_weakest_installs": math.log1p(field.get("weakest_installs") or 0),
        "field_weakest_rank":  1.0 - (field.get("weakest_rank") or 99) / 100.0,
        "field_has_featured":   1.0 if field.get("has_featured") else 0.0,
        "field_featured_relevant": float(field.get("featured_relevance") or 0.0),

        "field_installs_p10": math.log1p(p10),
        "field_newcomer_installs": math.log1p(field.get("newcomer_installs") or 0),
        "field_staleness_p50": float(field.get("staleness_p50") or 0.0),
        "field_age_spread":    float(field.get("age_spread") or 0.0),
        "field_age_known":     float(field.get("age_known_frac") or 0.0),
        "field_spread":       math.log1p(p90) - math.log1p(p10),
        "kw_tokens":          float(len(kw_t)),
        "days_since_release": _days_since(app.get("released_at")) / 365.0,
        "days_since_update":  _days_since(app.get("updated_at")) / 365.0,
        **{k: float((sugg or {}).get(k, v)) for k, v in SUGG_DEFAULTS.items()},
    }


def _intent_stats(group: dict | None, n_ranked: int, exclude_pkg: str | None) -> dict:
    if not group:
        return {"intent_rivals": 0, "intent_median": 0, "intent_weakest": 0,
                "intent_reach": 0.0, "intent_share": 0.0}
    # Leave the app itself out of its own rival count, same as the field stats.
    rivals = group["size"] - (1 if exclude_pkg in group["packages"] else 0)
    return {
        "intent_rivals": max(rivals, 0),
        "intent_median": group["median_installs"],
        "intent_weakest": group["weakest_installs"],
        # Its group already reaching rank 1 is proof this meaning can rank here.
        "intent_reach": 1.0 - (group["best_rank"] - 1) / max(n_ranked, 1),
        "intent_share": group["size"] / max(n_ranked, 1),
    }


def _vec_for(row, cache):
    return None if cache is None else cache.get(row.get("pkg"))


def compute_field(rows: list[dict], keyword: str, top_n: int = 10,
                  exclude_pkg: str | None = None,
                  featured: list[dict] | None = None,
                  kw_vec=None, app_vecs: dict | None = None,
                  intent_group: dict | None = None) -> dict:
    """Summarise the competitive field by QUANTILES, not means.

    A field holding one giant and nine small apps behaves nothing like ten
    mid-size apps, and three floats capture that difference for a fraction of
    what attention over the full set would cost.
    """
    # Leave-one-out. An app that is itself in the SERP must not contribute to the
    # field it is being scored against, or its own installs leak into the features
    # and the task becomes artificially easy. At prediction time your app is not in
    # the field either, so LOO is also what makes train and serve agree.
    pool = (r for r in rows if r.get("position") and r.get("pkg") != exclude_pkg)
    ranked = sorted(pool, key=lambda r: r["position"])[:top_n]
    if not ranked:
        return {"n": 0, "installs_p10": 0, "installs_p50": 0, "installs_p90": 0,
                "rating_p50": 0, "reviews_p50": 0, "exact_match_count": 0,
                "age_p50": 0, "newcomers": 0, "newcomer_installs": 0,
                "staleness_p50": 0, "newest_entrant_age": 0, "velocity_p50": 0,
                "age_spread": 0, "age_known_frac": 0, "relevance_p50": 0,
                "relevant_count": 0, "installs_p50_relevant": 0, "has_featured": 0,
                "featured_relevance": 0.0, "featured_installs": 0,
                "weakest_installs": 0, "weakest_rank": 99, "weakest_title": None,
                "intent_rivals": 0, "intent_median": 0, "intent_weakest": 0,
                "intent_reach": 0.0, "intent_share": 0.0}
    inst = np.array([r.get("installs") or 0 for r in ranked], dtype="float64")
    rate = np.array([r.get("rating") or 0 for r in ranked], dtype="float64")
    revs = np.array([r.get("reviews") or 0 for r in ranked], dtype="float64")
    # Age in years, and how many of the field are recent arrivals. A recent
    # arrival holding a top slot is direct evidence the field is penetrable.
    ages = np.array([_days_since(r.get("released_at")) / 365.0 for r in ranked])
    stale = np.array([_days_since(r.get("updated_at")) for r in ranked])
    known = ages[ages > 0]
    recent = [r for r, a in zip(ranked, ages) if 0 < a <= 1.0]

    # Installs per year, floored at a quarter year so a brand new app does not
    # divide by ~0 and report an absurd rate.
    vel = np.array([(r.get("installs") or 0) / max(a, 0.25)
                    for r, a in zip(ranked, ages) if a > 0])

    rel = np.array([app_relevance(r, keyword, kw_vec, _vec_for(r, app_vecs))
                    for r in ranked])
    weakest = min(ranked, key=lambda r: (r.get("installs") or 0))
    on_intent = [r for r, v in zip(ranked, rel) if v >= RELEVANT]
    on_inst = np.array([r.get("installs") or 0 for r in on_intent], dtype="float64")

    return {
        "n": len(ranked),
        "relevance_p50": float(np.percentile(rel, 50)) if rel.size else 0.0,
        "relevant_count": len(on_intent),
        # The bar that actually matters. Falls back to 0 when nothing on the
        # page is on-intent, which is the strongest opportunity signal there is.
        "installs_p50_relevant": float(np.percentile(on_inst, 50)) if on_inst.size else 0.0,
        "weakest_installs": weakest.get("installs") or 0,
        "weakest_rank": weakest.get("position") or 99,
        "weakest_title": weakest.get("title"),
        **_intent_stats(intent_group, len(ranked), exclude_pkg),
        "has_featured": int(bool(featured) or any(r.get("featured") for r in rows)),
        "featured_relevance": max(
            (app_relevance(r, keyword, kw_vec, _vec_for(r, app_vecs))
             for r in (featured or [])), default=0.0),
        "featured_installs": max((r.get("installs") or 0 for r in (featured or [])),
                                 default=0),
        "age_p50": float(np.percentile(known, 50)) if known.size else 0.0,
        # 0.0 would read as "someone arrived today", so an unknown door reports
        # nothing and `age_known_frac` tells the caller not to trust it.
        "newest_entrant_age": float(known.min()) if known.size else 0.0,
        "age_spread": float(np.percentile(known, 90) - np.percentile(known, 10))
                      if known.size else 0.0,
        "age_known_frac": float(known.size / max(len(ranked), 1)),
        "velocity_p50": float(np.percentile(vel, 50)) if vel.size else 0.0,
        "newcomers": len(recent),
        "newcomer_installs": max((r.get("installs") or 0) for r in recent) if recent else 0,
        "staleness_p50": float(np.percentile(stale, 50)) if stale.size else 0.0,
        "installs_p10": float(np.percentile(inst, 10)),
        "installs_p50": float(np.percentile(inst, 50)),
        "installs_p90": float(np.percentile(inst, 90)),
        "rating_p50":   float(np.percentile(rate, 50)),
        "reviews_p50":  float(np.percentile(revs, 50)),
        "exact_match_count": int(sum(exact_match(r.get("title") or "", keyword)
                                     for r in ranked)),
    }


class Scaler:
    """Affine, per feature. A positive scale preserves every monotone guarantee."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    DEGENERATE = 1e-3      # below this a feature had no spread worth normalising
    CLIP = 12.0            # standard deviations; beyond this is not information

    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean = X.mean(0)
        std = X.std(0)
        # A feature that was CONSTANT when the scaler was fitted has no scale to
        # divide by. Flooring its std at 1e-6 meant a later value of 1.0 became
        # 1,000,000, which drove the network's output to 200,000 and made expm1
        # overflow to infinity in the downloads head. Leave such a feature
        # unscaled instead: it is on its own units, which is honest, rather than
        # amplified by a million, which is not.
        self.std = np.where(std < self.DEGENERATE, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Clip as a second line of defence. A feature that gains spread after the
        # scaler was fitted is normal as the dataset grows; letting it reach the
        # network as a 20-sigma spike is not.
        z = (X - self.mean) / self.std
        return np.clip(z, -self.CLIP, self.CLIP).astype("float32")

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def load(d):
        return Scaler(np.array(d["mean"], "float32"), np.array(d["std"], "float32"))


def vectorize(feats: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Split into the monotone block and the free block.

    'dec' features are negated here, so every weight on the monotone path can be
    constrained positive and one construction covers both directions.
    """
    xm = np.array([[f[n] for n in MONO_NAMES] for f in feats], dtype="float32") * SIGN
    xf = np.array([[f[n] for n in FREE_NAMES] for f in feats], dtype="float32")
    return xm, xf


# Features the DOWNLOADS head must not see.
#
# `app_velocity` is log1p(installs/age) and the downloads label is the same
# quantity, so handing it in meant the head learned to copy its own input rather
# than forecast anything. At inference the hypothetical app has zero installs,
# so it dutifully predicted zero downloads while the rank head called the same
# keyword a 94/100 build. log_installs and log_reviews leak the same answer more
# weakly.
#
# What remains is the honest question: an app matching like THIS, entering a
# field like THAT, at that rank - what do apps in that position actually get?
LEAKS_DOWNLOADS = {"app_velocity", "log_installs", "log_reviews"}

VEL_MASK_MONO = np.array([f.name not in LEAKS_DOWNLOADS for f in MONO], dtype="float32")
VEL_MASK_FREE = np.array([f.name not in LEAKS_DOWNLOADS for f in FREE], dtype="float32")
