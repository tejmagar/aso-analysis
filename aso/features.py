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
    Feat("field_no_intent", "free", "nothing on the page shares the leader's meaning"),
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
    # What a NEW app on this page actually earns, which is the question the
    # downloads head is being asked and was not being told.
    #
    # field_velocity_p50 above is the median across the whole page, so one
    # veteran with a lifetime total in the millions sets it however long ago
    # those installs arrived. On "energy ring" that put the estimate at 82 a day
    # for an entrant while the apps at ranks 1, 3 and 4 were taking 21, 53 and 5.
    # A ten-year-old app's lifetime average is not evidence about demand this
    # month; a recently launched one's is close to the only evidence there is.
    Feat("field_newcomer_velocity", "free",
         "installs per year among apps launched in the last year: what entering earns now"),
    Feat("field_newcomer_evidence", "free",
         "how many recent launches that rate is measured over"),
    # Growth measured over a real window, not divided out of a lifetime. Every
    # other install signal here is a total over an age, which averages an app's
    # whole history: one that took a million in year one and nothing since reads
    # exactly like one earning steadily today. Two dated snapshots say which.
    Feat("field_measured_velocity", "dec",
         "installs a day the page is actually gaining, measured between snapshots"),
    Feat("field_measured_evidence", "free",
         "how many apps on the page that rate is measured over"),
    # The same three rates, over the apps answering the same question rather
    # than the whole page. These are the ones that say anything about what
    # entering would earn: a page can rank giants that are not competing with
    # you, and their installs are not your ceiling.
    Feat("intent_velocity", "dec", "lifetime rate inside your meaning"),
    Feat("intent_recent_velocity", "free",
         "rate of apps launched within the year inside your meaning"),
    Feat("intent_measured_velocity", "dec",
         "measured growth inside your meaning, between two dated observations"),
    Feat("intent_evidence", "free", "how many on-intent apps those rates cover"),
    # Where Play's answer stops and its padding starts.
    #
    # A narrow phrase does not return a page of competitors, it returns the few
    # apps that answer it and then whatever fills the rest: "energy ring notch"
    # has seven real answers, then Roblox, Google and Clock. Nothing above told
    # the model that, so it read a 21-a-day niche through the download rate of
    # apps with three billion installs. The cut is the last rank of the unbroken
    # on-intent run from the top, so Play's own ordering decides it.
    Feat("intent_cut", "free", "last rank before the page stops answering the phrase"),
    Feat("intent_cut_share", "free", "what fraction of the visible page is real answers"),
    Feat("intent_relevance_drop", "free",
         "how far relevance falls from rank 1 to the bottom of the window"),
    # The recent on-intent apps IN RANK ORDER, which is the closest thing a page
    # holds to the question being asked: somebody published into this exact
    # meaning recently, and Play gave them a rank. Collapsed to a median these
    # say nothing about entering at a given depth, which is what is being asked.
    Feat("intent_recent_best_rank", "free",
         "best rank any recently published on-intent app holds"),
    Feat("intent_recent_rate_at_best", "free",
         "what that recently published app earns a year"),
    Feat("intent_neighbour_rate", "free",
         "what the on-intent app nearest your own rank earns a year"),
    # Rank and downloads are not the same axis, and conflating them is the
    # failure this guards. A ten-year-old giant can sit at rank 10 on a lifetime
    # average of thousands a day while genuinely new apps hold ranks 1 and 2 on
    # twenty. Reading the neighbour's rate without its age turns the first into
    # a forecast for the second, so the age travels with the rate and the model
    # decides what a rate from an app that old is worth.
    Feat("intent_neighbour_age", "free",
         "how old that nearest on-intent app is, in years"),
    Feat("intent_recent_neighbour_rate", "free",
         "what the nearest RECENTLY PUBLISHED on-intent app earns a year"),
    Feat("intent_recent_neighbour_gap", "free",
         "how many ranks away that recent app is: near is evidence, far is a guess"),
    # What a title match is actually worth on THIS page.
    #
    # A count of how many apps carry the phrase says nothing about whether
    # carrying it helps. On "falling blocks game" one app of thirty-six has the
    # phrase in its title, which reads as an open door - and the truth is the
    # opposite: Tetris holds rank 1 without a word of the phrase in its name,
    # and the apps that do carry it start at rank 3. That is Play saying it
    # knows what the phrase means and that spelling it out does not buy the top.
    #
    # The shape is common in games: a famous title holds the top and the
    # phrase-matching imitations queue up beneath it.
    Feat("leader_match", "free",
         "does the app Play ranked first carry the phrase in its title"),
    Feat("leader_relevance", "free", "how close the leader is to the phrase at all"),
    Feat("leader_lead", "free",
         "how far the leader towers over the rest of the page, in installs"),
    Feat("first_match_rank", "free",
         "where the highest app carrying the phrase sits, 0 if none does"),
    Feat("match_starts_below", "free",
         "how many ranks separate the top of the page from the first literal match"),
    # What a SLOT earns, regardless of who is judged to share your meaning.
    #
    # The intent features answer "what do apps like mine earn", and they are
    # only as good as the grouping. On "falling blocks game" the group came out
    # as the two official Tetris apps - the leader and its sequel, alike by name
    # and alike to nothing a newcomer could be - so every on-intent rate read
    # sixteen thousand a day while the page's own median was twenty.
    #
    # These ask a question with no judgement in it: what do the apps sitting
    # where you would sit actually earn. A grouping mistake cannot reach them,
    # which is the point - they are the floor under an answer that otherwise
    # rests entirely on getting the meaning right.
    Feat("band_rate", "dec", "what apps within a couple of ranks of you earn"),
    Feat("band_recent_rate", "free",
         "the same, for the ones that launched within the year"),
    Feat("band_evidence", "free", "how many apps that neighbourhood rate covers"),
    Feat("below_rate", "dec", "what the apps ranked beneath you earn"),
    # Whether this page's numbers were earned at the pace their ages imply.
    #
    # A page where apps of the same age earn wildly different amounts is a page
    # somebody is buying installs on, and its rates are not a standing start.
    # A page where same-age apps earn alike is one where the figures mean what
    # they appear to mean. Neither is asserted here - the spread is reported and
    # the model decides what a spread that size is worth.
    Feat("inorganic_p90", "free",
         "how inorganic the fastest-growing apps on this page look"),
    Feat("inorganic_p50", "free", "the same at the middle of the page"),
    Feat("leader_inorganic", "free",
         "how much of the leader's rate its own age does not account for"),
    Feat("leader_breadth", "free",
         "how many phrases the app holding the top is reachable through"),
    Feat("breadth_gap", "free",
         "how much wider the leader's reach is than the rest of the page's"),
    # The newest app here, and what it has earned.
    #
    # The single strongest piece of evidence about what entering pays, because
    # it is the only app on the page that started from nothing recently enough
    # for its number to still describe the market as it stands. Everything else
    # is a rate averaged over years that are gone.
    Feat("newest_rate", "dec", "what the most recently published app here earns"),
    Feat("newest_rank", "free", "where that app sits on the page"),
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
        "field_no_intent": float(field.get("no_intent") or 0),
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
        "field_newcomer_velocity": math.log1p(field.get("newcomer_velocity") or 0.0),
        "field_measured_velocity": math.log1p(field.get("measured_velocity") or 0.0),
        "intent_velocity": math.log1p(field.get("intent_velocity") or 0.0),
        "intent_recent_velocity": math.log1p(field.get("intent_recent_velocity") or 0.0),
        "intent_measured_velocity": math.log1p(field.get("intent_measured_velocity") or 0.0),
        "intent_evidence": float(field.get("intent_evidence") or 0),
        "intent_cut": float(field.get("intent_cut") or 0),
        "intent_cut_share": float(field.get("intent_cut_share") or 0.0),
        "intent_relevance_drop": float(field.get("intent_relevance_drop") or 0.0),
        "intent_recent_best_rank": float(field.get("intent_recent_best_rank") or 0),
        "intent_recent_rate_at_best":
            math.log1p(field.get("intent_recent_rate_at_best") or 0.0),
        "intent_neighbour_rate": math.log1p(field.get("intent_neighbour_rate") or 0.0),
        "intent_neighbour_age": float(field.get("intent_neighbour_age") or 0.0),
        "intent_recent_neighbour_rate":
            math.log1p(field.get("intent_recent_neighbour_rate") or 0.0),
        "intent_recent_neighbour_gap":
            float(field.get("intent_recent_neighbour_gap") or 0.0),
        "leader_match": float(field.get("leader_match") or 0.0),
        "leader_relevance": float(field.get("leader_relevance") or 0.0),
        "leader_lead": float(field.get("leader_lead") or 0.0),
        "first_match_rank": float(field.get("first_match_rank") or 0),
        "match_starts_below": float(field.get("match_starts_below") or 0),
        "band_rate": math.log1p(field.get("band_rate") or 0.0),
        "band_recent_rate": math.log1p(field.get("band_recent_rate") or 0.0),
        "band_evidence": float(field.get("band_evidence") or 0),
        "below_rate": math.log1p(field.get("below_rate") or 0.0),
        "inorganic_p90": float(field.get("inorganic_p90") or 0.0),
        "inorganic_p50": float(field.get("inorganic_p50") or 0.0),
        "leader_inorganic": float(field.get("leader_inorganic") or 0.0),
        "leader_breadth": math.log1p(field.get("leader_breadth") or 0),
        "breadth_gap": float(field.get("breadth_gap") or 0.0),
        "newest_rate": math.log1p(field.get("newest_rate") or 0.0),
        "newest_rank": float(field.get("newest_rank") or 0),
        "field_measured_evidence": float(field.get("measured_count") or 0),
        # Paired with the rate, because a median over one app is a number and not
        # a measurement, and the model should be able to tell the difference.
        "field_newcomer_evidence": float(field.get("newcomers") or 0),
        "field_staleness_p50": float(field.get("staleness_p50") or 0.0),
        "field_age_spread":    float(field.get("age_spread") or 0.0),
        "field_age_known":     float(field.get("age_known_frac") or 0.0),
        "field_spread":       math.log1p(p90) - math.log1p(p10),
        "kw_tokens":          float(len(kw_t)),
        "days_since_release": _days_since(app.get("released_at")) / 365.0,
        "days_since_update":  _days_since(app.get("updated_at")) / 365.0,
        **{k: float((sugg or {}).get(k, v)) for k, v in SUGG_DEFAULTS.items()},
    }


# How fast the inorganic score approaches 1. Not a threshold: the curve is
# smooth, so there is no value at which an app suddenly counts as bought. It
# only sets the scale - a tenfold lead over its cohort reads around 0.5 and a
# thousandfold around 0.9 - and what any score is worth stays the model's to
# learn.
INORGANIC_SCALE = 3.0


def _inorganic(row, rows) -> float:
    """0 to 1: how much of this app's rate its age does not account for.

    Installs bought with ads cannot be observed, but what they leave behind can.
    Six months old and earning a thousand a day, beside other six-month-olds
    earning three, is not a standing start - and treating it as evidence of what
    entering pays is how an estimate ends up an order of magnitude high.

    Peers are apps within a factor of two of this one's age, so the comparison
    is a standing start of the same length rather than the whole page, where a
    ten-year-old incumbent would drown it.

    It is NOT called ran_ads, and it is not a probability of anything. The same
    shape is left by a budget, by a genuine hit, and by a brand nobody could be:
    on "falling blocks game" the second Tetris title scores 0.95 at eight months
    old, and Tetris itself scores 0.89. So this reports the distance and says
    nothing about the cause. Whether a high score means "bought, discount it" or
    "big market, that is the prize" is learned from which pages actually
    produced which downloads.

    Only the fast side counts. An app earning less than its cohort is an app
    that is doing badly, which the rate itself already says; folding it in here
    would put "quiet" and "bought" at opposite ends of one axis, and they are
    not opposites.
    """
    age = _days_since(row.get("released_at")) / 365.0
    if age <= 0:
        return 0.0
    peers = []
    for r in rows:
        if r.get("pkg") == row.get("pkg"):
            continue
        a = _days_since(r.get("released_at")) / 365.0
        if a > 0 and 0.5 * age <= a <= 2.0 * age:
            peers.append(_rate_of(r))
    if not peers:
        return 0.0
    gap = math.log1p(_rate_of(row)) - math.log1p(float(np.median(peers)))
    if gap <= 0:
        return 0.0
    return 1.0 - math.exp(-gap / INORGANIC_SCALE)


def _cos(a, b) -> float:
    """Cosine between two app vectors, 0.0 when either is missing."""
    if a is None or b is None:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _rate_of(row) -> float:
    """One app's installs per year. Floored at a quarter year so an app released
    last week does not divide by ~0 and report an absurd rate."""
    age = _days_since(row.get("released_at")) / 365.0
    return (row.get("installs") or 0) / max(age, 0.25) if age > 0 else 0.0


def _rates(rows) -> dict:
    """How fast a set of apps is growing, three ways.

    Used for the whole page and again for the meaning being entered, because
    those answer different questions and only the second one is about you. A
    page can be full of giants answering something else: on "down detector" the
    ranked list is Speedtest, Norton and Fing, none of which tells you what a
    down-detector app earns.

    `measured` is growth between two dated observations, `recent` is the
    lifetime rate of apps launched within the year, and `all` is the lifetime
    rate of everything. They are in decreasing order of how much they say about
    entering now, which is the order the model should weigh them in.
    """
    ages = [_days_since(r.get("released_at")) / 365.0 for r in rows]
    lifetime = [(r.get("installs") or 0) / max(a, 0.25)
                for r, a in zip(rows, ages) if a > 0]
    fresh = [(r.get("installs") or 0) / max(a, 0.25)
             for r, a in zip(rows, ages) if 0 < a <= 1.0]
    measured = [r["measured_per_day"] * 365.0 for r in rows
                if r.get("measured_per_day") is not None]
    return {
        "all": float(np.median(lifetime)) if lifetime else 0.0,
        "recent": float(np.median(fresh)) if fresh else 0.0,
        "recent_n": len(fresh),
        "measured": float(np.median(measured)) if measured else 0.0,
        "measured_n": len(measured),
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
                  intent_group: dict | None = None,
                  at_rank: int | None = None) -> dict:
    """Summarise the competitive field by QUANTILES, not means.

    A field holding one giant and nine small apps behaves nothing like ten
    mid-size apps, and three floats capture that difference for a fraction of
    what attention over the full set would cost.
    """
    # Leave-one-out. An app that is itself in the SERP must not contribute to the
    # field it is being scored against, or its own installs leak into the features
    # and the task becomes artificially easy. At prediction time your app is not in
    # the field either, so LOO is also what makes train and serve agree.
    pool = [r for r in rows if r.get("position") and r.get("pkg") != exclude_pkg]
    # The WHOLE page, in order, kept alongside the top-N window below. The
    # neighbourhood features need it: computed over the window, an app at rank
    # thirty has no neighbours inside the top ten and scores zero on every one
    # of them, while an app at rank three scores a real value - so the feature
    # stops meaning "what my neighbours earn" and starts meaning "am I in the
    # top ten", which is the label.
    whole = sorted(pool, key=lambda r: r["position"])
    ranked = sorted(pool, key=lambda r: r["position"])[:top_n]
    if not ranked:
        return {"n": 0, "installs_p10": 0, "installs_p50": 0, "installs_p90": 0,
                "rating_p50": 0, "reviews_p50": 0, "exact_match_count": 0,
                "age_p50": 0, "newcomers": 0, "newcomer_installs": 0,
                "newcomer_velocity": 0.0, "measured_velocity": 0.0,
                "measured_count": 0, "intent_velocity": 0.0,
                "intent_recent_velocity": 0.0, "intent_measured_velocity": 0.0,
                "intent_evidence": 0, "intent_cut": 0, "intent_cut_share": 0.0,
                "intent_relevance_drop": 0.0, "intent_recent_best_rank": 0,
                "intent_recent_rate_at_best": 0.0, "intent_neighbour_rate": 0.0,
                "intent_neighbour_age": 0.0, "intent_recent_neighbour_rate": 0.0,
                "intent_recent_neighbour_gap": 0.0,
                "leader_match": 0.0, "leader_relevance": 0.0, "leader_lead": 0.0,
                "first_match_rank": 0, "match_starts_below": 0,
                "band_rate": 0.0, "band_recent_rate": 0.0, "band_evidence": 0,
                "below_rate": 0.0, "inorganic_p90": 0.0, "inorganic_p50": 0.0,
                "leader_inorganic": 0.0, "leader_breadth": 0, "breadth_gap": 0.0,
                "newest_rate": 0.0, "newest_rank": 0,
                "staleness_p50": 0, "newest_entrant_age": 0, "velocity_p50": 0,
                "age_spread": 0, "age_known_frac": 0, "relevance_p50": 0,
                "relevant_count": 0, "installs_p50_relevant": 0,
                "no_intent": 0, "has_featured": 0,
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

    rel = np.array([app_relevance(r, keyword, kw_vec, _vec_for(r, app_vecs))
                    for r in ranked])
    weakest = min(ranked, key=lambda r: (r.get("installs") or 0))

    # Who counts as competition.
    #
    # Membership of the meaning being scored against, when there is one. That
    # meaning is Play's own answer: the group holding the top of the page, or
    # for an app already ranking, its own group.
    #
    # The fallback, and what this used to do everywhere, is to score each app's
    # similarity to the keyword itself and keep whatever clears a threshold.
    # That fails on exactly the phrases where the answer matters most. A short
    # phrase embeds weakly, so on "native cam" every app came in under the line,
    # the page read as having no competitors at all, and three lower-is-better
    # features hit zero together and called it wide open. Play had answered
    # perfectly clearly by ranking a camera app first.
    if intent_group and intent_group.get("packages"):
        members = set(intent_group["packages"])
        on_intent = [r for r in ranked if r.get("pkg") in members]
    else:
        on_intent = [r for r, v in zip(ranked, rel) if v >= RELEVANT]
    on_inst = np.array([r.get("installs") or 0 for r in on_intent], dtype="float64")

    # Growth, for the page and for the meaning. Separate because a page can rank
    # apps that are not competing with you at all, and their numbers are not
    # evidence about what you would earn.
    page, mine = _rates(ranked), _rates(on_intent)

    # Where the answer stops and the filler starts.
    #
    # An unbroken run of on-intent apps from rank 1, and the rank it ends at.
    # Contiguity is the whole point: Play puts what it believes answers the
    # phrase first, so a break in the run is the page changing subject. Anything
    # past it is padding, and its download rates are not a market a new app
    # could earn from.
    on_ranks = sorted(r["position"] for r in on_intent)
    cut = 0
    for i, pos in enumerate(on_ranks, start=1):
        if pos != i:
            break
        cut = i

    # How hard relevance falls across the window. A page that answers all the
    # way down and a page that gives up after two both have on-intent apps; only
    # this separates them.
    drop = float(rel[0] - rel[-1]) if rel.size > 1 else 0.0

    # The recent on-intent apps, in rank order. A newcomer holding rank 2 says
    # something a newcomer at rank 30 does not, and a median over the cohort
    # throws exactly that away.
    recent_intent = sorted(
        (r for r in on_intent
         if 0 < _days_since(r.get("released_at")) / 365.0 <= 1.0),
        key=lambda r: r["position"])
    best = recent_intent[0] if recent_intent else None

    # What the on-intent app nearest your own rank earns. For a row in the SERP
    # that rank is its own; for an app not on the page it is the rank it would
    # enter at, which the caller supplies after the rank head has run.
    own_rank = at_rank
    if own_rank is None and exclude_pkg:
        own = next((r for r in rows
                    if r.get("pkg") == exclude_pkg and r.get("position")), None)
        own_rank = own["position"] if own else None
    if own_rank and on_intent:
        near = min(on_intent, key=lambda r: abs(r["position"] - own_rank))
        neighbour = _rate_of(near)
        neighbour_age = _days_since(near.get("released_at")) / 365.0
    else:
        near, neighbour, neighbour_age = None, 0.0, 0.0

    # The same neighbour, restricted to apps that actually launched recently.
    # This is the one the question is really about: a rate earned by an app that
    # started from nothing near your rank, rather than a lifetime average banked
    # by an app that has held its slot since before the phrase existed. The gap
    # is how far away it sits, because the evidence is only as good as it is near.
    # The neighbourhood: apps sitting where this one would sit. Two ranks either
    # side, which on a ten-slot window is the part of the page a reader actually
    # compares against - wider and it is the page again, narrower and one
    # unusual neighbour is the whole answer.
    if own_rank:
        near_rows = [r for r in whole if abs(r["position"] - own_rank) <= 2
                     and r.get("position") != own_rank]
        # Capped, so a page of forty does not make "below me" mean "the page".
        below_rows = [r for r in whole
                      if own_rank < r["position"] <= own_rank + 5]
    else:
        near_rows, below_rows = [], []
    band = _rates(near_rows)
    below = _rates(below_rows)

    if own_rank and recent_intent:
        rnear = min(recent_intent, key=lambda r: abs(r["position"] - own_rank))
        recent_neighbour = _rate_of(rnear)
        recent_gap = abs(rnear["position"] - own_rank)
    else:
        recent_neighbour, recent_gap = 0.0, 0.0

    return {
        "n": len(ranked),
        "relevance_p50": float(np.percentile(rel, 50)) if rel.size else 0.0,
        "relevant_count": len(on_intent),
        # Three of the features above collapse to zero together when nothing is
        # on-intent, and each reads lower-is-better on its own, so a page we
        # could not read scored as the best opportunity available. This says
        # which kind of zero it is.
        "no_intent": int(len(on_intent) == 0),
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
        "velocity_p50": page["all"],
        # The same rate, over recent arrivals only. Their installs were earned
        # under today's demand rather than averaged across a decade of it.
        "newcomer_velocity": page["recent"],
        # Only the apps actually measured; the rest are absent rather than zero,
        # so a page with one measured app cannot look like a page standing still.
        "measured_velocity": page["measured"],
        "measured_count": page["measured_n"],
        "newcomers": len(recent),
        # And all of it again over the apps answering the same question, which
        # is the set a new app would actually be joining. Computed and then
        # dropped on the floor in the first version of this, so every one of
        # them trained as a constant zero.
        "intent_velocity": mine["all"],
        "intent_recent_velocity": mine["recent"],
        "intent_measured_velocity": mine["measured"],
        "intent_evidence": len(on_intent),
        "intent_cut": cut,
        "intent_cut_share": float(cut / max(len(ranked), 1)),
        "intent_relevance_drop": drop,
        "intent_recent_best_rank": best["position"] if best else 0,
        "intent_recent_rate_at_best": _rate_of(best) if best else 0.0,
        "intent_neighbour_rate": neighbour,
        "intent_neighbour_age": neighbour_age,
        "intent_recent_neighbour_rate": recent_neighbour,
        "intent_recent_neighbour_gap": recent_gap,
        **_inorganic_page(ranked),
        **_reach_and_newest(ranked),
        "band_rate": band["all"],
        "band_recent_rate": band["recent"],
        "band_evidence": len(near_rows),
        "below_rate": below["all"],
        "newcomer_installs": max((r.get("installs") or 0) for r in recent) if recent else 0,
        "staleness_p50": float(np.percentile(stale, 50)) if stale.size else 0.0,
        "installs_p10": float(np.percentile(inst, 10)),
        "installs_p50": float(np.percentile(inst, 50)),
        "installs_p90": float(np.percentile(inst, 90)),
        "rating_p50":   float(np.percentile(rate, 50)),
        "reviews_p50":  float(np.percentile(revs, 50)),
        "exact_match_count": int(sum(exact_match(r.get("title") or "", keyword)
                                     for r in ranked)),
        **_leader(ranked, keyword, rel, inst),
    }


def _reach_and_newest(ranked) -> dict:
    """Who is reachable how many ways, and what the newest arrival earns."""
    if not ranked:
        return {"leader_breadth": 0, "breadth_gap": 0.0,
                "newest_rate": 0.0, "newest_rank": 0}
    reach = [r.get("keyword_breadth") or 0 for r in ranked]
    rest = reach[1:] or reach
    fresh = [r for r in ranked if _days_since(r.get("released_at")) > 0]
    newest = min(fresh, key=lambda r: _days_since(r.get("released_at"))) if fresh else None
    return {
        "leader_breadth": reach[0],
        # In log units, so "twice as reachable" and "ten times" differ.
        "breadth_gap": math.log1p(reach[0]) - math.log1p(float(np.median(rest))),
        "newest_rate": _rate_of(newest) if newest else 0.0,
        "newest_rank": newest["position"] if newest else 0,
    }


def _inorganic_page(ranked) -> dict:
    """How much of this page's growth its apps' ages do not account for.

    A page where same-age apps earn alike is one whose figures mean what they
    appear to mean. A page where they differ by orders of magnitude is one
    somebody is buying installs on, and its rates are not a standing start.
    """
    if not ranked:
        return {"inorganic_p90": 0.0, "inorganic_p50": 0.0, "leader_inorganic": 0.0}
    scores = [_inorganic(r, ranked) for r in ranked]
    return {
        "inorganic_p90": float(np.percentile(scores, 90)),
        "inorganic_p50": float(np.percentile(scores, 50)),
        "leader_inorganic": scores[0],
    }


def _leader(ranked, keyword, rel, inst) -> dict:
    """What holding the top of this page took.

    Read together these separate two pages that look identical to a count of
    title matches: one where the phrase is unclaimed and a new app can take it,
    and one where a famous app owns the meaning and the apps spelling the phrase
    out are queued below it. The second is most of the games category.
    """
    if not ranked:
        return {"leader_match": 0.0, "leader_relevance": 0.0, "leader_lead": 0.0,
                "first_match_rank": 0, "match_starts_below": 0}
    top = ranked[0]
    matches = [i for i, r in enumerate(ranked, start=1)
               if exact_match(r.get("title") or "", keyword)]
    first = matches[0] if matches else 0
    others = inst[1:] if inst.size > 1 else inst
    lead = (math.log1p(inst[0]) - math.log1p(float(np.percentile(others, 50)))
            if others.size else 0.0)
    return {
        "leader_match": 1.0 if exact_match(top.get("title") or "", keyword) else 0.0,
        "leader_relevance": float(rel[0]) if rel.size else 0.0,
        # In log units, so "ten times the median" and "a thousand times" are
        # different numbers rather than both being "very large".
        "leader_lead": float(lead),
        "first_match_rank": first,
        # 0 when the leader itself matches, and large when the phrase is spelled
        # out only well down the page - which is the case that reads as an open
        # door and is not one.
        "match_starts_below": max(first - 1, 0) if first else 0,
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


# ---------------------------------------------------------------------------
# The page as a SET, not as a summary.
#
# Every field feature above is an aggregate: a median, a p90, a count, a rate
# over some subset we chose. Aggregates cannot express the relation that
# actually answers "how many downloads would I get", which is per-app and reads
# roughly: THIS app, at THIS rank, THIS old, earns THIS much. Collapsing the
# page to quantiles destroys it, and every attempt to rescue it by hand - the
# relevant subset, the on-intent subset, the nearest neighbour by rank - is
# another formula written by us about a page that Play assembled for reasons
# including gap-filling and taste. Those are not recoverable by arithmetic over
# medians, so the arithmetic stops here: the rows go to the model whole and it
# learns which ones matter.
#
# One row per app, fixed width, zero-padded to MAX_APPS with a mask.
PAGE_FEATS = [
    "rank",              # where Play put it
    "rank_gap",          # its rank minus yours: how the attention finds neighbours
    "gap_known",         # 0 when the asking app has no rank yet
    "log_installs",
    "log_rate",          # installs per year over its life
    "age",               # years since release
    "age_known",
    "log_measured",      # installs per year measured between two snapshots
    "measured_known",    # measured at zero and never measured are different
    "rating",
    "log_reviews",
    "relevance",         # cosine to the keyword
    "on_intent",         # in the group answering the same question
    # Intent is not a boundary, it is a slope.
    #
    # Play puts what it is most confident about first and the match fades down
    # the page, so on "down detector" the outage app at rank 1 and the speed
    # test at rank 2 are both answering the phrase while rank 18 is not. A
    # clustering threshold has to draw a line somewhere in that slope, and
    # wherever it draws it the copy ends up claiming one app answers the phrase
    # when two clearly do. These two carry the slope itself - how close this app
    # is to what Play ranked first, and to the phrase - and the rank column
    # carries the position, so the model can learn the fade instead of being
    # handed our guess at where it ends.
    "sim_to_top",        # cosine to the app Play ranked first
    "top_rank_gap",      # ranks between this app and that one
    # 0 to 1: how much of this app's rate its age does not account for. High is
    # the shape paid installs leave, and also the shape a hit leaves, which is
    # why it is a measurement and not a verdict.
    "inorganic",
    # How many different phrases this app is reachable through. The app that
    # owns a concept is reachable many ways; one renamed to avoid a trademark
    # is reachable only through the generic phrase.
    "breadth",
    "title_exact",       # carries the phrase in its title
    "staleness",         # years since it last shipped an update
    "featured",          # Play's promoted card
]
MAX_APPS = 12

# Every name distinct, and none of them accidentally glued to the next.
#
# A missing comma between two strings in a list is not an error in Python, it is
# concatenation: the list quietly loses an entry and the only symptom is a
# shape mismatch thrown much later, from a line that looks fine. This shipped
# once - "breadth" "title_exact" became one name, and every analysis died with
# "could not broadcast input array from shape (20,) into shape (19,)".
assert len(set(PAGE_FEATS)) == len(PAGE_FEATS), "duplicate page feature"
assert not [n for n in PAGE_FEATS if len(n) > 24], \
    f"page feature name too long, likely a missing comma: {PAGE_FEATS}"


def page_matrix(rows: list[dict], keyword: str, kw_vec=None,
                app_vecs: dict | None = None, intent_group: dict | None = None,
                exclude_pkg: str | None = None, at_rank: int | None = None,
                max_apps: int = MAX_APPS) -> tuple[np.ndarray, np.ndarray]:
    """The ranked page as (MAX_APPS, len(PAGE_FEATS)) plus a presence mask.

    Leave-one-out on the same terms as compute_field: an app is never shown its
    own row, because at prediction time the asking app is not on the page.
    """
    pool = [r for r in rows if r.get("position") and r.get("pkg") != exclude_pkg]
    ranked = sorted(pool, key=lambda r: r["position"])[:max_apps]
    members = set((intent_group or {}).get("packages") or ())
    kw = keyword.strip().lower()

    # The anchor is whatever Play ranked first among the apps actually present,
    # which after leave-one-out may not be position 1.
    top = ranked[0] if ranked else None
    top_vec = _vec_for(top, app_vecs) if top is not None else None
    top_rank = top["position"] if top is not None else 0

    X = np.zeros((max_apps, len(PAGE_FEATS)), dtype="float32")
    mask = np.zeros(max_apps, dtype="float32")
    for i, r in enumerate(ranked):
        age = _days_since(r.get("released_at")) / 365.0
        measured = r.get("measured_per_day")
        X[i] = [
            r["position"] / 10.0,
            ((r["position"] - at_rank) / 10.0) if at_rank else 0.0,
            1.0 if at_rank else 0.0,
            math.log1p(r.get("installs") or 0),
            math.log1p(_rate_of(r)),
            min(age, 15.0) if age > 0 else 0.0,
            1.0 if age > 0 else 0.0,
            math.log1p((measured or 0.0) * 365.0),
            1.0 if measured is not None else 0.0,
            float(r.get("rating") or 0.0),
            math.log1p(r.get("reviews") or 0),
            app_relevance(r, keyword, kw_vec, _vec_for(r, app_vecs)),
            1.0 if r.get("pkg") in members else 0.0,
            _cos(_vec_for(r, app_vecs), top_vec),
            (r["position"] - top_rank) / 10.0,
            _inorganic(r, ranked),
            math.log1p(r.get("keyword_breadth") or 0),
            1.0 if kw in (r.get("title") or "").lower() else 0.0,
            min(_days_since(r.get("updated_at")) / 365.0, 10.0),
            float(r.get("featured") or 0),
        ]
        mask[i] = 1.0
    return X, mask
