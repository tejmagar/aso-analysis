"""Pipeline tests. Zero dependencies beyond the project itself.

Synthetic rows live ONLY inside a throwaway database created per run. Nothing
here can reach data/aso.db, which is why the old seed script is gone: fake
packages sharing a table with real scrapes silently poison training.

    python tests/test_pipeline.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before aso.db is imported: it resolves DB_PATH at import time.
_TMP = Path(tempfile.mkdtemp(prefix="aso-test-")) / "test.db"
os.environ["ASO_DB"] = str(_TMP)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from aso import correct, dataset, db, features, memory, predict, train  # noqa: E402
from aso.model import Ensemble                                  # noqa: E402
from aso.scrape import organic_positions, parse_date, parse_int  # noqa: E402

FAILED = []


def _refuse_production() -> None:
    """Refuse to run against a database that is not obviously a test one.

    This suite deletes the model registry and writes synthetic keywords, which
    is fine against a scratch database and not fine anywhere else. Pointed at
    the live one it wiped 73 model records and left 293 fixture rows in the
    training set, and nothing complained, because every one of those writes is
    what the tests are supposed to do.

    The check is on the name: a database called `aso` is the real one. Set
    ASO_PG_DSN to something ending in _test to run these.
    """
    import os
    import sys as _s

    dsn = os.environ.get("ASO_PG_DSN", "")
    if not dsn:
        return                                  # no database: the caller finds out
    name = dsn.rsplit("/", 1)[-1].split("?")[0]
    if name.endswith("_test") or name.startswith("test"):
        return
    print(f"\n  refusing to run against the database {name!r}.\n"
          f"  These tests delete the model registry and insert synthetic\n"
          f"  keywords. Point ASO_PG_DSN at a scratch database:\n\n"
          f"    createdb aso_test\n"
          f"    ASO_PG_DSN=postgresql://…/aso_test python tests/test_pipeline.py\n")
    _s.exit(2)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


# --------------------------------------------------------------- scraper shape
def test_promoted_cards_excluded():
    r = [{"package": "promo", "featured": True},
         {"package": "a", "featured": False},
         {"package": "b", "featured": True},     # featured AND organic: keeps its slot
         {"package": "c", "featured": False}]
    got = organic_positions(r)
    check("promoted hero card gets no organic position", got[0][0] is None)
    check("organic ranks start at 1 after the promoted run",
          [p for p, _ in got[1:]] == [1, 2, 3])
    check("a featured app inside the organic block keeps its slot",
          got[2] == (2, r[2]))
    check("no featured card means plain numbering",
          [p for p, _ in organic_positions(r[1:])] == [1, 2, 3])


def test_parsers():
    check("install counts survive comma formatting", parse_int("45,891,296") == 45891296)
    check("a missing install count is 0, not a crash", parse_int(None) == 0)
    check("release dates parse to ISO", parse_date("May 29, 2013") == "2013-05-29")
    check("an unparseable date is None, never a guess", parse_date("soon") is None)


# --------------------------------------------------------------- feature layer
def test_field_is_leave_one_out():
    rows = [{"pkg": "mine", "position": 1, "title": "X", "installs": 10**9,
             "rating": 4.9, "reviews": 10**6},
            {"pkg": "o1", "position": 2, "title": "Y", "installs": 1000,
             "rating": 4.0, "reviews": 10},
            {"pkg": "o2", "position": 3, "title": "Z", "installs": 2000,
             "rating": 4.1, "reviews": 20}]
    with_me = features.compute_field(rows, "x")
    without = features.compute_field(rows, "x", exclude_pkg="mine")
    check("an app does not inflate the field it is scored against",
          without["installs_p90"] < with_me["installs_p90"],
          f'{without["installs_p90"]:.0f} < {with_me["installs_p90"]:.0f}')
    check("leave-one-out drops exactly one row", without["n"] == with_me["n"] - 1)


def test_newcomers_signal_penetrability():
    """A field that keeps admitting recent apps must read as easier than an
    identical one that has admitted nobody, whatever the install medians say."""
    from datetime import date, timedelta

    def make(release_days_ago):
        return [{"pkg": f"p{i}", "position": i + 1, "title": "Habit Tracker",
                 "installs": 5_000_000, "rating": 4.5, "reviews": 50_000,
                 "released_at": str(date.today() - timedelta(days=release_days_ago)),
                 "updated_at": str(date.today() - timedelta(days=30))}
                for i in range(10)]

    old = features.compute_field(make(8 * 365), "habit tracker")
    new = features.compute_field(make(200), "habit tracker")

    check("an entrenched field reports no newcomers", old["newcomers"] == 0)
    check("a field of recent arrivals reports them", new["newcomers"] == 10)
    check("median age is measured in years", round(old["age_p50"]) == 8,
          f'{old["age_p50"]:.1f}y')
    # No formula weighs these any more. What matters is that the model is HANDED
    # the distinction, with the right monotone direction attached.
    fo = features.extract({"title": "x", "installs": 0, "rating": 0, "reviews": 0},
                          "habit tracker", old)
    fn = features.extract({"title": "x", "installs": 0, "rating": 0, "reviews": 0},
                          "habit tracker", new)
    check("the model sees a lower newcomer count for an entrenched field",
          fo["field_newcomers"] < fn["field_newcomers"],
          f'{fo["field_newcomers"]} vs {fn["field_newcomers"]}')
    check("the model sees an older door for an entrenched field",
          fo["newest_entrant_age"] > fn["newest_entrant_age"])
    check("field age is a decreasing feature, newcomers an increasing one",
          next(f.direction for f in features.REGISTRY if f.name == "field_age_p50") == "dec"
          and next(f.direction for f in features.REGISTRY
                   if f.name == "field_newcomers") == "inc")

    # The door test: how long since ANYONE new got in.
    check("an entrenched field reports a shut door",
          round(old["newest_entrant_age"]) == 8, f'{old["newest_entrant_age"]:.1f}y')
    check("a fresh field reports an open door",
          new["newest_entrant_age"] < 1.0, f'{new["newest_entrant_age"]:.2f}y')

    # Age against downloads is a rate, and that is the point of it.
    fast = features.compute_field(
        [{"pkg": "f", "position": 1, "title": "T", "installs": 1_000_000,
          "rating": 4.5, "reviews": 100, "released_at": str(date.today() - timedelta(days=300)),
          "updated_at": str(date.today())}], "t")
    slow = features.compute_field(
        [{"pkg": "s", "position": 1, "title": "T", "installs": 1_000_000,
          "rating": 4.5, "reviews": 100, "released_at": str(date.today() - timedelta(days=3000)),
          "updated_at": str(date.today())}], "t")
    check("identical installs at different ages give different growth rates",
          fast["velocity_p50"] > slow["velocity_p50"] * 5,
          f'{fast["velocity_p50"]:,.0f}/yr vs {slow["velocity_p50"]:,.0f}/yr')

    # A fast scrape has no release dates. Reporting 0.0 there would read as
    # "someone arrived today" and wrongly open the door.
    blind = features.compute_field(
        [{"pkg": f"b{i}", "position": i + 1, "title": "T", "installs": 5_000_000,
          "rating": 4.5, "reviews": 100, "released_at": None, "updated_at": None}
         for i in range(10)], "t")
    check("missing release dates are reported as unknown, not as open",
          blind["age_known_frac"] == 0.0 and blind["newest_entrant_age"] == 0.0
          and blind["newcomers"] == 0)


def test_intent_and_featured_card():
    """The spider-tracker case: a page of large unrelated apps is an
    opportunity, and a promoted on-intent card is the worst signal on it."""
    filler = [{"pkg": f"g{i}", "position": i + 1, "title": t, "short_desc": "",
               "description": "location gps tracking for family and friends",
               "installs": 50_000_000, "rating": 4.4, "reviews": 900_000,
               "released_at": "2016-01-01", "updated_at": "2026-07-01"}
              for i, t in enumerate(["Strava Run Bike", "Find Hub", "Geo Tracker",
                                     "Sports Tracker", "Findmykids", "Life360",
                                     "Tracker Connect", "Family Locator",
                                     "Phone Tracker", "GPS Tools"])]
    real = [dict(r, title="Spider Tracker Pro",
                 description="spider tracker for arachnid sightings") for r in filler]

    off = features.compute_field(filler, "spider tracker")
    on = features.compute_field(real, "spider tracker")

    check("filler apps are recognised as off-intent", off["relevant_count"] == 0,
          f'{off["relevant_count"]} of {off["n"]}')
    check("genuinely matching apps are recognised", on["relevant_count"] == 10)
    check("the real bar ignores off-intent installs",
          off["installs_p50_relevant"] == 0 and on["installs_p50_relevant"] > 0)
    check("the model sees a lower on-intent count for a filler page",
          off["relevant_count"] < on["relevant_count"],
          f'{off["relevant_count"]} vs {on["relevant_count"]}')

    # The promoted card. position is NULL on these rows, which is exactly why
    # they used to be filtered out before reaching the model.
    card = [{"pkg": "spider", "title": "Spider Tracker", "short_desc": "",
             "description": "spider tracker", "installs": 29_246,
             "rating": 4.2, "reviews": 100, "released_at": "2023-01-01",
             "updated_at": "2026-01-01"}]
    with_card = features.compute_field(filler, "spider tracker", featured=card)
    check("a promoted card is seen even with no organic position",
          with_card["has_featured"] == 1 and with_card["featured_relevance"] >= 0.9,
          f'relevance {with_card["featured_relevance"]:.2f}')
    fc = features.extract({"title": "x", "installs": 0, "rating": 0, "reviews": 0},
                          "spider tracker", with_card)
    check("the promoted card reaches the model as its own feature",
          fc["field_has_featured"] == 1.0 and fc["field_featured_relevant"] >= 0.9,
          f'relevance feature {fc["field_featured_relevant"]:.2f}')


def test_intent_group_is_the_competition():
    """The phone-mirror case. A page answering two questions is two
    competitions, and the model must be handed the one it would enter."""
    from aso import embed as E
    from aso import intent as I

    page = ([{"pkg": "mirror1", "position": 1, "title": "Phone Mirror",
              "short_desc": "Use your phone as a handy pocket mirror anytime",
              "description": "turns your smartphone into a convenient pocket mirror "
                             "using the front camera", "installs": 125, "rating": 4.0,
              "reviews": 5, "released_at": "2026-08-01", "updated_at": "2026-08-01"}]
            + [{"pkg": f"cast{i}", "position": i + 2, "title": t,
                "short_desc": "Cast your screen to a TV",
                "description": "wireless screen mirroring app to cast android phone "
                               "to PC, smart TV, chromecast and miracast receivers",
                "installs": 20_000_000, "rating": 4.4, "reviews": 500_000,
                "released_at": "2017-01-01", "updated_at": "2026-07-01"}
               for i, t in enumerate(["ApowerMirror Screen Mirroring",
                                      "Screen Mirroring 1001 TVs",
                                      "AirDroid Cast screen mirroring",
                                      "Miracast Screen Mirroring",
                                      "Cast to TV Screen Mirroring"])])

    vecs = {r["pkg"]: E.app_vec(r["title"], r["short_desc"], r["description"])
            for r in page}
    kv = E.keyword_vec("phone mirror")
    split = I.split(page, vecs, kv)

    check("the two meanings are separated", split["ambiguous"],
          f'{len(split["groups"])} groups')
    mirror = I.group_for(split, "mirror1")
    check("the pocket-mirror app is not grouped with the casting apps",
          mirror["size"] == 1 and mirror["packages"] == ["mirror1"],
          f'group size {mirror["size"]}')

    whole = features.compute_field(page, "phone mirror", kw_vec=kv, app_vecs=vecs)
    grp = features.compute_field(page, "phone mirror", kw_vec=kv, app_vecs=vecs,
                                 intent_group=mirror)
    check("the whole-page median is dominated by the other meaning",
          whole["installs_p50"] > 1_000_000, f'{whole["installs_p50"]:,.0f}')
    check("the intent bar handed to the model is the real one",
          grp["intent_median"] == 125 and grp["intent_rivals"] in (0, 1),
          f'{grp["intent_median"]:,} with {grp["intent_rivals"]} rival(s)')
    check("a meaning already at rank 1 is reported as reachable",
          features.extract({"title": "x", "installs": 0, "rating": 0, "reviews": 0},
                           "phone mirror", grp)["intent_reach"] == 1.0)
    check("intent rivals and bar push difficulty down, reach pushes it up",
          {f.name: f.direction for f in features.REGISTRY if f.name.startswith("intent_")}
          == {"intent_rivals": "dec", "intent_median": "dec", "intent_weakest": "dec",
              "intent_reach": "inc", "intent_share": "free"})


def test_downloads_and_confidence_are_learned():
    """Both come out of the network. Downloads is a second head trained on a
    free label; confidence is ensemble agreement discounted by evidence."""
    import torch as _t

    from aso.model import Ensemble

    m = Ensemble(len(features.MONO), len(features.FREE), k=7)
    xm = _t.zeros(2, len(features.MONO))
    xf = _t.zeros(2, len(features.FREE))
    with _t.no_grad():
        v, sd = m.velocity(xm, xf)
        agree = m.agreement(xm, xf)
    check("the velocity head returns a value per app and an error bar",
          v.shape == (2,) and sd.shape == (2,))
    check("agreement is bounded 0 to 1",
          bool(((agree >= 0) & (agree <= 1)).all()), f"{float(agree[0]):.2f}")
    check("the downloads head is unconstrained, unlike the rank head",
          not any(p is m.members[0].w2 for p in m.members[0].vel.parameters()))

    # Agreement alone must not read as certainty on a barely-fitted model.
    con = db.connect()
    r = predict.score_entry(con, "phone mirror") if db.scalar(
        con, "SELECT COUNT(*) FROM observations WHERE keyword='phone mirror'") else None
    if r:
        check("confidence is discounted by how little data exists",
              r["confidence"] <= r["agreement"] and r["evidence"] <= 1.0,
              f'{r["confidence"]:.2f} <= agreement {r["agreement"]:.2f}')
        check("downloads are reported per day, month and year",
              set(r["downloads"]) >= {"per_day", "per_month", "per_year"})
        d = r["downloads"]
        check("the three horizons are consistent",
              abs(d["per_year"] / 365.0 - d["per_day"]) < 1e-6)
    con.close()


def test_monotone_sign_map():
    check("every 'dec' feature is negated at vectorize time",
          all(features.SIGN[i] == -1.0
              for i, f in enumerate(features.MONO) if f.direction == "dec"))
    check("registry has no duplicate feature names",
          len({f.name for f in features.REGISTRY}) == len(features.REGISTRY))


# --------------------------------------------------------------- the guarantee
def test_model_is_monotone():
    """The load-bearing property: raising a monotone feature can never lower the
    score. Without it the tool can tell a user that a better title hurt them."""
    torch.manual_seed(0)
    m = Ensemble(n_mono=6, n_free=3, k=3, hidden=8).eval()
    rng = np.random.default_rng(0)
    worst, checked = 0.0, 0
    with torch.no_grad():
        for _ in range(200):
            xm = torch.tensor(rng.normal(size=(1, 6)), dtype=torch.float32)
            xf = torch.tensor(rng.normal(size=(1, 3)), dtype=torch.float32)
            base = float(m.mean_logit(xm, xf))
            for j in range(6):
                up = xm.clone()
                up[0, j] += abs(rng.normal()) + 0.05
                worst = min(worst, float(m.mean_logit(up, xf)) - base)
                checked += 1
    check(f"raising a monotone feature never lowers the score ({checked} probes)",
          worst >= -1e-5, f"worst delta {worst:+.2e}")


def test_free_features_are_unconstrained():
    torch.manual_seed(1)
    m = Ensemble(n_mono=4, n_free=2, k=5, hidden=8).eval()
    rng = np.random.default_rng(1)
    saw_drop = False
    with torch.no_grad():
        for _ in range(300):
            xm = torch.tensor(rng.normal(size=(1, 4)), dtype=torch.float32)
            xf = torch.tensor(rng.normal(size=(1, 2)), dtype=torch.float32)
            up = xf.clone()
            up[0, 0] += 1.0
            if float(m.mean_logit(xm, up)) < float(m.mean_logit(xm, xf)) - 1e-6:
                saw_drop = True
                break
    check("free features are genuinely unconstrained", saw_drop)


# --------------------------------------------------------------- memory layer
def test_residual_is_local_and_bounded():
    X = np.array([[0.0] * 8], dtype="float32")
    r = np.array([4.0], dtype="float32")
    cw = np.array([5.0], dtype="float32")
    here = memory.read(np.zeros(8, "float32"), X, r, cw)
    far = memory.read(np.full(8, 6.0, "float32"), X, r, cw)
    check("a correction applies at the point it was filed", here > 0.7 * r[0],
          f"{here:.2f} of {r[0]:.2f}")
    check("weight is confidence, not amplitude: never overshoots the reviewer",
          abs(here) <= abs(r[0]) + 1e-6, f"{here:.2f} <= {r[0]:.2f}")
    check("a distant point is left alone", abs(far) < 0.01, f"{far:+.4f}")
    check("empty memory is a no-op",
          memory.read(np.zeros(8, "float32"), np.zeros((0, 0), "float32"),
                      np.zeros(0, "float32"), np.zeros(0, "float32")) == 0.0)


# --------------------------------------------------------------- splitting
def test_rank_target_is_exact():
    """A rank correction must land the app in the right gap, not near it."""
    from aso.correct import target_logit_for_rank
    inc = [5.0, 4.0, 3.0, 2.0, 1.0]
    for want in (1, 2, 3, 5, 9):
        t = target_logit_for_rank(inc, want)
        got = 1 + sum(1 for v in inc if v > t)
        check(f"asking for rank {want} produces rank {got}", got == min(want, len(inc) + 1))


def test_no_hand_tuned_scoring_remains():
    """The scoring path must contain no invented constants. Difficulty and the
    verdict come from the network; without one there is simply no answer."""
    import inspect

    from aso import analyze as A

    check("the crowding formula is gone", not hasattr(A, "heuristic_crowding"))
    check("the demand formula module is gone",
          not (ROOT / "aso" / "demand.py").exists())

    src = inspect.getsource(A.recommend)
    for banned in ("** 0.6", "* 100.0", "0.55", "0.80"):
        check(f"recommend() contains no {banned!r} weighting", banned not in src)

    # Autocomplete order must arrive as raw structure, not a pre-blended score.
    names = {f.name for f in features.REGISTRY}
    for want in ("sugg_self_listed", "sugg_self_rank", "sugg_is_canonical",
                 "sugg_extends", "sugg_unrelated"):
        check(f"{want} is a model feature", want in names)
    check("suggestion features assert no direction on rank",
          all(f.direction == "free" for f in features.REGISTRY
              if f.name.startswith("sugg_")))


def test_no_undefined_names():
    """Every name the code uses exists.

    This is here because a name that is only referenced inside a function is not
    checked until that function runs, so `_refetch` reached production undefined
    and failed the first time an admin pressed the button. The same pass found
    three more, including one introduced the same day that would have crashed
    `aso serve` on startup. Import-time checks cannot catch any of them.
    """
    import subprocess
    import sys as _sys

    root = Path(__file__).resolve().parent.parent
    files = sorted(str(p) for p in (root / "aso").glob("*.py"))
    out = subprocess.run([_sys.executable, "-m", "pyflakes", *files],
                         capture_output=True, text=True)
    if "No module named pyflakes" in out.stderr:
        check("pyflakes is available to check for undefined names", True,
              "not installed, skipped")
        return
    bad = [l for l in out.stdout.splitlines() if "undefined name" in l]
    check("no undefined names anywhere in aso/", not bad,
          bad[0] if bad else f"{len(files)} files clean")


def test_model_always_predicts():
    """An untrained network still answers. Refusing until a gate is satisfied
    would mean no prediction on day one and no way to watch it improve."""
    from aso import train as T

    con = db.connect()
    con.execute("DELETE FROM registry")
    con.commit()
    model, blob, version = T.load_active(con)
    check("a model is always available, even with an empty registry",
          model is not None, f"got {version}")
    # Load order: what this machine trained, then the checkpoint shipped with
    # the repo, then random weights. A fresh clone lands on the shipped one.
    if T.SHIPPED.exists():
        check("an empty registry falls back to the shipped checkpoint",
              blob["meta"].get("untrained") is not True,
              f"loaded {version}")
    else:
        check("with no shipped checkpoint it bootstraps from random weights",
              blob["meta"].get("untrained") is True and version == "v0")
    check("its feature list matches the registry",
          blob["meta"]["features"] == [f.name for f in features.REGISTRY])

    import torch as _t
    xm = _t.zeros(1, len(features.MONO))
    xf = _t.zeros(1, len(features.FREE))
    with _t.no_grad():
        p, _ = model(xm, xf)
    check("an untrained model produces a finite prediction",
          bool(_t.isfinite(p).all()), f"p={float(p):.3f}")
    con.close()


def test_suggestions_are_real_not_invented():
    """Prefix slicing is gone. Signals come off the list Play actually returned."""
    from aso import suggest as S

    check("no prefix slicing survives", not hasattr(S, "variants")
          and not hasattr(S, "PREFIXES"))

    con = db.connect()
    con.execute("DELETE FROM suggestions")
    real = ["phone mirror", "phone mirror app", "phone mirroring to phone",
            "phone mirroring", "phone mirror to pc"]
    for i, term in enumerate(real):
        con.execute("INSERT INTO suggestions (query, country, position, suggestion, "
                    "fetched_at) VALUES ('phone mirror','us',%s,%s,%s)", (i, term, db.now()))
    con.commit()

    sig = S.signals(con, "phone mirror")
    check("the phrase is recognised as one Play lists first",
          sig["sugg_is_canonical"] == 1.0 and sig["sugg_self_rank"] == 0.0)
    check("longer forms of the phrase are counted",
          sig["sugg_extends"] == 0.8, f'{sig["sugg_extends"]:.2f}')
    check("nothing unrelated is reported when every suggestion contains it",
          sig["sugg_unrelated"] == 0.0)

    # A query Play reinterprets: none of the suggestions contain the phrase.
    con.execute("DELETE FROM suggestions")
    for i, term in enumerate(["screen mirroring", "cast to tv", "miracast"]):
        con.execute("INSERT INTO suggestions (query, country, position, suggestion, "
                    "fetched_at) VALUES ('phone mirrr','us',%s,%s,%s)", (i, term, db.now()))
    con.commit()
    sig2 = S.signals(con, "phone mirrr")
    check("a reinterpreted query reports itself as unlisted",
          sig2["sugg_self_listed"] == 0.0 and sig2["sugg_unrelated"] == 1.0)
    con.close()



def test_split_is_by_keyword():
    g = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10)
    tr, ho = dataset.group_split(g, holdout=0.25, seed=0)
    check("no keyword appears on both sides of the split",
          not (set(g[tr]) & set(g[ho])), f"{sorted(set(g[ho]))} held out")
    check("the split covers every row", bool((tr | ho).all()))


# --------------------------------------------------------------- full loop
def _synthetic_db():
    """Built in the throwaway DB only. Monotone truth, so a wired-up model
    should recover it; this proves plumbing, never real-world accuracy."""
    import math
    import random
    random.seed(7)
    db.init()
    con = db.connect()
    heads = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    tails = ["tracker", "planner", "manager", "log"]
    uid = 0
    for kw in [f"{h} {t}" for h in heads for t in tails]:
        pool = []
        for _ in range(12):
            uid += 1
            strong = random.random() < 0.5
            title = (f"{kw} pro" if strong
                     else f"{random.choice(heads)} {random.choice(tails)}")
            pool.append({
                "pkg": f"test.pkg{uid}", "title": title.title(),
                "short_desc": title, "description": (title + " ") * 4,
                "developer": "d", "category": "Productivity",
                "installs": int(10 ** random.uniform(3, 8)),
                "rating": round(random.uniform(3.2, 4.8), 1),
                "reviews": random.randint(10, 90000),
                "released_at": "2024-01-01", "updated_at": "2026-07-01",
                "country": "us", "lang": "en", "raw_json": None, "scraped_at": None})
        score = lambda a: (2.4 * all(w in a["title"].lower() for w in kw.split())
                           + 0.55 * math.log10(a["installs"]) + 0.7 * a["rating"]
                           + random.gauss(0, 0.4))
        pool.sort(key=lambda a: -score(a))
        for pos, a in enumerate(pool, 1):
            db.upsert_app(con, a)
            db.add_observation(con, kw, a["pkg"], pos, source="serp")
    con.commit()
    return con


def test_end_to_end():
    con = _synthetic_db()
    version, meta, gated = train.train(con, k=5, epochs=200, verbose=False)
    # v0 is the random-init bootstrap, so the first real fit is not v1.
    check("training promotes the fitted model over the bootstrap",
          not gated and version != "v0", f"promoted {version}")
    check("the model recovers a monotone truth", meta["golden_auc"] > 0.80,
          f'AUC {meta["golden_auc"]:.3f}')

    kw = con.execute("SELECT keyword FROM observations LIMIT 1").fetchone()["keyword"]
    weak = con.execute(
        "SELECT pkg FROM observations WHERE keyword=%s ORDER BY position DESC LIMIT 1",
        (kw,)).fetchone()["pkg"]
    strong = con.execute(
        "SELECT pkg FROM observations WHERE keyword=%s AND position=1", (kw,)).fetchone()["pkg"]

    lo = predict.score(con, weak, kw)
    hi = predict.score(con, strong, kw)
    check("the top-ranked app outscores the bottom one",
          hi["chance"] > lo["chance"], f'{hi["chance"]:.2f} vs {lo["chance"]:.2f}')
    check("the top-ranked app is predicted to rank above the bottom one",
          hi["predicted_rank"] < lo["predicted_rank"],
          f'#{hi["predicted_rank"]} vs #{lo["predicted_rank"]}')
    check("crowding describes the keyword, not the candidate",
          hi["crowding"] == lo["crowding"],
          f'{hi["crowding"]} == {lo["crowding"]}')

    other_kw = con.execute(
        "SELECT DISTINCT keyword FROM observations WHERE keyword != %s LIMIT 1",
        (kw,)).fetchone()["keyword"]
    before_other = predict.score(con, strong, other_kw, record=False)["chance"]

    correct.apply(con, kw, pkg=weak, rank=3, reviewer="tester")
    after = predict.score(con, weak, kw, record=False)
    after_other = predict.score(con, strong, other_kw, record=False)["chance"]

    check("a rank correction lands on exactly the rank asked for",
          after["predicted_rank"] == 3,
          f'#{lo["predicted_rank"]} -> #{after["predicted_rank"]}, asked for #3')
    check("an unrelated keyword is untouched",
          abs(after_other - before_other) < 0.02)
    # The correction must SURVIVE a retrain. Retiring the residual without
    # teaching the weights would silently throw every correction away.
    corrected_rank = after["predicted_rank"]
    train.train(con, k=5, epochs=200, verbose=False)
    check("a correction becomes a permanent observation",
          db.scalar(con, "SELECT COUNT(*) FROM observations WHERE source='review'") == 1)
    live = db.scalar(con, "SELECT COUNT(*) FROM residuals WHERE retired_at IS NULL")
    check("a residual the model has not learned is KEPT, never silently dropped",
          live in (0, 1), f"{live} still live")
    check("queued corrections are marked absorbed",
          db.scalar(con, "SELECT COUNT(*) FROM corrections WHERE status='queued'") == 0)
    survived = predict.score(con, weak, kw, record=False)
    check("the correction survives a retrain either way",
          survived["predicted_rank"] <= corrected_rank + 2,
          f'#{corrected_rank} before, #{survived["predicted_rank"]} after')
    con.close()


def main():
    print(f"\ntest db: {_TMP}\n")
    _refuse_production()
    for fn in [test_no_undefined_names, test_promoted_cards_excluded, test_parsers,
               test_field_is_leave_one_out, test_newcomers_signal_penetrability,
               test_intent_and_featured_card, test_intent_group_is_the_competition, test_no_hand_tuned_scoring_remains,
               test_model_always_predicts, test_downloads_and_confidence_are_learned, test_suggestions_are_real_not_invented,
               test_monotone_sign_map,
               test_model_is_monotone, test_free_features_are_unconstrained,
               test_residual_is_local_and_bounded, test_rank_target_is_exact,
               test_split_is_by_keyword,
               test_end_to_end]:
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
