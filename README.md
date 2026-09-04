# aso-analysis

Predicts whether a new app can rank for a Google Play keyword, and what it would
earn if it did.

Not a ranking engine. Google's ranker already ran, and every SERP you scrape is
its answer, in order. This only learns a one-dimensional projection of that:
**where would a new app slot into an ordering that already exists?** That is a
far simpler question than the one Google answers, which is why the model is
9,254 parameters and trains on a laptop in about ten seconds.

```
  phone mirror  us
  ──────────────────────────────────────────────────────────

  PREDICTED BY THE MODEL

  rank          #2 of 18, for a new app published here
  downloads     27/day   822/month   9.9K/year
                range 321 to 301.6K a year
  competition   ██████████████░░░░░░  crowded

  OBSERVED IN THE STORE

  autocomplete  ████████████████████  listed 1 of 5, 4 longer variants
  same question █░░░░░░░░░░░░░░░░░░░  1 of 17 apps, weakest 125 at #1
  newest entry  ████████████████████  0.08y ago, growth bar 664.2K/yr

  intent split: Play is answering 4 different questions on this page
    ▸ mirror                      1 app   weakest     125 at #1
      screen / mirroring         15 apps  weakest  111.6K at #14
  ──────────────────────────────────────────────────────────
  CONCLUSION

  chance of top 10 ██████████████████░░  91/100  the model's odds of ranking
  model certainty  █████████████████░░░  85/100  how sure it is of that

  worth building, it clears every threshold you set
```

---

## Install

```bash
git clone https://github.com/<you>/aso-analysis && cd aso-analysis
python3 -m venv .venv && source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel
pip install -e ".[all]"
```

That pulls everything, including the Play Store reader straight from git. If you
would rather be selective:

```bash
pip install -e .                # core: torch, numpy, the Play Store reader
pip install -e ".[embed]"       # + sentence-transformers  (strongly recommended)
pip install -e ".[publisher]"   # + selenium, only for scripts/fetch_publisher.py
```

**CPU only, by design.** No GPU is needed or used.

**Install `[embed]` unless you have a reason not to.** Without
`sentence-transformers`, `aso/embed.py` falls back to a character-trigram hash
and the intent clustering stops working — that is the part that separates *"use
your phone as a mirror"* from *"cast your screen to a TV"*, and without it a
wide-open keyword reads as crowded. The model is ~90MB, downloaded on first use.

### Check it worked

```bash
aso analyze "habit tracker"
```

A trained checkpoint ships with the repo (`models/latest.pt`, 55KB), so this
works on a fresh clone with no training and no data collection. It will scrape
that one keyword, which takes about 40 seconds.

The database is **not** in the repo — it is scraped state, it changes on every
command, and you rebuild it yourself by analyzing keywords you care about.

---

# For a person

## One command

```bash
aso analyze "habit tracker"
```

It scrapes the keyword if it has never seen it, predicts, and then asks what you
want to do next:

```
  ▸ Correct a value
    Full breakdown
    Compare an app
    Why is an app not ranking
    Another keyword
    Quit
```

Arrow keys or `j`/`k`, enter to choose, or press the option's letter. The prompt
only appears on a real terminal — piped or redirected it is a plain one-shot
command, and progress goes to stderr so `--json | jq` stays clean.

```bash
aso analyze "habit tracker" --depth        # field quantiles, the bar, the SERP
aso analyze "habit tracker" --once         # print and exit, no prompt
aso analyze "habit tracker" --json
aso analyze "habit tracker" --pkg com.you.app
aso analyze "habit tracker" --pkg "https://play.google.com/store/apps/details?id=com.you.app"
```

## Reading the output

The screen is split because the two halves are different kinds of claim.

**PREDICTED BY THE MODEL** is a forward pass. **OBSERVED IN THE STORE** is a fact
read off the page. Mixing them once made the model's guesses look like
measurements.

| Line | Means |
|---|---|
| `rank` | where a well-built new app would land, from the network |
| `downloads` | what apps in that position actually get. Range is the ensemble's disagreement |
| `competition` | the score needed to hold the last qualifying slot |
| `same question` | how many ranked apps share your **meaning**, not just your words |
| `intent split` | when Play is answering several different questions on one page |
| `newest entry` | how long since anyone new got in. The most decisive single number |

**`chance of top 10` is the model's odds of ranking. Whether that is worth
building is decided by your thresholds, not by the model.**

## Your thresholds

```bash
aso config                                 # list, marking (yours) vs (default)
aso config min download day 100            # or month, or year
aso config max_rank 5
aso config min_certainty 60
```

Or override for one call without saving:

```bash
aso analyze "habit tracker" --min-download 1 --min-download-unit day --max-rank 50
```

When something falls short it says which bar and by how much, rather than
folding it into a score:

```
  ⚠ not worth building, by your thresholds
    · it would land at #25, past your limit of #5
    · 4 downloads a day is under your minimum of 100 a day
```

## Why an app is not ranking

```bash
aso why "period tracker" --pkg alee.periodcalendar.mycycle
```

A **counterfactual walk through the real network**: take the app as it is,
replace one feature with the value held by the app in the last qualifying slot,
re-score, and measure the change. Repeat for every feature and sort.

```
  to displace   Ovia Cycle & Pregnancy Tracker at #10, 3.1M installs
  score gap     +0.79 logits to close

    intent_rivals   ██████████░░░░ +0.54   yours 19    theirs 0
    rating          █████░░░░░░░░░ +0.29   yours 0     theirs 4.5
    log_installs    █████░░░░░░░░░ +0.26   yours 160   theirs 3.1M
```

The top line is the useful one: *19 apps compete for the same meaning, while the
app at #10 has that meaning to itself.* Not "get more installs."

## Correcting it

```bash
aso correct "habit tracker" --rank 4 --pkg com.you.app   # it actually ranks #4
aso correct "habit tracker" --demand 60
aso correct "habit tracker" --crowding 45
```

Every correction takes effect on the **next** `analyze`, in under a second, with
no training. It is written to a residual memory read on each score. Weights move
later, in a batch, behind the gate.

| | takes effect | scope | undo |
|---|---|---|---|
| `aso correct` | next analyze, sub-second | that keyword and nearby ones | delete a row |
| `aso train` | when you run it | everything | registry rollback |

`--rank N` computes the exact score that lands the app in the gap you named, so
`--rank 4` produces rank 4. It also writes a reviewed observation, so the next
retrain learns it permanently — and the residual retires **only once the weights
demonstrably agree**, so a correction is never silently lost.

## Collecting data

```bash
aso analyze "kw"                    # scrapes it if new
aso scrape "a" "b" "c"              # pre-warm several
aso train --epochs 400
aso status
```

### Learning from apps you have already published

This is the highest-value data available, because you know something a scrape
cannot show: an app that is **absent** from a page you targeted is a confirmed
failure, whereas absence in a normal scrape means nothing.

```bash
pip install -e ".[publisher]"
python scripts/fetch_publisher.py "Your Developer Name" --json apps.json
aso ingest --from-publisher apps.json
aso train
```

Each app's title is read as the keyword it was built for, then the live page is
checked for it. Ranked is a confirmed positive; absent is a confirmed negative.

`aso ingest` with no arguments reads a private keyword-research workspace and is
only useful if you have one; point `ASO_RESEARCH_DB` at a sqlite file with a
`research` table, or ignore it.

Ingests are **resumable** — a keyword scraped within `fresh_days` is skipped, so
stop and restart freely. Play's developer page caps at 100 apps, so a larger
catalogue needs a Play Console export.

## Watching change over time

```bash
aso track            # re-scrape anything older than track_every_days, then diff
aso track --show     # just report what changed
aso backtest         # did the research call it right, months later
```

Every scrape appends a dated row rather than overwriting, so today's state is
automatically the before-half of a future comparison. `track` then extracts
**entered / left / moved** between snapshots — and an entrant is a natural
experiment you did not have to run, whoever shipped it.

---

# For an agent

## Run it as a server

Roughly 16 seconds of every CLI invocation is import and warm-up, 13.7s of it
the sentence encoder. A resident process pays that once.

```bash
aso serve                     # 127.0.0.1:8765
```

The CLI uses it automatically when it is up (`--no-server` to force in-process).
Same query: **~4s via the server, ~58s without.**

## API

```
GET  /health     {"ok":true,"model":"v59","uptime_seconds":33,"warm_seconds":16.4}
GET  /status     row counts and the active model
GET  /config     current thresholds

POST /analyze    {"keyword":"...", "pkg":"...", "country":"us", "learn":false,
                  "min_downloads":100, "min_downloads_unit":"day",
                  "max_rank":5, "min_build_score":0, "min_model_confidence":0}
POST /why        {"keyword":"...", "pkg":"..."}
POST /correct    {"keyword":"...", "pkg":"...", "rank":4, "reviewer":"agent"}
POST /train      {"epochs":400}
```

```bash
curl -s localhost:8765/analyze -d '{"keyword":"habit tracker"}' | jq .recommendation
```

## What an agent should read

```jsonc
{
  "recommendation": {
    "rank": 2,                    // where a new app would land
    "field_size": 18,
    "rank_chance": 91,            // model P(top K), 0-100
    "model_confidence": 85,       // agreement x evidence, 0-100
    "downloads": {"per_day": 27, "per_month": 822, "per_year": 9866,
                  "low_per_year": 321, "high_per_year": 301602},
    "meets_thresholds": true,
    "shortfalls": [],             // plain-English reasons if false
    "thresholds_used": {...},     // the bar actually applied
    "reason": "..."               // one paragraph, safe to surface verbatim
  }
}
```

**Read `model_confidence` before acting on `rank`.** It is ensemble agreement
multiplied by how much data exists. Agreement alone reads 1.00 on a model fitted
to four keywords, because seven members bootstrapped from the same handful of
rows converge on the same answer while knowing nothing.

**Pass your own thresholds per request** rather than writing to the config file.
Anything omitted falls back to the saved config, then to defaults.

**`/analyze` does not train by default.** A fit inside a request blocks every
other caller behind the write lock. Use `POST /train`, or `{"learn": true}` if
you want the old behaviour.

## Notes for automation

- Progress goes to **stderr**, results to stdout. `--json` output is clean.
- Exit code 1 with a plain message when Play returns nothing for a keyword.
- SQLite allows one writer; the server serialises `/correct` and `/train` rather
  than failing with `database is locked`. Two concurrent CLI `analyze` calls
  **will** collide — use the server.
- `aso analyze` scrapes on a cache miss: roughly 40s for a new keyword, ~4s for
  one already stored.
- Package arguments accept a bare package or any Play Store URL.

---

# How it works

## Two networks

| | Params | Role |
|---|---|---|
| `all-MiniLM-L6-v2` | 22,713,216 | Frozen. Turns app text into 384-d vectors. Does the *meaning* work |
| `MonotonicMLP × 7` | 9,254 | Trained here. 52 features → a score per app |

Producing one rank:

```
a. encoder forward pass on the field + the query      -> NN
b. cluster those vectors into intent groups           -> math on NN output
c. build 52 features                                  -> math
d. forward pass: 52 features -> 7 MLPs -> one logit   -> NN
e. sort the logits, count how many beat yours         -> math
```

## Monotone by construction

Weights on the monotone path are held positive through softplus, so the model
**cannot** claim that improving your title, rating or installs made things worse.
29 of 52 features carry a declared direction; the other 23 are genuinely
ambiguous and left free.

That constraint is most of why a few hundred rows can fit this at all, and it is
what keeps the counterfactual in `aso why` meaningful.

## Guarantees worth knowing

- **Leave-one-out fields.** An app never contributes to the field it is scored
  against, or its own installs leak into the features.
- **Fixed holdout.** 16 keywords chosen once and reused, so successive models are
  comparable. A fresh random split per run made every AUC incomparable with the
  last and turned the gate into a coin toss.
- **The downloads head cannot see its own answer.** `app_velocity`,
  `log_installs` and `log_reviews` are masked; without that it learned to echo
  its input and forecast zero for anything unlaunched.
- **Nothing promotes that regresses the holdout**, once there are enough keywords
  for that score to mean anything.

## Data

Nothing is hand-labelled. The rank Play assigned **is** the label; the downloads
label is `installs ÷ age`, already in the listing.

Rows are weighted by what they demonstrate, not by who owns the app: a
three-month-old app at rank 3 says far more about entering today than a
ten-year-old incumbent does.

| Source | Weight | Why |
|---|---|---|
| new app (< 18 months) | 3× | a real entry experiment, whoever shipped it |
| observed absence | 3× | we published for this and it never ranked — unobtainable from a scrape |
| human-confirmed rank | 5× | someone checked |
| ordinary incumbent | 1× | context |

## Tests

```bash
python tests/test_pipeline.py
```

Synthetic rows live only in a per-run temporary database — nothing in the suite
can reach `data/aso.db`. Among the invariants: the model is monotone (1,200
probes), a rank correction lands on exactly the rank asked for, an unlearned
residual is kept rather than silently dropped, and no hand-tuned scoring
constants have crept back into the answer path.

## Known gaps

- `country` is stored but not sent to the scraper, which is pinned to
  `hl=en-US`. Rankings are per-storefront, so non-US data is not yet trustworthy.
- Play's developer page caps at 100 apps; a larger catalogue needs a Console
  export.
- `aso why` explains a gap against the current field. It cannot tell you whether
  closing it is cheaper than picking a different keyword.
