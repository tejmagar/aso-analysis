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
| `downloads` | an order of magnitude, not a number — see below. Range is the ensemble's disagreement |
| `competition` | the score needed to hold the last qualifying slot |
| `same question` | how many ranked apps share your **meaning**, not just your words |
| `intent split` | when Play is answering several different questions on one page |
| `newest entry` | how long since anyone new got in. The most decisive single number |

**`chance of top 10` is the model's odds of ranking. Whether that is worth
building is decided by your thresholds, not by the model.**

### How much to trust the downloads figure

Measured on 170 held-out new apps, on keywords the model never trained on:

```
  median predicted / actual :  0.91x     <- no systematic bias
  median error              :  4.7x off
  within 10x of actual      :  70%
```

**Read it as an order of magnitude and lean on the printed range.** Two reasons
it cannot be much better, both structural:

- **Demand splits across ranks invisibly.** We observe position and installs,
  never impressions or taps. Rank #3 on a keyword nobody searches and rank #3 on
  a busy one look identical in the data.
- **The label is bucketed.** Play reports `100+`, `500+`, `1,000+`. For exactly
  the small apps you compete against, the ground truth is a range, so a 4.7x
  error against a label that is itself ±5x is near the measurement floor.

The **rank** prediction does not have this problem, because ordering needs only
relative comparisons and those are directly observed.

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

## Caching

Autocomplete is cached for **a day**, because it moves slowly and refetching on
every analyze spent a round trip to learn nothing.

```bash
aso suggest "warranty tracker"              # cached
aso suggest "warranty tracker" --no-cache   # refetch AND update the cache
aso config suggest_ttl_hours 6              # tighten it
```

`--no-cache` skips the read but still performs the write, so the next caller
serves the refreshed list rather than paying for the same fetch again.

If a refetch fails the stale list is returned rather than nothing — a day-old
ordering beats no ordering, and the next call retries. `--games` bypasses the
cache entirely, since it uses a different Play filter and would poison the
app-filtered rows.

## Reading the Play Store directly

Everything the underlying Play reader can do, wrapped so you can look at the raw
page without going through the model. Useful for checking what the analyser was
actually looking at when a result seems wrong.

```bash
aso suggest "warranty"                    # autocomplete, in Play's own order
aso suggest "puzzle" --games
aso search "spider tracker"               # top results with correct organic ranks
aso search "spider tracker" --details --limit 10
aso details com.duolingo                  # package name or full Play Store URL
aso publisher "Some Developer Name"       # caps near 50; see fetch_publisher.py
```

Two things these add over calling the library yourself.

**Organic ranks are right.** Play prepends a promoted hero card that is not part
of the ranked list, so numbering straight off the array records a paid placement
as #1 and shifts every real position by one:

```
  3 organic results for 'spider tracker', plus a promoted card

  ad   Spider Tracker                            10K+
    1  Tracker Network Stats               5,000,000+
    2  Strava: Run, Bike, Walk           100,000,000+
```

The card comes back as `position: null, featured: true` rather than being
silently numbered.

**`details` shows both shapes**, so you can see how a string was read:

```
  installs   500,000,000+   read as 963,136,826
  released   May 29, 2013   read as 2013-05-29
```

That is the same normalisation the model consumes, so a feature that looks wrong
can be traced here.

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
aso serve                                     # 127.0.0.1:8765
aso serve --max-concurrent 8 --timeout 300
```

The CLI uses it automatically when it is up (`--no-server` to force in-process).
Same query: **~4s via the server, ~58s without.**

### Ports

**`--port` is strict.** Naming a port means you want that port, so it is used or
the start fails. Omitting it means you do not care, so the default moves up if
busy.

```bash
aso serve                      # 8765, or the next free port if taken
aso serve --port 9000          # 9000 or fail
aso serve --port 9000 --auto-port   # 9000, but move up rather than fail
```

```
$ aso serve                    # 8765 busy
  port 8765 is held by another aso server (model v65), using 8766 instead
  listening on http://127.0.0.1:8766

$ aso serve --port 9200        # 9200 busy
  port 9200 is held by another aso server.
    You asked for this port specifically, so it was not moved.
    aso serve --port 9200 --auto-port   move up if busy
    aso serve                            use the default, 8765
```

The failure names *what* holds the port — another aso server or an unrelated
process — because those want different responses from you. It also binds before
loading the model, so a busy port fails in milliseconds rather than after the
16s warm-up.

**Clients find it on their own.** A running server records itself in
`data/server.json`, and the CLI resolves in order: `ASO_SERVER` → that runfile →
the default port. The recorded PID is checked before the file is trusted, so a
crashed server does not send requests into a void.

```bash
aso serve --port 9000         # in one terminal
aso analyze "habit tracker"   # in another - finds 9000 without being told
```

### Concurrency

Default is **4 requests at once**, deliberately low: a cold keyword holds its
slot for ~40s of scraping, and unbounded threads means concurrent scrapes that
Play rate-limits. Over the limit you get `503` with `retry_after_seconds`, not a
queue that silently grows.

It can be retuned **while the server is running** - no restart, so you never pay
the 16s warm-up to change a number:

```bash
curl -s localhost:8765/config -d '{"server_max_concurrent":12,"server_timeout":300}'
curl -s localhost:8765/config -d '{"server_max_concurrent":12,"persist":true}'   # and save it
aso config server_max_concurrent 6            # the server notices on its own
```

Lowering the limit lets in-flight work finish and admits fewer afterwards;
nothing is cancelled. A value passed to `aso serve` is **pinned** and will not be
overridden by a later config edit — `/health` reports which mode you are in.

## API

```
GET  /health     model, uptime, in_flight, max_concurrent, timeout, and counters
                 for rejected_busy / timed_out / client_disconnects
GET  /status     row counts and the active model
GET  /config     thresholds and the live server limits

POST /analyze    {"keyword":"...", "pkg":"...", "country":"us", "learn":false,
                  "timeout":60,
                  "min_downloads":100, "min_downloads_unit":"day",
                  "max_rank":5, "min_build_score":0, "min_model_confidence":0}
POST /why        {"keyword":"...", "pkg":"..."}
POST /correct    {"keyword":"...", "pkg":"...", "rank":4, "reviewer":"agent"}
POST /train      {"epochs":400}
POST /config     {"server_max_concurrent":12, "server_timeout":300, "persist":false}
```

Raw Play passthroughs — no model, no database, no scraping into the DB:

```
POST /suggest    {"keyword":"warranty", "games":false, "no_cache":false}
POST /search     {"keyword":"...", "details":false, "limit":10}
POST /details    {"pkg":"com.foo"}                  # package name or URL
POST /publisher  {"developer":"..."}
```

`/search` returns `position: null, featured: true` for a promoted card, and
`/details` returns both `raw` (Play's fields) and `normalised` (what the model
consumes), so an agent can check how a value was interpreted.

`/suggest` reports its own cache state, so a caller can decide whether to force
a refresh:

```json
{"query":"warranty", "suggestions":[...], "count":5,
 "cached":true, "age_hours":3.2}
```

### Errors

Every failure is JSON with an `error` and a `detail`, never a stack trace.

| Status | Means |
|---|---|
| `400` | bad request — `detail` says what was missing |
| `404` | no such route |
| `503` | at the concurrency limit; `retry_after_seconds` included |
| `504` | over the timeout. **The work continues in the background** — a thread cannot be killed safely and a half-finished scrape would leave a partial row, so the same request is fast once the scrape lands |
| `500` | anything else, with the exception type in `detail` |

A client that hangs up mid-answer (curl `--max-time`, an abandoned request) is
routine, not an error: it is counted as `client_disconnects` and logged as one
line rather than a traceback.

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
- Package arguments accept a bare package or any Play Store URL, everywhere.
- The Play passthroughs (`/search`, `/details`, `/publisher`) do not write to the
  database, so they are safe to call freely and never affect what the model later
  trains on. `/suggest` is the exception: it maintains its own day-long cache.
- A busy port is not an error. The server moves to the next free one and records
  where it landed, so scripts do not need to coordinate.

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
- **The downloads label comes only from apps under 18 months old.** The same
  arithmetic on an eight-year-old app gives a lifetime average earned over years
  of already ranking — median 262,952/yr against 4,947/yr for a newcomer. Training
  on both had it forecasting 57,000 downloads a day for an unlaunched app.
- **No connection is held across network I/O.** Scrape rows are buffered and
  written in one short transaction; the schema is created only when missing,
  because DDL opens a write transaction that pins the lock for the life of the
  connection. That single line was what made every concurrent command fail with
  `database is locked`.

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
