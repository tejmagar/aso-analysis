-- A rank is a rank: our apps and competitor apps share one table and one label space.
-- `source` does NOT mean ownership. It records whether we can observe this app's FAILURES.
--   'serp'   scraped from a result page. Survivor only: absence is unobserved.
--   'owned'  an app we published. We know the keywords it targeted and never ranked for.
--   'manual' hand-entered.

CREATE TABLE IF NOT EXISTS apps (
    pkg          TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    short_desc   TEXT DEFAULT '',
    description  TEXT DEFAULT '',
    developer    TEXT DEFAULT '',
    category     TEXT DEFAULT '',
    installs     INTEGER DEFAULT 0,
    rating       REAL    DEFAULT 0.0,
    reviews      INTEGER DEFAULT 0,
    released_at  TEXT,
    updated_at   TEXT,
    country      TEXT DEFAULT 'us',
    lang         TEXT DEFAULT 'en',
    raw_json     TEXT,
    scraped_at   TEXT NOT NULL
);

-- One (keyword, app) rank observation.
-- position: 1..250 observed | 251 = checked and absent | NULL = never checked.
-- The 251 rows are the only true negatives in the whole dataset.
CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY,
    keyword     TEXT NOT NULL,
    country     TEXT NOT NULL DEFAULT 'us',
    pkg         TEXT NOT NULL REFERENCES apps(pkg),
    position    INTEGER,
    source      TEXT NOT NULL DEFAULT 'serp',
    featured    INTEGER NOT NULL DEFAULT 0,   -- promoted hero card, not organic
    observed_at TEXT NOT NULL,
    UNIQUE (keyword, country, pkg, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_obs_kw  ON observations(keyword, country);
CREATE INDEX IF NOT EXISTS idx_obs_pkg ON observations(pkg);

-- Cached competitive field per keyword. Depends only on the keyword, never on the
-- candidate, so it is computed once per scrape and read at prediction time.
CREATE TABLE IF NOT EXISTS fields (
    keyword           TEXT NOT NULL,
    country           TEXT NOT NULL DEFAULT 'us',
    n                 INTEGER NOT NULL,
    installs_p10      REAL, installs_p50 REAL, installs_p90 REAL,
    rating_p50        REAL, reviews_p50   REAL,
    exact_match_count INTEGER,
    computed_at       TEXT NOT NULL,
    PRIMARY KEY (keyword, country)
);

CREATE TABLE IF NOT EXISTS predictions (
    id            TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    model_version TEXT NOT NULL,
    pkg           TEXT NOT NULL,
    keyword       TEXT NOT NULL,
    country       TEXT NOT NULL DEFAULT 'us',
    chance        REAL, uncertainty REAL, crowding REAL, fit REAL,
    logit         REAL,
    features_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id            INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,
    prediction_id TEXT NOT NULL REFERENCES predictions(id),
    reviewer      TEXT NOT NULL,
    kind          TEXT NOT NULL,          -- outcome | chance | preference
    value         REAL,
    other_pred_id TEXT,
    weight        REAL NOT NULL DEFAULT 1.0,
    status        TEXT NOT NULL DEFAULT 'queued'   -- queued | absorbed | rejected
);

-- T0 residual memory. Read on every prediction, written the instant a reviewer
-- disagrees, retired once a scheduled retrain absorbs it into the weights.
CREATE TABLE IF NOT EXISTS residuals (
    id            INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,
    correction_id INTEGER REFERENCES corrections(id),
    keyword       TEXT NOT NULL,
    pkg           TEXT,            -- exact target, so a re-query of the same
                                   -- (keyword, app) matches fully even after the
                                   -- reviewed rank shifts the field around it
    features_json TEXT NOT NULL,   -- RAW features, not scaled: retraining moves the
                                   -- scaler, and a key stored in old scaled space
                                   -- would silently drift away from its own point
    residual      REAL NOT NULL,
    target_logit  REAL,            -- what the reviewer asked for, so retirement
                                   -- can be verified instead of assumed
    weight        REAL NOT NULL DEFAULT 1.0,
    retired_at    TEXT
);

CREATE TABLE IF NOT EXISTS registry (
    version    TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    n_rows     INTEGER,
    golden_auc REAL,
    golden_ece REAL,
    active     INTEGER NOT NULL DEFAULT 0
);

-- Demand proxy from Play autocomplete. Separate from the ranker on purpose:
-- nothing in a SERP encodes search volume, so this is its own small problem.
CREATE TABLE IF NOT EXISTS demand (
    keyword     TEXT NOT NULL,
    country     TEXT NOT NULL DEFAULT 'us',
    score       REAL NOT NULL,
    hits        INTEGER,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (keyword, country)
);

-- Human overrides, addressed by keyword. A reviewer's value simply wins for the
-- fields that are not learned (demand, crowding); the learned rank goes through
-- the residual memory instead, because it has to generalize.
CREATE TABLE IF NOT EXISTS overrides (
    keyword  TEXT NOT NULL,
    country  TEXT NOT NULL DEFAULT 'us',
    field    TEXT NOT NULL,          -- demand | crowding
    value    REAL NOT NULL,
    reviewer TEXT,
    ts       TEXT NOT NULL,
    PRIMARY KEY (keyword, country, field)
);

-- Exactly what Play's autocomplete returned, in order.
-- No prefixes are invented and nothing is scored here: the package hands back
-- the real keyword family for a query, and its ORDER is the signal.
CREATE TABLE IF NOT EXISTS suggestions (
    query      TEXT NOT NULL,
    country    TEXT NOT NULL DEFAULT 'us',
    position   INTEGER NOT NULL,      -- 0-based, as Play ordered them
    suggestion TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (query, country, position)
);

-- Cached keyword-family observations. Probing the family means live
-- autocomplete calls, which must not happen inside a training loop.
CREATE TABLE IF NOT EXISTS families (
    keyword       TEXT NOT NULL,
    country       TEXT NOT NULL DEFAULT 'us',
    features_json TEXT NOT NULL,
    detail_json   TEXT,
    computed_at   TEXT NOT NULL,
    PRIMARY KEY (keyword, country)
);

-- The state of a keyword's field at a point in time.
-- Written on every scrape and backfilled from research notes, so the model can
-- be shown not just what a field looks like but which way it is MOVING. A field
-- whose bar doubled in three months is a different proposition from an identical
-- one that has been flat, and a single snapshot cannot tell them apart.
CREATE TABLE IF NOT EXISTS field_history (
    keyword      TEXT NOT NULL,
    country      TEXT NOT NULL DEFAULT 'us',
    observed_at  TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'scrape',   -- scrape | research
    n_apps       INTEGER,
    installs_p50 REAL,
    installs_p90 REAL,
    rating_p50   REAL,
    exact_match  INTEGER,
    newcomers    INTEGER,
    PRIMARY KEY (keyword, country, observed_at)
);

-- The fixed evaluation set. Chosen once and never changed, so successive models
-- are scored on identical keywords and their numbers can be compared. A fresh
-- random split per training run made every AUC incomparable with the last.
CREATE TABLE IF NOT EXISTS holdout (
    keyword TEXT NOT NULL,
    country TEXT NOT NULL DEFAULT 'us',
    chosen_at TEXT NOT NULL,
    PRIMARY KEY (keyword, country)
);
