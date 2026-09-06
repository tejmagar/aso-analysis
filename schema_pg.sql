-- Postgres schema for the analyser.
--
-- A port of the SQLite one, not a redesign: same tables, same columns, same
-- names, so a statement that ran against one runs against the other. Three
-- deliberate differences, each noted where it appears:
--
--   INTEGER PRIMARY KEY -> BIGSERIAL   SQLite's rowid alias assigns itself;
--                                      Postgres needs the sequence spelled out.
--   INTEGER -> BIGINT                  SQLite integers are 64-bit and Postgres
--                                      INTEGER is 32-bit; installs pass 2.1e9.
--   REAL -> DOUBLE PRECISION           Postgres REAL is 32-bit and would round
--                                      the model's own numbers under it.
--   timestamps stay TEXT               They are written by db.now() as ISO-8601
--                                      and compared as strings throughout, and
--                                      ISO-8601 sorts correctly as text.

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
    installs     BIGINT DEFAULT 0,
    rating       DOUBLE PRECISION    DEFAULT 0.0,
    reviews      BIGINT DEFAULT 0,
    -- The store's own icon URL. Available on both the search result and the
    -- detail payload, so it costs nothing extra to keep.
    icon         TEXT,
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
    id            BIGSERIAL PRIMARY KEY,
    keyword     TEXT NOT NULL,
    country     TEXT NOT NULL DEFAULT 'us',
    pkg         TEXT NOT NULL REFERENCES apps(pkg),
    position    BIGINT,
    source      TEXT NOT NULL DEFAULT 'serp',
    featured    BIGINT NOT NULL DEFAULT 0,   -- promoted hero card, not organic
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
    n                 BIGINT NOT NULL,
    installs_p10      DOUBLE PRECISION, installs_p50 DOUBLE PRECISION, installs_p90 DOUBLE PRECISION,
    rating_p50        DOUBLE PRECISION, reviews_p50   DOUBLE PRECISION,
    exact_match_count BIGINT,
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
    chance        DOUBLE PRECISION, uncertainty DOUBLE PRECISION, crowding DOUBLE PRECISION, fit DOUBLE PRECISION,
    logit         DOUBLE PRECISION,
    features_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id            BIGSERIAL PRIMARY KEY,
    ts            TEXT NOT NULL,
    prediction_id TEXT NOT NULL REFERENCES predictions(id),
    reviewer      TEXT NOT NULL,
    kind          TEXT NOT NULL,          -- outcome | chance | preference
    value         DOUBLE PRECISION,
    other_pred_id TEXT,
    weight        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    status        TEXT NOT NULL DEFAULT 'queued'   -- queued | absorbed | rejected
);

-- T0 residual memory. Read on every prediction, written the instant a reviewer
-- disagrees, retired once a scheduled retrain absorbs it into the weights.
CREATE TABLE IF NOT EXISTS residuals (
    id            BIGSERIAL PRIMARY KEY,
    ts            TEXT NOT NULL,
    correction_id BIGINT REFERENCES corrections(id),
    keyword       TEXT NOT NULL,
    pkg           TEXT,            -- exact target, so a re-query of the same
                                   -- (keyword, app) matches fully even after the
                                   -- reviewed rank shifts the field around it
    features_json TEXT NOT NULL,   -- RAW features, not scaled: retraining moves the
                                   -- scaler, and a key stored in old scaled space
                                   -- would silently drift away from its own point
    residual      DOUBLE PRECISION NOT NULL,
    target_logit  DOUBLE PRECISION,            -- what the reviewer asked for, so retirement
                                   -- can be verified instead of assumed
    weight        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    retired_at    TEXT
);

CREATE TABLE IF NOT EXISTS registry (
    version    TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    n_rows     BIGINT,
    golden_auc DOUBLE PRECISION,
    golden_ece DOUBLE PRECISION,
    -- What the run actually fitted: per-member training loss, and the downloads
    -- head's held-out error as a FACTOR rather than in log units. Kept per row
    -- so two runs can be compared, which is the only way to tell a model that
    -- improved from one that moved.
    metrics    JSONB,
    active     BIGINT NOT NULL DEFAULT 0
);
-- A registry that predates the column above is upgraded in place. IF NOT EXISTS
-- keeps this file re-runnable, which is what stands in for a migration step.
ALTER TABLE registry ADD COLUMN IF NOT EXISTS metrics JSONB;

-- Demand proxy from Play autocomplete. Separate from the ranker on purpose:
-- nothing in a SERP encodes search volume, so this is its own small problem.
CREATE TABLE IF NOT EXISTS demand (
    keyword     TEXT NOT NULL,
    country     TEXT NOT NULL DEFAULT 'us',
    score       DOUBLE PRECISION NOT NULL,
    hits        BIGINT,
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
    value    DOUBLE PRECISION NOT NULL,
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
    position   BIGINT NOT NULL,      -- 0-based, as Play ordered them
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
    n_apps       BIGINT,
    installs_p50 DOUBLE PRECISION,
    installs_p90 DOUBLE PRECISION,
    rating_p50   DOUBLE PRECISION,
    exact_match  BIGINT,
    newcomers    BIGINT,
    -- How old the field was AT THE TIME, not as it reads today. Every app in a
    -- June page is three months older now, so dating them from the present would
    -- report a field that had aged without anything happening in it.
    age_p50      DOUBLE PRECISION,
    velocity_p50 DOUBLE PRECISION,
    -- What share of that page carried a release date. A median age over two of
    -- thirty apps is not the field's age, so the reader is told the weight.
    age_known_frac DOUBLE PRECISION,
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

-- What an app looked like at a moment, so growth can be measured rather than
-- inferred.
--
-- Everything else records a lifetime total and a release date, and dividing one
-- by the other gives an average across the app's whole life. That is not what a
-- new app would earn now: an app that took a million installs in its first year
-- and nothing since reads the same as one earning steadily today. Two rows here
-- a month apart give the real rate over a known window.
CREATE TABLE IF NOT EXISTS app_snapshots (
    pkg         TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    installs    BIGINT,
    rating      DOUBLE PRECISION,
    reviews     BIGINT,
    updated_at  TEXT,
    source      TEXT NOT NULL DEFAULT 'scrape',   -- scrape | research
    PRIMARY KEY (pkg, observed_at)
);
CREATE INDEX IF NOT EXISTS app_snapshots_pkg_idx ON app_snapshots (pkg, observed_at DESC);

-- Everything one publisher has shipped, by name.
--
-- Play's SEARCH caps near 50 results however many a developer actually has, so
-- "show me this publisher's apps" cannot be answered from the store alone. What
-- we have already scraped fills the rest in, and that lookup is by developer
-- name - which was a sequential scan until this index.
CREATE INDEX IF NOT EXISTS apps_developer_idx ON apps (developer) WHERE developer <> '';
