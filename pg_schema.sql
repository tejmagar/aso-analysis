-- Historical result pages, one row per app per snapshot.
--
-- The SQLite side keeps only per-field summaries, which was enough to difference
-- two pages but threw away the page itself. This keeps the page: every app, at
-- the position it held, on the date it held it. That is the raw material the
-- summaries are derived from, so anything derived can be recomputed later
-- without re-scraping, and a summary defect does not become permanent.

CREATE TABLE IF NOT EXISTS snapshots (
    id           BIGSERIAL PRIMARY KEY,
    keyword      TEXT        NOT NULL,
    country      TEXT        NOT NULL DEFAULT 'us',
    observed_at  TIMESTAMPTZ NOT NULL,
    source       TEXT        NOT NULL,          -- research | scrape
    n_apps       INTEGER     NOT NULL,
    -- Whether this page's order can be trusted as rank. A list that happens to
    -- run strictly descending by installs may be page order or may have been
    -- sorted before it was written down, and there is no way to tell them apart
    -- after the fact. Recording the ambiguity beats silently treating a sorted
    -- list as a ranking.
    order_is_rank BOOLEAN,
    UNIQUE (keyword, country, observed_at, source)
);

CREATE TABLE IF NOT EXISTS snapshot_apps (
    snapshot_id  BIGINT  NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    rank         INTEGER NOT NULL,              -- 1-based position on the page
    title        TEXT,
    pkg          TEXT,
    developer    TEXT,
    -- Two install numbers on purpose. `installs` is the displayed bucket
    -- ("100,000+"), `installs_real` the true count where the session captured
    -- it. Collapsing them would lose the distinction between a measured value
    -- and a floor.
    installs      BIGINT,
    installs_real BIGINT,
    rating        REAL,
    reviews       BIGINT,
    released      DATE,
    -- 'day' when a full date was recorded, 'year' when only "2017" was, so a
    -- date resolved to mid-year is never mistaken for one that was observed.
    released_precision TEXT,
    PRIMARY KEY (snapshot_id, rank)
);

CREATE INDEX IF NOT EXISTS snapshots_keyword_idx    ON snapshots (keyword, country, observed_at);
CREATE INDEX IF NOT EXISTS snapshot_apps_pkg_idx    ON snapshot_apps (pkg) WHERE pkg IS NOT NULL;
CREATE INDEX IF NOT EXISTS snapshot_apps_title_idx  ON snapshot_apps (lower(title));
