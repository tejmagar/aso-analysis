"""The HTTP API, on FastAPI.

Every endpoint is a plain `def`, not `async def`, on purpose: the work is
blocking (torch, the database, scraping over the network) and FastAPI runs sync
handlers in a threadpool. Declaring them async would run them on the event loop
and block every other request for the ~40 seconds a cold keyword takes.

Concurrency is capped by a semaphore rather than left to the threadpool, because
the limit that matters is how many Play scrapes run at once, not how many
threads exist.
"""
from __future__ import annotations

import json

import os
import threading
import time
from contextlib import contextmanager

from fastapi.responses import StreamingResponse
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pathlib import Path

from . import config, db
from . import env as _env

_env.load()

API_TOKEN = os.environ.get("ASO_API_TOKEN", "").strip()
HARD_MAX = 32

_write = threading.Lock()
_state = {"started": time.time(), "requests": 0, "model": None,
          "rejected": 0, "warm_seconds": 0.0}
_limits = {"timeout": float(config.get("server_timeout", 180))}


class Gate:
    """Admission limit that can be retuned while requests are in flight."""

    def __init__(self, limit: int):
        self._cv = threading.Condition()
        self._limit, self._held = limit, 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def held(self) -> int:
        return self._held

    def resize(self, limit: int) -> int:
        with self._cv:
            self._limit = max(1, min(int(limit), HARD_MAX))
            self._cv.notify_all()
        return self._limit

    @contextmanager
    def slot(self):
        with self._cv:
            if self._held >= self._limit:
                raise HTTPException(
                    503, {"error": "server busy",
                          "detail": f"{self._limit} requests already running; a "
                                    f"cold keyword takes ~40s to scrape",
                          "in_flight": self._held,
                          "retry_after_seconds": 30})
            self._held += 1
        try:
            yield
        finally:
            with self._cv:
                self._held -= 1
                self._cv.notify()


_gate = Gate(int(config.get("server_max_concurrent", 4)))

if not API_TOKEN:
    raise SystemExit(
        "ASO_API_TOKEN is not set.\n"
        "  Every endpoint requires it. Generate one and put it in the environment:\n"
        "    ASO_API_TOKEN=$(openssl rand -hex 24)")

app = FastAPI(title="aso-analysis", version="0.1.0",
              summary="Can a new app rank for this Play Store keyword")


# ---------------------------------------------------------------- auth
def auth(authorization: str = Header(default=""),
         x_api_key: str = Header(default="")) -> None:
    """Required, always.

    This used to be skipped when ASO_API_TOKEN was unset, on the reasoning that
    a loopback bind is unreachable from outside. That reasoning does not survive
    a container: everything on the same docker network can reach it, and an
    unset variable is exactly how a token goes missing. Refusing to start
    without one turns a silent hole into a failure at boot.
    """
    if not API_TOKEN:
        raise HTTPException(503, {"error": "misconfigured",
                                  "detail": "ASO_API_TOKEN is not set on the server"})
    import hmac
    sent = authorization.removeprefix("Bearer ").strip() or x_api_key.strip()
    if not (sent and hmac.compare_digest(sent, API_TOKEN)):
        raise HTTPException(401, {"error": "unauthorised",
                                  "detail": "send Authorization: Bearer <token> "
                                            "or X-API-Key"})


@contextmanager
def session():
    """A connection held only as long as it is used, never across network I/O."""
    con = db.connect()
    try:
        yield con
    finally:
        con.close()


# ---------------------------------------------------------------- models
class Analyze(BaseModel):
    keyword: str = Field(min_length=1, description="the search phrase")
    pkg: str | None = Field(None, description="package name or Play Store URL")
    country: str | None = None
    learn: bool = Field(False, description="train on this keyword before answering")
    refresh: bool = Field(False, description="fetch the page again even if it is stored")
    max_age_days: float | None = Field(
        None, description="re-fetch when the stored page is older than this")
    min_downloads: float | None = None
    min_downloads_unit: str | None = Field(None, pattern="^(day|month|year)$")
    max_rank: int | None = None
    min_build_score: int | None = None
    min_model_confidence: int | None = None


class Why(BaseModel):
    keyword: str = Field(min_length=1)
    pkg: str = Field(min_length=1)
    country: str | None = None


class Correct(BaseModel):
    keyword: str = Field(min_length=1)
    pkg: str | None = None
    rank: int | None = None
    demand: float | None = None
    crowding: float | None = None
    reviewer: str = "api"
    country: str | None = None


class Query(BaseModel):
    keyword: str = Field(min_length=1)
    games: bool = False
    no_cache: bool = False
    details: bool = False
    limit: int | None = None


class Settings(BaseModel):
    """Anything a caller may change on a live server.

    Server limits apply immediately and are lost on restart unless persisted.
    Thresholds are a value judgement about what is worth building, so they are
    always written: a bar that quietly reverted would be worse than no bar.
    """
    server_max_concurrent: int | None = Field(None, ge=1, le=HARD_MAX)
    server_timeout: float | None = Field(None, gt=0)
    min_downloads: float | None = Field(None, ge=0)
    min_downloads_unit: str | None = Field(None, pattern="^(day|month|year)$")
    max_rank: int | None = Field(None, ge=0)
    min_build_score: int | None = Field(None, ge=0, le=100)
    min_model_confidence: int | None = Field(None, ge=0, le=100)
    top_k: int | None = Field(None, ge=1, le=250)
    persist: bool = False


# ---------------------------------------------------------------- errors
@app.exception_handler(HTTPException)
def _http_error(_: Request, e: HTTPException):
    body = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
    return JSONResponse(body, status_code=e.status_code)


@app.exception_handler(Exception)
def _unhandled(_: Request, e: Exception):
    # SystemExit is how the library reports a user-fixable problem; it is a 400,
    # not a crash.
    if isinstance(e, SystemExit):
        return JSONResponse({"error": "bad request", "detail": str(e)}, 400)
    return JSONResponse({"error": type(e).__name__, "detail": str(e)}, 500)


# ---------------------------------------------------------------- routes
@app.get("/health", summary="liveness, model version and live limits")
def health():
    return {"ok": True, "model": _state["model"],
            "uptime_seconds": round(time.time() - _state["started"], 1),
            "requests": _state["requests"], "in_flight": _gate.held,
            "max_concurrent": _gate.limit, "timeout_seconds": _limits["timeout"],
            "hard_max": HARD_MAX, "rejected_busy": _state["rejected"],
            "auth": bool(API_TOKEN),
            "warm_seconds": round(_state["warm_seconds"], 1)}


@app.get("/status", dependencies=[Depends(auth)])
def status():
    with session() as con:
        q = lambda s: db.scalar(con, s)
        return {"apps": q("SELECT COUNT(*) FROM apps"),
                "keywords": q("SELECT COUNT(DISTINCT keyword) FROM observations"),
                "ranked": q("SELECT COUNT(*) FROM observations "
                            "WHERE position IS NOT NULL"),
                "corrections": q("SELECT COUNT(*) FROM corrections"),
                "model": _state["model"]}


@app.get("/config", dependencies=[Depends(auth)])
def get_config():
    return {**config.load(reload=True),
            "server_max_concurrent": _gate.limit,
            "server_timeout": _limits["timeout"]}


@app.post("/config", dependencies=[Depends(auth)], summary="retune a live server")
def set_config(body: Settings):
    changed, persist_now = {}, {}
    if body.server_max_concurrent is not None:
        changed["server_max_concurrent"] = _gate.resize(body.server_max_concurrent)
    if body.server_timeout is not None:
        _limits["timeout"] = body.server_timeout
        changed["server_timeout"] = body.server_timeout

    for key in ("min_downloads", "min_downloads_unit", "max_rank",
                "min_build_score", "min_model_confidence", "top_k"):
        v = getattr(body, key)
        if v is not None:
            changed[key] = persist_now[key] = v

    if not changed:
        raise HTTPException(400, {
            "error": "bad request",
            "detail": "nothing to change",
            "accepts": sorted(Settings.model_fields)})

    # Thresholds always persist; server limits only when asked.
    to_save = dict(persist_now)
    if body.persist:
        to_save.update({k: v for k, v in changed.items() if k.startswith("server_")})
    if to_save:
        config.save(to_save)
    return {"applied": changed, "persisted": sorted(to_save),
            "in_flight": _gate.held}


@app.post("/analyze", dependencies=[Depends(auth)],
          summary="can a new app rank for this keyword")
def analyze(body: Analyze):
    from .analyze import NoResults
    from .analyze import analyze as run
    from .analyze import recommend
    _state["requests"] += 1
    try:
        return _analyze(body, run, recommend)
    except NoResults as e:
        # 422: the request was well formed, the keyword simply has no field.
        raise HTTPException(422, {"error": "no_field", "detail": str(e)}) from None


def _analyze(body, run, recommend):
    with _gate.slot(), session() as con:
        o = run(con, body.keyword, country=body.country or config.get("country", "us"),
                pkg=body.pkg, verbose=False, learn=body.learn,
                refresh=body.refresh, max_age_days=body.max_age_days)
        o["recommendation"] = recommend(o, body.model_dump(
            include={"min_downloads", "min_downloads_unit", "max_rank",
                     "min_build_score", "min_model_confidence"},
            exclude_none=True))
        o.pop("_your_group", None)
        return o


@app.post("/analyze/stream", dependencies=[Depends(auth)],
          summary="the same analysis, narrating each stage as it happens")
def analyze_stream(body: Analyze):
    """Server-sent events: one `stage` per step, then one `result` with the
    same payload /analyze returns.

    The work is blocking and lives in a thread; the generator drains a queue it
    fills. That keeps the pipeline free of any knowledge that it is being
    watched, which is why the same functions serve both endpoints.
    """
    import queue
    import threading

    from .analyze import analyze as run
    from .analyze import recommend

    events: queue.Queue = queue.Queue()

    def work():
        try:
            with _gate.slot(), session() as con:
                o = run(con, body.keyword,
                        country=body.country or config.get("country", "us"),
                        pkg=body.pkg, verbose=False, learn=body.learn,
                        refresh=body.refresh, max_age_days=body.max_age_days,
                        progress=lambda line: events.put(("stage", line)))
                o["recommendation"] = recommend(o, body.model_dump(
                    include={"min_downloads", "min_downloads_unit", "max_rank",
                             "min_build_score", "min_model_confidence"},
                    exclude_none=True))
                o.pop("_your_group", None)
                events.put(("result", o))
        except Exception as e:                       # noqa: BLE001 - reported, not raised
            # NoResults carries a sentence written for a reader. Anything else
            # is a fault, and its str() is usually a fragment that means nothing
            # outside a traceback, so it is named rather than quoted.
            from .analyze import NoResults
            detail = (str(e) if isinstance(e, NoResults)
                      else f"The analysis failed ({e.__class__.__name__}).")
            events.put(("error", {"detail": detail}))
        finally:
            events.put((None, None))

    _state["requests"] += 1
    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            kind, payload = events.get()
            if kind is None:
                break
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        # Proxies that buffer would hold every line until the end, which is
        # exactly the wait this endpoint exists to remove.
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/why", dependencies=[Depends(auth)],
          summary="why an app is not ranking, feature by feature")
def why(body: Why):
    from . import diagnose, scrape
    _state["requests"] += 1
    with _gate.slot(), session() as con:
        return diagnose.why(con, body.keyword, scrape.parse_package(body.pkg),
                            country=body.country or config.get("country", "us"))


@app.post("/correct", dependencies=[Depends(auth)])
def correct(body: Correct):
    from . import correct as fix
    from . import scrape
    with _write, session() as con:
        done = fix.apply(con, body.keyword,
                         pkg=scrape.parse_package(body.pkg) if body.pkg else None,
                         rank=body.rank, demand=body.demand, crowding=body.crowding,
                         reviewer=body.reviewer,
                         country=body.country or config.get("country", "us"))
        return {"keyword": body.keyword, "applied": done}


# Training runs in a background thread and reports through this. It holds the
# write lock only for the moment it registers the model, so analysis
# keeps working on the current weights throughout - a fit takes ~30s and
# blocking every caller for it would make the tool feel broken.
_job: dict = {"state": "idle", "started": None, "finished": None,
              "result": None, "error": None, "epochs": None,
              "phase": None, "step": 0, "steps": 0, "note": ""}
_job_lock = threading.Lock()

# Progress is mirrored to a file as well as held in memory. The API process can
# be restarted, and a run started from a chat should still be explicable
# afterwards rather than vanishing with the process that owned it.
# It used to sit beside the SQLite file. The database is remote now, so it goes
# where the other rebuildable local artefacts do, next to the embedding cache.
PROGRESS = Path(os.environ.get(
    "ASO_TRAIN_PROGRESS",
    Path(os.environ.get("ASO_EMBED_CACHE", "data/emb")).parent / "training.txt"))


def _write_progress() -> None:
    try:
        PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        started = _job.get("started") or time.time()
        lines = [f"state    {_job['state']}",
                 f"phase    {_job.get('phase') or '-'}",
                 f"step     {_job.get('step')}/{_job.get('steps')}",
                 f"note     {_job.get('note') or ''}",
                 f"elapsed  {time.time() - started:.0f}s",
                 f"updated  {db.now()}"]
        if _job.get("result"):
            r = _job["result"]
            lines += [f"version  {r.get('version')}",
                      f"auc      {r.get('golden_auc', 0):.3f}",
                      f"promoted {r.get('promoted')}"]
        if _job.get("error"):
            lines.append(f"error    {_job['error']}")
        PROGRESS.write_text("\n".join(lines) + "\n")
    except OSError:
        pass                       # progress reporting must never fail a run


def _run_training(epochs: int) -> None:
    from . import train as tr

    def progress(phase, done, total, note):
        _job.update(phase=phase, step=done, steps=total, note=note)
        _write_progress()

    try:
        with session() as con:
            version, meta, gated = tr.train(con, epochs=epochs, verbose=False,
                                            progress=progress)
            active = tr.load_active(con)[2]
        _job.update(state="done", finished=time.time(),
                    result={"version": version, "promoted": not gated,
                            "active": active,
                            "rows": (meta or {}).get("n_rows"),
                            "keywords": (meta or {}).get("n_keywords"),
                            "golden_auc": (meta or {}).get("golden_auc"),
                            "golden_ece": (meta or {}).get("golden_ece")})
        _state["model"] = active
        _job.update(phase="done", step=_job.get("steps", 0))
    except Exception as e:                                # noqa: BLE001
        _job.update(state="failed", finished=time.time(),
                    error=f"{type(e).__name__}: {e}")
    finally:
        _write_progress()


@app.post("/train", dependencies=[Depends(auth)],
          summary="start a training run in the background")
def train(epochs: int = 400):
    with _job_lock:
        if _job["state"] == "running":
            raise HTTPException(409, {
                "error": "already training",
                "detail": "one run at a time; poll GET /train for progress",
                "started_seconds_ago": round(time.time() - (_job["started"] or 0), 1)})
        _job.update(state="running", started=time.time(), finished=None,
                    result=None, error=None, epochs=epochs,
                    phase="starting", step=0, steps=0, note="")
    _write_progress()
    threading.Thread(target=_run_training, args=(epochs,), daemon=True,
                     name="aso-train").start()
    return {"state": "running", "epochs": epochs,
            "detail": "training started; analysis continues on the current model",
            "poll": "GET /train"}


@app.get("/train", dependencies=[Depends(auth)], summary="training progress")
def train_status():
    out = {k: v for k, v in _job.items()}
    if _job["started"]:
        end = _job["finished"] or time.time()
        out["elapsed_seconds"] = round(end - _job["started"], 1)
    out["serving"] = _state["model"]
    return out


# --- raw Play passthroughs: no model, no scraping into the database -------
@app.post("/suggest", dependencies=[Depends(auth)])
def suggest(body: Query):
    from . import play
    from . import suggest as sg
    out = play.suggest(body.keyword, games=body.games, no_cache=body.no_cache)
    with session() as con:
        age = sg.age_hours(con, body.keyword)
    kw = " ".join(body.keyword.strip().lower().split())
    return {"query": kw, "suggestions": out, "count": len(out),
            "cached": not body.no_cache,
            "age_hours": None if age is None else round(age, 2),
            # The same partition /analyze reports, so a caller wanting the list
            # on its own does not have to run an analysis to get the split, and
            # does not have to reimplement it to avoid one.
            "self_rank": out.index(kw) if kw in out else None,
            "extends": [x for x in out if x != kw and kw in x],
            "unrelated": [x for x in out if kw not in x]}


@app.post("/expand", dependencies=[Depends(auth)],
          summary="the whole keyword family, not just the first ten")
def expand(body: Query):
    """Autocomplete answers a prefix, so one query returns the ten commonest
    completions and hides the tail. This asks a seed per letter and unions the
    answers, which is slow the first time and cached afterwards."""
    from . import suggest as sugg_mod
    _state["requests"] += 1
    with _gate.slot(), session() as con:
        return sugg_mod.expand(con, body.keyword,
                               country=config.get("country", "us"),
                               ttl_hours=0 if body.no_cache else 24.0)


class Refetch(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    train_after: bool = False


@app.get("/keywords", dependencies=[Depends(auth)],
         summary="every keyword on file and when it was last fetched")
def keywords(country: str = "us"):
    with session() as con:
        rows = con.execute(
            """SELECT o.keyword,
                      MAX(o.observed_at) AS last_seen,
                      COUNT(DISTINCT o.pkg) AS apps
                 FROM observations o
                WHERE o.country = %s AND o.position IS NOT NULL
                GROUP BY o.keyword ORDER BY MAX(o.observed_at)""",
            (country,)).fetchall()
    return {"keywords": [dict(r) for r in rows]}


@app.get("/refetch", dependencies=[Depends(auth)], summary="refetch progress")
def refetch_status():
    return dict(_refetch)


@app.post("/refetch", dependencies=[Depends(auth)],
          summary="fetch these keywords' pages again, then optionally retrain")
def refetch(body: Refetch):
    """Queue pages to be fetched again, in the background.

    Deliberately one keyword at a time. The point of refetching is to have
    current pages before training, and doing it in parallel would take every
    slot the gate has and leave anyone analysing a keyword waiting behind a
    maintenance job they did not ask for.
    """
    import threading

    if _refetch.get("state") == "running":
        raise HTTPException(409, {"error": "busy",
                                  "detail": "a refetch is already running"})
    words = [" ".join(k.strip().lower().split()) for k in body.keywords if k.strip()]
    if not words:
        raise HTTPException(400, {"error": "empty", "detail": "no keywords given"})

    _refetch.update(state="running", done=0, total=len(words), keyword=None,
                    started=time.time(), finished=None, failed=0,
                    train_after=body.train_after, error=None, trained=None)

    def work():
        from . import scrape
        for kw in words:
            _refetch["keyword"] = kw
            try:
                with _gate.slot(), session() as con:
                    scrape.scrape_keyword(con, kw, verbose=False)
            except Exception as e:                       # noqa: BLE001
                _refetch["failed"] += 1
                _refetch["error"] = f"{kw}: {e}"
            _refetch["done"] += 1
        _refetch["keyword"] = None
        if body.train_after:
            # Inline rather than spawned: the point of the flag is to train on
            # what was just fetched, so it has to wait for the fetching.
            _refetch["state"] = "training"
            _run_training(400)
            _refetch["trained"] = _job.get("result")
        _refetch.update(state="done", finished=time.time())

    threading.Thread(target=work, daemon=True).start()
    return dict(_refetch)


@app.post("/search", dependencies=[Depends(auth)])
def search(body: Query):
    from . import play
    with _gate.slot():
        out = play.search(body.keyword, with_details=body.details, limit=body.limit)
    return {"query": body.keyword, "results": out,
            "organic": sum(1 for r in out if r["position"])}


@app.post("/details", dependencies=[Depends(auth)])
def details(pkg: str):
    from . import play
    return play.details(pkg)


@app.post("/publisher", dependencies=[Depends(auth)])
def publisher(developer: str):
    from . import play
    out = play.publisher(developer)
    return {"developer": developer, "apps": out, "count": len(out),
            "note": "Play search caps near 50; scripts/fetch_publisher.py reaches 100"}


@app.on_event("startup")
def _warm():
    """Pay the model load once, at startup, where it is visible in the logs."""
    from . import embed, train
    t0 = time.time()
    embed.encoder()
    with session() as con:
        _state["model"] = train.load_active(con)[2]
    _state["warm_seconds"] = time.time() - t0
    _state["started"] = time.time()
    print(f"  model {_state['model']} warm in {_state['warm_seconds']:.1f}s  "
          f"| {_gate.limit} concurrent | auth "
          f"{'on' if API_TOKEN else 'off'}", flush=True)
