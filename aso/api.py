"""The HTTP API, on FastAPI.

Every endpoint is a plain `def`, not `async def`, on purpose: the work is
blocking (torch, sqlite, scraping over the network) and FastAPI runs sync
handlers in a threadpool. Declaring them async would run them on the event loop
and block every other request for the ~40 seconds a cold keyword takes.

Concurrency is capped by a semaphore rather than left to the threadpool, because
the limit that matters is how many Play scrapes run at once, not how many
threads exist.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

app = FastAPI(title="aso-analysis", version="0.1.0",
              summary="Can a new app rank for this Play Store keyword")


# ---------------------------------------------------------------- auth
def auth(authorization: str = Header(default=""),
         x_api_key: str = Header(default="")) -> None:
    """Optional. Unset ASO_API_TOKEN means no auth, which is right for loopback."""
    if not API_TOKEN:
        return
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


class Limits(BaseModel):
    server_max_concurrent: int | None = Field(None, ge=1, le=HARD_MAX)
    server_timeout: float | None = Field(None, gt=0)
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
        q = lambda s: con.execute(s).fetchone()[0]
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
def set_config(body: Limits):
    changed = {}
    if body.server_max_concurrent is not None:
        changed["server_max_concurrent"] = _gate.resize(body.server_max_concurrent)
    if body.server_timeout is not None:
        _limits["timeout"] = body.server_timeout
        changed["server_timeout"] = body.server_timeout
    if not changed:
        raise HTTPException(400, {"error": "bad request",
                                  "detail": "send server_max_concurrent "
                                            "and/or server_timeout"})
    if body.persist:
        config.save(changed)
    return {"applied": changed, "persisted": body.persist, "in_flight": _gate.held}


@app.post("/analyze", dependencies=[Depends(auth)],
          summary="can a new app rank for this keyword")
def analyze(body: Analyze):
    from .analyze import analyze as run
    from .analyze import recommend
    _state["requests"] += 1
    with _gate.slot(), session() as con:
        o = run(con, body.keyword, country=body.country or config.get("country", "us"),
                pkg=body.pkg, verbose=False, learn=body.learn)
        o["recommendation"] = recommend(o, body.model_dump(
            include={"min_downloads", "min_downloads_unit", "max_rank",
                     "min_build_score", "min_model_confidence"},
            exclude_none=True))
        o.pop("_your_group", None)
        return o


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
# sqlite write lock only for the moment it registers the model, so analysis
# keeps working on the current weights throughout - a fit takes ~30s and
# blocking every caller for it would make the tool feel broken.
_job: dict = {"state": "idle", "started": None, "finished": None,
              "result": None, "error": None, "epochs": None}
_job_lock = threading.Lock()


def _run_training(epochs: int) -> None:
    from . import train as tr
    try:
        with session() as con:
            version, meta, gated = tr.train(con, epochs=epochs, verbose=False)
            active = tr.load_active(con)[2]
        _job.update(state="done", finished=time.time(),
                    result={"version": version, "promoted": not gated,
                            "active": active,
                            "rows": (meta or {}).get("n_rows"),
                            "keywords": (meta or {}).get("n_keywords"),
                            "golden_auc": (meta or {}).get("golden_auc"),
                            "golden_ece": (meta or {}).get("golden_ece")})
        _state["model"] = active
    except Exception as e:                                # noqa: BLE001
        _job.update(state="failed", finished=time.time(),
                    error=f"{type(e).__name__}: {e}")


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
                    result=None, error=None, epochs=epochs)
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
    return {"query": body.keyword, "suggestions": out, "count": len(out),
            "cached": not body.no_cache,
            "age_hours": None if age is None else round(age, 2)}


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
