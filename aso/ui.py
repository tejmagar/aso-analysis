"""Terminal progress, no dependencies.

Everything writes to STDERR on purpose: `aso analyze --json | jq` has to stay
clean, and progress is not part of the result.

Degrades honestly. On a real terminal you get an in-place spinner and counter;
piped or redirected, you get one plain line per task and no escape codes.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
CLEAR = "\r\033[2K"


def _tty() -> bool:
    return sys.stderr.isatty()


class Task:
    """One unit of visible work.

        with Task("searching Play") as t:
            ...
            t.done("30 results")

        with Task("fetching details", total=30) as t:
            for pkg in pkgs:
                t.step(pkg)
    """

    def __init__(self, label: str, total: int | None = None, quiet: bool = False):
        self.label, self.total, self.quiet = label, total, quiet
        self.n = 0
        self.note = ""
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        if self.quiet:
            return self
        if not _tty():
            print(f"  {self.label}...", file=sys.stderr, flush=True)
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._halt()
        if exc_type and not self.quiet:
            print(f"{CLEAR if _tty() else ''}  {self.label}: "
                  f"{exc_type.__name__}", file=sys.stderr, flush=True)
        return False

    def _halt(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

    # -- updates -----------------------------------------------------------
    def step(self, note: str = "", n: int = 1):
        self.n += n
        self.note = note

    def done(self, msg: str | None = None):
        self._halt()
        if self.quiet:
            return
        secs = time.monotonic() - self.t0
        line = f"  {self.label}: {msg}" if msg else f"  {self.label}"
        tail = f"  ({secs:.1f}s)" if secs >= 1.0 else ""
        print(f"{CLEAR if _tty() else ''}{line}{tail}", file=sys.stderr, flush=True)

    # -- the animation -----------------------------------------------------
    def _spin(self):
        for frame in itertools.cycle(FRAMES):
            if self._stop.is_set():
                return
            secs = time.monotonic() - self.t0
            count = f" {self.n}/{self.total}" if self.total else ""
            note = f"  {self.note[:42]}" if self.note else ""
            sys.stderr.write(f"{CLEAR}  {frame} {self.label}{count}{note}"
                             f"  {secs:.0f}s")
            sys.stderr.flush()
            time.sleep(0.08)


def say(msg: str, quiet: bool = False):
    if not quiet:
        print(f"  {msg}", file=sys.stderr, flush=True)
