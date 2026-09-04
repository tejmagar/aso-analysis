"""Colour and arrow-key menus. No dependencies.

Hand-rolled rather than pulling in prompt_toolkit: the whole need is a palette
and a vertical picker, and this project stays lean on purpose. Everything
degrades to plain text when stdout is not a terminal, and honours NO_COLOR.
"""
from __future__ import annotations

import os
import sys

_ENABLED = (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") not in (None, "dumb"))


def _c(code: str) -> str:
    return code if _ENABLED else ""


RESET = _c("\033[0m")
BOLD = _c("\033[1m")
DIM = _c("\033[2m")
RED = _c("\033[38;5;203m")
GREEN = _c("\033[38;5;114m")
AMBER = _c("\033[38;5;179m")
CYAN = _c("\033[38;5;080m")
GRAY = _c("\033[38;5;249m")
# 238 was effectively black on a dark terminal and made every label unreadable.
# FAINT is for text a person still has to READ; TRACK is for the bar's empty
# half, which is decoration and is the only thing that should disappear.
FAINT = _c("\033[38;5;246m")
TRACK = _c("\033[38;5;238m")
WHITE = _c("\033[38;5;255m")
INV = _c("\033[7m")

VERDICT_COLOR = {"BUILD IT": GREEN, "WORTH A TRY": AMBER, "SKIP": RED}


def color_for(score: float, invert: bool = False) -> str:
    """Green is good. For competition, high is bad, so the scale inverts."""
    s = 100 - score if invert else score
    return GREEN if s >= 60 else AMBER if s >= 30 else RED


def bar(frac: float | None, width: int = 20, tint: str = CYAN) -> str:
    if frac is None:
        return f"{TRACK}{'░' * width}{RESET}"
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return f"{tint}{'█' * n}{TRACK}{'░' * (width - n)}{RESET}"


def rule(width: int = 58) -> str:
    return f"{TRACK}{'─' * width}{RESET}"


# ------------------------------------------------------------------ keyboard

def _read_key() -> str:
    """One keypress. Returns 'up', 'down', 'enter', 'quit', or the character."""
    try:
        import termios
        import tty
    except ImportError:                                  # not a unix terminal
        return (sys.stdin.readline().strip() or "enter")

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return (sys.stdin.readline().strip() or "enter")

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "":                                     # EOF: piped input ran out.
            return "quit"                                # returning "" would spin forever
        if ch == "\x1b":                                 # escape sequence
            nxt = sys.stdin.read(2)
            return {"[A": "up", "[B": "down"}.get(nxt, "esc")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":                                 # ctrl-c
            return "quit"
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select(options: list[tuple[str, str]], hint: str = "") -> str | None:
    """Vertical picker. options is [(key, label)]; returns the key, or None.

    Arrow keys or j/k to move, Enter to choose, or press an option's own letter
    to jump straight to it. Falls back to a typed prompt when stdin is not a tty.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        for key, label in options:
            print(f"  [{key}] {label}")
        try:
            return input("  > ").strip().lower() or None
        except (EOFError, KeyboardInterrupt):
            return None

    idx = 0
    lines = len(options) + (1 if hint else 0)
    first = True
    while True:
        if not first:
            sys.stdout.write(f"\033[{lines}A")           # rewind over the menu
        first = False
        for i, (key, label) in enumerate(options):
            if i == idx:
                sys.stdout.write(f"\033[2K  {CYAN}▸{RESET} {BOLD}{WHITE}{label}{RESET}"
                                 f"  {FAINT}{key}{RESET}\n")
            else:
                sys.stdout.write(f"\033[2K    {GRAY}{label}{RESET}\n")
        if hint:
            sys.stdout.write(f"\033[2K  {FAINT}{hint}{RESET}\n")
        sys.stdout.flush()

        k = _read_key()
        if k in ("up", "k"):
            idx = (idx - 1) % len(options)
        elif k in ("down", "j"):
            idx = (idx + 1) % len(options)
        elif k == "enter":
            return options[idx][0]
        elif k in ("quit", "esc"):
            return None
        else:
            for key, _ in options:
                if k == key:
                    return key


def prompt(label: str) -> str | None:
    try:
        v = input(f"  {CYAN}?{RESET} {label} {FAINT}>{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return v or None
