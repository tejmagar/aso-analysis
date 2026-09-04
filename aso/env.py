"""Read a .env file, without a dependency.

Values already in the real environment win, so a container's `-e` flag or a
systemd unit is never overridden by a file someone left in the working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = (os.environ.get("ASO_ENV_FILE"), ".env", str(ROOT / ".env"))

_loaded = False


def load(force: bool = False) -> dict:
    """Populate os.environ from the first .env found. Idempotent."""
    global _loaded
    if _loaded and not force:
        return {}
    _loaded = True
    for path in CANDIDATES:
        if not path or not Path(path).is_file():
            continue
        found = {}
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            found[key] = val
            os.environ.setdefault(key, val)      # the real environment wins
        return found
    return {}
