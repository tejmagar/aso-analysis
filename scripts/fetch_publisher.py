"""List every app by a publisher, by scrolling their Play developer page.

The package's `fetch_publisher_apps` uses Play SEARCH, which caps at ~50 results
however many apps a publisher actually has. The developer page itself is a lazy
grid: it renders a first batch and appends more as you scroll, so reading all of
it needs a real browser.

    python scripts/fetch_publisher.py "Flash Software Solution"
    python scripts/fetch_publisher.py "Flash Software Solution" --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path


def fetch(developer: str, headless: bool = True, max_rounds: int = 80,
          quiet_rounds: int = 5, verbose: bool = True) -> list[dict]:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    for flag in ("--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                 "--window-size=1400,2000", "--log-level=3"):
        opts.add_argument(flag)
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    url = ("https://play.google.com/store/apps/developer?id="
           + urllib.parse.quote_plus(developer))
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(url)
        time.sleep(3)
        seen: dict[str, dict] = {}
        stale = 0
        for _ in range(max_rounds):
            for a in driver.find_elements(By.CSS_SELECTOR,
                                          'a[href*="/store/apps/details?id="]'):
                href = a.get_attribute("href") or ""
                m = re.search(r"[?&]id=([^&]+)", href)
                if not m:
                    continue
                pkg = urllib.parse.unquote(m.group(1))
                if pkg in seen:
                    continue
                lines = [x.strip() for x in (a.text or "").splitlines() if x.strip()]
                seen[pkg] = {"package": pkg, "title": lines[0] if lines else None,
                             "url": href.split("&")[0]}
            before = len(seen)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.8)
            if verbose:
                print(f"\r  {len(seen)} apps found", end="", file=sys.stderr, flush=True)
            # Stop only after several quiet rounds: the grid pauses between
            # batches, so one round adding nothing is not the end of the list.
            stale = stale + 1 if len(seen) == before else 0
            if stale >= quiet_rounds:
                break
        if verbose:
            print(file=sys.stderr)
        return sorted(seen.values(), key=lambda x: x["package"])
    finally:
        driver.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("developer")
    ap.add_argument("--json", help="write the list to this file")
    ap.add_argument("--show", action="store_true", help="run with a visible window")
    a = ap.parse_args()

    apps = fetch(a.developer, headless=not a.show)
    print(f"{len(apps)} apps for {a.developer!r}", file=sys.stderr)
    if a.json:
        Path(a.json).write_text(json.dumps(apps, indent=2))
        print(f"written to {a.json}", file=sys.stderr)
    else:
        for x in apps:
            print(f"{x['package']}\t{x['title']}")


if __name__ == "__main__":
    main()
