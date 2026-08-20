"""Polite HTTP client: robots.txt enforcement, rate limiting, retries.

Deliberately conservative. Per docs/DATA-SOURCES.md §3.9 the scraping decision
is taken with eyes open, and rate limiting plus robots compliance are the two
cheapest things that keep it defensible.
"""
from __future__ import annotations

import logging
import random
import time
from urllib.parse import urlparse

import requests

from .robots import RobotsRules

log = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}


class BlockedError(RuntimeError):
    """Raised when a host actively refuses us (bot wall), as opposed to a transient error."""


class RobotsDisallowed(RuntimeError):
    """Raised when robots.txt forbids the path for our user-agent."""


def _looks_blocked(text: str, status: int) -> bool:
    if status in (403, 429) or status == 503:
        markers = ("just a moment", "captcha", "access denied", "datadome",
                   "px-captcha", "attention required", "enable javascript and cookies")
        low = text[:4000].lower()
        return status in (403, 429) or any(m in low for m in markers)
    return False


class PoliteSession:
    """Rate-limited requests session that honours robots.txt.

    min_interval is enforced per-host; a small random jitter avoids a metronomic
    request pattern.
    """

    def __init__(self, user_agent: str = DEFAULT_UA, min_interval: float = 1.5,
                 timeout: int = 40, respect_robots: bool = True, max_retries: int = 3):
        self.ua = user_agent
        self.min_interval = min_interval
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.max_retries = max_retries
        self.s = requests.Session()
        self.s.headers.update({**BROWSER_HEADERS, "User-Agent": user_agent})
        self._last: dict[str, float] = {}
        self._robots: dict[str, RobotsRules | None] = {}

    # -- robots ---------------------------------------------------------
    def _robots_for(self, url: str) -> RobotsRules | None:
        host = urlparse(url).netloc
        if host in self._robots:
            return self._robots[host]
        rules: RobotsRules | None = None
        robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
        try:
            r = self.s.get(robots_url, timeout=self.timeout)
            body = r.text
            if r.status_code == 200 and "<html" not in body[:200].lower():
                rules = RobotsRules(body, self.ua)
                if rules.crawl_delay:
                    self.min_interval = max(self.min_interval, rules.crawl_delay)
                    log.info("%s declares crawl-delay %.1fs", host, rules.crawl_delay)
            elif r.status_code == 404:
                log.info("%s has no robots.txt (404) - nothing to obey", host)
            else:
                log.warning("%s robots.txt unreadable (HTTP %s) - treating as blocked host",
                            host, r.status_code)
        except requests.RequestException as e:
            log.warning("could not fetch robots for %s: %s", host, e)
        self._robots[host] = rules
        return rules

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rules = self._robots_for(url)
        return True if rules is None else rules.can_fetch(url)

    # -- fetching -------------------------------------------------------
    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last.get(host)
        if last is not None:
            wait = self.min_interval + random.uniform(0, 0.4) - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
        self._last[host] = time.time()

    def get(self, url: str, *, expect: str = "text") -> requests.Response:
        if not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle(url)
            try:
                r = self.s.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                last_exc = e
                time.sleep(2 ** attempt)
                continue
            body = r.text if expect == "text" else ""
            if _looks_blocked(body, r.status_code):
                raise BlockedError(
                    f"{urlparse(url).netloc} returned {r.status_code} with a bot wall. "
                    "This host needs a real browser and a residential IP."
                )
            if r.status_code == 200:
                return r
            if 500 <= r.status_code < 600:
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        raise RuntimeError(f"GET failed after {self.max_retries} attempts: {url} ({last_exc})")

    def download(self, url: str, dest) -> int:
        """Fetch a binary asset to `dest`. Returns bytes written."""
        from pathlib import Path
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._throttle(url)
        r = self.s.get(url, timeout=self.timeout, stream=True)
        r.raise_for_status()
        n = 0
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(65536):
                fh.write(chunk)
                n += len(chunk)
        return n
