"""A robots.txt matcher that actually handles wildcards.

Python's urllib.robotparser ignores `*` and `$` in paths, so it silently
permits paths the site disallows (verified against Rightmove's robots.txt,
which uses `*` in most of its rules). This implements the de-facto standard:
`*` matches any sequence, `$` anchors the end, and where an Allow and a
Disallow both match, the longest pattern wins (ties go to Allow).
"""
from __future__ import annotations

import re
from urllib.parse import urlparse, unquote


def _compile(pattern: str) -> re.Pattern:
    anchored_end = pattern.endswith("$")
    if anchored_end:
        pattern = pattern[:-1]
    parts = [re.escape(p) for p in pattern.split("*")]
    rx = ".*".join(parts)
    return re.compile("^" + rx + ("$" if anchored_end else ""))


class RobotsRules:
    """Rules for one user-agent group, parsed from a robots.txt body."""

    def __init__(self, body: str, user_agent: str = "*"):
        self.rules: list[tuple[str, re.Pattern, bool]] = []  # (raw, regex, allow)
        self.crawl_delay: float | None = None
        self._parse(body, user_agent)

    def _parse(self, body: str, user_agent: str) -> None:
        ua_l = user_agent.lower()
        groups: dict[str, list[tuple[str, str]]] = {}
        current: list[str] = []
        prev_was_ua = False
        for raw_line in body.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            field, _, value = line.partition(":")
            field = field.strip().lower()
            value = value.strip()
            if field == "user-agent":
                if not prev_was_ua:
                    current = []
                current.append(value.lower())
                groups.setdefault(value.lower(), [])
                prev_was_ua = True
            elif field in ("allow", "disallow", "crawl-delay"):
                prev_was_ua = False
                for ua in current:
                    groups.setdefault(ua, []).append((field, value))

        # Most specific matching group: exact UA token match, else '*'
        chosen: list[tuple[str, str]] = []
        for ua, directives in groups.items():
            if ua != "*" and ua in ua_l:
                chosen = directives
                break
        else:
            chosen = groups.get("*", [])

        for field, value in chosen:
            if field == "crawl-delay":
                try:
                    self.crawl_delay = float(value)
                except ValueError:
                    pass
                continue
            if field == "disallow" and value == "":
                continue  # "Disallow:" with no value means allow everything
            if not value:
                continue
            self.rules.append((value, _compile(value), field == "allow"))

    def can_fetch(self, url: str) -> bool:
        path = urlparse(url).path or "/"
        if urlparse(url).query:
            path += "?" + urlparse(url).query
        path = unquote(path)
        best: tuple[int, bool] | None = None
        for raw, rx, allow in self.rules:
            if rx.match(path):
                specificity = len(raw)
                if best is None or specificity > best[0] or (specificity == best[0] and allow):
                    best = (specificity, allow)
        return True if best is None else best[1]
