#!/usr/bin/env python3
"""Retrieval backends for the agent: live Wikipedia and a local passage index.

Both backends expose the same tiny interface so the agent does not care which
one it is talking to:

    search(query, limit) -> [{"title": str, "passage": str}, ...]

``WikipediaSearch`` uses the public MediaWiki API over plain HTTP (no extra
dependencies). ``LocalIndexSearch`` reads the SQuAD FTS5 index built by
data_prep/prepare_grounded_qa.py, so the agent still works offline.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.knowledge_store import query_terms


class WikipediaSearch:
    """Live Wikipedia search + plain-text extracts via the MediaWiki API."""

    def __init__(self, lang: str = "en", timeout: int = 12, user_agent: str = "AE-Assistant/1.0 (educational)",
                 delay: float = 1.0):
        self.lang = lang
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.timeout = timeout
        self.user_agent = user_agent
        self.delay = delay
        self._last_request = 0.0
        self._cache: dict[tuple, object] = {}

    def _throttle(self) -> None:
        """Space out requests so Wikipedia does not rate-limit us."""
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, params: dict, attempts: int = 3) -> dict:
        cache_key = tuple(sorted(params.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]
        url = self.api + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last_error = None
        for attempt in range(attempts):
            try:
                self._throttle()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self._cache[cache_key] = data
                return data
            except Exception as error:  # noqa: BLE001 - network errors are expected
                last_error = error
                time.sleep(1.5 * (attempt + 1))
        raise last_error if last_error else RuntimeError("Wikipedia request failed")

    def search(self, query: str, limit: int = 6) -> list[dict]:
        try:
            data = self._get({
                "action": "query", "format": "json", "list": "search",
                "srsearch": query, "srlimit": max(1, limit), "srprop": "snippet",
            })
        except Exception:
            return []
        return [
            {"title": hit.get("title", ""), "snippet": hit.get("snippet", "")}
            for hit in data.get("query", {}).get("search", [])
        ]

    def fetch(self, title: str, max_chars: int = 4000) -> str:
        try:
            data = self._get({
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": 1, "exlimit": 1, "titles": title, "redirects": 1,
            })
        except Exception:
            return ""
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract")
            if text:
                return text[:max_chars]
        return ""

    def fetch_lead(self, title: str, max_chars: int = 3000) -> str:
        """The opening of the article, long enough to contain the key fact.

        The REST summary endpoint is fast but often truncates before the fact
        we need ("Its capital ... is Paris"), so we read the plain-text extract
        instead and cap it. This keeps requests cheap and answers present.
        """
        return self.fetch(title, max_chars)

    def exact_passage(self, title: str) -> dict | None:
        """Fetch the article with exactly this title, if it exists.

        Much more reliable than search when we already know the subject
        ("France", "Hamlet"): search may rank a disambiguation page or a
        same-named novel above the article we actually want.
        """
        try:
            data = self._get({
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": 1, "exlimit": 1, "redirects": 1, "titles": title,
            })
        except Exception:
            return None
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("missing"):
                continue
            text = page.get("extract")
            if text:
                return {"title": page.get("title", title), "passage": text[:3000]}
        return None

    def search_passages(self, query: str, limit: int = 6) -> list[dict]:
        results = []
        for hit in self.search(query, limit):
            passage = self.fetch_lead(hit["title"]) or self.fetch(hit["title"])
            if passage:
                results.append({"title": hit["title"], "passage": passage})
        return results


class LocalIndexSearch:
    """Offline retrieval from the SQuAD FTS5 passage index."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)

    def close(self) -> None:
        self.connection.close()

    def search_passages(self, query: str, limit: int = 6) -> list[dict]:
        terms = list(dict.fromkeys(query_terms(query)))
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"*' for term in terms)
        try:
            rows = self.connection.execute(
                "SELECT title, passage FROM knowledge WHERE knowledge MATCH ? "
                "ORDER BY bm25(knowledge) LIMIT ?",
                (expression, max(1, limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"title": title, "passage": passage} for title, passage in rows]