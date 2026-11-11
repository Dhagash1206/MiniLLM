"""Wikipedia as the knowledge source: search the MediaWiki API, download plain-text pages, cache them on disk.

Standard library only (urllib). Searches and pages are cached as JSON under `cache_dir`, so a repeated
question needs no network and `offline=True` answers from cached pages alone (this is what the tests use).
Set the WIKI_USER_AGENT environment variable to identify yourself to Wikimedia if you fetch a lot of pages.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from MiniLLM.RAG.BM25 import BM25

DEFAULT_CACHE_DIR = "data/wiki_cache"
DEFAULT_USER_AGENT = "MiniLLM-RAG/0.1 (educational retrieval demo; Python urllib)"
API_URL_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"
MIN_SECONDS_BETWEEN_REQUESTS = 1.0
MAX_RETRY_DELAY_SECONDS = 60.0
RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
# Wikipedia's search matches keywords; question words and fillers only pull in unrelated pages.
QUESTION_WORDS = frozenset("""
    a an and are as at be by can could did do does for from had has have how i in is it me my of on or please
    tell that the their there these this those to was were what when where which who whom whose why will with
    would you your about name named called give list explain describe
""".split())

JsonFetcher = Callable[[str, dict], dict]


class OfflineError(RuntimeError):
    """Raised when a page or search is not cached and the source is offline."""


@dataclass(frozen=True)
class WikiPage:
    title: str
    url: str
    text: str


def search_query_from_question(question: str) -> str:
    words = re.findall(r"[\w'.-]+", question)
    keywords = [word.strip(".'") for word in words if word.lower().strip(".'") not in QUESTION_WORDS]
    return " ".join(word for word in keywords if word) or question.strip()


def cache_filename(key: str) -> str:
    """Readable and filesystem-safe; the hash keeps titles differing only in case or punctuation apart."""
    readable = re.sub(r"[^\w-]+", "_", key).strip("_")[:80] or "page"
    return f"{readable}-{hashlib.sha1(key.encode()).hexdigest()[:10]}.json"


def retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Honour the server's Retry-After on 429/503, else back off exponentially (2, 4, 8, ... s), capped at 60 s."""
    retry_after = getattr(error, "headers", None) and error.headers.get("Retry-After")
    if retry_after and retry_after.strip().isdigit():
        return min(float(retry_after), MAX_RETRY_DELAY_SECONDS)
    return min(2.0 ** (attempt + 1), MAX_RETRY_DELAY_SECONDS)


def fetch_json_over_http(url: str, params: dict, user_agent: str, timeout_seconds: float = 20.0,
                         retries: int = 5) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": user_agent})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_CODES or attempt == retries - 1:
                raise
            time.sleep(retry_delay_seconds(error, attempt))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries - 1:
                raise
            time.sleep(retry_delay_seconds(error, attempt))
    raise RuntimeError("retries must be at least 1")


class WikiSource:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR, language: str = "en", offline: bool = False,
                 fetch_json: JsonFetcher | None = None,
                 min_seconds_between_requests: float = MIN_SECONDS_BETWEEN_REQUESTS):
        self.cache_dir = Path(cache_dir)
        self.pages_dir = self.cache_dir / "pages"
        self.searches_dir = self.cache_dir / "searches"
        self.api_url = API_URL_TEMPLATE.format(language=language)
        self.offline = offline
        user_agent = os.environ.get("WIKI_USER_AGENT", DEFAULT_USER_AGENT)
        self.fetch_json = fetch_json or (lambda url, params: fetch_json_over_http(url, params, user_agent))
        self.min_seconds_between_requests = min_seconds_between_requests
        self._last_request_time = float("-inf")

    def _call_api(self, params: dict) -> dict:
        if self.offline:
            raise OfflineError("not cached and offline")
        # Wikimedia rate-limits bursts from anonymous clients; spacing requests avoids HTTP 429.
        wait = self.min_seconds_between_requests - (time.monotonic() - self._last_request_time)
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()
        return self.fetch_json(self.api_url, params)

    def cached_pages(self) -> list[WikiPage]:
        return [self._read_page(path) for path in sorted(self.pages_dir.glob("*.json"))]

    @staticmethod
    def _read_page(path: Path) -> WikiPage:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WikiPage(data["title"], data["url"], data["text"])

    def save_page(self, page: WikiPage, requested_title: str | None = None) -> None:
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({**asdict(page), "fetched_at": time.strftime("%Y-%m-%d")}, ensure_ascii=False)
        for title in {page.title, requested_title or page.title}:  # redirects: cache under both names
            (self.pages_dir / cache_filename(title)).write_text(payload, encoding="utf-8")

    def search(self, query: str, limit: int = 4) -> list[str]:
        """Titles of the best-matching articles. Offline, cached pages are ranked with BM25 instead."""
        cache_path = self.searches_dir / cache_filename(f"{query.lower()}|{limit}")
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if self.offline:
            pages = self.cached_pages()
            # Titles are repeated so a title match outweighs a passing mention deep in a long article.
            ranker = BM25([f"{page.title} " * 3 + page.text for page in pages])
            return [pages[index].title for index, _ in ranker.top_k(query, limit)]
        response = self._call_api({"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
                                   "srprop": "", "srwhat": "text"})
        titles = [hit["title"] for hit in response.get("query", {}).get("search", [])]
        self.searches_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
        return titles

    def page(self, title: str) -> WikiPage | None:
        """Plain-text article (sections marked `== Heading ==`), from cache when possible; None if missing."""
        cache_path = self.pages_dir / cache_filename(title)
        if cache_path.exists():
            return self._read_page(cache_path)
        if self.offline:
            return None
        response = self._call_api({"action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
                                   "inprop": "url", "titles": title})
        pages = response.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or not pages[0].get("extract", "").strip():
            return None
        found = pages[0]
        page = WikiPage(found["title"], found.get("fullurl") or f"https://en.wikipedia.org/wiki/{found['title']}",
                        found["extract"])
        self.save_page(page, requested_title=title)
        return page

    def pages_for_question(self, question: str, limit: int = 4) -> list[WikiPage]:
        pages, seen_urls = [], set()
        for title in self.search(search_query_from_question(question), limit):
            page = self.page(title)
            if page is not None and page.url not in seen_urls:
                seen_urls.add(page.url)
                pages.append(page)
        return pages
