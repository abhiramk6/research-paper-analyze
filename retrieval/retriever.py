import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml

from models.schema import EvidenceItem


logger = logging.getLogger(__name__)

DEFAULT_SNIPPET_MAX_CHARS = 350
SCHOLARLY_DOMAINS = (
    "arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "proceedings.mlr.press",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "semanticscholar.org",
    "springer.com",
    "sciencedirect.com",
    "nature.com",
    "neurips.cc",
    "papers.nips.cc",
)

BLOCKED_DOMAINS = (
    "baidu.com",
    "zhidao.baidu.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "quora.com",
)

BLOCKED_TITLE_TERMS = (
    "lyrics",
    "song",
    "mp3",
    "torrent",
    "百度",
)


def _load_retrieval_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "retrieval.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing retrieval config: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


_RETRIEVAL_CONFIG = _load_retrieval_config()


def _config_int(key: str, default: int) -> int:
    return int(_RETRIEVAL_CONFIG.get(key, default))


def _config_bool(key: str, default: bool) -> bool:
    value = _RETRIEVAL_CONFIG.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


MAX_RESULTS_PER_QUERY = _config_int("max_results_per_query", 5)
GLOBAL_EVIDENCE_CAP = _config_int("global_evidence_cap", 10)
SNIPPET_MAX_CHARS = _config_int("snippet_max_chars", DEFAULT_SNIPPET_MAX_CHARS)


def _make_evidence_id(source: str, index: int) -> str:
    digest = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:8]
    return f"ev_{digest}"


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower()


def domain_tier(url: str) -> str:
    domain = _domain_from_url(url)
    return "scholarly" if any(domain.endswith(token) for token in SCHOLARLY_DOMAINS) else "web"


def _normalized_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path, "", "", ""))


def _is_blocked_result(title: str, url: str, snippet: str) -> bool:
    domain = _domain_from_url(url)
    lowered_title = title.lower()
    lowered_snippet = snippet.lower()
    if any(domain.endswith(token) for token in BLOCKED_DOMAINS):
        return True
    if any(term in lowered_title or term in lowered_snippet for term in BLOCKED_TITLE_TERMS):
        return True
    return False


class DDGSRetriever:
    def retrieve(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[EvidenceItem]:
        try:
            from ddgs import DDGS
        except ImportError:
            raise RuntimeError("ddgs is not installed; external retrieval is unavailable.")

        items: list[EvidenceItem] = []
        time.sleep(0.35)
        with DDGS() as ddgs:
            next_index = 0
            for result in ddgs.text(query, max_results=max_results):
                url = result.get("href") or ""
                title = (result.get("title") or "")[:140]
                snippet = (result.get("body") or "")[:SNIPPET_MAX_CHARS]
                if not url or _is_blocked_result(title, url, snippet):
                    continue
                domain = _domain_from_url(url)
                items.append(
                    EvidenceItem(
                        evidence_id=_make_evidence_id(url or query, next_index),
                        query=query,
                        title=title,
                        url=url,
                        domain=domain,
                        domain_tier=domain_tier(url),
                        snippet=snippet,
                        retrieval_score=0.0,
                    )
                )
                next_index += 1
                if len(items) >= max_results:
                    break
        return items


class EvidenceRetriever:
    def __init__(self, backend: "DDGSRetriever | None" = None) -> None:
        retrieval_enabled = _config_bool("enable_web_retrieval", True)
        if backend is not None:
            self._backend = backend
        elif retrieval_enabled:
            self._backend = DDGSRetriever()
        else:
            raise RuntimeError("External retrieval is disabled in config.")

    def retrieve_for_plan(
        self,
        query_plan: dict[str, list[str]],
        max_per_query: int = MAX_RESULTS_PER_QUERY,
        global_cap: int = GLOBAL_EVIDENCE_CAP,
    ) -> list[EvidenceItem]:
        merged: list[EvidenceItem] = []
        seen_urls: set[str] = set()

        def collect(queries: list[str]) -> None:
            for query in queries:
                for item in self._backend.retrieve(query, max_results=max_per_query):
                    dedupe_key = _normalized_url(item.url) or f"{item.title}:{item.snippet}"
                    if dedupe_key in seen_urls:
                        continue
                    seen_urls.add(dedupe_key)
                    merged.append(item)
                    if len(merged) >= global_cap:
                        return

        collect(query_plan.get("scholarly", []))
        scholarly_count = sum(1 for item in merged if item.domain_tier == "scholarly")
        if scholarly_count < min(3, global_cap):
            collect(query_plan.get("general", []))
        return merged[:global_cap]
