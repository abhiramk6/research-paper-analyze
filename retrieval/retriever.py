from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from models.schema import EvidenceItem


logger = logging.getLogger(__name__)
_MISSING_DDG_WARNED = False

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


def _load_retrieval_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "retrieval.yaml"
    try:
        import yaml  # type: ignore[import]

        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


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


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[EvidenceItem]:
        ...


def _make_evidence_id(source: str, index: int) -> str:
    digest = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:8]
    return f"ev_{digest}"


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower()


def domain_tier(url: str) -> str:
    domain = _domain_from_url(url)
    return "scholarly" if any(domain.endswith(token) for token in SCHOLARLY_DOMAINS) else "web"


class DuckDuckGoRetriever:
    def retrieve(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[EvidenceItem]:
        global _MISSING_DDG_WARNED
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            if not _MISSING_DDG_WARNED:
                logger.warning("duckduckgo_search not installed; external retrieval disabled.")
                _MISSING_DDG_WARNED = True
            return []

        items: list[EvidenceItem] = []
        try:
            time.sleep(0.35)
            with DDGS() as ddgs:
                for index, result in enumerate(ddgs.text(query, max_results=max_results)):
                    url = result.get("href") or ""
                    domain = _domain_from_url(url)
                    items.append(
                        EvidenceItem(
                            evidence_id=_make_evidence_id(url or query, index),
                            query=query,
                            title=(result.get("title") or "")[:140],
                            url=url,
                            domain=domain,
                            domain_tier=domain_tier(url),
                            snippet=(result.get("body") or "")[:SNIPPET_MAX_CHARS],
                            retrieval_score=0.0,
                        )
                    )
        except Exception as exc:
            logger.warning("Retrieval failed for query %r: %s", query[:80], exc)
        return items


class EvidenceRetriever:
    def __init__(self, backend: RetrieverProtocol | None = None) -> None:
        retrieval_enabled = _config_bool("enable_web_retrieval", True)
        disabled_via_env = os.getenv("DISABLE_WEB_RETRIEVAL", "").lower() in {"1", "true", "yes"}
        if backend is not None:
            self._backend = backend
        elif retrieval_enabled and not disabled_via_env:
            self._backend = DuckDuckGoRetriever()
        else:
            self._backend = None

    def retrieve_for_plan(
        self,
        query_plan: dict[str, list[str]],
        max_per_query: int = MAX_RESULTS_PER_QUERY,
        global_cap: int = GLOBAL_EVIDENCE_CAP,
    ) -> list[EvidenceItem]:
        if self._backend is None:
            return []

        merged: list[EvidenceItem] = []
        seen_urls: set[str] = set()

        def collect(queries: list[str]) -> None:
            for query in queries:
                for item in self._backend.retrieve(query, max_results=max_per_query):
                    dedupe_key = item.url or f"{item.title}:{item.snippet}"
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
