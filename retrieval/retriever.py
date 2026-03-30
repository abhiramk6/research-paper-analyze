from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Protocol

from models.schema import EvidenceItem


logger = logging.getLogger(__name__)

# Per-query result cap — keeps evidence token budget bounded.
MAX_RESULTS_PER_QUERY = 5

# Snippet character cap — each snippet is kept short to stay under the 16k token budget.
SNIPPET_MAX_CHARS = 350


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[EvidenceItem]:
        ...


def _make_evidence_id(source: str, index: int) -> str:
    h = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:8]
    return f"ev_{h}"


class DuckDuckGoRetriever:
    """
    Lightweight web retriever using the duckduckgo-search package.

    Falls back silently to an empty list if the package is not installed
    or the search fails (rate limit, network error, etc.).

    Token budget: snippets are hard-capped at SNIPPET_MAX_CHARS chars each,
    and at most MAX_RESULTS_PER_QUERY items are returned per query.
    """

    def retrieve(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[EvidenceItem]:
        try:
            from duckduckgo_search import DDGS  # optional dependency
        except ImportError:
            logger.debug("duckduckgo_search not installed — falling back to stub retriever.")
            return StubRetriever().retrieve(query, max_results)

        items: list[EvidenceItem] = []
        try:
            # Small sleep to avoid hammering the DDG rate limiter.
            time.sleep(0.4)
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.text(query, max_results=max_results)):
                    snippet = (result.get("body") or "")[:SNIPPET_MAX_CHARS]
                    items.append(
                        EvidenceItem(
                            evidence_id=_make_evidence_id(result.get("href", query), i),
                            title=(result.get("title") or "")[:120],
                            url=result.get("href") or "",
                            snippet=snippet,
                            source_type="web",
                            retrieval_score=0.0,
                        )
                    )
        except Exception as exc:
            logger.warning("DDG retrieval failed for query %r: %s", query[:80], exc)
        return items


class StubRetriever:
    """
    Fallback retriever that returns empty evidence.

    Used when no search provider is available or when the claim type
    does not warrant external lookup (e.g., methodology_assertion).
    """

    def retrieve(self, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[EvidenceItem]:
        return []


class EvidenceRetriever:
    """
    Pluggable retriever facade.

    Chooses between real web search and stub based on environment / availability.
    Run all queries, deduplicate by URL, and return merged results.

    Architectural token budget:
    - Each query returns at most MAX_RESULTS_PER_QUERY snippets.
    - Each snippet is capped at SNIPPET_MAX_CHARS.
    - Total evidence passed downstream is further bounded by the evidence ranker.
    """

    def __init__(self, backend: RetrieverProtocol | None = None) -> None:
        if backend is not None:
            self._backend = backend
        elif os.getenv("DISABLE_WEB_RETRIEVAL", "").lower() in ("1", "true", "yes"):
            self._backend = StubRetriever()
        else:
            self._backend = DuckDuckGoRetriever()

    def retrieve_for_queries(
        self,
        queries: list[str],
        max_per_query: int = MAX_RESULTS_PER_QUERY,
        global_cap: int = 10,
    ) -> list[EvidenceItem]:
        """
        Run all queries, deduplicate by URL, cap total results at global_cap.

        global_cap keeps the evidence token cost bounded when multiple queries
        return overlapping results.
        """
        seen_urls: set[str] = set()
        merged: list[EvidenceItem] = []
        for query in queries:
            for item in self._backend.retrieve(query, max_results=max_per_query):
                if item.url and item.url in seen_urls:
                    continue
                seen_urls.add(item.url)
                merged.append(item)
                if len(merged) >= global_cap:
                    return merged
        return merged
