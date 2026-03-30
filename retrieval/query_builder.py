from __future__ import annotations

import re

from models.schema import Claim


# Common stopwords to filter out of keyword extraction.
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would could should"
    " may might shall of in on at to for with by from that this these those it its we our".split()
)

# Well-known benchmark names used to weight entity-focused queries.
_BENCHMARK_PATTERNS = re.compile(
    r"\b(BLEU|ROUGE|WMT|SQuAD|GLUE|SuperGLUE|ImageNet|COCO|MNIST|CIFAR|"
    r"BenchmarkName|MMLU|HumanEval|GSM8K|ARC|TruthfulQA|HellaSwag|"
    r"F1|mAP|perplexity|accuracy|FLOP|FLOPs)\b",
    re.IGNORECASE,
)

_CITED_TITLE_RE = re.compile(r'"([^"]{10,80})"')


def _extract_keywords(text: str, top_n: int = 6) -> str:
    words = re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", text)
    keywords = [w for w in words if w.lower() not in _STOPWORDS]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for kw in keywords:
        lw = kw.lower()
        if lw not in seen:
            seen.add(lw)
            unique.append(kw)
    return " ".join(unique[:top_n])


def _shorten_claim(text: str, max_chars: int = 180) -> str:
    """Return the first meaningful sentence of a claim, capped at max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[0][:max_chars] if sentences else text[:max_chars]


def build_queries(
    claim: Claim,
    paper_title: str = "",
    max_queries: int = 4,
) -> list[str]:
    """
    Generate 2–4 search queries for a claim.

    Strategy (in priority order):
    1. Shortened claim text — direct retrieval.
    2. Entity/benchmark-focused query — precision for numerical claims.
    3. Cited paper title if available — grounds query in prior work.
    4. Paper title + claim keywords — broad fallback.

    All queries are kept short to avoid search engine truncation and stay
    within the 16k architectural token budget when concatenated with evidence.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)

    # 1. Shortened claim.
    _add(_shorten_claim(claim.text))

    # 2. Benchmark / entity focused.
    benchmarks = _BENCHMARK_PATTERNS.findall(claim.text)
    if benchmarks:
        entity_q = f"{benchmarks[0]} {_extract_keywords(claim.text, top_n=4)}"
        _add(entity_q)

    # 3. Cited paper title from nearby_citations or cited_refs.
    cited = (claim.nearby_citations or []) + (claim.cited_refs or [])
    if cited:
        # Prefer refs that look like quoted titles.
        title_match = _CITED_TITLE_RE.search(claim.text)
        if title_match:
            _add(f"{title_match.group(1)} {_extract_keywords(claim.text, top_n=3)}")
        else:
            _add(f"{cited[0]} {_extract_keywords(claim.text, top_n=3)}")

    # 4. Paper title + claim keyword fallback.
    if paper_title:
        keywords = _extract_keywords(claim.text, top_n=5)
        _add(f"{paper_title[:60]} {keywords}")

    # Ensure at least 2 queries even if everything above collapsed.
    if len(queries) < 2:
        _add(_extract_keywords(claim.text, top_n=8))

    return queries[:max_queries]
