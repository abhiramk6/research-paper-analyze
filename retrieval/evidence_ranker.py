from __future__ import annotations

import re

from models.schema import Claim, EvidenceItem


# Sources that receive a trust bonus in scoring.
_TRUSTED_DOMAINS = frozenset(
    [
        "arxiv.org",
        "semanticscholar.org",
        "dl.acm.org",
        "ieeexplore.ieee.org",
        "openreview.net",
        "proceedings.mlr.press",
        "aclanthology.org",
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "papers.nips.cc",
        "neurips.cc",
    ]
)

_BENCHMARK_RE = re.compile(
    r"\b(BLEU|ROUGE|WMT|SQuAD|GLUE|SuperGLUE|ImageNet|COCO|MNIST|CIFAR|"
    r"MMLU|HumanEval|GSM8K|F1|mAP|perplexity|accuracy)\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|x|k|M|B)?\b")


def _token_overlap(a: str, b: str) -> float:
    """Fraction of unique tokens in `a` that appear in `b`."""
    tokens_a = set(re.findall(r"\b\w{3,}\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w{3,}\b", b.lower()))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _benchmark_overlap(claim_text: str, snippet: str) -> float:
    """Bonus score when both claim and evidence share benchmark/metric names."""
    claim_benchmarks = set(m.lower() for m in _BENCHMARK_RE.findall(claim_text))
    snippet_benchmarks = set(m.lower() for m in _BENCHMARK_RE.findall(snippet))
    if not claim_benchmarks:
        return 0.0
    overlap = len(claim_benchmarks & snippet_benchmarks)
    return min(1.0, overlap / len(claim_benchmarks))


def _number_overlap(claim_text: str, snippet: str) -> float:
    """Bonus when evidence contains numbers that also appear in the claim."""
    claim_nums = set(_NUMBER_RE.findall(claim_text))
    snippet_nums = set(_NUMBER_RE.findall(snippet))
    if not claim_nums:
        return 0.0
    overlap = len(claim_nums & snippet_nums)
    return min(1.0, overlap / len(claim_nums))


def _domain_trust(url: str) -> float:
    """Return a trust bonus [0, 1] based on the source domain."""
    url_lower = url.lower()
    return 0.25 if any(domain in url_lower for domain in _TRUSTED_DOMAINS) else 0.0


def _snippet_usefulness(snippet: str) -> float:
    """Penalise very short or empty snippets."""
    if not snippet or len(snippet) < 40:
        return 0.0
    if len(snippet) < 100:
        return 0.3
    return 1.0


def score_evidence_item(claim: Claim, item: EvidenceItem) -> float:
    """
    Compute a [0, 1] relevance score for a single evidence item against a claim.

    Weighted components:
      40% lexical overlap between claim text and snippet
      20% benchmark/entity overlap
      15% number overlap (important for result claims)
      25% source trust + snippet usefulness
    """
    combined_text = f"{item.title} {item.snippet}"
    lexical = _token_overlap(claim.text, combined_text)
    benchmark = _benchmark_overlap(claim.text, combined_text)
    number = _number_overlap(claim.text, item.snippet)
    trust = _domain_trust(item.url)
    usefulness = _snippet_usefulness(item.snippet)
    source_signal = (trust + usefulness) / 2.0

    score = (
        0.40 * lexical
        + 0.20 * benchmark
        + 0.15 * number
        + 0.25 * source_signal
    )
    return round(min(1.0, score), 4)


def rank_evidence(
    claim: Claim,
    items: list[EvidenceItem],
    top_k: int = 5,
) -> list[EvidenceItem]:
    """
    Score and rank evidence items for a claim, returning the top_k most relevant.

    top_k is kept small by default to bound the token cost when evidence is
    passed to the verifier LLM call (each snippet ≤ SNIPPET_MAX_CHARS chars).
    """
    if not items:
        return []
    scored = []
    for item in items:
        item_scored = item.model_copy(update={"retrieval_score": score_evidence_item(claim, item)})
        scored.append(item_scored)
    scored.sort(key=lambda x: x.retrieval_score, reverse=True)
    return scored[:top_k]
