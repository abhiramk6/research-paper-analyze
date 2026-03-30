from __future__ import annotations

import re

from models.schema import Claim, EvidenceItem


_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|x|k|M|B)?\b")


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\b\w{3,}\b", a.lower()))
    tokens_b = set(re.findall(r"\b\w{3,}\b", b.lower()))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _number_overlap(a: str, b: str) -> float:
    nums_a = set(_NUMBER_RE.findall(a))
    nums_b = set(_NUMBER_RE.findall(b))
    if not nums_a:
        return 0.0
    return len(nums_a & nums_b) / len(nums_a)


def _domain_bonus(item: EvidenceItem) -> float:
    return 1.0 if item.domain_tier == "scholarly" else 0.45


def _snippet_signal(snippet: str) -> float:
    if not snippet:
        return 0.0
    if len(snippet) < 80:
        return 0.25
    if len(snippet) < 140:
        return 0.6
    return 1.0


def score_evidence_item(claim: Claim, item: EvidenceItem) -> float:
    combined = f"{item.title} {item.snippet}"
    lexical = _token_overlap(claim.text, combined)
    numeric = _number_overlap(claim.text, item.snippet)
    score = (
        0.45 * lexical
        + 0.15 * numeric
        + 0.25 * _domain_bonus(item)
        + 0.15 * _snippet_signal(item.snippet)
    )
    return round(min(1.0, score), 4)


def rank_evidence(claim: Claim, items: list[EvidenceItem], top_k: int = 5) -> list[EvidenceItem]:
    scored: list[EvidenceItem] = []
    for item in items:
        scored.append(
            item.model_copy(update={"retrieval_score": score_evidence_item(claim, item)})
        )
    scored.sort(key=lambda item: item.retrieval_score, reverse=True)
    return scored[:top_k]
