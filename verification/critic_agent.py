from __future__ import annotations

import re

from models.schema import Claim, EvidenceFactCheckItem
from verification.claim_router import is_high_importance


# Verdicts that indicate a retrieval retry may be worthwhile.
_WEAK_EVIDENCE_VERDICTS = frozenset(["insufficient_evidence", "paper_supported_only"])


def should_retry(result: EvidenceFactCheckItem, claim: Claim) -> bool:
    """
    Return True when the evidence is weak AND the claim is high importance.

    This is the minimal agentic loop gate: only high-importance claims get
    a second retrieval attempt. Low-importance claims accept weak evidence.
    """
    if result.verdict not in _WEAK_EVIDENCE_VERDICTS:
        return False
    return is_high_importance(claim)


def build_retry_query(claim: Claim, paper_title: str = "") -> str:
    """
    Build a broader fallback query for the retry retrieval pass.

    Strips domain-specific jargon and expands to a more general
    formulation so the second pass fetches different results.
    """
    # Remove numbers and special chars, keep main noun phrases.
    simplified = re.sub(r"[\d%\(\)\[\]{}=<>]", " ", claim.text)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    words = simplified.split()

    # Take first 12 words as the core of the expanded query.
    core = " ".join(words[:12])

    if paper_title:
        return f"{paper_title[:50]} {core} evidence research"
    return f"{core} machine learning evidence research"
