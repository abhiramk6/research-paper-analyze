from __future__ import annotations

import re

from models.schema import Claim, EvidenceFactCheckItem
from verification.claim_router import is_high_importance


_WEAK_EVIDENCE_VERDICTS = frozenset(["insufficient_evidence", "paper_supported_only"])


def should_retry(result: EvidenceFactCheckItem, claim: Claim) -> bool:
    if result.verdict not in _WEAK_EVIDENCE_VERDICTS:
        return False
    return is_high_importance(claim)


def build_retry_query(claim: Claim, paper_title: str = "") -> str:
    simplified = re.sub(r"[\d%\(\)\[\]{}=<>]", " ", claim.text)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    words = simplified.split()

    core = " ".join(words[:12])

    if paper_title:
        return f"{paper_title[:50]} {core} evidence research"
    return f"{core} machine learning evidence research"
