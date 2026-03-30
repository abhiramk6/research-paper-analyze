from __future__ import annotations

from typing import Literal

from models.schema import Claim


# Claim types that warrant external evidence retrieval.
_EXTERNAL_CHECK_TYPES: frozenset[str] = frozenset(
    [
        "benchmark_result",
        "prior_work_comparison",
        "factual_background",
        # Legacy equivalents
        "result",
        "factual",
        "novelty",
    ]
)

# Claim types that only need internal consistency review — no external retrieval.
_CONSISTENCY_ONLY_TYPES: frozenset[str] = frozenset(["methodology_assertion"])

# Claim types routed to low-priority / no forced retrieval.
_LOW_PRIORITY_TYPES: frozenset[str] = frozenset(
    [
        "contribution_claim",
        "unsupported_general_statement",
        # Legacy equivalents
        "contribution",
        "background",
    ]
)

# Importance heuristic — claim types that imply high stakes regardless of the
# `importance` field value set by the extractor.
_INHERENTLY_HIGH_IMPORTANCE: frozenset[str] = frozenset(
    ["benchmark_result", "result", "prior_work_comparison", "novelty"]
)

RoutingDecision = Literal["external_factcheck", "consistency_only", "skip"]


def needs_external_retrieval(claim: Claim) -> bool:
    return claim.claim_type in _EXTERNAL_CHECK_TYPES


def is_high_importance(claim: Claim) -> bool:
    return claim.importance == "high" or claim.claim_type in _INHERENTLY_HIGH_IMPORTANCE


def routing_decision(claim: Claim) -> RoutingDecision:
    """
    Decide which review path a claim should take.

    Returns:
      "external_factcheck"  — retrieve evidence + verify
      "consistency_only"    — run internal consistency review only
      "skip"                — low priority, no forced review
    """
    if claim.claim_type in _EXTERNAL_CHECK_TYPES:
        return "external_factcheck"
    if claim.claim_type in _CONSISTENCY_ONLY_TYPES:
        return "consistency_only"
    return "skip"
