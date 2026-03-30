from __future__ import annotations

from typing import Literal

from models.schema import Claim


_EXTERNAL_CHECK_TYPES: frozenset[str] = frozenset(
    {
        "benchmark_result",
        "prior_work_comparison",
        "factual_background",
        "result",
        "factual",
        "novelty",
    }
)

_CONSISTENCY_ONLY_TYPES: frozenset[str] = frozenset({"methodology_assertion"})

_INHERENTLY_HIGH_IMPORTANCE: frozenset[str] = frozenset(
    {"benchmark_result", "result", "prior_work_comparison", "novelty"}
)

RoutingDecision = Literal["external_factcheck", "consistency_only", "skip"]


def is_high_importance(claim: Claim) -> bool:
    return claim.importance == "high" or claim.claim_type in _INHERENTLY_HIGH_IMPORTANCE


def routing_decision(claim: Claim) -> RoutingDecision:
    if claim.claim_type in _EXTERNAL_CHECK_TYPES:
        return "external_factcheck"
    if claim.claim_type in _CONSISTENCY_ONLY_TYPES:
        return "consistency_only"
    return "skip"
