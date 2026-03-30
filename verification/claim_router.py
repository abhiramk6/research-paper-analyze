from typing import Literal

from models.schema import Claim


RoutingDecision = Literal["external_evidence", "internal_only"]

_EXTERNAL_TYPES = frozenset({"result", "comparison", "factual"})


def is_high_importance(claim: Claim) -> bool:
    return claim.importance == "high" or claim.claim_type in {"result", "comparison"}


def routing_decision(claim: Claim) -> RoutingDecision:
    if claim.claim_type in _EXTERNAL_TYPES:
        return "external_evidence"
    return "internal_only"
