from __future__ import annotations

from models.schema import Claim, ClaimCheckResult, ConsistencyResult, FabricationBreakdown


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _high_impact_claims(claims: list[Claim]) -> list[Claim]:
    return [claim for claim in claims if claim.importance == "high"]


def compute_fabrication_breakdown(
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
) -> FabricationBreakdown:
    check_map = {item.claim_id: item for item in claim_checks}
    checked_claims = [claim for claim in claims if claim.claim_id in check_map]

    contradicted = sum(1 for claim in checked_claims if check_map[claim.claim_id].verdict == "contradicted")
    mixed = sum(1 for claim in checked_claims if check_map[claim.claim_id].verdict == "mixed")
    insufficient = sum(1 for claim in checked_claims if check_map[claim.claim_id].verdict == "insufficient_evidence")

    high_impact = _high_impact_claims(checked_claims)
    high_impact_insufficient = sum(
        1
        for claim in high_impact
        if check_map[claim.claim_id].verdict == "insufficient_evidence"
    )

    externally_checkable = sum(
        1 for claim in claims if claim.claim_type in {"result", "comparison", "factual"}
    )
    checked_count = len(checked_claims)

    contradicted_ratio = _ratio(contradicted, checked_count)
    mixed_ratio = _ratio(mixed, checked_count)
    insufficient_ratio = _ratio(insufficient, checked_count)
    high_impact_insufficient_ratio = _ratio(high_impact_insufficient, len(high_impact))
    evidence_coverage_ratio = _ratio(checked_count, externally_checkable)
    consistency_penalty_ratio = max(0.0, (100 - consistency.score) / 100.0)

    score = 100 * (
        0.40 * contradicted_ratio
        + 0.10 * mixed_ratio
        + 0.20 * insufficient_ratio
        + 0.15 * high_impact_insufficient_ratio
        + 0.10 * (1 - evidence_coverage_ratio)
        + 0.05 * consistency_penalty_ratio
    )
    score = max(0.0, min(100.0, score))

    score_components = {
        "contradicted_component": round(0.40 * contradicted_ratio * 100, 2),
        "mixed_component": round(0.10 * mixed_ratio * 100, 2),
        "insufficient_evidence_component": round(0.20 * insufficient_ratio * 100, 2),
        "high_impact_insufficient_component": round(0.15 * high_impact_insufficient_ratio * 100, 2),
        "coverage_gap_component": round(0.10 * (1 - evidence_coverage_ratio) * 100, 2),
        "consistency_component": round(0.05 * consistency_penalty_ratio * 100, 2),
    }

    explanation = (
        f"Fabrication risk is grounded in {checked_count} evidence-checked claim(s). "
        f"Contradicted ratio: {contradicted_ratio:.0%}; mixed ratio: {mixed_ratio:.0%}; "
        f"insufficient-evidence ratio: {insufficient_ratio:.0%}; "
        f"high-impact insufficient ratio: {high_impact_insufficient_ratio:.0%}; "
        f"evidence coverage: {evidence_coverage_ratio:.0%}; consistency penalty: {consistency_penalty_ratio:.0%}."
    )

    return FabricationBreakdown(
        contradicted_ratio=round(contradicted_ratio, 3),
        mixed_ratio=round(mixed_ratio, 3),
        insufficient_evidence_ratio=round(insufficient_ratio, 3),
        high_impact_insufficient_ratio=round(high_impact_insufficient_ratio, 3),
        evidence_coverage_ratio=round(evidence_coverage_ratio, 3),
        consistency_penalty_ratio=round(consistency_penalty_ratio, 3),
        final_score=round(score, 2),
        score_components=score_components,
        explanation=explanation,
    )
