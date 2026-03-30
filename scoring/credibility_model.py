from models.schema import Claim, ClaimCheckResult, ConsistencyResult, FabricationBreakdown


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _claim_weight(claim: Claim) -> int:
    if claim.importance == "high":
        return 3
    if claim.importance == "medium":
        return 2
    return 1


def compute_fabrication_breakdown(
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
) -> FabricationBreakdown:
    check_map = {item.claim_id: item for item in claim_checks}
    external_claims = [
        claim for claim in claims if claim.claim_type in {"result", "comparison", "factual"}
    ]
    checked_claims = [claim for claim in external_claims if claim.claim_id in check_map]

    contradicted = sum(1 for claim in checked_claims if check_map[claim.claim_id].verdict == "contradicted")
    mixed = sum(1 for claim in checked_claims if check_map[claim.claim_id].verdict == "mixed")
    insufficient = sum(1 for claim in checked_claims if check_map[claim.claim_id].verdict == "insufficient_evidence")

    high_impact = [claim for claim in external_claims if claim.importance == "high"]
    high_impact_insufficient = sum(
        1
        for claim in high_impact
        if claim.claim_id not in check_map or check_map[claim.claim_id].verdict == "insufficient_evidence"
    )

    checked_count = len(checked_claims)
    externally_checkable = len(external_claims)
    contradicted_ratio = _ratio(contradicted, checked_count)
    mixed_ratio = _ratio(mixed, checked_count)
    insufficient_ratio = _ratio(insufficient, checked_count)
    high_impact_insufficient_ratio = _ratio(high_impact_insufficient, len(high_impact))
    evidence_coverage_ratio = 1.0 if externally_checkable == 0 else _ratio(checked_count, externally_checkable)

    total_weight = sum(_claim_weight(claim) for claim in external_claims)
    contradicted_weight = sum(
        _claim_weight(claim)
        for claim in checked_claims
        if check_map[claim.claim_id].verdict == "contradicted"
    )
    mixed_weight = sum(
        _claim_weight(claim) * 0.5
        for claim in checked_claims
        if check_map[claim.claim_id].verdict == "mixed"
    )
    insufficient_weight = sum(
        _claim_weight(claim)
        for claim in checked_claims
        if check_map[claim.claim_id].verdict == "insufficient_evidence"
    )
    unchecked_weight = sum(
        _claim_weight(claim)
        for claim in external_claims
        if claim.claim_id not in check_map
    )
    adverse_weight = contradicted_weight + mixed_weight + insufficient_weight + unchecked_weight
    score = round(100 * adverse_weight / total_weight, 2) if total_weight else 0.0

    score_components = {
        "contradicted_claim_weight_percent": round(100 * contradicted_weight / total_weight, 2) if total_weight else 0.0,
        "mixed_claim_weight_percent": round(100 * mixed_weight / total_weight, 2) if total_weight else 0.0,
        "insufficient_claim_weight_percent": round(100 * insufficient_weight / total_weight, 2) if total_weight else 0.0,
        "unchecked_claim_weight_percent": round(100 * unchecked_weight / total_weight, 2) if total_weight else 0.0,
    }

    if externally_checkable == 0:
        explanation = (
            "Fabrication risk is based on externally checkable result/comparison/factual claims. "
            "No such claims were extracted, so this score is not very informative for this paper. "
            f"Consistency is reported separately at {consistency.score}/100."
        )
    else:
        explanation = (
            f"Fabrication risk is the share of externally checkable claim weight that remains adverse after verification. "
            f"{checked_count} of {externally_checkable} external claim(s) were checked. "
            f"Contradicted ratio: {contradicted_ratio:.0%}; mixed ratio: {mixed_ratio:.0%}; "
            f"insufficient-evidence ratio: {insufficient_ratio:.0%}; "
            f"high-impact insufficient ratio: {high_impact_insufficient_ratio:.0%}; "
            f"evidence coverage: {evidence_coverage_ratio:.0%}. "
            f"Consistency is reported separately at {consistency.score}/100."
        )

    return FabricationBreakdown(
        contradicted_ratio=round(contradicted_ratio, 3),
        mixed_ratio=round(mixed_ratio, 3),
        insufficient_evidence_ratio=round(insufficient_ratio, 3),
        high_impact_insufficient_ratio=round(high_impact_insufficient_ratio, 3),
        evidence_coverage_ratio=round(evidence_coverage_ratio, 3),
        consistency_penalty_ratio=round(max(0.0, (100 - consistency.score) / 100.0), 3),
        final_score=round(score, 2),
        score_components=score_components,
        explanation=explanation,
    )
