from models.schema import Claim, ClaimCheckResult, ConsistencyResult, FabricationRiskResult
from scoring.credibility_model import compute_fabrication_breakdown


def risk_band(score: float) -> str:
    if score < 34:
        return "low"
    if score < 67:
        return "medium"
    return "high"


def run_credibility_agent(
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
) -> FabricationRiskResult:
    external_claim_count = sum(
        1 for claim in claims if claim.claim_type in {"result", "comparison", "factual"}
    )
    breakdown = compute_fabrication_breakdown(
        claims=claims,
        claim_checks=claim_checks,
        consistency=consistency,
    )
    score = breakdown.final_score
    band = risk_band(score)

    risk_factors: list[str] = []
    if breakdown.contradicted_ratio > 0:
        risk_factors.append("At least one evidence-checked claim is contradicted by external evidence.")
    if breakdown.mixed_ratio > 0:
        risk_factors.append("Some evidence-checked claims have mixed support rather than clear confirmation.")
    if breakdown.insufficient_evidence_ratio > 0:
        risk_factors.append("Some checked claims could not be verified with strong external evidence.")
    if breakdown.high_impact_insufficient_ratio > 0:
        risk_factors.append("Some high-importance claims could not be supported with strong external evidence.")
    if external_claim_count and breakdown.evidence_coverage_ratio < 1.0:
        risk_factors.append("Not every externally checkable claim completed evidence checking.")
    if consistency.unresolved_claim_count > consistency.grounded_claim_count:
        risk_factors.append("Consistency is weakened by unresolved or contradictory claim support.")
    if external_claim_count == 0:
        risk_factors.append("No externally checkable factual/result claims were extracted, so fabrication risk is limited.")
    if not risk_factors:
        risk_factors.append("No major grounded fabrication signals were detected in the checked claim set.")

    reasoning = (
        f"Fabrication risk is {score:.1f}/100 ({band} risk). "
        + breakdown.explanation
    )

    return FabricationRiskResult(
        score=score,
        risk_band=band,
        risk_factors=risk_factors,
        reasoning=reasoning,
        breakdown=breakdown,
    )
