from __future__ import annotations

from models.schema import Claim, ClaimCheckResult, ConsistencyResult, FabricationRiskResult
from scoring.credibility_model import compute_fabrication_breakdown


def _risk_band(score: float) -> str:
    if score <= 30:
        return "low"
    if score <= 60:
        return "medium"
    return "high"


def run_credibility_agent(
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
) -> FabricationRiskResult:
    breakdown = compute_fabrication_breakdown(
        claims=claims,
        claim_checks=claim_checks,
        consistency=consistency,
    )
    score = breakdown.final_score
    band = _risk_band(score)

    risk_factors: list[str] = []
    if breakdown.contradicted_ratio > 0:
        risk_factors.append("At least one evidence-checked claim is contradicted by external evidence.")
    if breakdown.insufficient_evidence_ratio >= 0.5:
        risk_factors.append("A large share of checked claims could not be verified with strong external evidence.")
    if breakdown.high_impact_insufficient_ratio > 0:
        risk_factors.append("Some high-importance claims could not be supported with strong external evidence.")
    if breakdown.evidence_coverage_ratio < 0.5:
        risk_factors.append("Less than half of externally checkable claims received usable evidence coverage.")
    if consistency.score < 70:
        risk_factors.append("Consistency is weakened by unresolved or contradictory claim support.")
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
