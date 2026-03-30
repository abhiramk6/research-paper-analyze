from __future__ import annotations

from models.schema import (
    Claim,
    ClaimCheckResult,
    ConsistencyResult,
    FabricationRiskResult,
    FinalReport,
    GrammarResult,
    NoveltyResult,
    PaperDocument,
)


def _recommendation(score: float, consistency_score: int, grammar_rating: str) -> str:
    if score > 60 or consistency_score < 55 or grammar_rating == "Low":
        return "Fail"
    if score > 40 or consistency_score < 65:
        return "Borderline"
    return "Pass"


def _build_summary(
    document: PaperDocument,
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    novelty: NoveltyResult,
    fabrication: FabricationRiskResult,
    recommendation: str,
) -> str:
    checked = len(claim_checks)
    supported = sum(1 for item in claim_checks if item.verdict == "supported")
    contradicted = sum(1 for item in claim_checks if item.verdict == "contradicted")
    unresolved = sum(1 for item in claim_checks if item.verdict == "insufficient_evidence")
    contribution_count = sum(1 for claim in claims if claim.claim_type == "contribution")
    result_count = sum(1 for claim in claims if claim.claim_type in {"result", "comparison"})

    return "\n".join(
        [
            f"This evaluator reviewed `{document.title}` using bounded claim extraction, external evidence retrieval, and grounded scoring.",
            (
                f"It extracted {len(claims)} claims ({contribution_count} contribution, "
                f"{result_count} result/comparison) and externally checked {checked} of them."
            ),
            (
                f"Evidence outcomes: {supported} supported, {contradicted} contradicted, "
                f"{unresolved} insufficient-evidence."
            ),
            (
                f"Consistency scored {consistency.score}/100, grammar was rated {grammar.rating}, "
                f"novelty was rated {novelty.rating}, and fabrication risk was {fabrication.score:.1f}/100."
            ),
            f"Overall recommendation: **{recommendation}**.",
        ]
    )


def build_report(
    document: PaperDocument,
    claims: list[Claim],
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    novelty: NoveltyResult,
    fabrication: FabricationRiskResult,
    claim_checks: list[ClaimCheckResult],
) -> FinalReport:
    recommendation = _recommendation(
        fabrication.score,
        consistency.score,
        grammar.rating,
    )
    summary = _build_summary(
        document,
        claims,
        claim_checks,
        consistency,
        grammar,
        novelty,
        fabrication,
        recommendation,
    )
    return FinalReport(
        title=document.title,
        summary=summary,
        consistency_score=consistency.score,
        grammar_rating=grammar.rating,
        novelty_rating=novelty.rating,
        fabrication_risk=f"{round(fabrication.score)}% risk",
        recommendation=recommendation,
    )
