from __future__ import annotations

from models.schema import (
    CitationResult,
    Claim,
    ConsistencyResult,
    CredibilityResult,
    FactCheckResult,
    FinalReport,
    GrammarResult,
    NoveltyResult,
    PaperDocument,
)


def _recommendation(score: float) -> str:
    if score > 60:
        return "Fail"
    if score > 30:
        return "Borderline"
    return "Pass"


def _guardrailed_recommendation(
    credibility_score: float,
    consistency_score: int,
    citation_score: int,
    grammar_rating: str,
) -> str:
    base = _recommendation(credibility_score)
    if citation_score < 50 or consistency_score < 55 or grammar_rating == "Low":
        return "Fail"
    if citation_score < 65 or consistency_score < 70 or grammar_rating == "Medium":
        return "Borderline" if base == "Pass" else base
    return base


def _build_summary(
    document: PaperDocument,
    claims: list[Claim],
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    citation: CitationResult,
    factcheck: FactCheckResult,
    novelty: NoveltyResult,
    credibility: CredibilityResult,
    recommendation: str,
) -> str:
    suspicious_count = sum(1 for item in factcheck.items if item.status == "suspicious")
    verified_count = sum(1 for item in factcheck.items if item.status == "verified")
    contribution_claims = [c for c in claims if c.claim_type == "contribution"]
    result_claims = [c for c in claims if c.claim_type == "result"]

    # Score interpretation
    consistency_desc = (
        "strong internal coherence" if consistency.score >= 80
        else "moderate internal coherence" if consistency.score >= 65
        else "weak internal coherence"
    )
    citation_desc = (
        "strong citation grounding" if citation.score >= 80
        else "moderate citation grounding" if citation.score >= 65
        else "weak citation grounding"
    )
    novelty_desc = novelty.rating.lower() + " novelty"
    grammar_desc = grammar.rating.lower() + " writing quality"
    risk_desc = (
        "low fabrication risk" if credibility.score <= 30
        else "moderate fabrication risk" if credibility.score <= 60
        else "high fabrication risk"
    )

    # Fact check summary
    if not factcheck.items:
        factcheck_summary = "No factual claims were eligible for independent fact-checking."
    elif suspicious_count == 0 and verified_count == 0:
        factcheck_summary = f"All {len(factcheck.items)} fact-checked claim(s) were classified as unverifiable from general knowledge alone."
    elif suspicious_count == 0:
        factcheck_summary = (
            f"Fact-checking produced {verified_count} verified and "
            f"{len(factcheck.items) - verified_count} unverifiable claim(s); no suspicious findings."
        )
    else:
        factcheck_summary = (
            f"Fact-checking flagged {suspicious_count} claim(s) as suspicious, "
            f"verified {verified_count}, and found {len(factcheck.items) - suspicious_count - verified_count} unverifiable."
        )

    # Consistency issues summary
    if consistency.issues:
        issues_summary = f"The consistency agent identified {len(consistency.issues)} specific issue(s): {'; '.join(consistency.issues[:2])}{'...' if len(consistency.issues) > 2 else ''}."
    else:
        issues_summary = "No internal consistency issues were flagged."

    # Citation gaps summary
    if citation.citation_gaps:
        gaps_summary = f"The citation agent identified {len(citation.citation_gaps)} gap(s) in prior-work grounding."
    else:
        gaps_summary = "No significant citation gaps were identified."

    lines = [
        f"This paper was evaluated across six dimensions: internal consistency, writing quality, "
        f"citation grounding, fact-checking, novelty, and overall credibility.",
        "",
        f"The system extracted {len(claims)} high-signal claims "
        f"({len(contribution_claims)} contribution, {len(result_claims)} result-type). "
        f"{factcheck_summary}",
        "",
        f"Evaluation signals: {consistency_desc} (score {consistency.score}/100), "
        f"{grammar_desc}, {citation_desc} (score {citation.score}/100), "
        f"{novelty_desc}, and {risk_desc} ({round(credibility.score)}% fabrication probability). "
        f"{issues_summary} {gaps_summary}",
    ]

    if credibility.risk_factors and credibility.risk_factors != ["No major cross-signal risk factors were identified."]:
        top_risk = credibility.risk_factors[0]
        lines.append("")
        lines.append(f"Primary credibility concern: {top_risk}")

    lines.append("")
    lines.append(f"Overall recommendation: **{recommendation}**.")

    return "\n".join(lines)


def build_report(
    document: PaperDocument,
    claims: list[Claim],
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    citation: CitationResult,
    factcheck: FactCheckResult,
    novelty: NoveltyResult,
    credibility: CredibilityResult,
) -> FinalReport:
    recommendation = _guardrailed_recommendation(
        credibility.score,
        consistency.score,
        citation.score,
        grammar.rating,
    )
    summary = _build_summary(
        document, claims, consistency, grammar, citation,
        factcheck, novelty, credibility, recommendation,
    )
    return FinalReport(
        title=document.title,
        summary=summary,
        consistency_score=consistency.score,
        grammar_rating=grammar.rating,
        citation_score=citation.score,
        novelty_rating=novelty.rating,
        credibility_score=credibility.score,
        fabrication_probability=f"{round(credibility.score)}% risk",
        recommendation=recommendation,
    )
