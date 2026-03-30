from __future__ import annotations

from pathlib import Path

from models.schema import (
    Claim,
    ClaimCheckResult,
    ConsistencyResult,
    FabricationRiskResult,
    FinalReport,
    GrammarResult,
    NoveltyResult,
)


def _bullet_list(items: list[str], empty_message: str) -> str:
    if not items:
        return f"- {empty_message}"
    return "\n".join(f"- {item}" for item in items)


def _claim_table(claims: list[Claim], checks: list[ClaimCheckResult]) -> str:
    check_map = {item.claim_id: item for item in checks}
    rows = []
    for claim in claims:
        verdict = check_map.get(claim.claim_id)
        verdict_text = verdict.verdict if verdict else "not checked"
        rows.append(
            f"| `{claim.claim_id}` | {claim.claim_type} | {claim.importance} | "
            f"{claim.source_chunk_id or '—'} | {verdict_text} | {claim.text[:110].replace('|', '/')} |"
        )
    return (
        "| Claim ID | Type | Importance | Chunk | Verdict | Text |\n"
        "|---|---|---|---|---|---|\n"
        + ("\n".join(rows) if rows else "| — | — | — | — | — | No claims extracted |")
    )


def _factcheck_section(checks: list[ClaimCheckResult]) -> str:
    if not checks:
        return "## Fact Check Log\n\n*No claims were routed to external evidence checking.*\n"

    rows = []
    for item in checks:
        evidence_titles = "; ".join(evidence.title[:60] for evidence in item.evidence_items) or "none"
        rows.append(
            f"| `{item.claim_id}` | {item.verdict} | {round(item.confidence * 100)}% | "
            f"{evidence_titles.replace('|', '/')} | {item.reasoning[:120].replace('|', '/')} |"
        )
    return (
        "## Fact Check Log\n\n"
        "| Claim ID | Verdict | Confidence | Evidence | Reasoning |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )


def _fabrication_section(fabrication: FabricationRiskResult) -> str:
    breakdown = fabrication.breakdown
    rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {value:.2f} |"
        for name, value in breakdown.score_components.items()
    )
    return (
        "## Fabrication Risk\n\n"
        f"**Score:** {fabrication.score:.1f}/100\n\n"
        f"**Band:** {fabrication.risk_band.upper()}\n\n"
        f"{fabrication.reasoning}\n\n"
        "### Score Components\n\n"
        "| Component | Points |\n"
        "|---|---|\n"
        f"{rows}\n\n"
        "### Risk Factors\n\n"
        f"{_bullet_list(fabrication.risk_factors, 'No major grounded risk factors were identified.')}\n"
    )


def save_report(
    report: FinalReport,
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    novelty: NoveltyResult,
    fabrication: FabricationRiskResult,
    paper_id: str,
) -> Path:
    output_dir = Path("reports") / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.md"

    content = f"""# Judgement Report: {report.title}

## Executive Summary
{report.summary}

## Scores At A Glance
| Metric | Value |
|---|---|
| Consistency Score | {report.consistency_score}/100 |
| Grammar Rating | {report.grammar_rating} |
| Novelty Rating | {report.novelty_rating} |
| Fabrication Risk | {report.fabrication_risk} |
| Recommendation | {report.recommendation} |

## Claims Extracted
{_claim_table(claims, claim_checks)}

## Consistency Analysis
**Score:** {report.consistency_score}/100

{consistency.reasoning}

**Issues**
{_bullet_list(consistency.issues, 'No major consistency issues were identified.')}

## Grammar And Language
**Rating:** {grammar.rating}

{grammar.reasoning}

**Issues**
{_bullet_list(grammar.issues, 'No major writing issues were identified in the sampled text.')}

## Novelty Assessment
**Rating:** {novelty.rating}

{novelty.reasoning}

{_factcheck_section(claim_checks)}

{_fabrication_section(fabrication)}

## Final Recommendation
**{report.recommendation}**
"""

    output_path.write_text(content, encoding="utf-8")
    return output_path
