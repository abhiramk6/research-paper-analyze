import json

from agents.llm_client import call_llm_json_checked
from chunker.token_chunker import truncate_to_token_limit
from models.schema import (
    AssessmentSynthesisResult,
    Claim,
    ClaimCheckResult,
    ConsistencyResult,
    FabricationRiskResult,
    GrammarResult,
    NoveltyResult,
    PaperDocument,
)


def _claim_samples(claims: list[Claim], limit: int = 3) -> list[str]:
    return [claim.text for claim in claims[:limit]]


def _check_samples(
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    verdict: str,
    limit: int = 3,
) -> list[str]:
    claim_map = {claim.claim_id: claim for claim in claims}
    samples: list[str] = []
    for item in claim_checks:
        if item.verdict != verdict:
            continue
        claim = claim_map.get(item.claim_id)
        if not claim:
            continue
        samples.append(f"{claim.text} | {item.reasoning}")
        if len(samples) >= limit:
            break
    return samples


def _parser_warnings(document: PaperDocument) -> list[str]:
    warnings: list[str] = []
    if len(document.sections) <= 1:
        warnings.append("Section parsing recovered only one coarse body section.")
    if any(section.heading in {"Front Matter", "Body"} for section in document.sections):
        warnings.append("Some section structure is generic, so parser quality is limited.")
    if not document.references:
        warnings.append("No structured references were parsed from the PDF.")
    return warnings


def _payload(
    document: PaperDocument,
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    novelty: NoveltyResult,
    fabrication: FabricationRiskResult,
) -> dict:
    external_claims = [
        claim for claim in claims if claim.claim_type in {"result", "comparison", "factual"}
    ]
    contribution_claims = [claim for claim in claims if claim.claim_type == "contribution"]
    method_claims = [claim for claim in claims if claim.claim_type == "method"]
    check_map = {item.claim_id: item for item in claim_checks}

    supported = [item for item in claim_checks if item.verdict == "supported"]
    contradicted = [item for item in claim_checks if item.verdict == "contradicted"]
    mixed = [item for item in claim_checks if item.verdict == "mixed"]
    insufficient = [item for item in claim_checks if item.verdict == "insufficient_evidence"]

    unchecked_external = [
        claim for claim in external_claims if claim.claim_id not in check_map
    ]
    high_importance_external = [
        claim for claim in external_claims if claim.importance == "high"
    ]

    return {
        "paper": {
            "title": document.title,
            "source": document.source_url,
            "abstract_summary": truncate_to_token_limit(document.abstract, 500),
            "section_count": len(document.sections),
            "reference_count": len(document.references),
            "parser_warnings": _parser_warnings(document),
        },
        "claim_summary": {
            "total_claims": len(claims),
            "contribution_claims": len(contribution_claims),
            "method_claims": len(method_claims),
            "external_claims": len(external_claims),
            "high_importance_external_claims": len(high_importance_external),
            "unchecked_external_claims": len(unchecked_external),
            "sample_contributions": _claim_samples(contribution_claims),
            "sample_unchecked_external_claims": _claim_samples(unchecked_external),
        },
        "evidence_summary": {
            "checked_claims": len(claim_checks),
            "supported_claims": len(supported),
            "contradicted_claims": len(contradicted),
            "mixed_claims": len(mixed),
            "insufficient_claims": len(insufficient),
            "supported_examples": _check_samples(claims, claim_checks, "supported"),
            "contradicted_examples": _check_samples(claims, claim_checks, "contradicted"),
            "mixed_examples": _check_samples(claims, claim_checks, "mixed"),
            "insufficient_examples": _check_samples(claims, claim_checks, "insufficient_evidence"),
        },
        "baseline_assessments": {
            "consistency_score": consistency.score,
            "consistency_issues": consistency.issues,
            "consistency_reasoning": consistency.reasoning,
            "grammar_rating": grammar.rating,
            "grammar_issues": grammar.issues,
            "grammar_reasoning": grammar.reasoning,
            "novelty_rating": novelty.rating,
            "novelty_reasoning": novelty.reasoning,
            "fabrication_score": fabrication.score,
            "fabrication_band": fabrication.risk_band,
            "fabrication_risk_factors": fabrication.risk_factors,
            "fabrication_reasoning": fabrication.reasoning,
        },
    }


def run_assessment_synthesis_agent(
    document: PaperDocument,
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    novelty: NoveltyResult,
    fabrication: FabricationRiskResult,
) -> AssessmentSynthesisResult:
    schema = {
        "consistency_score": 0,
        "fabrication_score": 0,
        "recommendation": "Pass|Borderline|Fail",
        "summary": "string",
        "key_findings": ["string"],
        "consistency_reasoning": "string",
        "fabrication_reasoning": "string",
    }
    payload = _payload(
        document=document,
        claims=claims,
        claim_checks=claim_checks,
        consistency=consistency,
        grammar=grammar,
        novelty=novelty,
        fabrication=fabrication,
    )
    prompt = (
        "You are calibrating a research-paper evaluation from structured findings only.\n"
        "Do not invent facts beyond the payload.\n"
        "Do not use outside knowledge, paper reputation, citation counts, or field history.\n"
        "Do not call a paper seminal, foundational, famous, influential, or highly cited unless that exact information appears in the payload.\n"
        "Do not punish the paper for parser artifacts or retrieval weakness alone.\n"
        "Contradicted or mixed evidence should drive fabrication risk strongly.\n"
        "Insufficient evidence should be treated as a caution signal, not as direct proof of fabrication.\n"
        "A strong paper with no contradictions and several supported claims should usually remain strong even if some claims lack independent corroboration.\n"
        "Grammar rating should reflect writing quality, not PDF extraction noise.\n"
        "Use the baseline assessments as inputs, but override them when they are clearly too harsh or too lenient.\n"
        "Consistency score is 0-100.\n"
        "Fabrication score is 0-100 where higher means more concern.\n"
        "Recommendation must be one of Pass, Borderline, or Fail.\n"
        "Return one JSON object only.\n"
        "Use these exact keys: consistency_score, fabrication_score, recommendation, summary, key_findings, consistency_reasoning, fabrication_reasoning.\n"
        "Do not omit keys. Use an empty list for key_findings if needed.\n"
        "Return valid JSON matching this schema exactly:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Input findings:\n"
        f"{json.dumps(payload, indent=2)}"
    )
    response = call_llm_json_checked(
        prompt,
        required_fields=[
            "consistency_score",
            "fabrication_score",
            "recommendation",
            "summary",
            "key_findings",
            "consistency_reasoning",
            "fabrication_reasoning",
        ],
        nonempty_fields=[
            "recommendation",
            "summary",
            "consistency_reasoning",
            "fabrication_reasoning",
        ],
        enum_fields={"recommendation": {"Pass", "Borderline", "Fail"}},
    )

    recommendation = str(response.get("recommendation", "")).strip()
    if recommendation not in {"Pass", "Borderline", "Fail"}:
        raise ValueError(f"Unexpected synthesis recommendation: {recommendation!r}")

    consistency_score = int(round(float(response.get("consistency_score"))))
    fabrication_score = round(float(response.get("fabrication_score")), 2)
    summary = str(response.get("summary", "")).strip()
    consistency_reasoning = str(response.get("consistency_reasoning", "")).strip()
    fabrication_reasoning = str(response.get("fabrication_reasoning", "")).strip()
    key_findings = [str(item).strip() for item in response.get("key_findings", []) if str(item).strip()]

    if not summary or not consistency_reasoning or not fabrication_reasoning:
        raise ValueError("Assessment synthesis returned incomplete reasoning.")

    return AssessmentSynthesisResult(
        consistency_score=max(0, min(100, consistency_score)),
        fabrication_score=max(0.0, min(100.0, fabrication_score)),
        recommendation=recommendation,
        summary=summary,
        key_findings=key_findings,
        consistency_reasoning=consistency_reasoning,
        fabrication_reasoning=fabrication_reasoning,
    )
