from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from models.schema import CitationResult, Claim, ReferenceItem
from prompt_loader import load_prompt


def _baseline_citation_score(claims: list[Claim], references: list[ReferenceItem]) -> int:
    if not claims:
        return 85 if references else 70

    external_claims = [claim for claim in claims if claim.claim_type in {"background", "factual", "novelty"}]
    uncited_external = [claim for claim in external_claims if not claim.cited_refs]
    gap_ratio = len(uncited_external) / max(len(external_claims), 1)

    if not references:
        return 45 if external_claims else 65
    if len(references) >= 25:
        if gap_ratio <= 0.35:
            return 92
        return 84
    if len(references) >= 12:
        if gap_ratio <= 0.35:
            return 86
        return 78
    if gap_ratio <= 0.15:
        return 92
    if gap_ratio <= 0.35:
        return 84
    return 72


def run_citation_agent(claims: list[Claim], references: list[ReferenceItem]) -> CitationResult:
    payload_claims = [claim.model_dump() for claim in claims]
    payload_references = [reference.model_dump() for reference in references]
    baseline_score = _baseline_citation_score(claims, references)
    prompt = load_prompt(
        "citation_review.txt",
        schema_json='{"score": 0, "citation_gaps": ["string"], "reasoning": "string"}',
        claims_json=json.dumps(payload_claims, indent=2),
        references_json=json.dumps(payload_references, indent=2),
    )
    payload = call_llm_json(prompt, fallback={"score": baseline_score, "citation_gaps": [], "reasoning": "Fallback citation analysis used because the LLM response was invalid."})
    model_score = max(0, min(100, int(payload.get("score", baseline_score))))
    calibrated_score = max(baseline_score, model_score - 5)
    return CitationResult(
        score=calibrated_score,
        citation_gaps=[str(gap) for gap in payload.get("citation_gaps", [])],
        reasoning=str(payload.get("reasoning", "")),
    )
