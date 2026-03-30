from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from models.schema import Claim, EvidenceItem, NoveltyResult, PaperDocument


def _fallback(contribution_claims: list[Claim], evidence_items: list[EvidenceItem]) -> dict:
    if not contribution_claims:
        return {
            "rating": "Low",
            "reasoning": "No concrete contribution claims were extracted, so novelty could not be justified.",
            "supporting_evidence_ids": [],
        }
    if not evidence_items:
        return {
            "rating": "Low",
            "reasoning": "No meaningful prior-work evidence was retrieved, so novelty is rated conservatively.",
            "supporting_evidence_ids": [],
        }
    scholarly_hits = [item for item in evidence_items if item.domain_tier == "scholarly"]
    rating = "Moderate" if scholarly_hits else "Low"
    return {
        "rating": rating,
        "reasoning": "Novelty fallback used. Prior-work evidence exists but the structured novelty synthesis could not be completed reliably.",
        "supporting_evidence_ids": [item.evidence_id for item in scholarly_hits[:3]],
    }


def run_novelty_agent(
    document: PaperDocument,
    contribution_claims: list[Claim],
    evidence_items: list[EvidenceItem],
) -> NoveltyResult:
    if not contribution_claims or not evidence_items:
        fallback = _fallback(contribution_claims, evidence_items)
        return NoveltyResult(
            rating=fallback["rating"],
            reasoning=fallback["reasoning"],
            supporting_evidence_ids=fallback["supporting_evidence_ids"],
        )

    schema = {
        "rating": "High|Moderate|Low",
        "reasoning": "string",
        "supporting_evidence_ids": ["string"],
    }
    payload = {
        "title": document.title,
        "abstract": document.abstract[:1200],
        "contribution_claims": [claim.model_dump() for claim in contribution_claims[:5]],
        "prior_work_evidence": [item.model_dump() for item in evidence_items[:8]],
    }
    prompt = (
        "You are judging novelty for a research paper using contribution claims and retrieved prior-work evidence.\n"
        "Be conservative. If the evidence is thin or indirect, do not overstate novelty.\n"
        "High means the contribution appears materially distinct from retrieved prior work.\n"
        "Moderate means there may be differentiation but the evidence is incomplete or mixed.\n"
        "Low means the contribution is weakly differentiated or not well supported.\n"
        f"Return valid JSON matching:\n{json.dumps(schema, indent=2)}\n\n"
        f"Input:\n{json.dumps(payload, indent=2)}"
    )
    response = call_llm_json(prompt, fallback=_fallback(contribution_claims, evidence_items))
    rating = str(response.get("rating", "Low"))
    if rating not in {"High", "Moderate", "Low"}:
        rating = "Low"
    supporting_ids = [str(item) for item in response.get("supporting_evidence_ids", [])]
    return NoveltyResult(
        rating=rating,
        reasoning=str(response.get("reasoning", "")),
        supporting_evidence_ids=supporting_ids,
    )
