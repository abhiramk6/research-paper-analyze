from __future__ import annotations

import json

from agents.common import summarize_chunks
from agents.llm_client import call_llm_json
from models.schema import Claim, NoveltyResult, PaperDocument
from prompt_loader import load_prompt


def _novelty_fallback(document: PaperDocument, claims: list[Claim], related_summary: str) -> dict:
    contribution_claims = [claim for claim in claims if claim.claim_type in {"contribution", "novelty"}]
    has_related_work = related_summary != "Related work section not found."
    strong_novelty_language = any(
        keyword in f"{document.title} {document.abstract}".lower()
        for keyword in ("new ", "novel", "first", "we propose", "we introduce")
    )

    if (len(contribution_claims) >= 2 and has_related_work) or (contribution_claims and has_related_work and strong_novelty_language):
        rating = "High"
        reasoning = (
            "Heuristic novelty analysis was used. The paper states multiple concrete contribution claims and includes "
            "related-work context, which is consistent with a strong novelty case."
        )
    elif contribution_claims and has_related_work:
        rating = "Moderate"
        reasoning = (
            "Heuristic novelty analysis was used. The paper makes at least one concrete contribution claim "
            "and includes related-work context, which supports a moderate novelty estimate without overclaiming."
        )
    elif contribution_claims:
        rating = "Moderate"
        reasoning = (
            "Heuristic novelty analysis was used. The paper states concrete contribution claims, but related-work "
            "context was limited, so the novelty estimate is kept moderate rather than high."
        )
    else:
        rating = "Low"
        reasoning = (
            "Heuristic novelty analysis was used. Few explicit contribution claims were detected, so novelty could "
            "not be argued confidently from the parsed text alone."
        )
    return {"rating": rating, "reasoning": reasoning}


def run_novelty_agent(document: PaperDocument, claims: list[Claim], prepared_context: str = "") -> NoveltyResult:
    related_work = prepared_context or "\n\n".join(
        section.content
        for section in document.sections
        if "related work" in section.heading.lower() or "background" in section.heading.lower()
    )
    related_summary = summarize_chunks("related work", related_work) if related_work else "Related work section not found."
    contribution_claims = [claim.model_dump() for claim in claims if claim.claim_type in {"contribution", "novelty"}]
    prompt = load_prompt(
        "novelty_review.txt",
        schema_json='{"rating": "High|Moderate|Low", "reasoning": "string"}',
        title=document.title,
        abstract=document.abstract,
        claims_json=json.dumps(contribution_claims, indent=2),
        related_summary=related_summary,
    )
    payload = call_llm_json(prompt, fallback=_novelty_fallback(document, claims, related_summary))
    rating = str(payload.get("rating", "Moderate"))
    if rating not in {"High", "Moderate", "Low"}:
        rating = "Moderate"
    return NoveltyResult(rating=rating, reasoning=str(payload.get("reasoning", "")))
