from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from models.schema import Claim, FactCheckItem, FactCheckResult
from prompt_loader import load_prompt


def _normalize_status(status: str, reasoning: str) -> str:
    status = status.strip().lower()
    if status not in {"verified", "unverifiable", "suspicious"}:
        return "unverifiable"
    if status == "suspicious":
        reasoning_lower = reasoning.lower()
        if any(phrase in reasoning_lower for phrase in ["cannot verify", "not sure", "unclear", "insufficient context"]):
            return "unverifiable"
    return status


def run_factcheck_agent(claims: list[Claim], prepared_context: str = "") -> FactCheckResult:
    target_claims = [
        claim.model_dump()
        for claim in claims
        if claim.claim_type in {"factual", "result"} and claim.confidence >= 0.45
    ]
    if not target_claims:
        return FactCheckResult(items=[])

    prompt = load_prompt(
        "factcheck_review.txt",
        schema_json='{"items": [{"claim_id": "string", "status": "verified|unverifiable|suspicious", "reasoning": "string"}]}',
        claims_json=json.dumps(
            {
                "claims": target_claims,
                "supporting_context": prepared_context[:4000],
            },
            indent=2,
        ),
    )
    payload = call_llm_json(prompt, fallback={"items": []})
    items: list[FactCheckItem] = []
    for raw_item in payload.get("items", []):
        reasoning = str(raw_item.get("reasoning", ""))
        status = _normalize_status(str(raw_item.get("status", "unverifiable")), reasoning)
        items.append(
            FactCheckItem(
                claim_id=str(raw_item.get("claim_id", "")),
                status=status,
                reasoning=reasoning,
            )
        )
    return FactCheckResult(items=items)
