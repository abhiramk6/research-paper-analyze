from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from models.schema import (
    Claim,
    EvidenceFactCheckItem,
    FactCheckItem,
    FactCheckResult,
)
from prompt_loader import load_prompt

_INTERNAL_CHECK_TYPES = frozenset(
    {
        "factual",
        "result",
        "benchmark_result",
        "prior_work_comparison",
        "factual_background",
    }
)


def _normalize_status(status: str, reasoning: str) -> str:
    status = status.strip().lower()
    if status not in {"verified", "unverifiable", "suspicious"}:
        return "unverifiable"
    if status == "suspicious":
        reasoning_lower = reasoning.lower()
        if any(
            phrase in reasoning_lower
            for phrase in ["cannot verify", "not sure", "unclear", "insufficient context"]
        ):
            return "unverifiable"
    return status


def _is_eligible(claim: Claim) -> bool:
    return claim.claim_type in _INTERNAL_CHECK_TYPES and claim.confidence >= 0.45


def run_factcheck_agent(claims: list[Claim], prepared_context: str = "") -> FactCheckResult:
    target_claims = [claim.model_dump() for claim in claims if _is_eligible(claim)]
    if not target_claims:
        return FactCheckResult(items=[])

    prompt = load_prompt(
        "factcheck_review.txt",
        schema_json=json.dumps(
            {
                "items": [
                    {
                        "claim_id": "string",
                        "status": "verified|unverifiable|suspicious",
                        "reasoning": "string",
                    }
                ]
            }
        ),
        claims_json=json.dumps(
            {
                "claims": target_claims,
                "supporting_context": prepared_context[:4_000],
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


def evidence_factcheck_to_legacy(
    evidence_results: list[EvidenceFactCheckItem],
) -> FactCheckResult:
    verdict_map = {
        "supported": "verified",
        "contradicted": "suspicious",
        "mixed": "suspicious",
        "insufficient_evidence": "unverifiable",
        "paper_supported_only": "unverifiable",
    }
    items = [
        FactCheckItem(
            claim_id=r.claim_id,
            status=verdict_map.get(r.verdict, "unverifiable"),
            reasoning=r.reasoning,
        )
        for r in evidence_results
    ]
    return FactCheckResult(items=items)
