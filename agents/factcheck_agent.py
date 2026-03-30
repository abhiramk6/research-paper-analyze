from __future__ import annotations

import json
import logging

from agents.llm_client import call_llm_json
from models.schema import (
    Claim,
    EvidenceFactCheckItem,
    FactCheckItem,
    FactCheckResult,
)
from prompt_loader import load_prompt


logger = logging.getLogger(__name__)

# Legacy claim types eligible for internal fact-check (unchanged from v1).
_INTERNAL_CHECK_TYPES = frozenset(["factual", "result"])

# New v2 claim types eligible for internal fact-check when external evidence unavailable.
_INTERNAL_CHECK_TYPES_V2 = frozenset(
    ["benchmark_result", "prior_work_comparison", "factual_background"]
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
    """A claim is eligible for internal fact-check when type + confidence meet the bar."""
    type_ok = claim.claim_type in _INTERNAL_CHECK_TYPES or claim.claim_type in _INTERNAL_CHECK_TYPES_V2
    return type_ok and claim.confidence >= 0.45


def run_factcheck_agent(claims: list[Claim], prepared_context: str = "") -> FactCheckResult:
    """
    Internal fact-check agent (v1 interface, preserved for backward compatibility).

    Runs a single LLM call against eligible claims using general domain knowledge
    plus the bounded paper context.  Returns legacy FactCheckResult.

    Token budget: prepared_context is capped at 4 000 chars before being embedded
    in the prompt, keeping the total input well under the 16k architectural limit.
    """
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
    """
    Convert evidence-backed EvidenceFactCheckItem list to legacy FactCheckResult.

    Mapping:
      supported              → verified
      contradicted           → suspicious
      mixed                  → suspicious
      insufficient_evidence  → unverifiable
      paper_supported_only   → unverifiable
    """
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
