from __future__ import annotations

import json
import logging

from agents.llm_client import call_llm_json
from chunker.token_chunker import VERIFIER_MAX_INPUT_TOKENS, count_tokens, truncate_to_token_limit
from models.schema import Claim, ClaimCheckResult, EvidenceItem


logger = logging.getLogger(__name__)

PAPER_CONTEXT_CAP = 1_200
MAX_EVIDENCE_ITEMS = 5


def _trim_evidence(items: list[EvidenceItem]) -> list[dict]:
    trimmed: list[dict] = []
    for item in items[:MAX_EVIDENCE_ITEMS]:
        trimmed.append(
            {
                "evidence_id": item.evidence_id,
                "query": item.query,
                "title": item.title,
                "url": item.url,
                "domain": item.domain,
                "domain_tier": item.domain_tier,
                "snippet": item.snippet,
                "retrieval_score": round(item.retrieval_score, 3),
            }
        )
    return trimmed


def _build_prompt(claim: Claim, paper_context: str, evidence_items: list[EvidenceItem]) -> str:
    schema = {
        "verdict": "supported|contradicted|mixed|insufficient_evidence",
        "confidence": 0.0,
        "reasoning": "string",
        "matched_evidence_ids": ["string"],
    }
    payload = {
        "claim": {
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type,
            "importance": claim.importance,
            "text": claim.text,
        },
        "paper_context": truncate_to_token_limit(paper_context, PAPER_CONTEXT_CAP),
        "external_evidence": _trim_evidence(evidence_items),
    }
    return (
        "You are verifying a research-paper claim using the paper text and retrieved external evidence.\n\n"
        "Be conservative.\n"
        "- supported: the evidence clearly corroborates the claim\n"
        "- contradicted: the evidence clearly conflicts with the claim\n"
        "- mixed: some evidence supports and some conflicts\n"
        "- insufficient_evidence: evidence is too weak, indirect, or absent\n"
        "- Only cite evidence IDs you actually used.\n"
        f"Return valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Input:\n{json.dumps(payload, indent=2)}"
    )


def verify_claim(
    claim: Claim,
    paper_context: str,
    evidence_items: list[EvidenceItem],
) -> ClaimCheckResult:
    if not evidence_items:
        return ClaimCheckResult(
            claim_id=claim.claim_id,
            verdict="insufficient_evidence",
            confidence=0.2,
            reasoning="No external evidence was retrieved for this claim.",
            matched_evidence_ids=[],
            paper_context_excerpt=truncate_to_token_limit(paper_context, PAPER_CONTEXT_CAP),
            evidence_items=[],
        )

    prompt = _build_prompt(claim, paper_context, evidence_items)
    if count_tokens(prompt) > VERIFIER_MAX_INPUT_TOKENS:
        logger.warning("Verifier prompt exceeded budget for %s; trimming evidence.", claim.claim_id)
        prompt = _build_prompt(claim, paper_context, evidence_items[:2])

    fallback = {
        "verdict": "insufficient_evidence",
        "confidence": 0.2,
        "reasoning": "The verifier could not produce a reliable structured judgement.",
        "matched_evidence_ids": [],
    }
    payload = call_llm_json(prompt, fallback=fallback)
    verdict = str(payload.get("verdict", "insufficient_evidence")).strip().lower()
    if verdict not in {"supported", "contradicted", "mixed", "insufficient_evidence"}:
        verdict = "insufficient_evidence"
    matched_ids = [str(item) for item in payload.get("matched_evidence_ids", [])]
    used_items = [item for item in evidence_items if item.evidence_id in matched_ids] or evidence_items[:2]

    return ClaimCheckResult(
        claim_id=claim.claim_id,
        verdict=verdict,
        confidence=round(float(payload.get("confidence") or 0.0), 3),
        reasoning=str(payload.get("reasoning", "")),
        matched_evidence_ids=matched_ids,
        paper_context_excerpt=truncate_to_token_limit(paper_context, PAPER_CONTEXT_CAP),
        evidence_items=used_items,
    )
