from __future__ import annotations

import json
import logging

from agents.llm_client import call_llm_json
from chunker.token_chunker import count_tokens
from models.schema import Claim, EvidenceFactCheckItem, EvidenceItem


logger = logging.getLogger(__name__)

VERIFIER_MAX_INPUT_TOKENS = 9_000

EVIDENCE_SNIPPET_CAP = 350
MAX_EVIDENCE_ITEMS = 5

PAPER_CONTEXT_CAP = 1_200


def _trim_evidence(items: list[EvidenceItem]) -> list[dict]:
    trimmed = []
    for item in items[:MAX_EVIDENCE_ITEMS]:
        trimmed.append(
            {
                "evidence_id": item.evidence_id,
                "title": item.title[:100],
                "url": item.url[:200],
                "snippet": item.snippet[:EVIDENCE_SNIPPET_CAP],
                "source_type": item.source_type,
                "retrieval_score": round(item.retrieval_score, 3),
            }
        )
    return trimmed


def _build_verifier_prompt(
    claim: Claim,
    paper_context: str,
    evidence_items: list[EvidenceItem],
) -> str:
    schema = {
        "verdict": "supported|contradicted|mixed|insufficient_evidence|paper_supported_only",
        "confidence": 0.0,
        "reasoning": "string",
        "used_evidence_ids": ["string"],
    }
    payload = {
        "claim": {
            "claim_id": claim.claim_id,
            "text": claim.text[:400],
            "claim_type": claim.claim_type,
            "importance": claim.importance,
        },
        "paper_local_context": paper_context[:PAPER_CONTEXT_CAP],
        "external_evidence": _trim_evidence(evidence_items),
    }
    instructions = (
        "You are a rigorous scientific fact-checker.\n\n"
        "Given a claim extracted from a research paper, paper-local context, and external evidence items, "
        "decide whether the claim is supported, contradicted, mixed, or has insufficient external evidence.\n\n"
        "Rules:\n"
        "- verdict='supported' requires at least one evidence item that corroborates the claim.\n"
        "- verdict='paper_supported_only' when only the paper's own text supports it and no external evidence was found.\n"
        "- verdict='insufficient_evidence' when external evidence exists but does not clearly address the claim.\n"
        "- verdict='contradicted' when evidence directly conflicts with the claim.\n"
        "- verdict='mixed' when evidence both supports and contradicts.\n"
        "- Always list the evidence_ids you actually used in used_evidence_ids.\n"
        "- Be conservative: do not mark 'supported' unless the evidence clearly addresses the specific claim.\n\n"
        f"Respond only with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Input:\n{json.dumps(payload, indent=2)}"
    )
    return instructions


def _validate_verdict(verdict: str) -> str:
    valid = {"supported", "contradicted", "mixed", "insufficient_evidence", "paper_supported_only"}
    return verdict if verdict in valid else "insufficient_evidence"


def verify_claim(
    claim: Claim,
    paper_context: str = "",
    evidence_items: list[EvidenceItem] | None = None,
) -> EvidenceFactCheckItem:
    evidence_items = evidence_items or []

    if not evidence_items:
        return EvidenceFactCheckItem(
            claim_id=claim.claim_id,
            verdict="paper_supported_only",
            confidence=0.3,
            reasoning="No external evidence was retrieved for this claim. Verdict is based on paper context only.",
            used_evidence_ids=[],
            paper_context_excerpt=paper_context[:PAPER_CONTEXT_CAP],
            evidence_items=[],
        )

    prompt = _build_verifier_prompt(claim, paper_context, evidence_items)

    if count_tokens(prompt) > VERIFIER_MAX_INPUT_TOKENS:
        logger.warning(
            "Verifier prompt for %s exceeds %d tokens — truncating evidence.",
            claim.claim_id,
            VERIFIER_MAX_INPUT_TOKENS,
        )
        prompt = _build_verifier_prompt(claim, paper_context, evidence_items[:2])

    fallback = {
        "verdict": "insufficient_evidence",
        "confidence": 0.2,
        "reasoning": "Verifier LLM call failed or returned unparseable output.",
        "used_evidence_ids": [],
    }
    result = call_llm_json(prompt, fallback=fallback)

    verdict = _validate_verdict(str(result.get("verdict", "insufficient_evidence")))
    confidence = float(result.get("confidence") or 0.0)
    reasoning = str(result.get("reasoning", ""))
    used_ids = [str(eid) for eid in result.get("used_evidence_ids", [])]

    used_items = [e for e in evidence_items if e.evidence_id in used_ids] or evidence_items[:2]

    return EvidenceFactCheckItem(
        claim_id=claim.claim_id,
        verdict=verdict,
        confidence=round(confidence, 3),
        reasoning=reasoning,
        used_evidence_ids=used_ids,
        paper_context_excerpt=paper_context[:PAPER_CONTEXT_CAP],
        evidence_items=used_items,
    )
