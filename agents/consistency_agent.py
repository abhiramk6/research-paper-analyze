from __future__ import annotations

import json

from agents.common import summarize_chunks
from agents.llm_client import call_llm_json
from models.schema import Claim, ConsistencyResult, PaperDocument
from prompt_loader import load_prompt


METHOD_SECTION_KEYWORDS = (
    "method",
    "methodology",
    "architecture",
    "approach",
    "model",
    "encoder",
    "decoder",
    "attention",
    "training",
)

RESULT_SECTION_KEYWORDS = (
    "result",
    "experiment",
    "evaluation",
    "analysis",
    "benchmark",
)


def _consistency_fallback(document: PaperDocument, claims: list[Claim]) -> dict:
    has_method = any(any(keyword in section.heading.lower() for keyword in METHOD_SECTION_KEYWORDS) for section in document.sections)
    has_results = any(any(keyword in section.heading.lower() for keyword in RESULT_SECTION_KEYWORDS) for section in document.sections)
    result_claims = [claim for claim in claims if claim.claim_type == "result"]
    noisy_claims = [
        claim
        for claim in claims
        if any(marker in claim.text.lower() for marker in ("keywords:", "taxonomy", "dataset which are", "shown a significant improvement"))
        or len(claim.text.split()) < 10
    ]
    issues: list[str] = []
    score = 72

    if has_method:
        score += 8
    else:
        issues.append("Method section was not clearly detected, so the evidence trail is harder to judge.")
    if has_results:
        score += 8
    else:
        issues.append("Results section was not clearly detected, so empirical support is harder to judge.")
    if len(result_claims) >= 2:
        score += 6
    elif not result_claims:
        issues.append("No explicit result claims were extracted, which limits consistency checking.")
        score -= 8
    if noisy_claims:
        issues.append("Some extracted claims look noisy or weakly formed, which reduces confidence in the evidence chain.")
        score -= min(18, 6 * len(noisy_claims))

    score = max(45, min(92, score))
    reasoning = (
        "Heuristic consistency analysis was used. The paper structure shows "
        f"{'a detected method section' if has_method else 'no clear method section'} and "
        f"{'a detected results section' if has_results else 'no clear results section'}, "
        f"with {len(result_claims)} extracted result claims available for cross-checking and "
        f"{len(noisy_claims)} potentially noisy extracted claims."
    )
    return {"score": score, "issues": issues, "reasoning": reasoning}


def run_consistency_agent(document: PaperDocument, claims: list[Claim], prepared_context: str = "") -> ConsistencyResult:
    if prepared_context:
        method_text = prepared_context
        results_text = prepared_context
    else:
        method_text = "\n\n".join(
            section.content
            for section in document.sections
            if any(keyword in section.heading.lower() for keyword in METHOD_SECTION_KEYWORDS)
        )
        results_text = "\n\n".join(
            section.content
            for section in document.sections
            if any(keyword in section.heading.lower() for keyword in RESULT_SECTION_KEYWORDS)
        )
    contribution_claims = [claim.model_dump() for claim in claims if claim.claim_type in {"contribution", "result"}]
    method_summary = summarize_chunks("method", method_text) if method_text else "Method section not found."
    results_summary = summarize_chunks("results", results_text) if results_text else "Results section not found."

    prompt = load_prompt(
        "consistency_review.txt",
        schema_json='{"score": 0, "issues": ["string"], "reasoning": "string"}',
        method_summary=method_summary,
        results_summary=results_summary,
        claims_json=json.dumps(contribution_claims, indent=2),
    )
    payload = call_llm_json(prompt, fallback=_consistency_fallback(document, claims))
    return ConsistencyResult(
        score=max(0, min(100, int(payload.get("score", 72)))),
        issues=[str(issue) for issue in payload.get("issues", [])],
        reasoning=str(payload.get("reasoning", "")),
    )
