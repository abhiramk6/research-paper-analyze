from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from models.schema import (
    CitationResult,
    ConsistencyResult,
    CredibilityResult,
    FactCheckResult,
    GrammarResult,
    NoveltyResult,
)
from prompt_loader import load_prompt


def compute_credibility_score(
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    citation: CitationResult,
    factcheck: FactCheckResult,
    novelty: NoveltyResult,
) -> tuple[float, str]:
    novelty_risk = {"High": 5, "Moderate": 12, "Low": 28}[novelty.rating]
    suspicious_count = sum(1 for item in factcheck.items if item.status == "suspicious")
    suspicious_ratio = suspicious_count / max(len(factcheck.items), 1)
    fact_risk = suspicious_ratio * 35
    consistency_penalty = max(0, 85 - consistency.score) * 1.0
    citation_penalty = max(0, 80 - citation.score) * 1.1
    grammar_adjustment = {"High": -3, "Medium": 4, "Low": 12}[grammar.rating]
    hard_penalty = 0.0
    if citation.score < 55:
        hard_penalty += 18
    if consistency.score < 65:
        hard_penalty += 14
    if grammar.rating == "Low":
        hard_penalty += 10
    if not factcheck.items:
        hard_penalty += 6
    credibility_score = max(
        0.0,
        min(100.0, consistency_penalty + citation_penalty + novelty_risk + fact_risk + grammar_adjustment + hard_penalty),
    )
    return credibility_score, f"{round(credibility_score)}% risk"


def _local_risk_factors(
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    citation: CitationResult,
    factcheck: FactCheckResult,
    novelty: NoveltyResult,
) -> list[str]:
    factors: list[str] = []
    if consistency.score < 70:
        factors.append("Internal consistency appears mixed rather than clearly strong.")
    if citation.score < 65:
        factors.append("Citation support is weak enough that important claims may not be adequately grounded.")
    if any(item.status == "suspicious" for item in factcheck.items):
        factors.append("At least one extracted claim was marked suspicious during fact checking.")
    if not factcheck.items:
        factors.append("No fact-check confirmations were produced, which leaves core factual claims less supported.")
    if novelty.rating == "Low":
        factors.append("The novelty case appears weak relative to the parsed related-work context.")
    if grammar.rating in {"Medium", "Low"}:
        factors.append("Writing quality may make the technical claims harder to evaluate confidently.")
    return factors or ["No major cross-signal risk factors were identified."]


def run_credibility_agent(
    consistency: ConsistencyResult,
    grammar: GrammarResult,
    citation: CitationResult,
    factcheck: FactCheckResult,
    novelty: NoveltyResult,
    critic_feedback: dict | None = None,
) -> CredibilityResult:
    score, fallback_probability = compute_credibility_score(consistency, grammar, citation, factcheck, novelty)
    fallback_factors = _local_risk_factors(consistency, grammar, citation, factcheck, novelty)
    critic_feedback = critic_feedback or {}
    prompt = load_prompt(
        "credibility_review.txt",
        schema_json='{"risk_factors": ["string"], "reasoning": "string"}',
        signals_json=json.dumps(
            {
                "consistency": consistency.model_dump(),
                "grammar": grammar.model_dump(),
                "citation": citation.model_dump(),
                "factcheck": factcheck.model_dump(),
                "novelty": novelty.model_dump(),
                "critic": critic_feedback,
            },
            indent=2,
        ),
    )
    payload = call_llm_json(
        prompt,
        fallback={
            "risk_factors": fallback_factors,
            "reasoning": (
                "Heuristic credibility analysis was used. The overall risk estimate was derived from the "
                "consistency, citation, novelty, fact-check, and grammar signals already produced in the pipeline."
            ),
        },
    )
    return CredibilityResult(
        score=round(score, 2),
        risk_factors=[str(factor) for factor in payload.get("risk_factors", [])],
        reasoning=str(payload.get("reasoning", fallback_probability)),
    )
