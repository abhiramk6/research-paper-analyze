from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from chunker.token_chunker import GRAMMAR_SAMPLE_TOKENS, count_tokens, truncate_to_token_limit
from models.schema import GrammarResult, PaperDocument


def _sample_text(document: PaperDocument) -> str:
    candidate_parts = [document.abstract]
    for section in document.sections:
        heading = section.heading.lower()
        if any(token in heading for token in ("introduction", "conclusion", "discussion")):
            candidate_parts.append(section.content)
    text = "\n\n".join(part.strip() for part in candidate_parts if part.strip())
    return truncate_to_token_limit(text, GRAMMAR_SAMPLE_TOKENS)


def _fallback(text: str) -> dict:
    long_sentences = sum(1 for part in text.split(".") if len(part.split()) > 35)
    issues: list[str] = []
    rating = "High"
    if long_sentences >= 5:
        rating = "Medium"
        issues.append("Several sentences are long enough to reduce clarity.")
    if "et al" not in text and text.count(",") > 30 and long_sentences >= 8:
        rating = "Low"
        issues.append("The sampled prose appears difficult to parse cleanly.")
    return {
        "rating": rating,
        "issues": issues,
        "reasoning": "Grammar rating was produced from conservative local checks because structured LLM output was unavailable.",
    }


def run_grammar_agent(document: PaperDocument) -> GrammarResult:
    text = _sample_text(document)
    schema = {"rating": "High|Medium|Low", "issues": ["string"], "reasoning": "string"}
    prompt = (
        "You are reviewing academic writing quality.\n"
        "Rate only grammar, clarity, and professional tone.\n"
        "Do not score technical correctness.\n"
        f"Return valid JSON matching:\n{json.dumps(schema, indent=2)}\n\n"
        f"Sampled paper text:\n{text}"
    )
    payload = call_llm_json(prompt, fallback=_fallback(text))
    rating = str(payload.get("rating", "Medium"))
    if rating not in {"High", "Medium", "Low"}:
        rating = "Medium"
    issues = [str(item) for item in payload.get("issues", []) if str(item).strip()]
    return GrammarResult(
        rating=rating,
        reasoning=str(payload.get("reasoning", "")),
        issues=issues,
    )
