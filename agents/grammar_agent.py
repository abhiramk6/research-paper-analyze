from __future__ import annotations

from agents.llm_client import call_llm_json
from chunker.token_chunker import chunk_text
from models.schema import GrammarResult, PaperDocument
from prompt_loader import load_prompt


def _grammar_fallback(text_for_review: str) -> dict:
    long_sentences = sum(1 for sentence in text_for_review.split(".") if len(sentence.split()) > 35)
    noisy_markers = sum(
        1
        for marker in (
            "keywords:",
            "dataset which are",
            "shown a significant improvement",
            "this thesis proposes",
            "domain specific",
        )
        if marker in text_for_review.lower()
    )
    if noisy_markers >= 2:
        rating = "Low"
    elif noisy_markers >= 1 or long_sentences > 4:
        rating = "Medium"
    else:
        rating = "High"
    reasoning = (
        "Heuristic grammar analysis was used. The sampled prose appears "
        f"{'generally polished and publication-grade' if rating == 'High' else 'readable but somewhat uneven' if rating == 'Medium' else 'noticeably uneven and below publication-grade in places'}, "
        "with no obvious signal of severe writing issues in the extracted text."
    )
    return {"rating": rating, "reasoning": reasoning}


def run_grammar_agent(document: PaperDocument, prepared_context: str = "") -> GrammarResult:
    combined_text = prepared_context.strip() or "\n\n".join(
        [
            document.abstract,
            *[
                section.content
                for section in document.sections
                if any(name in section.heading.lower() for name in ("introduction", "conclusion"))
            ],
        ]
    ).strip()

    samples = chunk_text(combined_text)
    text_for_review = "\n\n".join(samples[:2])
    prompt = load_prompt(
        "grammar_review.txt",
        schema_json='{"rating": "High|Medium|Low", "reasoning": "string"}',
        paper_text=text_for_review,
    )
    payload = call_llm_json(prompt, fallback=_grammar_fallback(text_for_review))
    rating = str(payload.get("rating", "Medium"))
    if rating not in {"High", "Medium", "Low"}:
        rating = "Medium"
    return GrammarResult(rating=rating, reasoning=str(payload.get("reasoning", "")))
