import json

from agents.llm_client import call_llm_json_checked
from chunker.token_chunker import GRAMMAR_SAMPLE_TOKENS, truncate_to_token_limit
from models.schema import GrammarResult, PaperDocument


def _sample_text(document: PaperDocument) -> str:
    candidate_parts = [document.abstract]
    for section in document.sections:
        heading = section.heading.lower()
        if any(token in heading for token in ("introduction", "conclusion", "discussion")):
            candidate_parts.append(section.content)
    text = "\n\n".join(part.strip() for part in candidate_parts if part.strip())
    return truncate_to_token_limit(text, GRAMMAR_SAMPLE_TOKENS)


def run_grammar_agent(document: PaperDocument) -> GrammarResult:
    text = _sample_text(document)
    schema = {"rating": "High|Medium|Low", "issues": ["string"], "reasoning": "string"}
    prompt = (
        "You are reviewing academic writing quality.\n"
        "Rate only grammar, clarity, and professional tone.\n"
        "Do not score technical correctness.\n"
        "Ignore PDF extraction artifacts such as repeated headers, page numbers, broken section ordering, or metadata fragments unless they clearly reflect bad writing by the authors.\n"
        "Do not downgrade strong academic prose merely because the extracted text is concatenated awkwardly.\n"
        "Return one JSON object only.\n"
        "Use these exact keys: rating, issues, reasoning.\n"
        "rating must be exactly one of High, Medium, Low.\n"
        "If there are no notable issues, return an empty list for issues.\n"
        f"Return valid JSON matching:\n{json.dumps(schema, indent=2)}\n\n"
        f"Sampled paper text:\n{text}"
    )
    payload = call_llm_json_checked(
        prompt,
        required_fields=["rating", "issues", "reasoning"],
        nonempty_fields=["rating", "reasoning"],
        enum_fields={"rating": {"High", "Medium", "Low"}},
    )
    rating = str(payload.get("rating", "")).strip()
    if rating not in {"High", "Medium", "Low"}:
        raise ValueError(f"Unexpected grammar rating: {rating!r}")
    issues = [str(item) for item in payload.get("issues", []) if str(item).strip()]
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("Grammar agent returned empty reasoning.")
    return GrammarResult(rating=rating, reasoning=reasoning, issues=issues)
