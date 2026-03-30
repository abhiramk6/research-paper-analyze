import json

from agents.llm_client import call_llm_json_checked
from models.schema import Claim, EvidenceItem, NoveltyResult, PaperDocument


def run_novelty_agent(
    document: PaperDocument,
    contribution_claims: list[Claim],
    evidence_items: list[EvidenceItem],
) -> NoveltyResult:
    if not contribution_claims:
        raise ValueError("Novelty evaluation requires at least one contribution claim.")
    if not evidence_items:
        raise ValueError("Novelty evaluation requires retrieved prior-work evidence.")

    schema = {
        "rating": "High|Moderate|Low",
        "reasoning": "string",
        "supporting_evidence_ids": ["string"],
    }
    payload = {
        "title": document.title,
        "abstract": document.abstract[:1200],
        "contribution_claims": [claim.model_dump() for claim in contribution_claims[:5]],
        "prior_work_evidence": [item.model_dump() for item in evidence_items[:8]],
    }
    prompt = (
        "You are judging novelty for a research paper using contribution claims and retrieved prior-work evidence.\n"
        "Be conservative. If the evidence is thin or indirect, do not overstate novelty.\n"
        "High means the contribution appears materially distinct from retrieved prior work.\n"
        "Moderate means there may be differentiation but the evidence is incomplete or mixed.\n"
        "Low means the contribution is weakly differentiated or not well supported.\n"
        "Return one JSON object only.\n"
        "Use these exact keys: rating, reasoning, supporting_evidence_ids.\n"
        "rating must be exactly one of High, Moderate, Low.\n"
        "If no evidence IDs are directly used, return an empty list for supporting_evidence_ids.\n"
        f"Return valid JSON matching:\n{json.dumps(schema, indent=2)}\n\n"
        f"Input:\n{json.dumps(payload, indent=2)}"
    )
    response = call_llm_json_checked(
        prompt,
        required_fields=["rating", "reasoning", "supporting_evidence_ids"],
        nonempty_fields=["rating", "reasoning"],
        enum_fields={"rating": {"High", "Moderate", "Low"}},
    )
    rating = str(response.get("rating", "")).strip()
    if rating not in {"High", "Moderate", "Low"}:
        raise ValueError(f"Unexpected novelty rating: {rating!r}")
    reasoning = str(response.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("Novelty agent returned empty reasoning.")
    supporting_ids = [str(item) for item in response.get("supporting_evidence_ids", [])]
    return NoveltyResult(
        rating=rating,
        reasoning=reasoning,
        supporting_evidence_ids=supporting_ids,
    )
