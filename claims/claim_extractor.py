from __future__ import annotations

import json
import re

from agents.llm_client import call_llm_json
from chunker.token_chunker import CLAIM_CHUNK_TOKENS, build_all_chunks, count_tokens
from models.schema import Claim, PaperChunk, PaperDocument, PaperSection


CLAIM_SECTIONS = (
    "abstract", "introduction", "method", "methodology",
    "results", "conclusion", "related work", "background",
    "experiments", "evaluation", "analysis",
)

_VALID_CLAIM_TYPES = frozenset(
    {
        "benchmark_result",
        "prior_work_comparison",
        "factual_background",
        "methodology_assertion",
        "contribution_claim",
        "unsupported_general_statement",
        "contribution",
        "result",
        "novelty",
        "factual",
        "background",
    }
)

_IMPORTANCE_TYPES = frozenset(["benchmark_result", "result", "prior_work_comparison", "novelty"])
_MEDIUM_IMPORTANCE_TYPES = frozenset(
    ["factual_background", "factual", "contribution_claim", "contribution", "methodology_assertion"]
)


def _section_is_claim_relevant(section: PaperSection) -> bool:
    heading = section.heading.lower()
    return any(name in heading for name in CLAIM_SECTIONS)


def _extract_citations(text: str) -> list[str]:
    citations = re.findall(r"\[(.*?)\]", text)
    extracted: list[str] = []
    for citation in citations:
        extracted.extend([part.strip() for part in citation.split(",") if part.strip()])
    return extracted


def _normalize_claim_type(value: str) -> str:
    v = value.strip().lower().replace(" ", "_")
    if v in _VALID_CLAIM_TYPES:
        return v
    if "benchmark" in v or "metric" in v:
        return "benchmark_result"
    if "prior" in v or "comparison" in v or "baseline" in v:
        return "prior_work_comparison"
    if "method" in v or "approach" in v or "architecture" in v:
        return "methodology_assertion"
    if "contribution" in v or "propose" in v:
        return "contribution_claim"
    if "factual" in v or "background" in v:
        return "factual_background"
    return "unsupported_general_statement"


def _infer_importance(claim_type: str, confidence: float) -> str:
    if claim_type in _IMPORTANCE_TYPES or confidence >= 0.8:
        return "high"
    if claim_type in _MEDIUM_IMPORTANCE_TYPES or confidence >= 0.6:
        return "medium"
    return "low"


def _build_chunk_prompt(chunk: PaperChunk) -> str:
    schema = {
        "claims": [
            {
                "claim_id": "string",
                "text": "string",
                "claim_type": (
                    "benchmark_result|prior_work_comparison|factual_background|"
                    "methodology_assertion|contribution_claim|unsupported_general_statement"
                ),
                "source_section": chunk.section_name,
                "source_chunk_id": chunk.chunk_id,
                "nearby_citations": ["string"],
                "importance": "high|medium|low",
                "confidence": 0.0,
            }
        ]
    }
    instructions = (
        "You are a precise scientific claim extractor.\n\n"
        "Extract the most important, verifiable claims from the section below. "
        "Assign each claim the correct type:\n"
        "  benchmark_result     — numerical results, metric scores, performance comparisons\n"
        "  prior_work_comparison — comparisons to other methods, baselines, prior papers\n"
        "  factual_background   — external facts, dataset descriptions, field-wide stats\n"
        "  methodology_assertion — claims about how the proposed method works\n"
        "  contribution_claim   — top-level contribution assertions ('we propose', 'we introduce')\n"
        "  unsupported_general_statement — vague or unverifiable assertions\n\n"
        "Rules:\n"
        "- Do NOT paraphrase — use close to the original text.\n"
        "- Capture nearby citations as nearby_citations (e.g. ['12', '15']).\n"
        "- Set importance='high' for numerical results and external comparisons.\n"
        "- Set confidence in [0, 1] for how confidently this is a real extractable claim.\n"
        "- Skip sub-8-word fragments, headers, and figure captions.\n"
        f"- Respond only with valid JSON matching:\n{json.dumps(schema, indent=2)}\n\n"
        f"Section: [{chunk.section_name}]  Chunk: {chunk.chunk_id}\n\n"
        f"{chunk.content}"
    )
    return instructions


def _infer_claim_type_fallback(sentence: str, source_section: str) -> str:
    s = sentence.lower()
    sec = source_section.lower()
    if any(k in s for k in ("we propose", "our approach", "introduce", "present")):
        return "contribution_claim"
    if any(k in s for k in ("outperform", "achieve", "improve", "accuracy", "score", "%", "bleu", "rouge")):
        return "benchmark_result"
    if any(k in s for k in ("novel", "first", "unlike prior", "compared to")):
        return "prior_work_comparison"
    if any(k in s for k in ("method", "architecture", "layer", "trained")):
        return "methodology_assertion"
    if "result" in sec or "conclusion" in sec:
        return "benchmark_result"
    return "factual_background"


def _local_claim_fallback(document: PaperDocument, starting_counter: int = 1) -> list[Claim]:
    candidate_sections = [s for s in document.sections if _section_is_claim_relevant(s)]
    fallback_claims: list[Claim] = []

    for section in candidate_sections:
        sentences = re.split(r"(?<=[.!?])\s+", section.content)
        for sentence in sentences:
            clean = re.sub(r"\s+", " ", sentence).strip()
            if len(clean.split()) < 8:
                continue
            if not any(
                keyword in clean.lower()
                for keyword in ("we ", "our ", "show", "demonstrate", "achieve", "improve",
                                "propose", "introduce", "outperform", "%", "bleu", "rouge")
            ):
                continue
            ctype = _infer_claim_type_fallback(clean, section.heading)
            fallback_claims.append(
                Claim(
                    claim_id=f"claim_{starting_counter}",
                    text=clean[:400],
                    claim_type=ctype,
                    source_section=section.heading,
                    source_chunk_id=None,
                    importance=_infer_importance(ctype, 0.45),
                    nearby_citations=_extract_citations(clean),
                    cited_refs=_extract_citations(clean),
                    confidence=0.45,
                )
            )
            starting_counter += 1
            if len(fallback_claims) >= 6:
                return fallback_claims
    return fallback_claims


def _relevant_chunks(document: PaperDocument) -> list[PaperChunk]:
    if document.chunks:
        return [
            chunk for chunk in document.chunks
            if any(name in chunk.section_name.lower() for name in CLAIM_SECTIONS)
        ]
    all_chunks = build_all_chunks(document, max_tokens=CLAIM_CHUNK_TOKENS)
    return [
        chunk for chunk in all_chunks
        if any(name in chunk.section_name.lower() for name in CLAIM_SECTIONS)
    ]


def extract_claims(document: PaperDocument) -> list[Claim]:
    claims: list[Claim] = []
    counter = 1
    seen_texts: set[str] = set()

    for chunk in _relevant_chunks(document):
        prompt = _build_chunk_prompt(chunk)
        payload = call_llm_json(prompt, fallback={"claims": []})
        raw_claims = payload.get("claims", [])

        for item in raw_claims:
            claim_text = str(item.get("text", "")).strip()
            if not claim_text:
                continue
            normalized_text = re.sub(r"\s+", " ", claim_text).strip().lower()
            if normalized_text in seen_texts:
                continue

            claim_type = _normalize_claim_type(str(item.get("claim_type", "unsupported_general_statement")))
            confidence = float(item.get("confidence") or 0.0)

            if claim_type == "unsupported_general_statement" and confidence < 0.5:
                continue
            if confidence < 0.30:
                continue

            importance_raw = str(item.get("importance", "")).lower()
            if importance_raw in ("high", "medium", "low"):
                importance = importance_raw
            else:
                importance = _infer_importance(claim_type, confidence)

            nearby = [str(r).strip() for r in item.get("nearby_citations", []) if str(r).strip()]
            if not nearby:
                nearby = _extract_citations(claim_text)

            seen_texts.add(normalized_text)
            claims.append(
                Claim(
                    claim_id=f"claim_{counter}",
                    text=claim_text[:500],
                    claim_type=claim_type,
                    source_section=str(item.get("source_section", chunk.section_name)),
                    source_chunk_id=str(item.get("source_chunk_id", chunk.chunk_id)),
                    importance=importance,
                    nearby_citations=nearby,
                    cited_refs=nearby,
                    confidence=round(confidence, 3),
                )
            )
            counter += 1
            if counter > 14:
                return claims

    if not claims:
        return _local_claim_fallback(document, starting_counter=counter)
    return claims
