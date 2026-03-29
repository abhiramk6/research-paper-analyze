from __future__ import annotations

import json
import re

from agents.llm_client import call_llm_json
from chunker.token_chunker import chunk_text
from models.schema import Claim, PaperDocument, PaperSection
from prompt_loader import load_prompt


CLAIM_SECTIONS = ("abstract", "introduction", "method", "methodology", "results", "conclusion")


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
    if value in {"contribution", "result", "novelty", "factual", "background"}:
        return value
    return "background"


def _build_prompt(section_heading: str, chunk: str) -> str:
    schema = {
        "claims": [
            {
                "claim_id": "string",
                "text": "string",
                "claim_type": "contribution|result|novelty|factual|background",
                "source_section": section_heading,
                "cited_refs": ["string"],
                "confidence": 0.0,
            }
        ]
    }
    return load_prompt(
        "claim_extraction.txt",
        schema_json=json.dumps(schema),
        section_heading=section_heading,
        chunk=chunk,
    )


def _group_sections(document: PaperDocument) -> list[tuple[str, str]]:
    grouped_sections: list[tuple[str, str]] = []
    current_heading = "paper_overview"
    current_parts: list[str] = []

    for section in document.sections:
        if not _section_is_claim_relevant(section):
            continue
        section_heading = section.heading.lower()
        if any(name in section_heading for name in ("method", "methodology", "results")):
            group_name = "technical_findings"
        elif any(name in section_heading for name in ("abstract", "introduction", "conclusion")):
            group_name = "paper_overview"
        else:
            group_name = "supporting_context"

        formatted_section = f"[{section.heading}]\n{section.content.strip()}"
        if current_parts and group_name != current_heading:
            grouped_sections.append((current_heading, "\n\n".join(current_parts)))
            current_parts = []
        current_heading = group_name
        current_parts.append(formatted_section)

    if current_parts:
        grouped_sections.append((current_heading, "\n\n".join(current_parts)))
    return grouped_sections


def _infer_claim_type(sentence: str, source_section: str) -> str:
    sentence_lower = sentence.lower()
    section_lower = source_section.lower()
    if any(keyword in sentence_lower for keyword in ("we propose", "our approach", "introduce", "present")):
        return "contribution"
    if any(keyword in sentence_lower for keyword in ("outperform", "achieve", "improve", "results", "accuracy", "score")):
        return "result"
    if any(keyword in sentence_lower for keyword in ("novel", "first", "new", "unlike prior")):
        return "novelty"
    if "result" in section_lower or "conclusion" in section_lower:
        return "result"
    return "factual"


def _local_claim_fallback(document: PaperDocument, starting_counter: int = 1) -> list[Claim]:
    candidate_sections = [
        section
        for section in document.sections
        if _section_is_claim_relevant(section)
    ]
    fallback_claims: list[Claim] = []

    for section in candidate_sections:
        sentences = re.split(r"(?<=[.!?])\s+", section.content)
        for sentence in sentences:
            clean_sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(clean_sentence.split()) < 8:
                continue
            if not any(
                keyword in clean_sentence.lower()
                for keyword in (
                    "we ",
                    "our ",
                    "show",
                    "demonstrate",
                    "achieve",
                    "improve",
                    "propose",
                    "introduce",
                    "outperform",
                )
            ):
                continue
            fallback_claims.append(
                Claim(
                    claim_id=f"claim_{starting_counter}",
                    text=clean_sentence[:400],
                    claim_type=_infer_claim_type(clean_sentence, section.heading),
                    source_section=section.heading,
                    cited_refs=_extract_citations(clean_sentence),
                    confidence=0.45,
                )
            )
            starting_counter += 1
            if len(fallback_claims) >= 6:
                return fallback_claims
    return fallback_claims


def extract_claims(document: PaperDocument) -> list[Claim]:
    claims: list[Claim] = []
    counter = 1
    seen_texts: set[str] = set()

    for group_heading, group_text in _group_sections(document):
        for chunk in chunk_text(group_text, max_tokens=9_000):
            payload = call_llm_json(_build_prompt(group_heading, chunk), fallback={"claims": []})
            raw_claims = payload.get("claims", [])

            for item in raw_claims:
                claim_text = str(item.get("text", "")).strip()
                if not claim_text:
                    continue
                normalized_text = re.sub(r"\s+", " ", claim_text).strip().lower()
                if normalized_text in seen_texts:
                    continue
                claim_type = _normalize_claim_type(str(item.get("claim_type", "background")))
                confidence = float(item.get("confidence", 0.0) or 0.0)
                if claim_type == "background" or confidence < 0.35:
                    continue
                cited_refs = [str(ref).strip() for ref in item.get("cited_refs", []) if str(ref).strip()]
                if not cited_refs:
                    cited_refs = _extract_citations(claim_text)
                seen_texts.add(normalized_text)
                claims.append(
                    Claim(
                        claim_id=f"claim_{counter}",
                        text=claim_text,
                        claim_type=claim_type,
                        source_section=group_heading,
                        cited_refs=cited_refs,
                        confidence=confidence,
                    )
                )
                counter += 1
                if counter > 12:
                    return claims

    if not claims:
        return _local_claim_fallback(document, starting_counter=counter)
    return claims
