from __future__ import annotations

import json
import re

from agents.llm_client import call_llm_json
from chunker.token_chunker import DEFAULT_WINDOW_TOKENS, build_all_chunks, count_tokens, truncate_to_token_limit
from models.schema import Claim, PaperChunk, PaperDocument


EXTRACTION_WINDOW_TOKENS = min(DEFAULT_WINDOW_TOKENS, 4_500)
MAX_CLAIMS = 18

_VALID_CLAIM_TYPES = {"contribution", "result", "comparison", "factual", "method"}
_RESULT_TYPES = {"result", "comparison", "factual"}
_HIGH_IMPORTANCE_TYPES = {"result", "comparison"}


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_citations(text: str) -> list[str]:
    matches = re.findall(r"\[(.*?)\]", text)
    citations: list[str] = []
    for match in matches:
        citations.extend(part.strip() for part in match.split(",") if part.strip())
    return citations


def _normalize_claim_type(raw: str) -> str:
    value = raw.strip().lower().replace(" ", "_")
    if value in _VALID_CLAIM_TYPES:
        return value
    if any(token in value for token in ("result", "benchmark", "metric", "accuracy", "score")):
        return "result"
    if any(token in value for token in ("compare", "comparison", "baseline", "prior")):
        return "comparison"
    if any(token in value for token in ("fact", "background", "dataset", "historical")):
        return "factual"
    if any(token in value for token in ("method", "approach", "architecture", "model")):
        return "method"
    return "contribution"


def _infer_importance(claim_type: str, text: str, confidence: float) -> str:
    lowered = text.lower()
    if claim_type in _HIGH_IMPORTANCE_TYPES:
        return "high"
    if claim_type == "factual" and any(token in lowered for token in ("%", "first", "state-of-the-art", "sota")):
        return "high"
    if claim_type in _RESULT_TYPES or confidence >= 0.8:
        return "high"
    if claim_type in {"contribution", "method"} or confidence >= 0.55:
        return "medium"
    return "low"


def _claim_prompt(chunk: PaperChunk) -> str:
    schema = {
        "claims": [
            {
                "text": "string",
                "claim_type": "contribution|result|comparison|factual|method",
                "importance": "high|medium|low",
                "confidence": 0.0,
                "nearby_citations": ["string"],
            }
        ]
    }
    content = truncate_to_token_limit(chunk.content, EXTRACTION_WINDOW_TOKENS)
    return (
        "You are extracting grounded claims from a research paper window.\n\n"
        "Return only the most important claims that are specific enough to evaluate.\n"
        "Use only these claim types:\n"
        "- contribution: the paper's proposed contribution or main idea\n"
        "- result: concrete quantitative or performance claims\n"
        "- comparison: comparisons against baselines or prior work\n"
        "- factual: external facts, historical statements, or dataset facts\n"
        "- method: claims describing how the method works\n\n"
        "Rules:\n"
        "- Prefer claims that are testable or evidence-bearing.\n"
        "- Use close-to-source wording, not broad paraphrase.\n"
        "- Skip generic hype or vague claims.\n"
        "- Keep nearby citations when visible.\n"
        "- Confidence must be in [0, 1].\n"
        f"Respond with valid JSON matching:\n{json.dumps(schema, indent=2)}\n\n"
        f"Window metadata: section={chunk.section_name}, chunk_id={chunk.chunk_id}\n\n"
        f"{content}"
    )


def _local_fallback(document: PaperDocument, start_index: int) -> list[Claim]:
    fallback: list[Claim] = []
    signal_words = (
        "we propose",
        "we introduce",
        "we present",
        "outperform",
        "improve",
        "achieve",
        "state-of-the-art",
        "%",
        "compared to",
        "baseline",
        "dataset",
    )
    chunks = document.chunks or build_all_chunks(document, max_tokens=EXTRACTION_WINDOW_TOKENS)
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.content)
        for sentence in sentences:
            clean = _normalize_spaces(sentence)
            if len(clean.split()) < 8:
                continue
            if not any(token in clean.lower() for token in signal_words):
                continue
            if len(fallback) >= 6:
                return fallback
            claim_type = "contribution"
            lowered = clean.lower()
            if any(token in lowered for token in ("outperform", "baseline", "compared to")):
                claim_type = "comparison"
            elif any(token in lowered for token in ("achieve", "accuracy", "score", "%", "state-of-the-art")):
                claim_type = "result"
            elif any(token in lowered for token in ("dataset", "corpus", "benchmark")):
                claim_type = "factual"
            elif any(token in lowered for token in ("architecture", "layer", "attention", "training")):
                claim_type = "method"
            fallback.append(
                Claim(
                    claim_id=f"claim_{start_index + len(fallback)}",
                    text=clean[:500],
                    claim_type=claim_type,
                    source_section=chunk.section_name,
                    source_chunk_id=chunk.chunk_id,
                    importance=_infer_importance(claim_type, clean, 0.45),
                    nearby_citations=_extract_citations(clean),
                    confidence=0.45,
                )
            )
    return fallback


def extract_claims(document: PaperDocument) -> list[Claim]:
    chunks = document.chunks or build_all_chunks(document, max_tokens=EXTRACTION_WINDOW_TOKENS)
    claims: list[Claim] = []
    seen_texts: set[str] = set()

    for chunk in chunks:
        prompt = _claim_prompt(chunk)
        payload = call_llm_json(prompt, fallback={"claims": []})
        for raw in payload.get("claims", []):
            text = _normalize_spaces(str(raw.get("text", "")))
            if not text or len(text.split()) < 6:
                continue
            normalized_text = text.lower()
            if normalized_text in seen_texts:
                continue

            confidence = float(raw.get("confidence") or 0.0)
            if confidence < 0.35:
                continue

            claim_type = _normalize_claim_type(str(raw.get("claim_type", "contribution")))
            importance_raw = str(raw.get("importance", "")).lower()
            if importance_raw not in {"high", "medium", "low"}:
                importance_raw = _infer_importance(claim_type, text, confidence)
            nearby = [
                str(item).strip()
                for item in raw.get("nearby_citations", [])
                if str(item).strip()
            ] or _extract_citations(text)

            seen_texts.add(normalized_text)
            claims.append(
                Claim(
                    claim_id=f"claim_{len(claims) + 1}",
                    text=text[:500],
                    claim_type=claim_type,
                    source_section=chunk.section_name,
                    source_chunk_id=chunk.chunk_id,
                    importance=importance_raw,
                    nearby_citations=nearby,
                    confidence=round(confidence, 3),
                )
            )
            if len(claims) >= MAX_CLAIMS:
                return claims

    if claims:
        return claims
    return _local_fallback(document, start_index=1)
