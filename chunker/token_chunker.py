from __future__ import annotations

import re

import tiktoken

from models.schema import PaperChunk, PaperDocument, PaperSection


# Hard architectural limit: no single LLM input should exceed this.
# Downstream agents get bounded evidence packets, so raw chunks should be smaller.
MAX_TOKENS = 14_000

# Chunk size for claim extraction — keeps per-claim context tight.
CLAIM_CHUNK_TOKENS = 6_000

# Overlap in tokens to preserve cross-paragraph context at chunk boundaries.
CHUNK_OVERLAP_TOKENS = 200

_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is not None:
        return _ENCODING
    try:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODING = False
    return _ENCODING


def count_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding:
        return len(encoding.encode(text or ""))
    # Offline-safe approximation when tokenizer assets are unavailable.
    return max(1, len((text or "").split()))


def _split_large_block(block: str, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", block.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks or [block]


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if count_tokens(paragraph) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_large_block(paragraph, max_tokens))
            continue

        candidate = f"{current}\n\n{paragraph}".strip()
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def chunk_section(section: PaperSection, max_tokens: int = MAX_TOKENS) -> list[str]:
    return chunk_text(section.content, max_tokens=max_tokens)


def chunk_all_sections(document: PaperDocument, max_tokens: int = MAX_TOKENS) -> PaperDocument:
    updated_sections = [
        section.model_copy(update={"token_count": count_tokens(section.content)})
        for section in document.sections
    ]
    return document.model_copy(update={"sections": updated_sections})


def _extract_overlap_tail(text: str, overlap_tokens: int) -> str:
    """Return the last `overlap_tokens` worth of text from a chunk for context continuity."""
    if overlap_tokens <= 0:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    tail = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {tail}".strip()
        if count_tokens(candidate) > overlap_tokens:
            break
        tail = candidate
    return tail


def recursive_chunk_section(
    section: PaperSection,
    max_tokens: int = CLAIM_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[PaperChunk]:
    """
    Recursively split a section into bounded PaperChunk objects.

    Each chunk preserves section name, a sequential chunk_id, and local order.
    Overlap ensures context is not severed at boundaries.
    Token budget is enforced at every level.
    """
    text = section.content
    section_name = section.heading

    # Fast path: fits in one chunk.
    if count_tokens(text) <= max_tokens:
        return [
            PaperChunk(
                chunk_id=f"{section.section_id}_c0",
                section_name=section_name,
                chunk_index=0,
                content=text,
                token_count=count_tokens(text),
            )
        ]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    raw_chunks: list[str] = []
    current = ""
    overlap_prefix = ""

    for paragraph in paragraphs:
        # If a single paragraph is too big, recursively split it by sentences.
        if count_tokens(paragraph) > max_tokens:
            if current:
                raw_chunks.append(current)
                overlap_prefix = _extract_overlap_tail(current, overlap_tokens)
                current = ""
            sub_chunks = _split_large_block(paragraph, max_tokens)
            for sub in sub_chunks:
                candidate = f"{overlap_prefix}\n\n{sub}".strip() if overlap_prefix else sub
                if count_tokens(candidate) <= max_tokens:
                    raw_chunks.append(candidate)
                    overlap_prefix = _extract_overlap_tail(candidate, overlap_tokens)
                else:
                    # Sub-chunk itself too large after overlap — emit without overlap.
                    raw_chunks.append(sub)
                    overlap_prefix = _extract_overlap_tail(sub, overlap_tokens)
            continue

        candidate = f"{current}\n\n{paragraph}".strip()
        if current and count_tokens(candidate) > max_tokens:
            raw_chunks.append(current)
            overlap_prefix = _extract_overlap_tail(current, overlap_tokens)
            # Start next chunk with overlap context.
            current = f"{overlap_prefix}\n\n{paragraph}".strip() if overlap_prefix else paragraph
        else:
            current = candidate

    if current:
        raw_chunks.append(current)

    return [
        PaperChunk(
            chunk_id=f"{section.section_id}_c{i}",
            section_name=section_name,
            chunk_index=i,
            content=chunk,
            token_count=count_tokens(chunk),
        )
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]


def build_all_chunks(
    document: PaperDocument,
    max_tokens: int = CLAIM_CHUNK_TOKENS,
) -> list[PaperChunk]:
    """Build PaperChunk list for the whole document, capped at max_tokens per chunk."""
    all_chunks: list[PaperChunk] = []
    for section in document.sections:
        all_chunks.extend(recursive_chunk_section(section, max_tokens=max_tokens))
    return all_chunks
