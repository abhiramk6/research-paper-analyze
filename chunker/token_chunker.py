from __future__ import annotations

import re

import tiktoken

from models.schema import PaperDocument, PaperSection


MAX_TOKENS = 14_000
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
    # Offline-safe approximation when the tokenizer assets are unavailable.
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
