from __future__ import annotations

import re

import tiktoken

from models.schema import PaperChunk, PaperDocument, PaperSection


MAX_LLM_TOKENS = 16_000
DEFAULT_SAFETY_MARGIN = 1_000
DEFAULT_WINDOW_TOKENS = 5_000
DEFAULT_OVERLAP_TOKENS = 500
GRAMMAR_SAMPLE_TOKENS = 3_500
VERIFIER_MAX_INPUT_TOKENS = 8_500

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
    return max(1, len((text or "").split()))


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text

    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    selected: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join(selected + [paragraph]).strip()
        if selected and count_tokens(candidate) > max_tokens:
            break
        selected.append(paragraph)
    if selected:
        return "\n\n".join(selected).strip()

    words = text.split()
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word]).strip()
        if current and count_tokens(candidate) > max_tokens:
            break
        current.append(word)
    return " ".join(current).strip()


def _split_large_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
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
    return chunks or [paragraph]


def _split_text_to_units(text: str, max_tokens: int) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if count_tokens(paragraph) > max_tokens:
            units.extend(_split_large_paragraph(paragraph, max_tokens))
        else:
            units.append(paragraph)
    return units or ([text.strip()] if text.strip() else [])


def _extract_overlap_tail(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    tail = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {tail}".strip()
        if tail and count_tokens(candidate) > overlap_tokens:
            break
        tail = candidate
        if count_tokens(tail) >= overlap_tokens:
            break
    return tail.strip()


def rolling_window_text(
    text: str,
    max_tokens: int = DEFAULT_WINDOW_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    if not text.strip():
        return []
    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    units = _split_text_to_units(text, max_tokens=max_tokens)
    chunks: list[str] = []
    current = ""
    overlap_prefix = ""

    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current.strip())
            overlap_prefix = _extract_overlap_tail(current, overlap_tokens)
            current = f"{overlap_prefix}\n\n{unit}".strip() if overlap_prefix else unit
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_all_sections(document: PaperDocument) -> PaperDocument:
    updated_sections = [
        section.model_copy(update={"token_count": count_tokens(section.content)})
        for section in document.sections
    ]
    return document.model_copy(update={"sections": updated_sections})


def build_all_chunks(
    document: PaperDocument,
    max_tokens: int = DEFAULT_WINDOW_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for section in document.sections:
        section_windows = rolling_window_text(
            section.content,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for index, window in enumerate(section_windows):
            chunks.append(
                PaperChunk(
                    chunk_id=f"{section.section_id}_c{index}",
                    section_id=section.section_id,
                    section_name=section.heading,
                    chunk_index=index,
                    content=window,
                    token_count=count_tokens(window),
                )
            )
    return chunks


def build_context_bundle(
    chunks: list[PaperChunk],
    max_tokens: int,
    heading_filter: set[str] | None = None,
) -> str:
    selected: list[str] = []
    for chunk in chunks:
        if heading_filter and chunk.section_name.lower() not in heading_filter:
            continue
        entry = f"[{chunk.section_name} | {chunk.chunk_id}]\n{chunk.content}".strip()
        candidate = "\n\n".join(selected + [entry]).strip()
        if selected and count_tokens(candidate) > max_tokens:
            break
        selected.append(entry)
    return "\n\n".join(selected).strip()


def safe_prompt_room(max_prompt_tokens: int = MAX_LLM_TOKENS) -> int:
    return max(1_000, max_prompt_tokens - DEFAULT_SAFETY_MARGIN)
