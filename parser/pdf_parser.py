from __future__ import annotations

import re
from pathlib import Path

import fitz

from models.schema import PaperDocument, PaperSection, ReferenceItem


HEADING_PATTERNS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methodology",
    "approach",
    "model architecture",
    "architecture",
    "encoder",
    "decoder",
    "attention",
    "experiments",
    "results",
    "training",
    "evaluation",
    "analysis",
    "why self-attention",
    "discussion",
    "conclusion",
    "references",
}


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _normalized_heading(line: str) -> str:
    clean = _normalize_line(line)
    clean = re.sub(r"^(?:\d+(?:\.\d+)*)\s+", "", clean)
    return clean.rstrip(".: ").lower()


def _is_noise_line(line: str) -> bool:
    lower = line.lower()
    return (
        lower.startswith("arxiv:")
        or "permission to reproduce" in lower
        or "journalistic or scholarly works" in lower
    )


def _looks_like_title_case_heading(line: str) -> bool:
    clean = _normalize_line(line)
    if not clean or "@" in clean or len(clean) > 80:
        return False
    if any(symbol in clean for symbol in ("=", "(", ")", "[", "]", "{", "}", "∈", "+", "/", "\\", ",")):
        return False
    if clean.endswith((".", "?", "!", ";", ",")):
        return False
    first_alpha = next((char for char in clean if char.isalpha()), "")
    if not first_alpha or not first_alpha.isupper():
        return False
    words = clean.split()
    if len(words) < 2 or len(words) > 8:
        return False
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not alpha_words:
        return False
    capitalized = sum(1 for word in alpha_words if word[:1].isupper())
    if capitalized < max(1, len(alpha_words) - 1):
        return False
    return True


def _heading_text_is_clean(line: str) -> bool:
    clean = _normalize_line(line)
    if any(symbol in clean for symbol in ("=", "(", ")", "[", "]", "{", "}", "∈", "+", "/", "\\", ",")):
        return False
    first_alpha = next((char for char in clean if char.isalpha()), "")
    return bool(first_alpha and first_alpha.isupper())


def _looks_like_heading(line: str) -> bool:
    bare = _normalized_heading(line)
    return bare in HEADING_PATTERNS or _looks_like_title_case_heading(line)


def _extract_lines(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    lines: list[str] = []
    try:
        for page in doc:
            text = page.get_text("text")
            for raw_line in text.splitlines():
                line = _normalize_line(raw_line)
                if not line or _is_noise_line(line):
                    continue
                lines.append(line)
    finally:
        doc.close()
    return lines


def _extract_title(pdf_path: Path, lines: list[str]) -> str:
    doc = fitz.open(pdf_path)
    try:
        first_page = doc[0].get_text("dict")
    finally:
        doc.close()

    candidates: list[tuple[float, str]] = []
    for block in first_page.get("blocks", []):
        for line in block.get("lines", []):
            text = _normalize_line("".join(span["text"] for span in line.get("spans", [])))
            if not text or _is_noise_line(text):
                continue
            if text.lower() == "abstract" or "@" in text:
                continue
            if len(text.split()) < 3:
                continue
            if re.search(r"\bgoogle (?:brain|research)\b", text.lower()):
                continue
            size = max(span["size"] for span in line.get("spans", []))
            candidates.append((size, text))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    for line in lines[:20]:
        if len(line.split()) >= 4 and not _looks_like_heading(line) and not _is_noise_line(line):
            return line
    raise ValueError("Could not extract a plausible title from the PDF")


def _split_sections(lines: list[str]) -> tuple[str, list[PaperSection], list[ReferenceItem]]:
    sections: list[PaperSection] = []
    references: list[ReferenceItem] = []
    current_heading = "Front Matter"
    current_lines: list[str] = []
    in_references = False
    abstract = ""
    body_started = False

    def flush() -> None:
        nonlocal current_lines, current_heading, sections, abstract
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        if current_heading.lower() == "abstract" and not abstract:
            abstract = content
        sections.append(
            PaperSection(
                section_id=f"sec_{len(sections) + 1}",
                heading=current_heading,
                content=content,
            )
        )
        current_lines = []

    ref_pattern = re.compile(r"^\[(\d+)\]\s+(.+)")
    footnote_pattern = re.compile(r"^[*∗†‡]|^[*∗†‡]*\s*equal contribution\b|^[*∗†‡]*\s*listing order is random\b", re.IGNORECASE)

    for line in lines:
        normalized_heading = _normalized_heading(line)
        is_explicit_heading = normalized_heading in HEADING_PATTERNS
        is_generic_heading = _looks_like_title_case_heading(line)

        if (is_explicit_heading and _heading_text_is_clean(line)) or (body_started and is_generic_heading):
            flush()
            current_heading = re.sub(r"^\d+(\.\d+)*\s+", "", line).strip().rstrip(".:")
            in_references = current_heading.lower() == "references"
            if current_heading.lower() in {"abstract", "introduction"}:
                body_started = True
            continue

        if in_references:
            match = ref_pattern.match(line)
            if match:
                references.append(
                    ReferenceItem(ref_id=match.group(1), raw_text=match.group(2).strip())
                )
            elif references:
                references[-1].raw_text = f"{references[-1].raw_text} {line}".strip()
            continue

        if current_heading.lower() == "abstract" and footnote_pattern.match(line):
            continue

        current_lines.append(line)

    flush()
    if not abstract:
        abstract = next((section.content for section in sections if "abstract" in section.heading.lower()), "")
    return abstract, sections, references


def parse_paper(pdf_path: str | Path, source_url: str = "") -> PaperDocument:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    lines = _extract_lines(path)
    if not lines:
        raise ValueError("PDF text extraction returned no content")

    title = _extract_title(path, lines)
    abstract, sections, references = _split_sections(lines)
    if not sections:
        raise ValueError("No sections could be extracted from the PDF")

    return PaperDocument(
        paper_id=path.stem,
        source_url=source_url,
        title=title,
        abstract=abstract,
        sections=sections,
        references=references,
    )
