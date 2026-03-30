from __future__ import annotations

import json
import logging
import re

from agents.llm_client import call_llm_json
from models.schema import PaperDocument, PaperSection


logger = logging.getLogger(__name__)

CANONICAL_CATEGORIES = [
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "experiments",
    "results",
    "analysis",
    "conclusion",
    "appendix",
    "references",
    "other",
]

_HEURISTIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\babstract\b", re.I), "abstract"),
    (re.compile(r"\bintroduct", re.I), "introduction"),
    (re.compile(r"\brelated\b|\bliterature\b|\bprior work\b|\bbackground\b", re.I), "related_work"),
    (re.compile(r"\bmethod|approach|model|architect|framework|system|propos|our\b", re.I), "methodology"),
    (re.compile(r"\bexperiment|setup|implement|training|dataset|baselines\b", re.I), "experiments"),
    (re.compile(r"\bresult|evaluat|benchmark|performance|metric|comparison|ablat", re.I), "results"),
    (re.compile(r"\banalys|discussion|insight|error analysis\b", re.I), "analysis"),
    (re.compile(r"\bconclus|future work|summary\b", re.I), "conclusion"),
    (re.compile(r"\bappendix|supplementary\b", re.I), "appendix"),
    (re.compile(r"\breference|bibliography\b", re.I), "references"),
]


def _heuristic_category(heading: str) -> str:
    for pattern, category in _HEURISTIC_PATTERNS:
        if pattern.search(heading):
            return category
    return "other"


def _build_prompt(headings: list[str]) -> str:
    categories_str = ", ".join(f'"{c}"' for c in CANONICAL_CATEGORIES)
    schema_example = {h: "canonical_category" for h in headings[:3]}

    return (
        "You are a research paper structure analyst.\n\n"
        "Below is a list of section headings extracted from a research paper PDF. "
        "For each heading, assign the single best canonical category from this list:\n"
        f"  {categories_str}\n\n"
        "Rules:\n"
        "- Use only the categories listed above — do not invent new ones.\n"
        "- If a heading describes the proposed model/approach/algorithm, use 'methodology'.\n"
        "- If a heading describes experiments, training setup, or datasets, use 'experiments'.\n"
        "- If a heading describes numerical results, benchmarks, or comparisons, use 'results'.\n"
        "- If a heading is about prior work, related papers, or literature review, use 'related_work'.\n"
        "- If unsure, use 'other'.\n\n"
        f"Return a single JSON object mapping each exact heading string to its category.\n"
        f"Example shape (not the real answer): {json.dumps(schema_example)}\n\n"
        f"Headings to classify:\n{json.dumps(headings, indent=2)}"
    )


def normalize_section_headings(document: PaperDocument) -> PaperDocument:
    headings = [section.heading for section in document.sections]
    if not headings:
        return document

    heuristic_map = {h: _heuristic_category(h) for h in headings}
    fallback = heuristic_map

    prompt = _build_prompt(headings)
    raw = call_llm_json(prompt, fallback=fallback)

    valid_categories = set(CANONICAL_CATEGORIES)
    llm_map: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            cat = value.strip().lower()
            if cat in valid_categories:
                llm_map[key] = cat

    final_map = {**heuristic_map, **llm_map}
    updated_sections = [
        section.model_copy(update={"canonical_category": final_map.get(section.heading, "other")})
        for section in document.sections
    ]

    logger.info(
        "Section normalization complete: %d sections → %s",
        len(updated_sections),
        {s.heading: s.canonical_category for s in updated_sections},
    )

    return document.model_copy(update={"sections": updated_sections})
