from __future__ import annotations

"""
Section normalizer: maps raw PDF section headings to canonical semantic categories
via a single LLM call.

Why: PDF papers use wildly different heading styles —
  "3. Our Proposed Architecture"  →  methodology
  "4.2 Quantitative Analysis"     →  results
  "5 Related Literature"          →  related_work
  ...etc.

The downstream pipeline uses keyword matching on headings to build evidence packets
(e.g. "give me the 'methodology' sections"). Without normalization, a heading like
"Our Approach" never matches the keyword "method", so evidence packets come back empty.

This module runs ONE LLM call over the full heading list and returns a
heading → canonical_category mapping that gets stored on each PaperSection.
Falls back gracefully to heuristic mapping if the LLM fails or quota is exhausted.
"""

import json
import logging
import re

from agents.llm_client import call_llm_json
from models.schema import PaperDocument, PaperSection


logger = logging.getLogger(__name__)

# Canonical categories the LLM must choose from.
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

# Heuristic fallback patterns for when LLM is unavailable.
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
    """
    Run one LLM call to map every raw section heading to a canonical category.
    Stores the result on each PaperSection.canonical_category.
    Falls back to heuristic patterns on LLM failure.
    """
    headings = [section.heading for section in document.sections]
    if not headings:
        return document

    # Build heuristic fallback mapping up front.
    heuristic_map = {h: _heuristic_category(h) for h in headings}

    fallback = heuristic_map  # fallback is the full heading→category dict

    prompt = _build_prompt(headings)
    raw = call_llm_json(prompt, fallback=fallback)

    # Validate: only keep keys that are known headings and values that are valid categories.
    valid_categories = set(CANONICAL_CATEGORIES)
    llm_map: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            cat = value.strip().lower()
            if cat in valid_categories:
                llm_map[key] = cat

    # Merge: LLM result takes priority; heuristic fills any gaps.
    final_map = {**heuristic_map, **llm_map}

    # Annotate sections.
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
