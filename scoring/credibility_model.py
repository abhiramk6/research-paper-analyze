from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from models.schema import (
    Claim,
    CitationResult,
    ConsistencyResult,
    CredibilityBreakdown,
    EvidenceFactCheckItem,
    GrammarResult,
)


logger = logging.getLogger(__name__)

# Default weights used when config/scoring.yaml is missing or unreadable.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "contradicted_ratio": 0.35,
    "insufficient_evidence_ratio": 0.20,
    "supported_ratio": 0.15,
    "consistency_penalty": 0.15,
    "citation_coverage": 0.10,
    "parser_uncertainty": 0.03,
    "high_impact_unverified": 0.02,
}

_DEFAULT_PENALTIES: dict[str, Any] = {
    "citation_score_floor": 55,
    "citation_floor_penalty": 15,
    "consistency_score_floor": 65,
    "consistency_floor_penalty": 12,
    "grammar_low_penalty": 8,
    "no_evidence_penalty": 5,
}

_DEFAULT_BANDS: dict[str, float] = {
    "low_max": 30.0,
    "medium_max": 60.0,
}


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config" / "scoring.yaml"
    try:
        import yaml  # type: ignore[import]
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_weights() -> dict[str, float]:
    cfg = _load_config()
    return {**_DEFAULT_WEIGHTS, **(cfg.get("weights") or {})}


def _get_penalties() -> dict[str, Any]:
    cfg = _load_config()
    return {**_DEFAULT_PENALTIES, **(cfg.get("hard_penalties") or {})}


def _get_bands() -> dict[str, float]:
    cfg = _load_config()
    raw = cfg.get("risk_bands") or {}
    return {
        "low_max": float(raw.get("low_max", _DEFAULT_BANDS["low_max"])),
        "medium_max": float(raw.get("medium_max", _DEFAULT_BANDS["medium_max"])),
    }


def _risk_band(score: float, bands: dict[str, float]) -> str:
    if score <= bands["low_max"]:
        return "low"
    if score <= bands["medium_max"]:
        return "medium"
    return "high"


def compute_credibility_breakdown(
    evidence_results: list[EvidenceFactCheckItem],
    claims: list[Claim],
    consistency: ConsistencyResult,
    citation: CitationResult,
    grammar: GrammarResult,
) -> CredibilityBreakdown:
    """
    Compute an interpretable, feature-level credibility breakdown.

    Every feature is a measurable signal, not a hidden heuristic.
    The formula and weights are configurable via config/scoring.yaml.

    Score is in [0, 100] where higher = more risk.
    """
    weights = _get_weights()
    penalties = _get_penalties()
    bands = _get_bands()

    n = len(evidence_results) or 1

    supported = sum(1 for r in evidence_results if r.verdict == "supported")
    contradicted = sum(1 for r in evidence_results if r.verdict == "contradicted")
    mixed = sum(1 for r in evidence_results if r.verdict == "mixed")
    insufficient = sum(1 for r in evidence_results if r.verdict in ("insufficient_evidence", "paper_supported_only"))

    supported_ratio = supported / n
    contradicted_ratio = contradicted / n
    insufficient_ratio = (insufficient + mixed * 0.5) / n  # mixed counts as partial insufficiency

    # Citation coverage: normalise to [0, 1].
    citation_coverage = min(1.0, citation.score / 100.0)

    # Consistency penalty: how far below 100 the score is, normalised.
    consistency_penalty_norm = max(0.0, (100 - consistency.score) / 100.0)

    # Parser uncertainty: fraction of claims with low confidence (< 0.5).
    low_conf_claims = sum(1 for c in claims if c.confidence < 0.5)
    parser_uncertainty = low_conf_claims / max(len(claims), 1)

    # High-impact unverified count: claims marked high importance with no external support.
    high_impact_unverified = sum(
        1 for r in evidence_results
        if r.verdict in ("insufficient_evidence", "paper_supported_only")
        for c in claims
        if c.claim_id == r.claim_id and c.importance == "high"
    )
    # Normalise to [0, 1]: 5 or more unverified high-impact claims = max penalty.
    high_impact_norm = min(1.0, high_impact_unverified / 5.0)

    # Weighted base score.
    raw = (
        weights["contradicted_ratio"] * contradicted_ratio
        + weights["insufficient_evidence_ratio"] * insufficient_ratio
        - weights["supported_ratio"] * supported_ratio
        + weights["consistency_penalty"] * consistency_penalty_norm
        - weights["citation_coverage"] * citation_coverage
        + weights["parser_uncertainty"] * parser_uncertainty
        + weights["high_impact_unverified"] * high_impact_norm
    )

    # Hard penalties for severely weak signals.
    hard = 0.0
    if citation.score < penalties["citation_score_floor"]:
        hard += penalties["citation_floor_penalty"]
    if consistency.score < penalties["consistency_score_floor"]:
        hard += penalties["consistency_floor_penalty"]
    if grammar.rating == "Low":
        hard += penalties["grammar_low_penalty"]
    if not evidence_results:
        hard += penalties["no_evidence_penalty"]

    # Convert to 0–100 scale and apply hard penalties.
    final_score = max(0.0, min(100.0, raw * 100 + hard))

    score_breakdown = {
        "contradicted_component": round(weights["contradicted_ratio"] * contradicted_ratio * 100, 2),
        "insufficient_component": round(weights["insufficient_evidence_ratio"] * insufficient_ratio * 100, 2),
        "supported_reduction": round(weights["supported_ratio"] * supported_ratio * 100, 2),
        "consistency_component": round(weights["consistency_penalty"] * consistency_penalty_norm * 100, 2),
        "citation_reduction": round(weights["citation_coverage"] * citation_coverage * 100, 2),
        "uncertainty_component": round(weights["parser_uncertainty"] * parser_uncertainty * 100, 2),
        "high_impact_component": round(weights["high_impact_unverified"] * high_impact_norm * 100, 2),
        "hard_penalties": round(hard, 2),
    }

    explanation = _build_explanation(
        final_score, supported_ratio, contradicted_ratio, insufficient_ratio,
        citation_coverage, consistency.score, high_impact_unverified, hard,
    )

    return CredibilityBreakdown(
        supported_claim_ratio=round(supported_ratio, 3),
        contradicted_claim_ratio=round(contradicted_ratio, 3),
        insufficient_evidence_ratio=round(insufficient_ratio, 3),
        citation_coverage_ratio=round(citation_coverage, 3),
        consistency_penalty=round(consistency_penalty_norm, 3),
        parser_uncertainty=round(parser_uncertainty, 3),
        high_impact_unverified_claim_count=high_impact_unverified,
        final_score=round(final_score, 2),
        risk_band=_risk_band(final_score, bands),
        score_breakdown=score_breakdown,
        explanation=explanation,
    )


def _build_explanation(
    score: float,
    supported_ratio: float,
    contradicted_ratio: float,
    insufficient_ratio: float,
    citation_coverage: float,
    consistency_score: int,
    high_impact_unverified: int,
    hard_penalties: float,
) -> str:
    lines = [f"Final credibility risk score: {score:.1f}/100."]

    if contradicted_ratio > 0:
        lines.append(
            f"{contradicted_ratio:.0%} of fact-checked claims were contradicted by external evidence — "
            "this is the strongest risk signal."
        )
    if supported_ratio > 0.5:
        lines.append(f"{supported_ratio:.0%} of checked claims were externally supported, reducing risk.")
    if insufficient_ratio > 0.5:
        lines.append(
            f"{insufficient_ratio:.0%} of claims had insufficient external evidence — "
            "they could not be independently verified."
        )
    if citation_coverage < 0.6:
        lines.append(f"Citation coverage is low ({citation_coverage:.0%}), increasing uncertainty.")
    if consistency_score < 70:
        lines.append(f"Consistency score is {consistency_score}/100 — internal coherence is weak.")
    if high_impact_unverified > 0:
        lines.append(
            f"{high_impact_unverified} high-importance claim(s) could not be externally verified."
        )
    if hard_penalties > 0:
        lines.append(
            f"Hard signal penalties ({hard_penalties:.0f} pts) were applied for severely weak dimensions."
        )

    return " ".join(lines)
