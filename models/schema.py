from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PaperSection(BaseModel):
    section_id: str
    heading: str
    content: str
    token_count: int = 0
    canonical_category: Optional[str] = None


class PaperChunk(BaseModel):
    chunk_id: str
    section_name: str
    chunk_index: int
    content: str
    token_count: int = 0


class ReferenceItem(BaseModel):
    ref_id: str
    raw_text: str
    title: Optional[str] = None
    abstract: Optional[str] = None


class PaperDocument(BaseModel):
    paper_id: str
    source_url: str
    title: str
    abstract: str
    sections: list[PaperSection]
    references: list[ReferenceItem]
    chunks: list[PaperChunk] = Field(default_factory=list)


ClaimType = Literal[
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
]


class Claim(BaseModel):
    claim_id: str
    text: str
    claim_type: ClaimType
    source_section: str
    source_chunk_id: Optional[str] = None
    importance: Literal["high", "medium", "low"] = "medium"
    nearby_citations: list[str] = Field(default_factory=list)
    cited_refs: list[str] = Field(default_factory=list)
    verification_status: Literal[
        "unverified",
        "supported",
        "weak_support",
        "citation_gap",
        "contradicted",
    ] = "unverified"
    confidence: float = 0.0


class EvidenceItem(BaseModel):
    evidence_id: str
    title: str
    url: str
    snippet: str
    source_type: Literal["web", "scholarly", "stub"]
    retrieval_score: float = 0.0
    publication_year: Optional[int] = None


class EvidenceFactCheckItem(BaseModel):
    claim_id: str
    verdict: Literal[
        "supported",
        "contradicted",
        "mixed",
        "insufficient_evidence",
        "paper_supported_only",
    ]
    confidence: float = 0.0
    reasoning: str
    used_evidence_ids: list[str] = Field(default_factory=list)
    paper_context_excerpt: str = ""
    evidence_items: list[EvidenceItem] = Field(default_factory=list)


class CredibilityBreakdown(BaseModel):
    supported_claim_ratio: float = 0.0
    contradicted_claim_ratio: float = 0.0
    insufficient_evidence_ratio: float = 0.0
    citation_coverage_ratio: float = 0.0
    consistency_penalty: float = 0.0
    parser_uncertainty: float = 0.0
    high_impact_unverified_claim_count: int = 0
    final_score: float = 0.0
    risk_band: Literal["low", "medium", "high"] = "medium"
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class ConsistencyResult(BaseModel):
    score: int
    issues: list[str]
    reasoning: str


class GrammarResult(BaseModel):
    rating: Literal["High", "Medium", "Low"]
    reasoning: str


class CitationResult(BaseModel):
    score: int
    citation_gaps: list[str]
    reasoning: str


class FactCheckItem(BaseModel):
    claim_id: str
    status: Literal["verified", "unverifiable", "suspicious"]
    reasoning: str


class FactCheckResult(BaseModel):
    items: list[FactCheckItem]


class NoveltyResult(BaseModel):
    rating: Literal["High", "Moderate", "Low"]
    reasoning: str


class CredibilityResult(BaseModel):
    score: float
    risk_factors: list[str]
    reasoning: str
    breakdown: Optional[CredibilityBreakdown] = None


class FinalReport(BaseModel):
    title: str
    summary: str
    consistency_score: int
    grammar_rating: str
    citation_score: int
    novelty_rating: str
    credibility_score: float
    fabrication_probability: str
    recommendation: Literal["Pass", "Borderline", "Fail"]
