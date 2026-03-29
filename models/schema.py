from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PaperSection(BaseModel):
    section_id: str
    heading: str
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


class Claim(BaseModel):
    claim_id: str
    text: str
    claim_type: Literal["contribution", "result", "novelty", "factual", "background"]
    source_section: str
    cited_refs: list[str] = Field(default_factory=list)
    verification_status: Literal[
        "unverified",
        "supported",
        "weak_support",
        "citation_gap",
        "contradicted",
    ] = "unverified"
    confidence: float = 0.0


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

