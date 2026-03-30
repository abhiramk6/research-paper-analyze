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
    section_id: str
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


ClaimType = Literal["contribution", "result", "comparison", "factual", "method"]
ClaimImportance = Literal["high", "medium", "low"]
ClaimStatus = Literal[
    "unverified",
    "supported",
    "contradicted",
    "mixed",
    "insufficient_evidence",
]


class Claim(BaseModel):
    claim_id: str
    text: str
    claim_type: ClaimType
    source_section: str
    source_chunk_id: Optional[str] = None
    importance: ClaimImportance = "medium"
    nearby_citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    verification_status: ClaimStatus = "unverified"


class EvidenceItem(BaseModel):
    evidence_id: str
    query: str
    title: str
    url: str
    domain: str
    domain_tier: Literal["scholarly", "web"]
    snippet: str
    retrieval_score: float = 0.0


class ClaimCheckResult(BaseModel):
    claim_id: str
    verdict: Literal["supported", "contradicted", "mixed", "insufficient_evidence"]
    confidence: float = 0.0
    reasoning: str
    matched_evidence_ids: list[str] = Field(default_factory=list)
    paper_context_excerpt: str = ""
    evidence_items: list[EvidenceItem] = Field(default_factory=list)


class ConsistencyResult(BaseModel):
    score: int
    issues: list[str]
    reasoning: str
    grounded_claim_count: int = 0
    unresolved_claim_count: int = 0
    raw_score: Optional[int] = None


class GrammarResult(BaseModel):
    rating: Literal["High", "Medium", "Low"]
    reasoning: str
    issues: list[str] = Field(default_factory=list)


class NoveltyResult(BaseModel):
    rating: Literal["High", "Moderate", "Low"]
    reasoning: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class FabricationBreakdown(BaseModel):
    contradicted_ratio: float = 0.0
    mixed_ratio: float = 0.0
    insufficient_evidence_ratio: float = 0.0
    high_impact_insufficient_ratio: float = 0.0
    evidence_coverage_ratio: float = 0.0
    consistency_penalty_ratio: float = 0.0
    final_score: float = 0.0
    score_components: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class FabricationRiskResult(BaseModel):
    score: float
    risk_band: Literal["low", "medium", "high"]
    risk_factors: list[str]
    reasoning: str
    breakdown: FabricationBreakdown
    raw_score: Optional[float] = None


class AssessmentSynthesisResult(BaseModel):
    consistency_score: int
    fabrication_score: float
    recommendation: Literal["Pass", "Borderline", "Fail"]
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    consistency_reasoning: str
    fabrication_reasoning: str


class FinalReport(BaseModel):
    title: str
    summary: str
    consistency_score: int
    grammar_rating: str
    novelty_rating: str
    fabrication_risk: str
    recommendation: Literal["Pass", "Borderline", "Fail"]
