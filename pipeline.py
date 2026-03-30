from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.consistency_agent import run_consistency_agent
from agents.credibility_agent import run_credibility_agent
from agents.grammar_agent import run_grammar_agent
from agents.novelty_agent import run_novelty_agent
from aggregator.report_builder import build_report
from chunker.token_chunker import DEFAULT_WINDOW_TOKENS, build_all_chunks, chunk_all_sections
from claims.claim_extractor import extract_claims
from ingestion.downloader import resolve_pdf_input
from parser.pdf_parser import parse_paper
from reporter.report_writer import save_report
from retrieval.evidence_ranker import rank_evidence
from retrieval.query_builder import build_query_plan
from retrieval.retriever import EvidenceRetriever
from verification.claim_router import routing_decision
from verification.verifier_agent import verify_claim


load_dotenv()
logger = logging.getLogger(__name__)

_RETRIEVER = EvidenceRetriever()


class EvaluatorState(TypedDict, total=False):
    source: str
    pdf_path: str
    document: Any
    claims: Any
    claim_checks: Any
    novelty_evidence: Any
    grammar: Any
    consistency: Any
    novelty: Any
    fabrication: Any
    report: Any
    report_path: str
    paper_id: str
    retrieval_log: list[dict[str, Any]]
    evidence_catalog: list[Any]


def _paper_context_for_claim(document, claim) -> str:
    if claim.source_chunk_id:
        for chunk in document.chunks:
            if chunk.chunk_id == claim.source_chunk_id:
                return chunk.content[:1_200]
    for section in document.sections:
        if claim.source_section.lower() == section.heading.lower():
            return section.content[:1_200]
    return document.abstract[:800]


def _merge_evidence(existing: list, incoming: list) -> list:
    by_id = {item.evidence_id: item for item in existing}
    for item in incoming:
        by_id[item.evidence_id] = item
    return list(by_id.values())


def ingest_paper_node(state: EvaluatorState) -> EvaluatorState:
    pdf_path, source_reference = resolve_pdf_input(state["source"])
    document = parse_paper(pdf_path, source_url=source_reference)
    document = chunk_all_sections(document)
    chunks = build_all_chunks(document, max_tokens=DEFAULT_WINDOW_TOKENS)
    document = document.model_copy(update={"chunks": chunks})
    return {
        "pdf_path": str(pdf_path),
        "document": document,
        "paper_id": document.paper_id,
    }


def claim_extraction_node(state: EvaluatorState) -> EvaluatorState:
    return {"claims": extract_claims(state["document"])}


def evidence_node(state: EvaluatorState) -> EvaluatorState:
    document = state["document"]
    claims = state.get("claims", [])
    claim_checks = []
    retrieval_log: list[dict[str, Any]] = []
    evidence_catalog: list[Any] = []

    for claim in claims:
        decision = routing_decision(claim)
        log_entry: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type,
            "importance": claim.importance,
            "routing_decision": decision,
        }
        if decision != "external_evidence":
            retrieval_log.append(log_entry)
            continue

        query_plan = build_query_plan(claim, paper_title=document.title)
        raw_evidence = _RETRIEVER.retrieve_for_plan(query_plan)
        ranked_evidence = rank_evidence(claim, raw_evidence, top_k=5)
        result = verify_claim(
            claim,
            paper_context=_paper_context_for_claim(document, claim),
            evidence_items=ranked_evidence,
        )
        claim_checks.append(result)
        evidence_catalog = _merge_evidence(evidence_catalog, ranked_evidence)
        log_entry.update(
            {
                "queries": query_plan,
                "retrieved_count": len(raw_evidence),
                "used_evidence_ids": result.matched_evidence_ids,
                "final_verdict": result.verdict,
            }
        )
        retrieval_log.append(log_entry)
        logger.info("Claim %s checked with verdict %s", claim.claim_id, result.verdict)

    return {
        "claim_checks": claim_checks,
        "retrieval_log": retrieval_log,
        "evidence_catalog": evidence_catalog,
    }


def consistency_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "consistency": run_consistency_agent(
            state["document"],
            state.get("claims", []),
            state.get("claim_checks", []),
        )
    }


def grammar_node(state: EvaluatorState) -> EvaluatorState:
    return {"grammar": run_grammar_agent(state["document"])}


def novelty_node(state: EvaluatorState) -> EvaluatorState:
    document = state["document"]
    contribution_claims = [
        claim for claim in state.get("claims", [])
        if claim.claim_type == "contribution"
    ][:3]
    novelty_evidence: list[Any] = []
    for claim in contribution_claims:
        query_plan = build_query_plan(claim, paper_title=document.title)
        ranked = rank_evidence(
            claim,
            _RETRIEVER.retrieve_for_plan(query_plan),
            top_k=3,
        )
        novelty_evidence = _merge_evidence(novelty_evidence, ranked)
    return {
        "novelty": run_novelty_agent(document, contribution_claims, novelty_evidence),
        "novelty_evidence": novelty_evidence,
        "evidence_catalog": _merge_evidence(state.get("evidence_catalog", []), novelty_evidence),
    }


def fabrication_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "fabrication": run_credibility_agent(
            state.get("claims", []),
            state.get("claim_checks", []),
            state["consistency"],
        )
    }


def report_builder_node(state: EvaluatorState) -> EvaluatorState:
    report = build_report(
        state["document"],
        state.get("claims", []),
        state["consistency"],
        state["grammar"],
        state["novelty"],
        state["fabrication"],
        state.get("claim_checks", []),
    )
    report_path = save_report(
        report=report,
        claims=state.get("claims", []),
        claim_checks=state.get("claim_checks", []),
        consistency=state["consistency"],
        grammar=state["grammar"],
        novelty=state["novelty"],
        fabrication=state["fabrication"],
        paper_id=state["paper_id"],
    )
    return {"report": report, "report_path": str(report_path)}


def build_graph():
    workflow = StateGraph(EvaluatorState)

    workflow.add_node("ingest_paper", ingest_paper_node)
    workflow.add_node("extract_claims", claim_extraction_node)
    workflow.add_node("retrieve_evidence", evidence_node)
    workflow.add_node("evaluate_consistency", consistency_node)
    workflow.add_node("evaluate_grammar", grammar_node)
    workflow.add_node("evaluate_novelty", novelty_node)
    workflow.add_node("evaluate_fabrication", fabrication_node)
    workflow.add_node("build_report", report_builder_node)

    workflow.add_edge(START, "ingest_paper")
    workflow.add_edge("ingest_paper", "extract_claims")
    workflow.add_edge("extract_claims", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "evaluate_consistency")
    workflow.add_edge("evaluate_consistency", "evaluate_grammar")
    workflow.add_edge("evaluate_grammar", "evaluate_novelty")
    workflow.add_edge("evaluate_novelty", "evaluate_fabrication")
    workflow.add_edge("evaluate_fabrication", "build_report")
    workflow.add_edge("build_report", END)

    return workflow.compile()


GRAPH = build_graph()


def run_pipeline(source: str) -> tuple:
    final_state = GRAPH.invoke({"source": source})
    return final_state["report"], {
        "document": final_state["document"],
        "claims": final_state.get("claims", []),
        "claim_checks": final_state.get("claim_checks", []),
        "consistency": final_state["consistency"],
        "grammar": final_state["grammar"],
        "novelty": final_state["novelty"],
        "fabrication": final_state["fabrication"],
        "evidence_catalog": final_state.get("evidence_catalog", []),
        "retrieval_log": final_state.get("retrieval_log", []),
        "report_path": final_state["report_path"],
        "paper_id": final_state["paper_id"],
        "graph_mode": "langgraph-grounded-evaluator",
    }
