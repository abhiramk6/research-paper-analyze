from __future__ import annotations

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.citation_agent import run_citation_agent
from agents.consistency_agent import run_consistency_agent
from agents.credibility_agent import run_credibility_agent
from agents.factcheck_agent import evidence_factcheck_to_legacy, run_factcheck_agent
from agents.grammar_agent import run_grammar_agent
from agents.llm_client import call_llm_json
from agents.novelty_agent import run_novelty_agent
from aggregator.report_builder import build_report
from chunker.token_chunker import build_all_chunks, chunk_all_sections, chunk_text, count_tokens
from claims.claim_extractor import extract_claims
from ingestion.downloader import download_pdf
from parser.pdf_parser import parse_paper
from parser.section_normalizer import normalize_section_headings
from prompt_loader import load_prompt
from reporter.report_writer import save_report
from retrieval.evidence_ranker import rank_evidence
from retrieval.query_builder import build_queries
from retrieval.retriever import EvidenceRetriever
from verification.claim_router import routing_decision
from verification.critic_agent import build_retry_query, should_retry
from verification.verifier_agent import verify_claim


load_dotenv()
logger = logging.getLogger(__name__)

_RETRIEVER = EvidenceRetriever()
_VERIFIER_TOP_K = 5


class EvaluatorState(TypedDict, total=False):
    url: str
    pdf_path: str
    document: Any
    planner: dict[str, Any]
    planner_notes: str
    evidence_packets: dict[str, str]
    claims: Any
    consistency: Any
    grammar: Any
    citation: Any
    factcheck: Any
    novelty: Any
    critic: dict[str, Any]
    credibility: Any
    report: Any
    report_path: str
    paper_id: str
    evidence_results: list[Any]
    routing_log: list[dict[str, Any]]


def _section_snapshot(document) -> list[dict[str, Any]]:
    return [
        {
            "heading": section.heading,
            "canonical_category": section.canonical_category or "unknown",
            "token_count": section.token_count,
            "preview": section.content[:500],
        }
        for section in document.sections
    ]


def _section_matches(section_heading: str, keywords: list[str], canonical: str | None = None) -> bool:
    kw_lower = [k.lower() for k in keywords]
    if canonical:
        cat = canonical.lower()
        if any(k in cat or cat in k for k in kw_lower):
            return True
    return any(k in section_heading.lower() for k in kw_lower)


def _bounded_join(parts: list[str], max_tokens: int = 12_000) -> str:
    selected: list[str] = []
    for part in parts:
        candidate = "\n\n".join(selected + [part]).strip()
        if selected and count_tokens(candidate) > max_tokens:
            break
        selected.append(part)
    return "\n\n".join(selected).strip()


def _packet_from_keywords(document, keywords: list[str], fallback_headings: list[str], max_tokens: int) -> str:
    selected = [
        f"[{section.heading}]\n{section.content.strip()}"
        for section in document.sections
        if _section_matches(section.heading, keywords, canonical=section.canonical_category)
    ]
    if not selected:
        selected = [
            f"[{section.heading}]\n{section.content.strip()}"
            for section in document.sections
            if (
                section.heading in fallback_headings
                or (section.canonical_category and section.canonical_category in
                    [h.lower() for h in fallback_headings])
            )
        ]
    bounded_parts: list[str] = []
    for section_text in selected:
        bounded_parts.extend(chunk_text(section_text, max_tokens=min(max_tokens, 4_000)))
    return _bounded_join(bounded_parts, max_tokens=max_tokens)


def _paper_context_for_claim(document, claim) -> str:
    target_heading = claim.source_section or ""
    for section in document.sections:
        if target_heading.lower() in section.heading.lower() or section.heading.lower() in target_heading.lower():
            return section.content[:1_200]
    return document.abstract[:800]

def ingest_paper_node(state: EvaluatorState) -> EvaluatorState:
    pdf_path = download_pdf(state["url"])
    document = parse_paper(pdf_path, source_url=state["url"])
    document = chunk_all_sections(document)
    document = normalize_section_headings(document)
    chunks = build_all_chunks(document)
    document = document.model_copy(update={"chunks": chunks})
    return {
        "pdf_path": str(pdf_path),
        "document": document,
        "paper_id": document.paper_id,
    }


def planner_agent_node(state: EvaluatorState) -> EvaluatorState:
    document = state["document"]
    schema = {
        "paper_quality_hint": "strong|mixed|weak",
        "consistency_sections": ["string"],
        "grammar_sections": ["string"],
        "citation_mode": "claims_and_references",
        "factcheck_sections": ["string"],
        "novelty_sections": ["string"],
        "planner_notes": "string",
    }
    fallback = {
        "paper_quality_hint": "mixed",
        "consistency_sections": ["method", "architecture", "training", "results", "evaluation", "analysis"],
        "grammar_sections": ["abstract", "introduction", "conclusion"],
        "citation_mode": "claims_and_references",
        "factcheck_sections": ["results", "evaluation", "analysis"],
        "novelty_sections": ["abstract", "background", "related work", "introduction"],
        "planner_notes": (
            "Evidence-backed agentic pipeline: chunks → claim extraction → "
            "routing → external retrieval → verification (with retry) → "
            "interpretable credibility scoring. "
            "All LLM inputs are bounded to stay under 16k tokens."
        ),
    }
    prompt = load_prompt(
        "planner_agent.txt",
        schema_json=json.dumps(schema),
        paper_metadata_json=json.dumps(
            {
                "title": document.title,
                "abstract": document.abstract[:1200],
                "reference_count": len(document.references),
                "section_count": len(document.sections),
                "chunk_count": len(document.chunks),
            },
            indent=2,
        ),
        sections_json=json.dumps(_section_snapshot(document), indent=2),
    )
    planner = call_llm_json(prompt, fallback=fallback)
    return {
        "planner": planner,
        "planner_notes": str(planner.get("planner_notes", fallback["planner_notes"])),
    }


def evidence_prep_node(state: EvaluatorState) -> EvaluatorState:
    document = state["document"]
    planner = state["planner"]
    packets = {
        "consistency": _packet_from_keywords(
            document,
            [str(item) for item in planner.get("consistency_sections", [])],
            fallback_headings=["Model Architecture", "Training", "Results", "Evaluation"],
            max_tokens=11_000,
        ),
        "grammar": _packet_from_keywords(
            document,
            [str(item) for item in planner.get("grammar_sections", [])],
            fallback_headings=["Abstract", "Introduction", "Conclusion"],
            max_tokens=8_000,
        ),
        "factcheck": _packet_from_keywords(
            document,
            [str(item) for item in planner.get("factcheck_sections", [])],
            fallback_headings=["Results", "Evaluation", "Analysis"],
            max_tokens=9_000,
        ),
        "novelty": _packet_from_keywords(
            document,
            [str(item) for item in planner.get("novelty_sections", [])],
            fallback_headings=["Abstract", "Background", "Introduction"],
            max_tokens=10_000,
        ),
    }
    return {"evidence_packets": packets}


def claim_extraction_node(state: EvaluatorState) -> EvaluatorState:
    claims = extract_claims(state["document"])
    return {"claims": claims}


def consistency_expert_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "consistency": run_consistency_agent(
            state["document"],
            state["claims"],
            prepared_context=state.get("evidence_packets", {}).get("consistency", ""),
        )
    }


def grammar_expert_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "grammar": run_grammar_agent(
            state["document"],
            prepared_context=state.get("evidence_packets", {}).get("grammar", ""),
        )
    }


def citation_expert_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "citation": run_citation_agent(state["claims"], state["document"].references),
    }


def evidence_retrieval_node(state: EvaluatorState) -> EvaluatorState:
    claims = state.get("claims", [])
    document = state["document"]
    paper_title = document.title

    evidence_results = []
    routing_log = []

    for claim in claims:
        decision = routing_decision(claim)
        log_entry: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type,
            "importance": claim.importance,
            "routing_decision": decision,
        }

        if decision != "external_factcheck":
            log_entry["action"] = "skipped"
            routing_log.append(log_entry)
            continue

        queries = build_queries(claim, paper_title=paper_title)
        log_entry["queries"] = queries

        raw_evidence = _RETRIEVER.retrieve_for_queries(
            queries,
            max_per_query=5,
            global_cap=10,
        )
        ranked_evidence = rank_evidence(claim, raw_evidence, top_k=_VERIFIER_TOP_K)

        paper_context = _paper_context_for_claim(document, claim)

        result = verify_claim(claim, paper_context=paper_context, evidence_items=ranked_evidence)
        log_entry["verdict_pass1"] = result.verdict

        if should_retry(result, claim):
            retry_query = build_retry_query(claim, paper_title=paper_title)
            log_entry["retry_query"] = retry_query
            retry_raw = _RETRIEVER.retrieve_for_queries(
                [retry_query],
                max_per_query=5,
                global_cap=8,
            )
            retry_ranked = rank_evidence(claim, retry_raw, top_k=_VERIFIER_TOP_K)
            if retry_ranked:
                retry_result = verify_claim(
                    claim,
                    paper_context=paper_context,
                    evidence_items=retry_ranked,
                )
                log_entry["verdict_pass2"] = retry_result.verdict
                _WEAK = {"insufficient_evidence", "paper_supported_only"}
                if result.verdict in _WEAK and retry_result.verdict not in _WEAK:
                    result = retry_result
                    log_entry["used_retry"] = True
                else:
                    log_entry["used_retry"] = False
            else:
                log_entry["retry_found_evidence"] = False

        evidence_results.append(result)
        log_entry["final_verdict"] = result.verdict
        log_entry["action"] = "verified"
        routing_log.append(log_entry)

        logger.info(
            "Claim %s (%s / %s): %s → %s",
            claim.claim_id,
            claim.claim_type,
            claim.importance,
            decision,
            result.verdict,
        )

    return {"evidence_results": evidence_results, "routing_log": routing_log}


def factcheck_expert_node(state: EvaluatorState) -> EvaluatorState:
    evidence_results = state.get("evidence_results") or []
    if evidence_results:
        factcheck = evidence_factcheck_to_legacy(evidence_results)
    else:
        factcheck = run_factcheck_agent(
            state["claims"],
            prepared_context=state.get("evidence_packets", {}).get("factcheck", ""),
        )
    return {"factcheck": factcheck}


def novelty_expert_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "novelty": run_novelty_agent(
            state["document"],
            state["claims"],
            prepared_context=state.get("evidence_packets", {}).get("novelty", ""),
        )
    }


def reviewer_critic_node(state: EvaluatorState) -> EvaluatorState:
    schema = {
        "calibration_notes": ["string"],
        "summary_assessment": "string",
    }
    fallback = {
        "calibration_notes": ["All specialist signals appear well-calibrated."],
        "summary_assessment": (
            "Reviewer-critic fallback used. "
            "No calibration concerns identified beyond existing guardrails."
        ),
    }
    routing_log = state.get("routing_log", [])
    evidence_results = state.get("evidence_results", [])

    prompt = load_prompt(
        "reviewer_critic.txt",
        schema_json=json.dumps(schema),
        signals_json=json.dumps(
            {
                "planner": state.get("planner", {}),
                "consistency": state["consistency"].model_dump(),
                "grammar": state["grammar"].model_dump(),
                "citation": state["citation"].model_dump(),
                "factcheck": state["factcheck"].model_dump(),
                "novelty": state["novelty"].model_dump(),
                "claim_count": len(state.get("claims", [])),
                "evidence_checked_claim_count": len(evidence_results),
                "routing_summary": [
                    {"claim_id": e["claim_id"], "decision": e["routing_decision"], "verdict": e.get("final_verdict")}
                    for e in routing_log
                ],
            },
            indent=2,
        ),
    )
    return {"critic": call_llm_json(prompt, fallback=fallback)}


def credibility_synthesis_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "credibility": run_credibility_agent(
            state["consistency"],
            state["grammar"],
            state["citation"],
            state["factcheck"],
            state["novelty"],
            critic_feedback=state.get("critic", {}),
            evidence_results=state.get("evidence_results") or None,
            claims=state.get("claims") or None,
        )
    }


def report_builder_node(state: EvaluatorState) -> EvaluatorState:
    report = build_report(
        state["document"],
        state["claims"],
        state["consistency"],
        state["grammar"],
        state["citation"],
        state["factcheck"],
        state["novelty"],
        state["credibility"],
    )
    report_path = save_report(
        report,
        state["claims"],
        state["consistency"],
        state["citation"],
        state["factcheck"],
        state["novelty"],
        state["credibility"],
        state["paper_id"],
        grammar=state["grammar"],
        evidence_results=state.get("evidence_results") or None,
    )
    return {"report": report, "report_path": str(report_path)}


def build_graph():
    workflow = StateGraph(EvaluatorState)

    workflow.add_node("ingest_paper", ingest_paper_node)
    workflow.add_node("planner_agent", planner_agent_node)
    workflow.add_node("evidence_prep_agent", evidence_prep_node)
    workflow.add_node("extract_claims", claim_extraction_node)
    workflow.add_node("consistency_expert", consistency_expert_node)
    workflow.add_node("grammar_expert", grammar_expert_node)
    workflow.add_node("citation_expert", citation_expert_node)
    workflow.add_node("evidence_retrieval", evidence_retrieval_node)
    workflow.add_node("factcheck_expert", factcheck_expert_node)
    workflow.add_node("novelty_expert", novelty_expert_node)
    workflow.add_node("reviewer_critic", reviewer_critic_node)
    workflow.add_node("credibility_synthesis", credibility_synthesis_node)
    workflow.add_node("build_report", report_builder_node)

    workflow.add_edge(START, "ingest_paper")
    workflow.add_edge("ingest_paper", "planner_agent")
    workflow.add_edge("planner_agent", "evidence_prep_agent")
    workflow.add_edge("evidence_prep_agent", "extract_claims")
    workflow.add_edge("extract_claims", "consistency_expert")
    workflow.add_edge("consistency_expert", "grammar_expert")
    workflow.add_edge("grammar_expert", "citation_expert")
    workflow.add_edge("citation_expert", "evidence_retrieval")
    workflow.add_edge("evidence_retrieval", "factcheck_expert")
    workflow.add_edge("factcheck_expert", "novelty_expert")
    workflow.add_edge("novelty_expert", "reviewer_critic")
    workflow.add_edge("reviewer_critic", "credibility_synthesis")
    workflow.add_edge("credibility_synthesis", "build_report")
    workflow.add_edge("build_report", END)

    return workflow.compile()


GRAPH = build_graph()


def run_pipeline(url: str) -> tuple:
    final_state = GRAPH.invoke({"url": url})
    return final_state["report"], {
        "document": final_state["document"],
        "planner": final_state.get("planner", {}),
        "planner_notes": final_state.get("planner_notes", ""),
        "consistency": final_state["consistency"],
        "grammar": final_state["grammar"],
        "citation": final_state["citation"],
        "factcheck": final_state["factcheck"],
        "novelty": final_state["novelty"],
        "critic": final_state.get("critic", {}),
        "credibility": final_state["credibility"],
        "claims": final_state["claims"],
        "evidence_results": final_state.get("evidence_results", []),
        "routing_log": final_state.get("routing_log", []),
        "report_path": final_state["report_path"],
        "paper_id": final_state["paper_id"],
        "graph_mode": "langgraph-agentic-evidence-backed-pipeline-v2",
    }
