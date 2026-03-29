from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.citation_agent import run_citation_agent
from agents.consistency_agent import run_consistency_agent
from agents.credibility_agent import run_credibility_agent
from agents.factcheck_agent import run_factcheck_agent
from agents.llm_client import call_llm_json
from agents.grammar_agent import run_grammar_agent
from agents.novelty_agent import run_novelty_agent
from aggregator.report_builder import build_report
from chunker.token_chunker import chunk_all_sections, chunk_text, count_tokens
from claims.claim_extractor import extract_claims
from ingestion.downloader import download_pdf
from parser.pdf_parser import parse_paper
from prompt_loader import load_prompt
from reporter.report_writer import save_report


load_dotenv()


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


def _section_snapshot(document) -> list[dict[str, Any]]:
    return [
        {
            "heading": section.heading,
            "token_count": section.token_count,
            "preview": section.content[:500],
        }
        for section in document.sections
    ]


def _section_matches(section_heading: str, keywords: list[str]) -> bool:
    normalized_heading = section_heading.lower()
    return any(keyword.lower() in normalized_heading for keyword in keywords)


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
        if _section_matches(section.heading, keywords)
    ]
    if not selected:
        selected = [
            f"[{section.heading}]\n{section.content.strip()}"
            for section in document.sections
            if section.heading in fallback_headings
        ]

    bounded_parts: list[str] = []
    for section_text in selected:
        bounded_parts.extend(chunk_text(section_text, max_tokens=min(max_tokens, 4_000)))
    return _bounded_join(bounded_parts, max_tokens=max_tokens)


def ingest_paper_node(state: EvaluatorState) -> EvaluatorState:
    pdf_path = download_pdf(state["url"])
    document = parse_paper(pdf_path, source_url=state["url"])
    document = chunk_all_sections(document)
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
        "planner_notes": "Sequential LangGraph execution with bounded evidence packets to keep each downstream LLM call comfortably below the 16k token limit.",
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


def factcheck_expert_node(state: EvaluatorState) -> EvaluatorState:
    return {
        "factcheck": run_factcheck_agent(
            state["claims"],
            prepared_context=state.get("evidence_packets", {}).get("factcheck", ""),
        )
    }


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
        "summary_assessment": "Reviewer-critic fallback used. No calibration concerns identified beyond existing guardrails.",
    }
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
    workflow.add_edge("citation_expert", "factcheck_expert")
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
        "report_path": final_state["report_path"],
        "paper_id": final_state["paper_id"],
        "graph_mode": "langgraph-sequential-agent-pipeline",
    }
