from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.llm_client import get_quota_error_message, is_quota_exhausted, reset_quota_flag
from pipeline import run_pipeline
from utils.cache_manager import clear_runtime_cache


load_dotenv()


def claims_to_df(claims: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": claim.claim_id,
                "Claim": claim.text[:140] + ("..." if len(claim.text) > 140 else ""),
                "Type": claim.claim_type,
                "Importance": claim.importance,
                "Confidence": claim.confidence,
                "Chunk": claim.source_chunk_id or "—",
            }
            for claim in claims
        ]
    )


def checks_to_df(claim_checks: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Claim ID": item.claim_id,
                "Verdict": item.verdict,
                "Confidence": f"{round(item.confidence * 100)}%",
                "Evidence IDs": ", ".join(item.matched_evidence_ids) or "—",
                "Reasoning": item.reasoning[:160] + ("..." if len(item.reasoning) > 160 else ""),
            }
            for item in claim_checks
        ]
    )


def evidence_to_df(evidence_items: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Evidence ID": item.evidence_id,
                "Query": item.query[:100],
                "Title": item.title[:100],
                "Tier": item.domain_tier,
                "Domain": item.domain,
                "Score": item.retrieval_score,
                "URL": item.url,
            }
            for item in evidence_items
        ]
    )


def retrieval_log_to_df(retrieval_log: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Claim ID": item.get("claim_id", ""),
                "Type": item.get("claim_type", ""),
                "Routing": item.get("routing_decision", ""),
                "Retrieved": item.get("retrieved_count", "—"),
                "Verdict": item.get("final_verdict", "—"),
            }
            for item in retrieval_log
        ]
    )


def render_bullets(items: list[str], empty: str) -> None:
    st.markdown("\n".join(f"- {item}" for item in (items or [empty])))


st.set_page_config(page_title="Grounded Paper Evaluator", layout="wide")
st.title("Grounded Paper Evaluator")

top_left, top_right = st.columns([4, 1])
with top_right:
    if st.button("Clear Cache"):
        removed = clear_runtime_cache()
        st.success(f"Cleared cache and runtime artifacts ({len(removed)} items removed).")

input_mode = st.radio("Input mode", ["arXiv URL", "Local PDF Path"], horizontal=True)
if input_mode == "arXiv URL":
    source = st.text_input("arXiv URL", placeholder="https://arxiv.org/abs/1706.03762")
else:
    source = st.text_input("Local PDF Path", placeholder="/absolute/path/to/paper.pdf")

if st.button("Evaluate Paper", type="primary"):
    if not source.strip():
        st.error("Enter an arXiv URL or local PDF path before running the evaluator.")
    else:
        try:
            reset_quota_flag()
            with st.spinner("Running grounded evaluation... this may take 60–120 seconds."):
                report, details = run_pipeline(source.strip())

            if is_quota_exhausted():
                st.warning(
                    "Gemini quota was exhausted during part of this run.\n\n"
                    f"Detail: {get_quota_error_message()}"
                )

            colour = {"Pass": "green", "Borderline": "orange", "Fail": "red"}[report.recommendation]
            st.markdown(f"## Recommendation: :{colour}[{report.recommendation}]")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Consistency", f"{report.consistency_score}/100")
            col2.metric("Grammar", report.grammar_rating)
            col3.metric("Novelty", report.novelty_rating)
            col4.metric("Fabrication Risk", report.fabrication_risk)

            st.subheader("Executive Summary")
            st.write(report.summary)
            st.caption(f"Orchestration: {details['graph_mode']}")

            st.subheader("Claims Extracted")
            st.dataframe(claims_to_df(details["claims"]), use_container_width=True)

            st.subheader("Fact Check Log")
            st.dataframe(checks_to_df(details["claim_checks"]), use_container_width=True)

            with st.expander("Retrieved Evidence"):
                evidence_items = details.get("evidence_catalog", [])
                if evidence_items:
                    st.dataframe(evidence_to_df(evidence_items), use_container_width=True)
                else:
                    st.write("No evidence items captured.")

            with st.expander("Retrieval Routing Log"):
                retrieval_log = details.get("retrieval_log", [])
                if retrieval_log:
                    st.dataframe(retrieval_log_to_df(retrieval_log), use_container_width=True)
                else:
                    st.write("No retrieval activity recorded.")

            with st.expander("Consistency Analysis"):
                st.write(details["consistency"].reasoning)
                render_bullets(details["consistency"].issues, "No major consistency issues were identified.")

            with st.expander("Grammar And Language"):
                st.write(details["grammar"].reasoning)
                render_bullets(details["grammar"].issues, "No major writing issues were identified.")

            with st.expander("Novelty Assessment"):
                st.write(details["novelty"].reasoning)

            with st.expander("Fabrication Risk Breakdown"):
                fabrication = details["fabrication"]
                breakdown = fabrication.breakdown
                st.metric("Risk Band", fabrication.risk_band.upper())
                st.write(fabrication.reasoning)
                st.json(breakdown.model_dump())
                render_bullets(fabrication.risk_factors, "No major grounded risk factors were identified.")

            report_path = Path(details["report_path"])
            st.success(f"Saved Markdown report to {report_path}")
            st.download_button(
                "Download Report (Markdown)",
                data=report_path.read_text(encoding="utf-8"),
                file_name="report.md",
                mime="text/markdown",
            )
        except Exception as exc:
            if is_quota_exhausted():
                st.error(
                    "Gemini quota was exhausted and the pipeline could not complete the run.\n\n"
                    f"{get_quota_error_message()}"
                )
            else:
                st.error(f"Evaluation failed: {exc}")
