from __future__ import annotations

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
                "Claim": claim.text[:120] + ("..." if len(claim.text) > 120 else ""),
                "Type": claim.claim_type,
                "Importance": getattr(claim, "importance", "—"),
                "Citations": ", ".join(claim.cited_refs),
                "Confidence": claim.confidence,
                "Chunk": getattr(claim, "source_chunk_id", "—") or "—",
            }
            for claim in claims
        ]
    )


def factcheck_to_df(factcheck) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Claim ID": item.claim_id,
                "Status": item.status,
                "Reasoning": item.reasoning,
            }
            for item in factcheck.items
        ]
    )


def evidence_results_to_df(evidence_results: list) -> pd.DataFrame:
    rows = []
    for r in evidence_results:
        evidence_titles = "; ".join(e.title[:60] for e in r.evidence_items) if r.evidence_items else "none"
        rows.append(
            {
                "Claim ID": r.claim_id,
                "Verdict": r.verdict,
                "Confidence": f"{round(r.confidence * 100)}%",
                "Evidence Sources": evidence_titles,
                "Reasoning": r.reasoning[:150] + ("..." if len(r.reasoning) > 150 else ""),
            }
        )
    return pd.DataFrame(rows)


def routing_log_to_df(routing_log: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Claim ID": e.get("claim_id", ""),
                "Type": e.get("claim_type", ""),
                "Importance": e.get("importance", ""),
                "Routing": e.get("routing_decision", ""),
                "Action": e.get("action", ""),
                "Final Verdict": e.get("final_verdict", "—"),
                "Retried": "yes" if e.get("used_retry") else ("—" if "used_retry" not in e else "no"),
            }
            for e in routing_log
        ]
    )


def render_bullet_list(items: list[str], empty_message: str) -> None:
    values = items or [empty_message]
    st.markdown("\n".join(f"- {item}" for item in values))


st.set_page_config(page_title="arXiv Paper Evaluator", layout="wide")
st.title("arXiv Paper Evaluator")

top_left, top_right = st.columns([4, 1])
with top_right:
    if st.button("Clear Cache"):
        removed = clear_runtime_cache()
        st.success(f"Cleared cache and runtime artifacts ({len(removed)} items removed).")

url = st.text_input("arXiv URL", placeholder="https://arxiv.org/abs/1706.03762")

if st.button("Evaluate Paper", type="primary"):
    if not url.strip():
        st.error("Enter an arXiv URL before running the evaluator.")
    else:
        try:
            reset_quota_flag()
            with st.spinner("Running agents... this usually takes 60–120 seconds."):
                report, details = run_pipeline(url)

            if is_quota_exhausted():
                st.warning(
                    "⚠️ **Gemini API quota exhausted during this run.**\n\n"
                    "One or more agents fell back to heuristics instead of LLM reasoning. "
                    "Results may be less accurate than usual.\n\n"
                    f"**Detail:** {get_quota_error_message()}\n\n"
                    "To fix: wait for your free-tier quota to reset (usually midnight Pacific), "
                    "or add a paid API key to your `.env` file."
                )

            colour = {"Pass": "green", "Borderline": "orange", "Fail": "red"}[report.recommendation]
            st.markdown(f"## Recommendation: :{colour}[{report.recommendation}]")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Consistency", f"{report.consistency_score}/100")
            col2.metric("Grammar", report.grammar_rating)
            col3.metric("Citation", f"{report.citation_score}/100")
            col4.metric("Novelty", report.novelty_rating)
            col5.metric("Credibility Risk", report.fabrication_probability)

            st.subheader("Executive Summary")
            st.write(report.summary)
            st.caption(f"Orchestration: {details['graph_mode']}")
            if details.get("planner_notes"):
                st.caption(f"Planner notes: {details['planner_notes']}")

            st.subheader("Claims Extracted")
            st.dataframe(claims_to_df(details["claims"]), use_container_width=True)

            evidence_results = details.get("evidence_results", [])
            if evidence_results:
                st.subheader("Evidence-Backed Fact-Check")
                st.dataframe(evidence_results_to_df(evidence_results), use_container_width=True)
                with st.expander("Routing Log (Agentic Decisions)"):
                    routing_log = details.get("routing_log", [])
                    if routing_log:
                        st.dataframe(routing_log_to_df(routing_log), use_container_width=True)
                    else:
                        st.write("No routing log available.")
            else:
                st.subheader("Fact Check Log")
                st.dataframe(factcheck_to_df(details["factcheck"]), use_container_width=True)

            breakdown = getattr(details["credibility"], "breakdown", None)
            if breakdown:
                with st.expander("Credibility Score Breakdown (Interpretable)"):
                    b = breakdown
                    bcol1, bcol2, bcol3 = st.columns(3)
                    bcol1.metric("Supported Claim Ratio", f"{b.supported_claim_ratio:.1%}")
                    bcol2.metric("Contradicted Claim Ratio", f"{b.contradicted_claim_ratio:.1%}")
                    bcol3.metric("Insufficient Evidence Ratio", f"{b.insufficient_evidence_ratio:.1%}")
                    st.markdown(f"**Risk Band:** {b.risk_band.upper()}  |  **Final Score:** {b.final_score:.1f}/100")
                    st.write(b.explanation)

            with st.expander("Consistency Analysis"):
                st.write(details["consistency"].reasoning)
                render_bullet_list(details["consistency"].issues, "No major internal consistency issues were flagged.")

            with st.expander("Novelty Assessment"):
                st.write(details["novelty"].reasoning)

            with st.expander("Credibility Risk Factors"):
                render_bullet_list(details["credibility"].risk_factors, "No major risk factors identified.")
                st.write(details["credibility"].reasoning)

            if details.get("critic"):
                with st.expander("Reviewer-Critic"):
                    st.write(details["critic"].get("summary_assessment", details["critic"].get("reasoning", "")))
                    render_bullet_list(details["critic"].get("calibration_notes", []), "No additional calibration adjustments were suggested.")

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
                    "**Gemini API quota exhausted — pipeline could not complete.**\n\n"
                    f"{get_quota_error_message()}\n\n"
                    "Wait for your free-tier quota to reset or add a paid key to `.env`."
                )
            else:
                st.error(f"Evaluation failed: {exc}")
                st.info("Tip: use a standard arXiv abstract URL such as https://arxiv.org/abs/1706.03762")
