from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pipeline import run_pipeline
from utils.cache_manager import clear_runtime_cache


load_dotenv()


def claims_to_df(claims: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": claim.claim_id,
                "Claim": claim.text,
                "Type": claim.claim_type,
                "Citations": ", ".join(claim.cited_refs),
                "Confidence": claim.confidence,
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
            with st.spinner("Running 6 agents... this usually takes about 60-90 seconds."):
                report, details = run_pipeline(url)

            colour = {"Pass": "green", "Borderline": "orange", "Fail": "red"}[report.recommendation]
            st.markdown(f"## Recommendation: :{colour}[{report.recommendation}]")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Consistency", f"{report.consistency_score}/100")
            col2.metric("Grammar", report.grammar_rating)
            col3.metric("Citation", f"{report.citation_score}/100")
            col4.metric("Novelty", report.novelty_rating)
            col5.metric("Fabrication Risk", report.fabrication_probability)

            st.subheader("Executive Summary")
            st.write(report.summary)
            st.caption(f"Orchestration: {details['graph_mode']}")
            if details.get("planner_notes"):
                st.caption(f"Planner notes: {details['planner_notes']}")

            st.subheader("Claims Extracted")
            st.dataframe(claims_to_df(details["claims"]), use_container_width=True)

            st.subheader("Fact Check Log")
            st.dataframe(factcheck_to_df(details["factcheck"]), use_container_width=True)

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
                    st.write(details["critic"].get("reasoning", ""))
                    render_bullet_list(details["critic"].get("adjustments", []), "No additional calibration adjustments were suggested.")

            report_path = Path(details["report_path"])
            st.success(f"Saved Markdown report to {report_path}")
            st.download_button(
                "Download Report (Markdown)",
                data=report_path.read_text(encoding="utf-8"),
                file_name="report.md",
                mime="text/markdown",
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            st.info("Tip: use a standard arXiv abstract URL such as https://arxiv.org/abs/1706.03762")
