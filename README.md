# Grounded Paper Evaluator

This repo now implements a smaller LangGraph-based research-paper evaluator focused on assignment-required outputs and evidence-backed reasoning.

## What It Does

Given either:

- an arXiv URL such as `https://arxiv.org/abs/1706.03762`
- a local PDF path such as `/absolute/path/to/paper.pdf`

the app will:

1. parse the PDF into lightweight sections
2. build token-bounded rolling windows below the 16k call limit
3. extract grounded claims with a reduced taxonomy:
   - `contribution`
   - `result`
   - `comparison`
   - `factual`
   - `method`
4. retrieve scholarly-first external evidence for checkable claims
5. verify claims against paper-local context plus retrieved evidence
6. compute:
   - `Consistency Score (0-100)`
   - `Grammar Rating (High/Medium/Low)`
   - `Novelty Rating (High/Moderate/Low)`
   - `Fabrication Risk (0-100)`
7. write a Markdown judgement report under `reports/<paper_id>/report.md`

## Current Graph

The active LangGraph flow in [`pipeline.py`](/Users/abhiramkamini/Downloads/hivel assignment /pipeline.py) is:

`input -> ingest -> extract_claims -> retrieve_evidence -> evaluate_consistency -> evaluate_grammar -> evaluate_novelty -> evaluate_fabrication -> build_report`

Removed from the active architecture:

- planner agent
- evidence-prep packets
- citation score as a standalone dimension
- reviewer-critic layer
- legacy fact-check fallback path
- prompt-file-driven scoring stack

## Key Design Choices

- Section splitting is only a structural hint. The real context budget control comes from rolling token windows in [`chunker/token_chunker.py`](/Users/abhiramkamini/Downloads/hivel assignment /chunker/token_chunker.py).
- Numeric scores are deterministic and grounded in explicit signals, not direct LLM scoring.
- LLMs are used for:
  - claim extraction
  - evidence verdict synthesis
  - grammar review
  - novelty synthesis
- Retrieval is scholarly-first and falls back to general web only when needed.

## Main Files

- [`pipeline.py`](/Users/abhiramkamini/Downloads/hivel assignment /pipeline.py): LangGraph orchestration
- [`app.py`](/Users/abhiramkamini/Downloads/hivel assignment /app.py): Streamlit UI
- [`claims/claim_extractor.py`](/Users/abhiramkamini/Downloads/hivel assignment /claims/claim_extractor.py): bounded claim extraction
- [`retrieval/retriever.py`](/Users/abhiramkamini/Downloads/hivel assignment /retrieval/retriever.py): scholarly-first retrieval
- [`verification/verifier_agent.py`](/Users/abhiramkamini/Downloads/hivel assignment /verification/verifier_agent.py): evidence-backed claim verification
- [`agents/consistency_agent.py`](/Users/abhiramkamini/Downloads/hivel assignment /agents/consistency_agent.py): grounded consistency scoring
- [`agents/novelty_agent.py`](/Users/abhiramkamini/Downloads/hivel assignment /agents/novelty_agent.py): literature-backed novelty synthesis
- [`agents/credibility_agent.py`](/Users/abhiramkamini/Downloads/hivel assignment /agents/credibility_agent.py): fabrication risk synthesis
- [`reporter/report_writer.py`](/Users/abhiramkamini/Downloads/hivel assignment /reporter/report_writer.py): Markdown report generation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

The UI supports both arXiv URLs and local PDF paths.

## Notes

- If Gemini is unavailable or returns unusable structured output, the app falls back conservatively.
- If `duckduckgo-search` is unavailable in the environment, external retrieval is disabled and evidence-backed checks degrade conservatively.
- The repo still writes reports to `reports/` and clears runtime artifacts with the existing cache utility.
