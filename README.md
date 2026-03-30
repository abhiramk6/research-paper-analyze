# Grounded Paper Evaluator

This project is a LangGraph-based research-paper evaluator that audits a paper using bounded claim extraction, external evidence retrieval, and grounded scoring.

## Overview

The app accepts either:

- an arXiv URL such as `https://arxiv.org/abs/1706.03762`
- a local PDF path

It then:

1. parses the paper into lightweight sections
2. builds token-bounded rolling windows below the 16k call limit
3. extracts grounded claims using a compact operational taxonomy
4. retrieves scholarly-first external evidence for checkable claims
5. verifies claims against paper-local context plus retrieved evidence
6. produces a Markdown judgement report

## Features

- LangGraph orchestration for a traceable multi-step evaluation pipeline
- Support for both arXiv URLs and local PDF inputs
- Rolling token-window chunking with overlap for safe LLM context management
- Reduced claim taxonomy focused on evaluation behavior:
  - `contribution`
  - `result`
  - `comparison`
  - `factual`
  - `method`
- Scholarly-first evidence retrieval with fallback to broader web search
- Evidence-backed claim verification
- Grounded outputs for:
  - `Consistency Score (0-100)`
  - `Grammar Rating (High/Medium/Low)`
  - `Novelty Rating (High/Moderate/Low)`
  - `Fabrication Risk (0-100)`
- Markdown report generation under `reports/<paper_id>/report.md`
- Streamlit UI for interactive evaluation and report download

## Pipeline

The active flow is:

`input -> ingest -> extract_claims -> retrieve_evidence -> evaluate_consistency -> evaluate_grammar -> evaluate_novelty -> evaluate_fabrication -> build_report`

## Project Structure

- `pipeline.py`: LangGraph orchestration
- `app.py`: Streamlit UI
- `chunker/token_chunker.py`: rolling token windows and token accounting
- `claims/claim_extractor.py`: bounded claim extraction
- `retrieval/retriever.py`: scholarly-first retrieval
- `verification/verifier_agent.py`: evidence-backed claim verification
- `agents/consistency_agent.py`: grounded consistency scoring
- `agents/grammar_agent.py`: grammar and writing review
- `agents/novelty_agent.py`: literature-backed novelty synthesis
- `agents/credibility_agent.py`: fabrication risk synthesis
- `reporter/report_writer.py`: Markdown report generation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

## Notes

- If Gemini is unavailable or returns unusable structured output, the system falls back conservatively.
- If external retrieval is unavailable, evidence-backed checks degrade conservatively rather than inventing support.
- Reports are written to `reports/`, and runtime artifacts can be cleared from the UI.
