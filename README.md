# arXiv Paper Evaluator

A multi-agent research paper reviewer for arXiv PDFs. It downloads a paper, parses it into sections, extracts claims, runs specialist review stages, retrieves external evidence for selected claims, and produces a Markdown judgement report through a Streamlit UI.

This is an inspectable MVP for paper auditing, not a full peer-review replacement. The focus is clear orchestration, bounded context, structured outputs, and explainable risk scoring.

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

Run the app:

```bash
streamlit run app.py
```

## What It Does

Given an arXiv URL like `https://arxiv.org/abs/1706.03762`, the system:

1. downloads and parses the PDF
2. normalizes sections and chunks long content
3. extracts high-signal claims
4. runs review stages for consistency, grammar, citation grounding, fact-checking, novelty, and credibility
5. retrieves and ranks external evidence for selected claim types
6. verifies claims with bounded paper context plus retrieved evidence
7. writes a Markdown report to `reports/{paper_id}/report.md`

## Pipeline

High-level flow in [pipeline.py](/Users/abhiramkamini/Downloads/hivel assignment /pipeline.py):

`ingest -> planner -> evidence prep -> claim extraction -> consistency -> grammar -> citation -> evidence retrieval -> factcheck -> novelty -> reviewer critic -> credibility -> report`

Key components:

- [claims/claim_extractor.py](claims/claim_extractor.py): extracts typed claims with confidence, importance, and chunk provenance
- [retrieval/retriever.py](retrieval/retriever.py): retrieves external web evidence
- [retrieval/query_builder.py](retrieval/query_builder.py): builds claim-focused search queries
- [retrieval/evidence_ranker.py](retrieval/evidence_ranker.py): ranks evidence for verifier use
- [verification/verifier_agent.py](verification/verifier_agent.py): assigns claim verdicts from paper context plus external evidence
- [scoring/credibility_model.py](scoring/credibility_model.py): computes an interpretable credibility risk breakdown
- [reporter/report_writer.py](reporter/report_writer.py): writes the final Markdown report

## Output

The generated report includes:

- executive summary
- extracted claims
- evidence-backed fact-check log
- consistency, grammar, citation, and novelty analysis
- credibility risk breakdown
- final recommendation

The Streamlit UI in [app.py](/Users/abhiramkamini/Downloads/hivel assignment /app.py) lets you run the pipeline, inspect intermediate outputs, and download the report.

## Design Choices

- Sequential LangGraph orchestration for predictability and easy debugging
- Claim-first evaluation instead of one-shot paper scoring
- Bounded context and chunking to stay within prompt limits
- Pydantic schemas for structured intermediate outputs
- Heuristic fallbacks when LLM calls fail or quota is exhausted

## Current Limitations

- External fact-checking is retrieval-assisted, but still snippet-based rather than full scholarly source ingestion
- Novelty is only partially literature-grounded; it is still strongest at evaluating the paper's own novelty argument
- PDF parsing and reference extraction are heuristic and may struggle on unusual layouts or citation styles
- Standard modern arXiv URLs are supported best

## Good Demo Inputs

- `https://arxiv.org/abs/1706.03762`
- `https://arxiv.org/abs/1409.0473`

## Tech Stack

- Python
- Streamlit
- LangGraph
- Gemini via `langchain-google-genai`
- PyMuPDF
- Pydantic
- pandas
- tiktoken
