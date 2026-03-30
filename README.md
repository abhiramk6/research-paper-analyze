# arXiv Paper Evaluator

An agentic research-paper review app for arXiv PDFs. It downloads a paper, parses and normalizes sections, extracts claims, routes higher-value claims to external evidence retrieval, scores multiple review dimensions, and generates a Markdown report through a Streamlit UI.

This project is best understood as an inspectable paper-auditing pipeline, not a replacement for peer review. The repo is organized so each stage can be read, modified, and debugged independently.

## What The App Does

Given an arXiv URL such as `https://arxiv.org/abs/1706.03762`, the pipeline:

1. downloads the PDF
2. extracts title, abstract, sections, and references
3. normalizes section headings into canonical categories
4. chunks the paper into bounded contexts
5. extracts structured claims with type, confidence, citations, and importance
6. runs consistency, grammar, citation, fact-check, novelty, and credibility stages
7. retrieves external web evidence for selected claim types
8. verifies claims against paper-local context plus ranked evidence
9. writes a Markdown report to `reports/<paper_id>/report.md`

## Current Pipeline

The main orchestration lives in [`pipeline.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/pipeline.py) and runs this flow:

`ingest -> planner -> evidence prep -> claim extraction -> consistency -> grammar -> citation -> evidence retrieval -> factcheck -> novelty -> reviewer critic -> credibility -> report`

Important details from the current implementation:

- LangGraph drives the pipeline sequentially.
- A planner step chooses section groups for downstream reviewers.
- Claim routing decides whether a claim gets external retrieval, internal-only handling, or is skipped.
- External verification can retry once with a broader search query when evidence is weak for a high-importance claim.
- Credibility scoring is deterministic and formula-based, with the LLM used for narrative reasoning rather than raw numeric scoring.
- If Gemini quota is exhausted or an LLM call fails, the app falls back to heuristics where possible and surfaces that in the UI.

## UI

The Streamlit app is in [`app.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/app.py).

The UI lets you:

- run the evaluator from an arXiv URL
- inspect extracted claims
- inspect evidence-backed fact-check results and routing decisions
- inspect the credibility breakdown
- download the generated Markdown report
- clear cached runtime artifacts

## Repository Map

Core orchestration and entrypoints:

- [`app.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/app.py): Streamlit frontend
- [`pipeline.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/pipeline.py): LangGraph orchestration and stage wiring
- [`prompt_loader.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/prompt_loader.py): prompt file loader

Input and paper parsing:

- [`ingestion/downloader.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/ingestion/downloader.py): downloads arXiv PDFs
- [`parser/pdf_parser.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/parser/pdf_parser.py): extracts title, sections, and references from PDFs
- [`parser/section_normalizer.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/parser/section_normalizer.py): maps raw headings to canonical section categories
- [`chunker/token_chunker.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/chunker/token_chunker.py): token counting and section chunking

Claims and retrieval:

- [`claims/claim_extractor.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/claims/claim_extractor.py): typed claim extraction
- [`retrieval/query_builder.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/retrieval/query_builder.py): search query generation
- [`retrieval/retriever.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/retrieval/retriever.py): DuckDuckGo-backed retrieval with stub fallback
- [`retrieval/evidence_ranker.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/retrieval/evidence_ranker.py): evidence ranking before verification

Verification and scoring:

- [`verification/claim_router.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/verification/claim_router.py): routes claims to retrieval or skip paths
- [`verification/verifier_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/verification/verifier_agent.py): evidence-backed claim verification
- [`verification/critic_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/verification/critic_agent.py): retry decision and broader retry query generation
- [`agents/credibility_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/credibility_agent.py): final credibility synthesis
- [`scoring/credibility_model.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/scoring/credibility_model.py): interpretable credibility breakdown

Specialist review agents:

- [`agents/consistency_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/consistency_agent.py)
- [`agents/grammar_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/grammar_agent.py)
- [`agents/citation_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/citation_agent.py)
- [`agents/factcheck_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/factcheck_agent.py)
- [`agents/novelty_agent.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/novelty_agent.py)
- [`agents/llm_client.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/llm_client.py): Gemini client, JSON parsing, and quota fallback handling

Reporting:

- [`aggregator/report_builder.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/aggregator/report_builder.py): builds the summary report object
- [`reporter/report_writer.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/reporter/report_writer.py): writes Markdown reports
- [`reports/test-case/report.md`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/reports/test-case/report.md): committed sample output artifact

Configuration and prompts:

- [`config/scoring.yaml`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/config/scoring.yaml): credibility weighting and thresholds
- [`config/retrieval.yaml`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/config/retrieval.yaml): retrieval-related config
- [`prompts/`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/prompts): prompt templates for planner and reviewer stages

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add environment variables

Create `.env` in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

Notes:

- The app expects Gemini through `langchain-google-genai`.
- If `GEMINI_API_KEY` is missing, LLM-backed stages will fail early.
- If quota is exhausted, some stages fall back to heuristics and the app shows a warning.

### 3. Run the app

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, paste an arXiv abstract URL, and run the evaluation.

## Dependencies

From [`requirements.txt`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/requirements.txt):

- `streamlit`
- `langgraph`
- `langchain-google-genai`
- `PyMuPDF`
- `tiktoken`
- `pydantic`
- `requests`
- `python-dotenv`
- `pandas`
- `duckduckgo-search`
- `pyyaml`

## Outputs

For each evaluated paper, the app writes a report under `reports/<paper_id>/report.md`.

The generated report includes:

- executive summary
- extracted claims
- consistency analysis
- grammar analysis
- citation analysis
- fact-check or evidence-backed verification log
- novelty assessment
- credibility and risk analysis
- final recommendation

The Streamlit UI also exposes intermediate outputs such as:

- planner notes
- evidence-backed claim verdicts
- routing logs
- credibility score breakdown

## Claim Routing And Verification

The current repo distinguishes between claims that should and should not trigger retrieval.

Claims like these are prioritized for external checking:

- `benchmark_result`
- `prior_work_comparison`
- `factual_background`
- selected legacy equivalents such as `result` and `factual`

Claims like methodology descriptions are generally kept out of web retrieval and handled through internal analysis instead.

Verification uses:

- the claim itself
- a bounded excerpt from the source paper
- ranked external evidence snippets

If the first verification pass comes back as weak evidence for a high-importance claim, the pipeline can issue one broader retry query and prefer the retry only when it produces a stronger verdict.

## Caching And Cleanup

The UI’s Clear Cache button calls [`utils/cache_manager.py`](/Users/abhiramkamini/Downloads/hivel%20assignment%20/utils/cache_manager.py), which removes:

- Python `__pycache__` directories in the repo
- generated report folders under `reports/` except `reports/test-case`
- temporary PDF files in `/tmp`

## Supported Input

Best supported input format:

- standard arXiv abstract URLs such as `https://arxiv.org/abs/1706.03762`

Good demo inputs:

- `https://arxiv.org/abs/1706.03762`
- `https://arxiv.org/abs/1409.0473`

## Known Limitations

- PDF parsing is heuristic and can struggle with unusual layouts, section formatting, or reference styles.
- External retrieval is snippet-based web evidence, not full scholarly corpus ingestion.
- Novelty assessment is still weaker than a true literature review pipeline.
- Routing and verification are tuned for research-paper style claims, not arbitrary documents.
- There is no formal automated test suite in the repo right now.

## Design Notes

- Structured Pydantic models are used for key intermediate objects.
- Prompt files are stored separately under `prompts/`.
- Token budgets are enforced throughout chunking and evidence preparation to keep prompts bounded.
- The system favors readable, inspectable pipeline stages over opaque end-to-end scoring.

## If You Want To Extend It

Common next steps for this repo would be:

- add proper automated tests for parsing, routing, and scoring
- add stronger scholarly retrieval sources beyond DuckDuckGo snippets
- split large reporting/orchestration modules into smaller units
- expose a CLI entrypoint in addition to the Streamlit app
- add persistence for run history and intermediate artifacts
