# arXiv Paper Evaluator

A multi-agent research paper review system that takes an arXiv paper URL, downloads the PDF, parses the paper into structured sections, extracts high-signal claims, evaluates the paper with specialist reviewer agents, and generates a Markdown judgement report through a Streamlit UI.

This project is built as a practical, inspectable assignment submission: the emphasis is on modularity, clear orchestration, bounded LLM context, typed outputs, and reviewer-style reasoning rather than a single black-box summary call.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file with:

```bash
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser and evaluate an arXiv paper.

## What This Project Does

Given an arXiv URL such as `https://arxiv.org/abs/1706.03762`, the system:

1. downloads the paper PDF from arXiv
2. extracts the title, abstract, sections, and references
3. counts tokens and prepares bounded evidence windows
4. extracts important claims from the paper
5. runs 6 specialist evaluation agents
6. synthesizes their outputs into a final credibility and fabrication-risk style judgement
7. saves a detailed Markdown report to `reports/{paper_id}/report.md`

## Why This Repo Is Different

This repo is not just "LLM summarization over a PDF." It has a few design choices that make it more robust and easier to explain:

- **Claim-centric evaluation**
  The system does not ask the model to rate a paper directly from raw text. It first extracts explicit claims and then evaluates those claims across multiple reviewer dimensions.

- **Planner-driven evidence routing**
  Before the specialist agents run, a planner agent looks at the paper structure and suggests which sections matter most for consistency, grammar, novelty, and fact-checking.

- **Bounded evidence packets**
  Instead of passing the full paper into every downstream prompt, the pipeline builds compact context packets tailored to each agent. This keeps token usage controlled and makes each review more focused.

- **Long-section summarization before review**
  If a section is too long, the system summarizes chunks first and then sends the condensed version to the downstream expert. This lets the repo handle longer papers while staying inside safe context limits.

- **Typed outputs throughout**
  The pipeline uses Pydantic models as a shared contract across parsing, claim extraction, agent evaluation, and reporting.

- **Graceful fallback behavior**
  If the LLM fails, returns empty content, or produces invalid JSON, several modules fall back to local heuristic scoring or safe defaults rather than crashing immediately.

- **Sequential LangGraph orchestration**
  LangGraph is used, but intentionally in a sequential, easy-to-debug pattern. That makes the flow explicit and presentation-friendly.

## Tech Stack

- Python 3.10+
- Streamlit
- LangGraph
- LangChain Google GenAI
- Gemini via `langchain-google-genai`
- PyMuPDF
- tiktoken
- Pydantic
- pandas

Current LLM configuration is defined in [agents/llm_client.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/llm_client.py) and uses:

- `gemini-3.1-flash-lite-preview`

## High-Level Pipeline

The implemented flow in [pipeline.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/pipeline.py) is:

`URL -> ingest -> planner -> evidence prep -> claim extraction -> consistency -> grammar -> citation -> factcheck -> novelty -> reviewer critic -> credibility synthesis -> report builder`

### Step-by-step

1. **Ingestion**
   [ingestion/downloader.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/ingestion/downloader.py) validates the arXiv URL, converts `/abs/` to `/pdf/`, downloads the PDF, and stores it temporarily.

2. **Parsing**
   [parser/pdf_parser.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/parser/pdf_parser.py) uses PyMuPDF to extract:
   - title
   - abstract
   - sectioned body text
   - references

   Section splitting is heuristic and based on heading detection.

3. **Token accounting**
   [chunker/token_chunker.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/chunker/token_chunker.py) counts tokens with `tiktoken` and prepares safe chunks for long text.

4. **Planner agent**
   The planner inspects the paper metadata and section snapshots, then decides which sections are most relevant for each downstream review dimension.

5. **Evidence preparation**
   The pipeline builds compact evidence packets for consistency, grammar, fact-checking, and novelty. This is one of the main token-control mechanisms in the system.

6. **Claim extraction**
   [claims/claim_extractor.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/claims/claim_extractor.py) extracts high-signal contribution, result, novelty, or factual claims from claim-relevant sections.

7. **Specialist review agents**
   The extracted claims and prepared evidence are passed into 6 reviewer agents:
   - consistency
   - grammar
   - citation
   - factcheck
   - novelty
   - credibility

8. **Reviewer critic**
   A critic stage looks across the specialist outputs and adds calibration notes before credibility synthesis.

9. **Credibility synthesis**
   The credibility agent converts cross-agent signals into a final risk score.

10. **Report generation**
    The final report is assembled and written as Markdown to `reports/{paper_id}/report.md`.

## Core Modules

### Data models

[models/schema.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/models/schema.py) defines the shared contracts for:

- `PaperDocument`
- `PaperSection`
- `ReferenceItem`
- `Claim`
- `ConsistencyResult`
- `GrammarResult`
- `CitationResult`
- `FactCheckResult`
- `NoveltyResult`
- `CredibilityResult`
- `FinalReport`

These models keep the pipeline structured and make it easier to inspect intermediate outputs.

### LLM client

[agents/llm_client.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/llm_client.py) is the shared entry point for LLM calls.

It provides:

- lazy initialization of the Gemini client
- `call_llm()` for plain text responses
- `safe_parse_json()` for tolerant JSON extraction
- `call_llm_json()` for structured prompts
- fallback payload injection when the LLM fails or returns malformed JSON

This is one of the most important reliability layers in the repo.

### Prompt loading

[prompt_loader.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/prompt_loader.py) loads prompt templates from the `prompts/` directory and formats them with runtime values.

### Chunk summarization helper

[agents/common.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/common.py) contains the summarization helper used when a downstream expert should not receive the full raw section text.

## The 6 Specialist Agents

### 1. Consistency Agent

[agents/consistency_agent.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/consistency_agent.py)

Purpose:
- checks whether methods, results, and extracted claims align
- identifies overstated conclusions or weak evidence chains

Input:
- method summary
- results summary
- contribution/result claims

Output:
- score out of 100
- issue list
- reasoning

### 2. Grammar Agent

[agents/grammar_agent.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/grammar_agent.py)

Purpose:
- checks readability, tone, and publication-style writing quality

Input:
- abstract and selected prose-heavy sections

Output:
- `High`, `Medium`, or `Low`
- reasoning

### 3. Citation Agent

[agents/citation_agent.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/citation_agent.py)

Purpose:
- checks whether claims are grounded in the references
- looks for uncited external-style claims or weak prior-work grounding

Input:
- extracted claims
- parsed references

Output:
- score out of 100
- citation gaps
- reasoning

### 4. FactCheck Agent

[agents/factcheck_agent.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/factcheck_agent.py)

Purpose:
- reviews eligible factual and result claims

Input:
- selected claims
- bounded supporting context

Output per claim:
- `verified`
- `unverifiable`
- `suspicious`

Note:
- this version does **not** perform live web search fact-checking

### 5. Novelty Agent

[agents/novelty_agent.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/novelty_agent.py)

Purpose:
- checks how convincingly the paper differentiates itself from prior work

Input:
- title
- abstract
- contribution or novelty claims
- related-work summary

Output:
- `High`, `Moderate`, or `Low`
- reasoning

### 6. Credibility Agent

[agents/credibility_agent.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/credibility_agent.py)

Purpose:
- combines the outputs of all prior agents into a final risk score

Input:
- consistency
- grammar
- citation
- factcheck
- novelty
- critic feedback

Output:
- numeric credibility score
- risk factors
- reasoning

## Scoring and Recommendation

The repo uses mixed output types:

- **Consistency:** numeric score
- **Citation:** numeric score
- **Credibility:** numeric score
- **Grammar:** categorical rating
- **Novelty:** categorical rating
- **Fact-checking:** per-claim labels

The final recommendation is built in [aggregator/report_builder.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/aggregator/report_builder.py).

High-level logic:

- low credibility risk tends toward `Pass`
- medium risk tends toward `Borderline`
- high risk tends toward `Fail`

Guardrails are also applied so very weak citation support, very weak consistency, or poor grammar can force a harsher recommendation even if the blended score is otherwise moderate.

## Report Output

Reports are written by [reporter/report_writer.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/reporter/report_writer.py).

Each generated Markdown report includes:

- executive summary
- extracted claims table
- per-claim analysis
- consistency analysis
- writing quality analysis
- citation analysis
- fact-check log
- novelty assessment
- credibility and fabrication-risk section
- final recommendation

<!-- Example generated reports already exist in:

- `reports/1706.03762/report.md`
- `reports/2504.08753/report.md`
- `reports/test-case/report.md` -->

## Streamlit UI

[app.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/app.py) provides a simple interface for running the evaluator.

Features:

- paste an arXiv URL
- run the full evaluation pipeline
- view extracted claims and fact-check logs in tables
- inspect per-agent reasoning in expandable sections
- download the generated Markdown report
- clear runtime cache and old report artifacts

## Repository Structure

```text
.
├── agents/                  # LLM client and specialist reviewer agents
├── aggregator/              # Final report object construction
├── chunker/                 # Token counting and bounded text chunking
├── claims/                  # Claim extraction logic
├── ingestion/               # arXiv PDF download
├── models/                  # Shared Pydantic schemas
├── parser/                  # PDF parsing into structured paper sections
├── prompts/                 # Prompt templates for all LLM-backed stages
├── reporter/                # Markdown report writing
├── reports/                 # Generated report artifacts
├── utils/                   # Cache cleanup helpers
├── app.py                   # Streamlit app entry point
├── pipeline.py              # LangGraph orchestration
├── prompt_loader.py         # Prompt file loader
├── requirement.md           # Original assignment design notes
└── requirements.txt         # Python dependencies
```



## Example Inputs

Good demo examples:

- `https://arxiv.org/abs/1706.03762`  
  Attention Is All You Need

- `https://arxiv.org/abs/1409.0473`  
  Neural Machine Translation by Jointly Learning to Align and Translate

## Design Tradeoffs

This repo intentionally makes a few practical tradeoffs:

- **Sequential graph instead of parallel orchestration**
  Easier to debug, easier to present, and more predictable on token usage.

- **Heuristic PDF parsing instead of full scholarly layout recovery**
  Faster to implement and good enough for many standard arXiv papers.

- **No live external retrieval in fact-checking**
  Keeps the assignment simpler, but means some claims are only classified as unverifiable rather than deeply verified.

- **Prompt-based reviewers with local fallback heuristics**
  Better resilience when the LLM produces empty or invalid structured output.

## Limitations

- supports standard modern arXiv IDs best
- PDF parsing is heuristic and may struggle on unusual layouts
- fact checking does not browse the web or external scholarly databases
- reference extraction is text-pattern based rather than citation-graph aware
- output quality depends on the paper PDF text being extractable

## Good Files to Read First

If you want to understand the repo quickly, start with:

1. [pipeline.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/pipeline.py)
2. [models/schema.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/models/schema.py)
3. [agents/llm_client.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/agents/llm_client.py)
4. [claims/claim_extractor.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/claims/claim_extractor.py)
5. [aggregator/report_builder.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/aggregator/report_builder.py)
6. [app.py](/Users/abhiramkamini/Downloads/hivel%20assignment%20/app.py)

## Summary

This project turns an arXiv paper into a structured multi-agent review pipeline. Its main strengths are:

- explicit orchestration
- claim-based evaluation
- bounded context preparation
- typed intermediate outputs
- graceful LLM failure handling
- inspectable Markdown reporting

That makes it especially suitable for assignment demos, architecture walkthroughs, and explaining how an agentic evaluation pipeline can be built in a practical way.
