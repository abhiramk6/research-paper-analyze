from __future__ import annotations

from pathlib import Path

from models.schema import (
    CitationResult,
    Claim,
    ConsistencyResult,
    CredibilityResult,
    FactCheckResult,
    FinalReport,
    GrammarResult,
    NoveltyResult,
)


def _bullet_list(items: list[str], empty_message: str) -> str:
    if not items:
        return f"- {empty_message}"
    return "\n".join(f"- {item}" for item in items)


def _score_bar(score: int) -> str:
    filled = round(score / 10)
    return "█" * filled + "░" * (10 - filled)


def _risk_bar(score: float) -> str:
    filled = round(score / 10)
    return "█" * filled + "░" * (10 - filled)


def _score_label(score: int) -> str:
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Moderate"
    if score >= 55:
        return "Weak"
    return "Poor"


def _risk_label(score: float) -> str:
    if score <= 30:
        return "Low Risk"
    if score <= 60:
        return "Moderate Risk"
    return "High Risk"


def _recommendation_line(rec: str) -> str:
    return {"Pass": "✅  PASS", "Borderline": "⚠️  BORDERLINE", "Fail": "❌  FAIL"}.get(rec, rec)


def _factcheck_badge(status: str) -> str:
    return {
        "verified": "✅ verified",
        "unverifiable": "⚪ unverifiable",
        "suspicious": "🚩 suspicious",
    }.get(status, status)


def _claim_type_label(claim_type: str) -> str:
    return {
        "contribution": "contribution",
        "result": "result",
        "novelty": "novelty",
        "factual": "factual",
        "background": "background",
    }.get(claim_type, claim_type)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return f"high ({confidence:.2f})"
    if confidence >= 0.65:
        return f"medium ({confidence:.2f})"
    if confidence >= 0.45:
        return f"low ({confidence:.2f})"
    return f"marginal ({confidence:.2f})"


def _build_claims_section(claims: list[Claim], factcheck: FactCheckResult) -> str:
    if not claims:
        return "## Claims Extracted (0 total)\n\n*No claims were extracted from this paper.*\n"

    fc_map = {item.claim_id: item for item in factcheck.items}

    type_counts: dict[str, int] = {}
    for c in claims:
        type_counts[c.claim_type] = type_counts.get(c.claim_type, 0) + 1
    type_summary = " · ".join(f"{t}: {n}" for t, n in sorted(type_counts.items()))

    # Summary table
    header = "| ID | Type | Confidence | Citations | Fact-Check |\n|---|---|---|---|---|"
    rows = []
    for claim in claims:
        cites = ", ".join(f"`{r}`" for r in claim.cited_refs) if claim.cited_refs else "—"
        fc_item = fc_map.get(claim.claim_id)
        fc_cell = _factcheck_badge(fc_item.status) if fc_item else "not evaluated"
        conf = _confidence_label(claim.confidence) if claim.confidence > 0 else "—"
        rows.append(f"| `{claim.claim_id}` | {claim.claim_type} | {conf} | {cites} | {fc_cell} |")

    summary_table = header + "\n" + "\n".join(rows)

    # Per-claim detail blocks
    details = []
    for claim in claims:
        fc_item = fc_map.get(claim.claim_id)
        cites = ", ".join(f"`{r}`" for r in claim.cited_refs) if claim.cited_refs else "none detected in sentence"
        conf_label = _confidence_label(claim.confidence) if claim.confidence > 0 else "not scored"

        block = f"#### `{claim.claim_id}` · {_claim_type_label(claim.claim_type)}\n\n"
        block += f"> {claim.text}\n\n"
        block += f"| Field | Value |\n|---|---|\n"
        block += f"| Source section | {claim.source_section} |\n"
        block += f"| Extraction confidence | {conf_label} |\n"
        block += f"| Inline citations | {cites} |\n"
        block += f"| Verification status | {claim.verification_status} |\n"

        if fc_item:
            block += f"\n**Fact-check result:** {_factcheck_badge(fc_item.status)}\n\n"
            block += f"{fc_item.reasoning}\n"
        else:
            block += f"\n**Fact-check result:** Not evaluated — this claim did not meet the fact-check threshold (requires factual or result type with confidence ≥ 0.45).\n"

        details.append(block)

    detail_section = "\n---\n\n".join(details)

    return (
        f"## Claims Extracted ({len(claims)} total)\n\n"
        f"**Type breakdown:** {type_summary}\n\n"
        f"### Claims Summary Table\n\n"
        f"{summary_table}\n\n"
        f"### Per-Claim Detailed Analysis\n\n"
        f"{detail_section}\n"
    )


def _build_factcheck_log(factcheck: FactCheckResult) -> str:
    if not factcheck.items:
        return (
            "## Fact Check Log\n\n"
            "*No claims were eligible for independent fact-checking.*\n\n"
            "**Why:** The fact-check agent targets claims classified as `factual` or `result` type "
            "with extraction confidence ≥ 0.45. Claims that fall outside this scope are not evaluated "
            "here but are covered by the consistency and citation agents.\n"
        )

    verified = [i for i in factcheck.items if i.status == "verified"]
    unverifiable = [i for i in factcheck.items if i.status == "unverifiable"]
    suspicious = [i for i in factcheck.items if i.status == "suspicious"]

    summary = (
        f"**{len(verified)} verified · {len(unverifiable)} unverifiable · {len(suspicious)} suspicious** "
        f"out of {len(factcheck.items)} evaluated claims."
    )

    rows = "\n".join(
        f"| `{item.claim_id}` | {_factcheck_badge(item.status)} | {item.reasoning.replace('|', '/')} |"
        for item in factcheck.items
    )
    table = f"| Claim ID | Status | Reasoning |\n|---|---|---|\n{rows}"

    notes = []
    if suspicious:
        notes.append(
            "**Suspicious claims** conflict with established knowledge according to the fact-check agent. "
            "These are a strong credibility signal and are weighted heavily in the fabrication risk score."
        )
    if unverifiable:
        notes.append(
            "**Unverifiable claims** are plausible but cannot be confirmed from general world knowledge alone. "
            "This is the expected status for most paper-specific experimental results."
        )
    if verified:
        notes.append(
            "**Verified claims** are consistent with widely known, well-documented facts in the domain."
        )

    return (
        f"## Fact Check Log\n\n"
        f"{summary}\n\n"
        f"{table}\n\n"
        + ("\n\n".join(notes) + "\n" if notes else "")
    )


def save_report(
    report: FinalReport,
    claims: list[Claim],
    consistency: ConsistencyResult,
    citation: CitationResult,
    factcheck: FactCheckResult,
    novelty: NoveltyResult,
    credibility: CredibilityResult,
    paper_id: str,
    grammar: GrammarResult | None = None,
) -> Path:
    output_dir = Path("reports") / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.md"

    rec_line = _recommendation_line(report.recommendation)
    consistency_label = _score_label(report.consistency_score)
    citation_label = _score_label(report.citation_score)
    risk_label = _risk_label(report.credibility_score)

    # Grammar section
    if grammar:
        grammar_detail = {
            "High": "Publication-ready prose. Grammar, tone, and clarity meet or exceed the standard expected at peer-reviewed venues.",
            "Medium": "Comprehensible writing with recurring issues. Proofreading revision required before top-venue submission.",
            "Low": "Frequent grammatical problems or unprofessional register that impairs evaluability of technical claims.",
        }.get(grammar.rating, "")
        grammar_section = (
            f"## Writing Quality Analysis\n\n"
            f"**Rating:** {grammar.rating}\n\n"
            f"{grammar_detail}\n\n"
            f"**Detailed Assessment:**\n{grammar.reasoning}\n"
        )
    else:
        grammar_section = "## Writing Quality Analysis\n\n*Grammar analysis not available.*\n"

    # Citation gaps
    gaps_detail = _bullet_list(
        citation.citation_gaps,
        "No citation gaps identified — prior-work comparisons appear adequately grounded."
    )

    # Consistency issues
    issues_detail = _bullet_list(
        consistency.issues,
        "No internal consistency issues flagged."
    )

    # Risk factors
    risk_factors_text = _bullet_list(
        credibility.risk_factors,
        "No major cross-signal risk factors were identified."
    )

    claims_section = _build_claims_section(claims, factcheck)
    factcheck_section = _build_factcheck_log(factcheck)

    content = f"""# Judgement Report: {report.title}

**Paper ID:** `{paper_id}`
**Final Recommendation:** {rec_line}

---

## Executive Summary

{report.summary}

---

## Scores at a Glance

| Dimension | Score | Visual | Interpretation |
|---|---|---|---|
| Internal Consistency | {report.consistency_score}/100 | `{_score_bar(report.consistency_score)}` | {consistency_label} |
| Grammar & Writing | {report.grammar_rating} | — | — |
| Citation Grounding | {report.citation_score}/100 | `{_score_bar(report.citation_score)}` | {citation_label} |
| Novelty | {report.novelty_rating} | — | — |
| Fabrication Risk | {report.fabrication_probability} | `{_risk_bar(report.credibility_score)}` | {risk_label} |

---

{claims_section}

---

## Consistency Analysis

**Score:** {report.consistency_score}/100 — *{consistency_label}*

Measures whether the described methodology supports the claimed results and whether conclusions stay within what the evidence can justify.

**Detailed Reasoning:**
{consistency.reasoning}

**Issues Identified:**
{issues_detail}

---

{grammar_section}

---

## Citation Analysis

**Score:** {report.citation_score}/100 — *{citation_label}*

Measures whether comparative claims and prior-work references are adequately cited and whether the reference list provides adequate grounding for the paper's claims.

**Detailed Reasoning:**
{citation.reasoning}

**Citation Gaps:**
{gaps_detail}

---

{factcheck_section}

---

## Novelty Assessment

**Rating:** {report.novelty_rating}

Evaluates how convincingly the paper argues its novelty relative to the related work it cites and discusses.

**Detailed Reasoning:**
{novelty.reasoning}

---

## Credibility & Fabrication Risk Analysis

**Fabrication Probability:** {report.fabrication_probability} — *{risk_label}*
**Raw Credibility Score:** {report.credibility_score:.1f} / 100 *(higher = more risk)*

The fabrication risk score is computed from weighted cross-signal inputs: consistency penalty (30%), citation penalty (30%), novelty risk mapping (20%), and suspicious fact-check ratio (20%), with hard penalties for severely weak signals.

**Cross-Signal Risk Factors:**
{risk_factors_text}

**Synthesis Reasoning:**
{credibility.reasoning}

---

## Final Recommendation: {rec_line}

| Threshold | Condition |
|---|---|
| Pass | Fabrication risk ≤ 30% AND consistency ≥ 70 AND citation ≥ 65 AND grammar ≠ Low |
| Borderline | Fabrication risk 30–60% OR moderate weakness in one dimension |
| Fail | Fabrication risk > 60% OR citation < 50 OR consistency < 55 OR grammar = Low |

**This paper received: {report.recommendation}**

> {report.summary.split(chr(10))[0]}
"""

    output_path.write_text(content, encoding="utf-8")
    return output_path
