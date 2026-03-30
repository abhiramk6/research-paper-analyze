from __future__ import annotations

from pathlib import Path

from models.schema import (
    CitationResult,
    Claim,
    ConsistencyResult,
    CredibilityBreakdown,
    CredibilityResult,
    EvidenceFactCheckItem,
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


def _verdict_badge(verdict: str) -> str:
    return {
        "supported": "✅ supported",
        "contradicted": "🚩 contradicted",
        "mixed": "⚠️ mixed",
        "insufficient_evidence": "⚪ insufficient evidence",
        "paper_supported_only": "📄 paper-supported only",
    }.get(verdict, verdict)


def _claim_type_label(claim_type: str) -> str:
    labels = {
        "benchmark_result": "benchmark result",
        "prior_work_comparison": "prior work comparison",
        "factual_background": "factual background",
        "methodology_assertion": "methodology assertion",
        "contribution_claim": "contribution claim",
        "unsupported_general_statement": "general statement",
        "contribution": "contribution",
        "result": "result",
        "novelty": "novelty",
        "factual": "factual",
        "background": "background",
    }
    return labels.get(claim_type, claim_type)


def _importance_badge(importance: str) -> str:
    return {"high": "🔴 high", "medium": "🟡 medium", "low": "🟢 low"}.get(importance, importance)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return f"high ({confidence:.2f})"
    if confidence >= 0.65:
        return f"medium ({confidence:.2f})"
    if confidence >= 0.45:
        return f"low ({confidence:.2f})"
    return f"marginal ({confidence:.2f})"

def _build_evidence_factcheck_log(evidence_results: list[EvidenceFactCheckItem]) -> str:
    if not evidence_results:
        return (
            "## Evidence-Backed Fact-Check Log\n\n"
            "*No claims were routed to external evidence checking.*\n\n"
            "Claims classified as `methodology_assertion`, `contribution_claim`, or "
            "`unsupported_general_statement` are not sent for external retrieval.\n"
        )

    supported = [r for r in evidence_results if r.verdict == "supported"]
    contradicted = [r for r in evidence_results if r.verdict == "contradicted"]
    mixed = [r for r in evidence_results if r.verdict == "mixed"]
    insufficient = [r for r in evidence_results if r.verdict in ("insufficient_evidence", "paper_supported_only")]

    summary = (
        f"**{len(supported)} supported · {len(contradicted)} contradicted · "
        f"{len(mixed)} mixed · {len(insufficient)} insufficient/paper-only** "
        f"out of {len(evidence_results)} evidence-checked claims."
    )

    table_rows = "\n".join(
        f"| `{r.claim_id}` | {_verdict_badge(r.verdict)} | {round(r.confidence * 100)}% | "
        f"{r.reasoning[:120].replace('|', '/')}... |"
        for r in evidence_results
    )
    table = (
        "| Claim ID | Verdict | Confidence | Reasoning |\n"
        "|---|---|---|---|\n"
        + table_rows
    )

    detail_blocks = []
    for r in evidence_results:
        block = f"#### `{r.claim_id}` — {_verdict_badge(r.verdict)}\n\n"

        if r.paper_context_excerpt:
            block += f"**Paper context:** _{r.paper_context_excerpt[:300]}_\n\n"

        block += f"**Reasoning:** {r.reasoning}\n\n"
        block += f"**Confidence:** {round(r.confidence * 100)}%\n\n"

        if r.evidence_items:
            block += "**External Evidence Used:**\n\n"
            for ev in r.evidence_items:
                year_str = f" ({ev.publication_year})" if ev.publication_year else ""
                block += (
                    f"- **[{ev.title[:80]}]({ev.url})**{year_str} "
                    f"*(score: {ev.retrieval_score:.2f}, type: {ev.source_type})*\n"
                    f"  > {ev.snippet[:280]}\n"
                )
        else:
            note = {
                "paper_supported_only": "_No external evidence found. Verdict is based on paper text only._",
                "insufficient_evidence": "_External evidence retrieved but did not clearly address this claim._",
            }.get(r.verdict, "_No evidence items linked._")
            block += f"{note}\n"

        detail_blocks.append(block)

    detail_section = "\n---\n\n".join(detail_blocks)

    notes: list[str] = []
    if contradicted:
        notes.append(
            "**Contradicted claims** conflict with retrieved external evidence. "
            "These are a strong credibility signal weighted heavily in the risk score."
        )
    if insufficient:
        notes.append(
            "**Paper-supported-only / insufficient-evidence** claims could not be independently "
            "verified from external sources. This is expected for novel experimental results."
        )
    if supported:
        notes.append(
            "**Supported claims** are corroborated by at least one retrieved external source."
        )

    return (
        f"## Evidence-Backed Fact-Check Log\n\n"
        f"{summary}\n\n"
        f"### Summary Table\n\n"
        f"{table}\n\n"
        + ("\n\n".join(notes) + "\n\n" if notes else "")
        + f"### Per-Claim Evidence Detail\n\n"
        f"{detail_section}\n"
    )

def _build_factcheck_log(factcheck: FactCheckResult) -> str:
    if not factcheck.items:
        return (
            "## Fact Check Log\n\n"
            "*No claims were eligible for independent fact-checking.*\n\n"
            "**Why:** The fact-check agent targets claims classified as `factual` or `result` type "
            "with extraction confidence ≥ 0.45.\n"
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
            "**Suspicious claims** conflict with established knowledge. "
            "These are weighted heavily in the fabrication risk score."
        )
    if unverifiable:
        notes.append(
            "**Unverifiable claims** are plausible but cannot be confirmed from general knowledge alone."
        )
    if verified:
        notes.append("**Verified claims** are consistent with well-documented domain facts.")

    return (
        f"## Fact Check Log\n\n{summary}\n\n{table}\n\n"
        + ("\n\n".join(notes) + "\n" if notes else "")
    )

def _build_credibility_breakdown(breakdown: CredibilityBreakdown) -> str:
    band_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(breakdown.risk_band, "")

    feature_rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {value:.2f} |"
        for name, value in breakdown.score_breakdown.items()
    )
    feature_table = "| Feature | Contribution (pts) |\n|---|---|\n" + feature_rows

    signal_rows = [
        f"| Supported claim ratio | {breakdown.supported_claim_ratio:.1%} |",
        f"| Contradicted claim ratio | {breakdown.contradicted_claim_ratio:.1%} |",
        f"| Insufficient evidence ratio | {breakdown.insufficient_evidence_ratio:.1%} |",
        f"| Citation coverage ratio | {breakdown.citation_coverage_ratio:.1%} |",
        f"| Consistency penalty | {breakdown.consistency_penalty:.3f} |",
        f"| Parser uncertainty | {breakdown.parser_uncertainty:.3f} |",
        f"| High-impact unverified count | {breakdown.high_impact_unverified_claim_count} |",
    ]
    signal_table = "| Signal | Value |\n|---|---|\n" + "\n".join(signal_rows)

    return (
        f"### Credibility Score Breakdown\n\n"
        f"**Final Score:** {breakdown.final_score:.1f}/100 — "
        f"{band_emoji} **{breakdown.risk_band.upper()} RISK**\n\n"
        f"#### Input Signals\n\n{signal_table}\n\n"
        f"#### Score Components\n\n{feature_table}\n\n"
        f"#### Explanation\n\n{breakdown.explanation}\n"
    )

def _build_claims_section(
    claims: list[Claim],
    factcheck: FactCheckResult,
    evidence_results: list[EvidenceFactCheckItem] | None = None,
) -> str:
    if not claims:
        return "## Claims Extracted (0 total)\n\n*No claims were extracted from this paper.*\n"

    fc_map = {item.claim_id: item for item in factcheck.items}
    ev_map = {r.claim_id: r for r in (evidence_results or [])}

    type_counts: dict[str, int] = {}
    for c in claims:
        type_counts[c.claim_type] = type_counts.get(c.claim_type, 0) + 1
    type_summary = " · ".join(f"{t}: {n}" for t, n in sorted(type_counts.items()))

    header = "| ID | Type | Importance | Confidence | Chunk | Evidence Verdict |\n|---|---|---|---|---|---|"
    rows = []
    for claim in claims:
        ev_result = ev_map.get(claim.claim_id)
        fc_item = fc_map.get(claim.claim_id)
        verdict_cell = (
            _verdict_badge(ev_result.verdict) if ev_result
            else (_factcheck_badge(fc_item.status) if fc_item else "not evaluated")
        )
        conf = _confidence_label(claim.confidence) if claim.confidence > 0 else "—"
        chunk_id = claim.source_chunk_id or "—"
        rows.append(
            f"| `{claim.claim_id}` | {_claim_type_label(claim.claim_type)} | "
            f"{_importance_badge(claim.importance)} | {conf} | `{chunk_id}` | {verdict_cell} |"
        )
    summary_table = header + "\n" + "\n".join(rows)

    details = []
    for claim in claims:
        ev_result = ev_map.get(claim.claim_id)
        fc_item = fc_map.get(claim.claim_id)
        cites = ", ".join(f"`{r}`" for r in claim.cited_refs) if claim.cited_refs else "none detected"
        conf_label = _confidence_label(claim.confidence) if claim.confidence > 0 else "not scored"

        block = f"#### `{claim.claim_id}` · {_claim_type_label(claim.claim_type)}\n\n"
        block += f"> {claim.text}\n\n"
        block += "| Field | Value |\n|---|---|\n"
        block += f"| Source section | {claim.source_section} |\n"
        block += f"| Source chunk | {claim.source_chunk_id or '—'} |\n"
        block += f"| Importance | {_importance_badge(claim.importance)} |\n"
        block += f"| Extraction confidence | {conf_label} |\n"
        block += f"| Inline citations | {cites} |\n"
        block += f"| Verification status | {claim.verification_status} |\n"

        if ev_result:
            block += f"\n**Evidence verdict:** {_verdict_badge(ev_result.verdict)} (confidence: {round(ev_result.confidence * 100)}%)\n\n"
            block += f"{ev_result.reasoning}\n"
        elif fc_item:
            block += f"\n**Fact-check result:** {_factcheck_badge(fc_item.status)}\n\n{fc_item.reasoning}\n"
        else:
            block += "\n**Fact-check result:** Not evaluated — claim type or confidence did not meet the threshold.\n"

        details.append(block)

    detail_section = "\n---\n\n".join(details)
    return (
        f"## Claims Extracted ({len(claims)} total)\n\n"
        f"**Type breakdown:** {type_summary}\n\n"
        f"### Claims Summary Table\n\n{summary_table}\n\n"
        f"### Per-Claim Detailed Analysis\n\n{detail_section}\n"
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
    evidence_results: list[EvidenceFactCheckItem] | None = None,
) -> Path:
    output_dir = Path("reports") / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.md"

    rec_line = _recommendation_line(report.recommendation)
    consistency_label = _score_label(report.consistency_score)
    citation_label = _score_label(report.citation_score)
    risk_label = _risk_label(report.credibility_score)

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

    gaps_detail = _bullet_list(
        citation.citation_gaps,
        "No citation gaps identified — prior-work comparisons appear adequately grounded."
    )
    issues_detail = _bullet_list(
        consistency.issues,
        "No internal consistency issues flagged."
    )
    risk_factors_text = _bullet_list(
        credibility.risk_factors,
        "No major cross-signal risk factors were identified."
    )

    claims_section = _build_claims_section(claims, factcheck, evidence_results)

    if evidence_results:
        factcheck_section = _build_evidence_factcheck_log(evidence_results)
    else:
        factcheck_section = _build_factcheck_log(factcheck)

    breakdown = credibility.breakdown
    if breakdown:
        credibility_detail = _build_credibility_breakdown(breakdown)
        risk_method_note = (
            "The credibility risk score is computed from explicit, measurable evidence signals. "
            "See the breakdown table below for the exact feature contributions."
        )
    else:
        credibility_detail = ""
        risk_method_note = (
            "The fabrication risk score is computed from weighted cross-signal inputs: "
            "consistency penalty (30%), citation penalty (30%), novelty risk mapping (20%), "
            "and suspicious fact-check ratio (20%), with hard penalties for severely weak signals."
        )

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
| Credibility Risk | {report.fabrication_probability} | `{_risk_bar(report.credibility_score)}` | {risk_label} |

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

Measures whether comparative claims and prior-work references are adequately cited.

**Detailed Reasoning:**
{citation.reasoning}

**Citation Gaps:**
{gaps_detail}

---

{factcheck_section}

---

## Novelty Assessment

**Rating:** {report.novelty_rating}

{novelty.reasoning}

---

## Credibility & Risk Analysis

**Risk Score:** {report.fabrication_probability} — *{risk_label}*
**Raw Score:** {report.credibility_score:.1f} / 100 *(higher = more risk)*

{risk_method_note}

{credibility_detail}

**Cross-Signal Risk Factors:**
{risk_factors_text}

**Synthesis Reasoning:**
{credibility.reasoning}

---

## Final Recommendation: {rec_line}

| Threshold | Condition |
|---|---|
| Pass | Risk ≤ 30% AND consistency ≥ 70 AND citation ≥ 65 AND grammar ≠ Low |
| Borderline | Risk 30–60% OR moderate weakness in one dimension |
| Fail | Risk > 60% OR citation < 50 OR consistency < 55 OR grammar = Low |

**This paper received: {report.recommendation}**

> {report.summary.split(chr(10))[0]}
"""

    output_path.write_text(content, encoding="utf-8")
    return output_path
