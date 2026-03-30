from __future__ import annotations

from models.schema import Claim, ClaimCheckResult, ConsistencyResult, PaperDocument


def _result_like(claim: Claim) -> bool:
    return claim.claim_type in {"result", "comparison", "factual"}


def _method_like(claim: Claim) -> bool:
    return claim.claim_type in {"method", "contribution"}


def run_consistency_agent(
    document: PaperDocument,
    claims: list[Claim],
    claim_checks: list[ClaimCheckResult],
) -> ConsistencyResult:
    result_claims = [claim for claim in claims if _result_like(claim)]
    method_claims = [claim for claim in claims if _method_like(claim)]
    check_map = {item.claim_id: item for item in claim_checks}
    checked_results = [check_map[claim.claim_id] for claim in result_claims if claim.claim_id in check_map]

    contradicted = [item for item in checked_results if item.verdict == "contradicted"]
    mixed = [item for item in checked_results if item.verdict == "mixed"]
    unresolved = [item for item in checked_results if item.verdict == "insufficient_evidence"]
    grounded = [item for item in checked_results if item.verdict == "supported"]

    method_sections_present = any(
        keyword in section.heading.lower()
        for keyword in ("method", "approach", "architecture", "training")
        for section in document.sections
    )

    checked_count = len(checked_results)
    contradicted_ratio = len(contradicted) / max(checked_count, 1)
    mixed_ratio = len(mixed) / max(checked_count, 1)
    unresolved_ratio = len(unresolved) / max(checked_count, 1)
    grounded_ratio = len(grounded) / max(checked_count, 1)

    score = 100.0
    score -= contradicted_ratio * 45
    score -= mixed_ratio * 24
    score -= unresolved_ratio * 18
    if not method_claims and not method_sections_present:
        score -= 12
    if not result_claims:
        score -= 10
    if checked_count and not grounded:
        score -= 18
    elif checked_count and grounded_ratio < 0.34:
        score -= 5
    if checked_count >= 3 and unresolved_ratio >= 0.75:
        score -= 8

    score_int = max(0, min(100, round(score)))

    issues: list[str] = []
    if not method_claims and not method_sections_present:
        issues.append("The parsed paper does not expose a clear method signal to support the reported outcomes.")
    if not result_claims:
        issues.append("Few concrete result or comparison claims were extracted, so consistency is hard to judge.")
    if contradicted:
        issues.append(f"{len(contradicted)} externally checked claim(s) were contradicted by retrieved evidence.")
    if mixed:
        issues.append(f"{len(mixed)} checked claim(s) had mixed support and contradiction signals.")
    if unresolved:
        issues.append(f"{len(unresolved)} checked claim(s) could not be tied to strong external evidence.")
    if checked_count and not grounded:
        issues.append("None of the externally checked result/comparison claims received strong support.")

    reasoning = (
        "Consistency is computed from grounded claim evidence rather than a free-form score. "
        f"{len(grounded)} checked result/comparison claim(s) were supported, "
        f"{len(contradicted)} contradicted, {len(mixed)} mixed, and {len(unresolved)} unresolved. "
        f"Method signals were {'present' if method_claims or method_sections_present else 'not clearly present'}."
    )

    return ConsistencyResult(
        score=score_int,
        issues=issues,
        reasoning=reasoning,
        grounded_claim_count=len(grounded),
        unresolved_claim_count=len(unresolved) + len(contradicted) + len(mixed),
    )
