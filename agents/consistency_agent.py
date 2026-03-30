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
    support_units = len(grounded) + (0.5 * len(mixed))
    score_int = round(100 * support_units / max(len(result_claims), 1))
    score_int = max(0, min(100, score_int))

    issues: list[str] = []
    if not method_claims and not method_sections_present:
        issues.append("The parsed paper does not expose a clear method signal, so result interpretation is weaker.")
    if not result_claims:
        issues.append("Few concrete result or comparison claims were extracted, so consistency is hard to judge.")
    if contradicted:
        issues.append(f"{len(contradicted)} externally checked claim(s) were contradicted by retrieved evidence.")
    if mixed:
        issues.append(f"{len(mixed)} checked claim(s) had mixed support and contradiction signals.")
    if unresolved:
        issues.append(f"{len(unresolved)} checked claim(s) could not be tied to strong external evidence.")
    missing_checks = len(result_claims) - checked_count
    if missing_checks > 0:
        issues.append(f"{missing_checks} result-like claim(s) were extracted but never completed external checking.")

    reasoning = (
        "Consistency is the percentage of extracted result/comparison/factual claims that received grounded support, "
        "with mixed verdicts counting as partial support. "
        f"{len(grounded)} checked result/comparison claim(s) were supported, "
        f"{len(contradicted)} contradicted, {len(mixed)} mixed, and {len(unresolved)} unresolved. "
        f"Method signals were {'present' if method_claims or method_sections_present else 'not clearly present'}."
    )

    return ConsistencyResult(
        score=score_int,
        issues=issues,
        reasoning=reasoning,
        grounded_claim_count=len(grounded),
        unresolved_claim_count=max(len(result_claims) - len(grounded), 0),
    )
