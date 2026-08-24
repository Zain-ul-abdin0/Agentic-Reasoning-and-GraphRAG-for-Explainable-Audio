import re

FORBIDDEN_DIAGNOSIS_PATTERNS = [
    r"\bdiagnosed\s+with\s+depression\b",
    r"\bhas\s+depression\b",
    r"\bis\s+depressed\b",
    r"\bsuffers\s+from\s+depression\b",
    r"\bmajor\s+depressive\s+disorder\b",
    r"\bclinical\s+depression\b",
]

FORBIDDEN_SCORE_PATTERNS = [
    r"\bphq[- ]?8\s+(score\s+)?(is|=|of)\s*\d+\b",
    r"\bestimated\s+phq[- ]?8\b",
    r"\bpredicted\s+phq[- ]?8\b",
]

UNSUPPORTED_ACTION_PATTERNS = [
    r"\bmedication\b",
    r"\bantidepressant\b",
    r"\bprescribe\b",
    r"\bsuicid(e|al)\b",
    r"\bself[- ]harm\b",
]


def _normalise(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def _collect_supported_terms(explanations, findings, aggregate_evidence=None):
    terms = set()

    for finding in findings:
        if finding.get("matched"):
            terms.add(finding.get("label", ""))
            terms.add(finding.get("feature", ""))
            terms.add(finding.get("summary", ""))

    for explanation in explanations:
        for label in explanation.get("path_labels", []):
            terms.add(label)
        for step in explanation.get("steps", []):
            terms.add(step.get("name", ""))
            terms.add(step.get("type", ""))
            terms.add(step.get("relation_to_next", ""))

    if aggregate_evidence:
        terms.add(aggregate_evidence.get("evidence_level", ""))
        terms.add(aggregate_evidence.get("interpretation", ""))
        terms.add("PHQ-8 >= 10")
        terms.add("screening")
        terms.add("not a diagnosis")

    return sorted({term for term in terms if term})


def _matching_terms(text, terms):
    normalised = _normalise(text)
    matched = []

    for term in terms:
        clean_term = _normalise(str(term))
        if clean_term and clean_term in normalised:
            matched.append(term)

    return matched


def _pattern_warnings(text, patterns, message):
    warnings = []
    normalised = _normalise(text)

    for pattern in patterns:
        if re.search(pattern, normalised):
            warnings.append(message)
            break

    return warnings


def check_report_faithfulness(
    llm_report,
    explanations,
    findings,
    aggregate_evidence=None
):
    if not llm_report:
        return {
            "checked": False,
            "supported": None,
            "warnings": [],
            "supported_terms_mentioned": [],
        }

    supported_terms = _collect_supported_terms(
        explanations,
        findings,
        aggregate_evidence=aggregate_evidence
    )
    mentioned_terms = _matching_terms(llm_report, supported_terms)
    warnings = []
    warnings.extend(_pattern_warnings(
        llm_report,
        FORBIDDEN_DIAGNOSIS_PATTERNS,
        "The LLM phrased a screening result like a depression diagnosis."
    ))
    warnings.extend(_pattern_warnings(
        llm_report,
        FORBIDDEN_SCORE_PATTERNS,
        "The LLM appeared to infer a PHQ-8 score that was not supplied."
    ))
    warnings.extend(_pattern_warnings(
        llm_report,
        UNSUPPORTED_ACTION_PATTERNS,
        "The LLM mentioned high-risk or treatment actions not supported by the graph evidence."
    ))

    if len(mentioned_terms) < 2:
        warnings.append(
            "The LLM report mentions too little of the retrieved graph evidence."
        )

    return {
        "checked": True,
        "supported": len(warnings) == 0,
        "warnings": warnings,
        "supported_terms_mentioned": mentioned_terms[:20],
    }
