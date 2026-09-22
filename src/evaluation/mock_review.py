def _score_range(value):
    return max(1, min(5, int(value)))


def score_report_quality(report, persona):
    aggregate = report.get("aggregate_evidence") or {}
    paths = report.get("explanation_paths") or []
    faithfulness = report.get("faithfulness_check") or {}
    recursive_context = report.get("recursive_context") or {}
    prompt_chain = report.get("persona_prompt_chain") or []

    matched_count = aggregate.get("matched_biomarker_count") or 0
    available_count = aggregate.get("available_biomarker_count") or 0
    path_count = len(paths)
    supported = faithfulness.get("supported", True)

    source_backed_paths = 0
    for path in paths:
        steps = path.get("steps") or []
        if any(step.get("sources_to_next") for step in steps):
            source_backed_paths += 1

    grounding = 2 + min(3, matched_count // 2)
    traceability = 1 + min(4, path_count // 4)
    source_support = 1 + min(4, source_backed_paths // 4)
    caution = 5 if supported else 3
    persona_fit = 5 if persona in {"psychologist", "patient"} and prompt_chain else 3
    context_use = 5 if recursive_context.get("enabled") else 3
    completeness = 1 + min(4, available_count // 3)

    if persona == "patient":
        actionability = 4 if aggregate.get("interpretation") else 3
    else:
        actionability = 4 if path_count and aggregate.get("calibration") else 3

    scores = {
        "evidence_grounding": _score_range(grounding),
        "graph_traceability": _score_range(traceability),
        "source_support": _score_range(source_support),
        "clinical_caution": _score_range(caution),
        "persona_fit": _score_range(persona_fit),
        "recursive_context_use": _score_range(context_use),
        "biomarker_completeness": _score_range(completeness),
        "actionability": _score_range(actionability),
    }
    scores["overall_mean"] = round(sum(scores.values()) / len(scores), 3)
    return scores


def build_mock_review_row(participant_id, report, persona):
    scores = score_report_quality(report, persona)
    aggregate = report.get("aggregate_evidence") or {}
    return {
        "participant_id": participant_id,
        "persona": persona,
        "screen_positive": aggregate.get("screen_positive"),
        "evidence_level": aggregate.get("evidence_level"),
        **scores,
        "review_type": "mock qualitative rubric",
        "note": "Deterministic proxy review for Month 5; not a substitute for clinician or participant user study.",
    }

