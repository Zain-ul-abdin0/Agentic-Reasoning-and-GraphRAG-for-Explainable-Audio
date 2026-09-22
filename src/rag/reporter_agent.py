from src.rag.faithfulness import check_report_faithfulness
from src.rag.llm_client import LLMUnavailableError, LocalLLMConfig, generate_text


PERSONA_PROMPT_CHAINS = {
    "psychologist": [
        {
            "step": "clinical_scope",
            "instruction": (
                "Write for a psychologist using concise clinical language. "
                "Frame the result as screening-oriented decision support, not diagnosis."
            ),
        },
        {
            "step": "biomarker_interpretation",
            "instruction": (
                "Name the matched acoustic biomarkers, their thresholds, feature sources, "
                "and aggregate evidence level."
            ),
        },
        {
            "step": "graph_reasoning",
            "instruction": (
                "Explain the retrieved graph paths from biomarkers to clinical concepts, "
                "screening constructs, risk, and follow-up. Mention edge-level sources when useful."
            ),
        },
        {
            "step": "longitudinal_context",
            "instruction": (
                "If previous-week context is supplied, compare the current aggregate evidence "
                "with prior weeks while clearly marking simulated context as synthetic."
            ),
        },
        {
            "step": "clinical_safety",
            "instruction": (
                "Avoid inferred PHQ-8 scores, diagnosis, medication advice, or unsupported high-risk claims."
            ),
        },
    ],
    "patient": [
        {
            "step": "plain_language_scope",
            "instruction": (
                "Write for a patient in calm plain language. Clearly say the audio result is "
                "not a diagnosis and should be discussed with a professional if concerning."
            ),
        },
        {
            "step": "simple_audio_evidence",
            "instruction": (
                "Explain only the most understandable audio signs, such as pauses, speaking pace, "
                "voice energy, or voice stability. Avoid unnecessary technical terms."
            ),
        },
        {
            "step": "simple_graph_reasoning",
            "instruction": (
                "Translate the graph path into a human explanation without sounding alarming."
            ),
        },
        {
            "step": "supportive_context",
            "instruction": (
                "If previous-week context is supplied, describe whether the screening signal appears "
                "higher, lower, or similar, and say simulated context is only an example."
            ),
        },
        {
            "step": "next_step",
            "instruction": (
                "Recommend supportive follow-up or completing a validated questionnaire, not self-diagnosis."
            ),
        },
    ],
}


def get_persona_prompt_chain(persona):
    return PERSONA_PROMPT_CHAINS.get(
        persona,
        PERSONA_PROMPT_CHAINS["psychologist"]
    )


def _path_to_sentence(path):
    return " -> ".join(path)


def _path_labels(explanation):
    return explanation.get("path_labels", explanation["path"])


def _format_feature_value(finding):
    feature_name = finding["feature"]
    if finding.get("feature_index") is not None:
        feature_name = f"{feature_name}[{finding['feature_index']}]"

    source = finding.get("feature_source") or "unknown_source"
    calibration = ""
    if finding.get("calibration_youden_j") is not None:
        calibration = (
            f", sensitivity={finding.get('calibration_sensitivity')}, "
            f"specificity={finding.get('calibration_specificity')}, "
            f"youden_j={finding.get('calibration_youden_j')}"
        )

    return (
        f"{finding['label']}: {feature_name}={finding['value']:.4f}, "
        f"threshold={finding['threshold']}, direction={finding['direction']}, "
        f"source={source}{calibration}, summary={finding['summary']}"
    )


def _format_path_sources(explanation):
    source_lines = []

    for step in explanation.get("steps", []):
        sources = step.get("sources_to_next") or []
        if not sources:
            continue
        relation = step.get("relation_to_next") or "related_to"
        source_lines.append(
            f"{step['name']} --{relation}--> supported by {', '.join(sources)}"
        )

    return source_lines


def _aggregate_sentence(aggregate_evidence):
    if not aggregate_evidence:
        return "Aggregate evidence scoring was not computed."

    return (
        f"Aggregate evidence level: {aggregate_evidence['evidence_level']}; "
        f"matched {aggregate_evidence['matched_biomarker_count']} of "
        f"{aggregate_evidence['available_biomarker_count']} available calibrated biomarkers. "
        f"Screen positive: {aggregate_evidence['screen_positive']} using the "
        f"matched-count threshold of {aggregate_evidence['aggregate_match_threshold']}."
    )


def _recursive_context_sentence(recursive_context):
    if not recursive_context or not recursive_context.get("enabled"):
        return "No previous-week context was used."

    warning = recursive_context.get("warning")
    summary = recursive_context.get("summary")
    if warning:
        return f"Recursive context: {summary} Note: {warning}"

    return f"Recursive context: {summary}"


def _deterministic_summary(
    matched_findings,
    explanations,
    aggregate_evidence=None,
    recursive_context=None
):
    summary = [
        _aggregate_sentence(aggregate_evidence),
        _recursive_context_sentence(recursive_context),
    ]

    if not matched_findings:
        summary.append(
            "No rule-based risk markers crossed the current calibrated thresholds."
        )
        return summary

    for finding in matched_findings:
        feature_name = finding["feature"]
        if finding.get("feature_index") is not None:
            feature_name = f"{feature_name}[{finding['feature_index']}]"

        summary.append(
            f"{finding['label']} ({feature_name}={finding['value']:.4f}) crossed the "
            f"{finding['direction']} threshold of {finding['threshold']}. "
            f"{finding['summary']}"
        )

    for explanation in explanations:
        summary.append(
            f"Graph explanation: {_path_to_sentence(_path_labels(explanation))}."
        )

    return summary


def _plain_biomarker_label(label):
    replacements = {
        "High Jitter": "less steady voice quality",
        "High Pause Ratio": "more or longer pauses",
        "Low Speech Rate": "slower speaking pace",
        "Low Vocal Energy": "lower voice energy",
        "High Pitch Variability": "more variation in pitch",
        "High Mean Pitch": "higher average pitch",
        "Low Mean Pitch": "lower average pitch",
        "High MFCC 1 Mean": "broad voice-spectrum change",
        "High MFCC 2 Variability": "more variation in voice spectrum",
        "Low MFCC 3 Mean": "voice-spectrum balance change",
        "High MFCC 4 Variability": "more mid-spectrum voice variation",
    }
    return replacements.get(label, label.lower())


def _patient_facing_summary(
    matched_findings,
    aggregate_evidence=None,
    recursive_context=None
):
    aggregate_evidence = aggregate_evidence or {}
    matched_count = aggregate_evidence.get("matched_biomarker_count", len(matched_findings))
    available_count = aggregate_evidence.get("available_biomarker_count")
    screen_positive = aggregate_evidence.get("screen_positive")
    evidence_level = aggregate_evidence.get("evidence_level", "unknown")

    if screen_positive:
        opening = (
            "The audio screening result is elevated. This means several speech and voice signs "
            "crossed the system's calibrated screening thresholds."
        )
    else:
        opening = (
            "The audio screening result is below the calibrated operating point. Some speech or "
            "voice signs may still be present, but the overall pattern was not strong enough for "
            "a positive screen."
        )

    summary = [
        opening,
        (
            "This is not a diagnosis. It is a screening-support result that should be interpreted "
            "together with a validated questionnaire and, if needed, a qualified professional."
        ),
    ]

    if available_count is not None:
        summary.append(
            f"The system found {matched_count} matching audio signs out of {available_count} available signs. "
            f"The overall evidence level was {evidence_level}."
        )

    if matched_findings:
        simple_signs = [
            _plain_biomarker_label(finding["label"])
            for finding in matched_findings[:5]
        ]
        summary.append(
            "The most understandable audio signs were: "
            + ", ".join(simple_signs)
            + "."
        )
    else:
        summary.append(
            "No calibrated audio sign crossed its current screening threshold."
        )

    if recursive_context and recursive_context.get("enabled"):
        context = recursive_context.get("summary")
        warning = recursive_context.get("warning")
        if context:
            summary.append(f"Compared with previous context: {context}")
        if warning:
            summary.append(
                "The previous-week context shown here is simulated prototype context, not real patient history."
            )

    summary.append(
        "A reasonable next step is to review this result alongside PHQ-8 or another validated screening tool, "
        "rather than using the audio result alone."
    )
    return summary


def _user_facing_summary(
    persona,
    matched_findings,
    explanations,
    aggregate_evidence=None,
    recursive_context=None
):
    if persona == "patient":
        return _patient_facing_summary(
            matched_findings,
            aggregate_evidence=aggregate_evidence,
            recursive_context=recursive_context
        )

    return _deterministic_summary(
        matched_findings,
        explanations,
        aggregate_evidence=aggregate_evidence,
        recursive_context=recursive_context
    )


def _format_prompt_chain(chain):
    return "\n".join(
        f"- {item['step']}: {item['instruction']}"
        for item in chain
    )


def build_report_prompt(
    explanations,
    features,
    file_name,
    findings,
    persona,
    aggregate_evidence=None,
    recursive_context=None
):
    matched_findings = [
        finding for finding in findings
        if finding["matched"]
    ]
    prompt_chain = get_persona_prompt_chain(persona)
    if persona == "patient":
        finding_lines = [
            f"{_plain_biomarker_label(finding['label'])}: {finding['summary']}"
            for finding in matched_findings
        ] or ["No calibrated audio sign crossed its current screening threshold."]
        path_lines = [
            "The matched audio signs are connected in the knowledge graph to speech and voice patterns "
            "that can be relevant for depression screening."
        ]
        source_lines = [
            "Do not include paper names or citations in the patient-facing report."
        ]
    else:
        finding_lines = [
            _format_feature_value(finding)
            for finding in matched_findings
        ] or ["No calibrated biomarker threshold was crossed."]
        path_lines = [
            _path_to_sentence(_path_labels(explanation))
            for explanation in explanations
        ] or ["No graph path was retrieved because no biomarker rule matched."]
        source_lines = [
            source_line
            for explanation in explanations
            for source_line in _format_path_sources(explanation)
        ] or ["No edge-level sources were retrieved."]

    compact_features = {
        "duration_seconds": features.get("duration_seconds"),
        "energy": features.get("energy"),
        "pitch_mean": features.get("pitch_mean"),
        "pitch_std": features.get("pitch_std"),
        "pause_ratio": features.get("pause_ratio"),
        "speech_rate": features.get("speech_rate"),
        "jitter": features.get("jitter"),
        "mfcc_mean_first_4": features.get("mfcc_mean", [])[:4],
        "mfcc_std_first_4": features.get("mfcc_std", [])[:4],
        "feature_sources": features.get("feature_sources", {}),
    }

    shared_prompt = [
        "You are the Reporter Agent in an agentic Graph-RAG system for explainable audio biomarkers.",
        "Follow the persona prompt chain exactly.",
        "Use only the supplied acoustic findings, aggregate evidence, recursive context, graph paths, and edge sources.",
        "Do not diagnose depression. Describe this as screening-oriented decision support.",
        "Mention PHQ-8 only as the validated reference construct used for calibration, not as an inferred questionnaire score.",
        "Do not recommend medication, emergency action, or treatment steps unless they are explicitly present in the supplied graph evidence.",
        "If aggregate evidence is not screen-positive, say that the acoustic evidence is below the calibrated operating point even if some markers matched.",
    ]

    if persona == "patient":
        shared_prompt.extend([
            "Patient persona rules:",
            "Write directly to the patient using calm plain language.",
            "Do not mention paper names, citations, Graph-RAG, knowledge graph internals, MFCC, jitter, spectral shift, thresholds, or numeric cutoffs.",
            "Do not discuss PHQ-8 as if the patient's PHQ-8 score is known.",
            "Use phrases like audio screening result, speech and voice signs, and follow-up with a qualified professional.",
            "Keep the report short and supportive.",
        ])
        headings = [
            "Write a short patient-facing report with these headings:",
            "1. What This Means",
            "2. Audio Signs Noticed",
            "3. Why This Is Not A Diagnosis",
            "4. Suggested Next Step",
        ]
    else:
        headings = [
            "Write a short report with these headings:",
            "1. Screening Interpretation",
            "2. Evidence From Audio",
            "3. Graph Reasoning",
            "4. Previous-Week Context",
            "5. Recommended Follow-up",
        ]

    return "\n".join(shared_prompt + [
        "",
        f"Persona: {persona}",
        "Persona prompt chain:",
        _format_prompt_chain(prompt_chain),
        "",
        f"Audio file: {file_name}",
        "",
        "Extracted acoustic features:",
        str(compact_features),
        "",
        "Aggregate evidence:",
        str(aggregate_evidence or {}),
        "",
        "Recursive previous-week context:",
        str(recursive_context or {}),
        "",
        "Matched calibrated biomarker rules:",
        "\n".join(f"- {line}" for line in finding_lines),
        "",
        "Retrieved knowledge-graph paths:",
        "\n".join(f"- {line}" for line in path_lines),
        "",
        "Edge-level sources:",
        "\n".join(f"- {line}" for line in source_lines),
        "",
        *headings,
    ])


def _safe_display_report(persona, user_facing_summary, llm_report=None):
    if persona == "patient":
        return "\n\n".join(user_facing_summary)

    return llm_report or "\n\n".join(user_facing_summary)


def generate_report(
    explanations,
    features,
    file_name,
    findings,
    persona="psychologist",
    llm_config=None,
    aggregate_evidence=None,
    recursive_context=None
):
    matched_findings = [
        finding for finding in findings
        if finding["matched"]
    ]
    prompt_chain = get_persona_prompt_chain(persona)

    user_facing_summary = _user_facing_summary(
        persona,
        matched_findings,
        explanations,
        aggregate_evidence=aggregate_evidence,
        recursive_context=recursive_context
    )

    report = {
        "file": file_name,
        "biomarkers": features,
        "rule_findings": findings,
        "aggregate_evidence": aggregate_evidence,
        "recursive_context": recursive_context,
        "explanation_paths": explanations,
        "clinical_summary": _deterministic_summary(
            matched_findings,
            explanations,
            aggregate_evidence=aggregate_evidence,
            recursive_context=recursive_context
        ),
        "user_facing_summary": user_facing_summary,
        "display_report": _safe_display_report(persona, user_facing_summary),
        "persona": persona,
        "persona_prompt_chain": prompt_chain,
        "llm_status": {
            "provider": "none",
            "model": None,
            "used": False,
            "error": None
        },
        "llm_report": None,
        "faithfulness_check": {
            "checked": False,
            "supported": None,
            "warnings": [],
            "supported_terms_mentioned": [],
        }
    }

    if llm_config is None:
        return report

    prompt = build_report_prompt(
        explanations=explanations,
        features=features,
        file_name=file_name,
        findings=findings,
        persona=persona,
        aggregate_evidence=aggregate_evidence,
        recursive_context=recursive_context
    )
    report["llm_prompt"] = prompt
    report["llm_status"] = {
        "provider": llm_config.provider,
        "model": llm_config.model,
        "used": False,
        "error": None
    }

    try:
        report["llm_report"] = generate_text(prompt, llm_config)
        report["llm_status"]["used"] = True
    except LLMUnavailableError as exc:
        report["llm_status"]["error"] = str(exc)

    report["display_report"] = _safe_display_report(
        persona,
        user_facing_summary,
        llm_report=report["llm_report"]
    )

    report["faithfulness_check"] = check_report_faithfulness(
        report["llm_report"],
        explanations,
        findings,
        aggregate_evidence=aggregate_evidence
    )

    return report
