from src.rag.llm_client import LLMUnavailableError, LocalLLMConfig, generate_text


PERSONA_INSTRUCTIONS = {
    "psychologist": (
        "Write for a psychologist. Use concise clinical language, mention the "
        "matched acoustic biomarkers, explain the graph reasoning paths, and "
        "keep the conclusion as screening support rather than diagnosis."
    ),
    "patient": (
        "Write for a patient. Use plain, non-alarming language, avoid technical "
        "jargon where possible, and clearly say that the audio result is not a "
        "diagnosis."
    ),
}


def _path_to_sentence(path):
    return " -> ".join(path)


def _path_labels(explanation):
    return explanation.get("path_labels", explanation["path"])


def _format_feature_value(finding):
    feature_name = finding["feature"]
    if finding.get("feature_index") is not None:
        feature_name = f"{feature_name}[{finding['feature_index']}]"

    return (
        f"{finding['label']}: {feature_name}={finding['value']:.4f}, "
        f"threshold={finding['threshold']}, direction={finding['direction']}, "
        f"summary={finding['summary']}"
    )


def _deterministic_summary(matched_findings, explanations):
    summary = []

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


def build_report_prompt(explanations, features, file_name, findings, persona):
    matched_findings = [
        finding for finding in findings
        if finding["matched"]
    ]
    persona_instruction = PERSONA_INSTRUCTIONS.get(
        persona,
        PERSONA_INSTRUCTIONS["psychologist"]
    )
    finding_lines = [
        _format_feature_value(finding)
        for finding in matched_findings
    ] or ["No calibrated biomarker threshold was crossed."]
    path_lines = [
        _path_to_sentence(_path_labels(explanation))
        for explanation in explanations
    ] or ["No graph path was retrieved because no biomarker rule matched."]

    compact_features = {
        "duration_seconds": features.get("duration_seconds"),
        "energy": features.get("energy"),
        "pitch_mean": features.get("pitch_mean"),
        "pitch_std": features.get("pitch_std"),
        "pause_ratio": features.get("pause_ratio"),
        "jitter": features.get("jitter"),
        "mfcc_mean_first_4": features.get("mfcc_mean", [])[:4],
        "mfcc_std_first_4": features.get("mfcc_std", [])[:4],
    }

    return "\n".join([
        "You are the Reporter Agent in a Graph-RAG system for explainable audio biomarkers.",
        persona_instruction,
        "Use only the supplied acoustic findings and graph paths.",
        "Do not diagnose depression. Describe this as screening-oriented decision support.",
        "Mention PHQ-8 only as the clinical reference construct, not as an inferred questionnaire score.",
        "",
        f"Audio file: {file_name}",
        "",
        "Extracted acoustic features:",
        str(compact_features),
        "",
        "Matched calibrated biomarker rules:",
        "\n".join(f"- {line}" for line in finding_lines),
        "",
        "Retrieved knowledge-graph paths:",
        "\n".join(f"- {line}" for line in path_lines),
        "",
        "Write a short report with these headings:",
        "1. Screening Interpretation",
        "2. Evidence From Audio",
        "3. Graph Reasoning",
        "4. Recommended Follow-up",
    ])


def generate_report(
    explanations,
    features,
    file_name,
    findings,
    persona="psychologist",
    llm_config=None
):
    matched_findings = [
        finding for finding in findings
        if finding["matched"]
    ]

    report = {
        "file": file_name,
        "biomarkers": features,
        "rule_findings": findings,
        "explanation_paths": explanations,
        "clinical_summary": _deterministic_summary(matched_findings, explanations),
        "persona": persona,
        "llm_status": {
            "provider": "none",
            "model": None,
            "used": False,
            "error": None
        },
        "llm_report": None
    }

    if llm_config is None:
        return report

    prompt = build_report_prompt(
        explanations=explanations,
        features=features,
        file_name=file_name,
        findings=findings,
        persona=persona
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

    return report
