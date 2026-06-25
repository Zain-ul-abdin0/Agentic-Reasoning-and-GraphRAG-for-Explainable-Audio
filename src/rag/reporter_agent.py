def _path_to_sentence(path):
    return " -> ".join(path)


def generate_report(explanations, features, file_name, findings):
    matched_findings = [
        finding for finding in findings
        if finding["matched"]
    ]

    report = {
        "file": file_name,
        "biomarkers": features,
        "rule_findings": findings,
        "explanation_paths": explanations,
        "clinical_summary": []
    }

    if not matched_findings:
        report["clinical_summary"].append(
            "No rule-based risk markers crossed the current baseline thresholds."
        )
        return report

    for finding in matched_findings:
        report["clinical_summary"].append(
            f"{finding['feature']}={finding['value']:.4f} crossed the "
            f"{finding['direction']} threshold of {finding['threshold']}. "
            f"{finding['summary']}"
        )

    for explanation in explanations:
        report["clinical_summary"].append(
            f"Graph explanation: {_path_to_sentence(explanation['path'])}."
        )

    return report
