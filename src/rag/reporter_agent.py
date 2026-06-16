def generate_report(paths, features, file_name):

    report = {
        "file": file_name,
        "biomarkers": features,
        "explanation_paths": paths,
        "clinical_summary": []
    }

    for path in paths:
        if "Vocal Instability" in path:
            report["clinical_summary"].append(
                "Voice instability detected → possible anxiety markers"
            )

        if "Speech Disruption" in path:
            report["clinical_summary"].append(
                "Speech irregularity suggests cognitive load"
            )

    return report