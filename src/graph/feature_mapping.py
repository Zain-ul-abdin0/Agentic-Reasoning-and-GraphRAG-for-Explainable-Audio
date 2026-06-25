FEATURE_RULES = {
    "jitter": {
        "threshold": 0.015,
        "direction": "high",
        "node": "High Jitter",
        "summary": "Vocal instability can indicate autonomic arousal."
    },
    "pause_ratio": {
        "threshold": 0.80,
        "direction": "high",
        "node": "High Pause Ratio",
        "summary": "Long silence or hesitation can indicate speech disruption."
    },
    "energy": {
        "threshold": 0.015,
        "direction": "low",
        "node": "Low Energy",
        "summary": "Low vocal intensity can indicate reduced arousal."
    },
    "pitch_std": {
        "threshold": 80.0,
        "direction": "high",
        "node": "High Pitch Variability",
        "summary": "Large pitch variation can indicate unstable vocal control."
    }
}


def _rule_matches(value, threshold, direction):
    if direction == "high":
        return value > threshold

    if direction == "low":
        return value < threshold

    raise ValueError(f"Unsupported rule direction: {direction}")


def assess_features(features):
    findings = []

    for feature_name, rule in FEATURE_RULES.items():
        value = features.get(feature_name)
        if value is None:
            continue

        matched = _rule_matches(
            value,
            rule["threshold"],
            rule["direction"]
        )

        findings.append({
            "feature": feature_name,
            "value": value,
            "threshold": rule["threshold"],
            "direction": rule["direction"],
            "matched": matched,
            "node": rule["node"],
            "summary": rule["summary"]
        })

    return findings
