from src.graph.knowledge_graph import get_biomarker_rules


def _rule_matches(value, threshold, direction):
    if direction == "high":
        return value > threshold

    if direction == "low":
        return value < threshold

    raise ValueError(f"Unsupported rule direction: {direction}")


def _read_feature_value(features, feature_key, feature_index=None):
    value = features.get(feature_key)

    if value is None:
        return None

    if feature_index is None:
        return value

    if feature_index >= len(value):
        return None

    return value[feature_index]


def assess_features(features, graph=None):
    feature_rules = get_biomarker_rules(graph)
    findings = []

    for rule in feature_rules.values():
        value = _read_feature_value(
            features,
            rule["feature_key"],
            rule["feature_index"]
        )
        if value is None:
            continue

        matched = _rule_matches(
            value,
            rule["threshold"],
            rule["direction"]
        )

        findings.append({
            "feature": rule["feature_key"],
            "feature_index": rule["feature_index"],
            "value": value,
            "threshold": rule["threshold"],
            "direction": rule["direction"],
            "matched": matched,
            "node": rule["node"],
            "label": rule["label"],
            "summary": rule["summary"]
        })

    return findings
