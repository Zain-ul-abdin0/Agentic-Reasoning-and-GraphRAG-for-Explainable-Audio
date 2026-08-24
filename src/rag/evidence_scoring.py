AGGREGATE_MATCH_THRESHOLD = 6

AGGREGATE_CALIBRATION = {
    "reference_label": "PHQ-8 >= 10",
    "dataset": "DAIC-WOZ participant-level feature matrix",
    "method": "Rule-count threshold selected by balanced accuracy after single-feature ROC/Youden calibration.",
    "matched_count_threshold": AGGREGATE_MATCH_THRESHOLD,
    "true_positive": 38,
    "true_negative": 41,
    "false_positive": 15,
    "false_negative": 17,
    "sensitivity": 0.6909,
    "specificity": 0.7321,
    "balanced_accuracy": 0.7115,
}


def score_evidence(findings, threshold=AGGREGATE_MATCH_THRESHOLD):
    matched_findings = [
        finding for finding in findings
        if finding.get("matched")
    ]
    available_findings = [
        finding for finding in findings
        if finding.get("value") is not None
    ]
    matched_count = len(matched_findings)

    calibration_scores = []
    for finding in matched_findings:
        youden = finding.get("calibration_youden_j")
        if youden is not None:
            calibration_scores.append(max(float(youden), 0.0))

    calibration_support_score = sum(calibration_scores)
    screen_positive = matched_count >= threshold

    if screen_positive:
        evidence_level = "elevated"
    elif matched_count >= 4:
        evidence_level = "borderline"
    elif matched_count > 0:
        evidence_level = "limited"
    else:
        evidence_level = "none"

    return {
        "screen_positive": screen_positive,
        "evidence_level": evidence_level,
        "matched_biomarker_count": matched_count,
        "available_biomarker_count": len(available_findings),
        "aggregate_match_threshold": threshold,
        "calibration_support_score": round(calibration_support_score, 4),
        "matched_biomarkers": [
            finding["label"]
            for finding in matched_findings
        ],
        "interpretation": (
            "Aggregate acoustic evidence meets the calibrated screening operating point."
            if screen_positive
            else "Aggregate acoustic evidence does not meet the calibrated screening operating point."
        ),
        "calibration": AGGREGATE_CALIBRATION,
    }
