AGGREGATE_MATCH_THRESHOLD = 5

AGGREGATE_CALIBRATION = {
    "reference_label": "PHQ-8 >= 10",
    "dataset": "113 saved DAIC-WOZ participant results",
    "method": "Exploratory count-threshold optimization after excluding High MFCC 1 Mean.",
    "validation_status": "Same-sample feature selection and tuning; independent validation pending.",
    "excluded_biomarkers": [
        "High MFCC 1 Mean"
    ],
    "matched_count_threshold": 5,
    "true_positive": 52,
    "true_negative": 38,
    "false_positive": 19,
    "false_negative": 4,
    "sensitivity": 0.9285714285714286,
    "specificity": 0.6666666666666666,
    "balanced_accuracy": 0.7976190476190477
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
