def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def binary_classification_metrics(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")

    true_positive = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 1)
    true_negative = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 0)
    false_positive = sum(1 for label, pred in zip(labels, predictions) if label == 0 and pred == 1)
    false_negative = sum(1 for label, pred in zip(labels, predictions) if label == 1 and pred == 0)

    sensitivity = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    precision = safe_divide(true_positive, true_positive + false_positive)
    f1_score = safe_divide(2 * precision * sensitivity, precision + sensitivity)

    return {
        "n": len(labels),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": safe_divide(true_positive + true_negative, len(labels)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0

