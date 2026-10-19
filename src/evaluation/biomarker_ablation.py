from pathlib import Path
import argparse
import csv
import json

from src.evaluation.metrics import binary_classification_metrics


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PARTICIPANT_RESULTS = BASE_DIR / "data" / "month_05_evaluation" / "participant_results.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "month_05_evaluation" / "biomarker_ablation"


BIOMARKER_GROUPS = {
    "transcript_timing": {
        "description": "Timing biomarkers estimated from transcript speaker turns when available.",
        "biomarkers": {"High Pause Ratio", "Low Speech Rate"},
    },
    "covarep_voice": {
        "description": "Pitch and voice-source biomarkers from COVAREP when available, with audio fallback.",
        "biomarkers": {"High Jitter", "High Pitch Variability", "High Mean Pitch", "Low Mean Pitch"},
    },
    "audio_waveform_spectral": {
        "description": "Energy and MFCC biomarkers extracted directly from the waveform.",
        "biomarkers": {
            "Low Vocal Energy",
            "High MFCC 1 Mean",
            "High MFCC 2 Variability",
            "Low MFCC 3 Mean",
            "High MFCC 4 Variability",
        },
    },
}


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_matched_biomarkers(value):
    if not value:
        return set()
    return {item.strip() for item in value.split(";") if item.strip()}


def load_participant_rows(path):
    rows = []
    for row in read_csv(path):
        rows.append({
            "participant_id": row["participant_id"],
            "phq8_score": float(row["phq8_score"]) if row.get("phq8_score") else None,
            "label": int(float(row["phq8_label"])),
            "baseline_prediction": int(float(row["prediction"])),
            "matched_biomarkers": parse_matched_biomarkers(row.get("matched_biomarkers", "")),
        })
    return rows


def all_biomarkers(rows):
    markers = set()
    for row in rows:
        markers.update(row["matched_biomarkers"])
    for group in BIOMARKER_GROUPS.values():
        markers.update(group["biomarkers"])
    return sorted(markers)


def labels(rows):
    return [row["label"] for row in rows]


def predictions_from_counts(counts, threshold):
    return [int(count >= threshold) for count in counts]


def best_count_threshold(counts, truth):
    max_count = max(counts) if counts else 0
    candidates = range(1, max_count + 1) if max_count else [1]
    best = None
    for threshold in candidates:
        predictions = predictions_from_counts(counts, threshold)
        metrics = binary_classification_metrics(truth, predictions)
        candidate = {"threshold": threshold, "metrics": metrics}
        if best is None:
            best = candidate
            continue
        current = metrics["balanced_accuracy"]
        previous = best["metrics"]["balanced_accuracy"]
        if current > previous:
            best = candidate
        elif current == previous and metrics["accuracy"] > best["metrics"]["accuracy"]:
            best = candidate
    return best


def count_without(row, removed):
    return len(row["matched_biomarkers"] - set(removed))


def evaluate_removed_set(rows, removed, fixed_threshold):
    truth = labels(rows)
    counts = [count_without(row, removed) for row in rows]
    fixed_predictions = predictions_from_counts(counts, fixed_threshold)
    fixed_metrics = binary_classification_metrics(truth, fixed_predictions)
    optimized = best_count_threshold(counts, truth)
    return {
        "fixed_threshold": fixed_threshold,
        "fixed_metrics": fixed_metrics,
        "optimized_threshold": optimized["threshold"],
        "optimized_metrics": optimized["metrics"],
    }


def evaluate_single_biomarker(rows, biomarker):
    truth = labels(rows)
    predictions = [int(biomarker in row["matched_biomarkers"]) for row in rows]
    metrics = binary_classification_metrics(truth, predictions)
    return {
        "biomarker": biomarker,
        "matched_participants": sum(predictions),
        **metrics,
    }


def flatten_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_markdown(path, summary):
    baseline = summary["baseline_metrics"]
    optimized = summary["optimized_baseline_metrics"]
    lines = [
        "# Biomarker Contribution Analysis",
        "",
        "## Purpose",
        "",
        "This analysis estimates how individual biomarkers and feature-source groups affect screening performance against the PHQ-8 >= 10 reference label.",
        "",
        "## Baseline",
        "",
        f"- Participants: {summary['participants']}",
        f"- Deployed matched-biomarker threshold: {summary['deployed_threshold']}",
        f"- Deployed balanced accuracy: {baseline['balanced_accuracy']:.3f}",
        f"- Deployed sensitivity: {baseline['sensitivity']:.3f}",
        f"- Deployed specificity: {baseline['specificity']:.3f}",
        f"- Optimized baseline threshold: {summary['optimized_baseline_threshold']}",
        f"- Optimized baseline balanced accuracy: {optimized['balanced_accuracy']:.3f}",
        "",
        "## Most Important Biomarkers By Leave-One-Out",
        "",
    ]
    for row in summary["most_important_by_leave_one_out"]:
        lines.append(
            f"- {row['removed_biomarker']}: optimized balanced-accuracy delta "
            f"{row['optimized_balanced_accuracy_delta']:.3f}; matched in "
            f"{row['participants_where_biomarker_matched']} participants."
        )
    lines.extend(["", "## Best Single Biomarkers", ""])
    for row in summary["best_single_biomarkers"]:
        lines.append(
            f"- {row['biomarker']}: balanced accuracy {row['balanced_accuracy']:.3f}, "
            f"sensitivity {row['sensitivity']:.3f}, specificity {row['specificity']:.3f}."
        )
    lines.extend(["", "## Feature-Source Group Ablation", ""])
    for row in summary["group_ablation"]:
        lines.append(
            f"- {row['removed_group']}: optimized balanced-accuracy delta "
            f"{row['optimized_balanced_accuracy_delta']:.3f}. "
            f"Removed biomarkers: {row['removed_biomarkers']}."
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ablation(participant_results, output_dir, deployed_threshold=6):
    rows = load_participant_rows(participant_results)
    marker_names = all_biomarkers(rows)
    truth = labels(rows)
    baseline_predictions = [row["baseline_prediction"] for row in rows]
    baseline_metrics = binary_classification_metrics(truth, baseline_predictions)
    baseline_counts = [len(row["matched_biomarkers"]) for row in rows]
    optimized_baseline = best_count_threshold(baseline_counts, truth)

    leave_one_rows = []
    for biomarker in marker_names:
        result = evaluate_removed_set(rows, {biomarker}, deployed_threshold)
        leave_one_rows.append({
            "removed_biomarker": biomarker,
            "participants_where_biomarker_matched": sum(1 for row in rows if biomarker in row["matched_biomarkers"]),
            "fixed_balanced_accuracy_delta": result["fixed_metrics"]["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
            "optimized_balanced_accuracy_delta": result["optimized_metrics"]["balanced_accuracy"] - optimized_baseline["metrics"]["balanced_accuracy"],
            "optimized_threshold_after_removal": result["optimized_threshold"],
            **flatten_metrics("fixed", result["fixed_metrics"]),
            **flatten_metrics("optimized", result["optimized_metrics"]),
        })

    single_rows = [evaluate_single_biomarker(rows, biomarker) for biomarker in marker_names]

    group_rows = []
    for group_name, group in BIOMARKER_GROUPS.items():
        result = evaluate_removed_set(rows, group["biomarkers"], deployed_threshold)
        group_rows.append({
            "removed_group": group_name,
            "description": group["description"],
            "removed_biomarkers": "; ".join(sorted(group["biomarkers"])),
            "fixed_balanced_accuracy_delta": result["fixed_metrics"]["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
            "optimized_balanced_accuracy_delta": result["optimized_metrics"]["balanced_accuracy"] - optimized_baseline["metrics"]["balanced_accuracy"],
            "optimized_threshold_after_removal": result["optimized_threshold"],
            **flatten_metrics("fixed", result["fixed_metrics"]),
            **flatten_metrics("optimized", result["optimized_metrics"]),
        })

    leave_one_rows = sorted(leave_one_rows, key=lambda row: row["optimized_balanced_accuracy_delta"])
    single_rows = sorted(single_rows, key=lambda row: row["balanced_accuracy"], reverse=True)
    group_rows = sorted(group_rows, key=lambda row: row["optimized_balanced_accuracy_delta"])

    output_dir = resolve_path(output_dir)
    summary = {
        "analysis": "Biomarker contribution and ablation analysis",
        "participant_results": str(resolve_path(participant_results)),
        "participants": len(rows),
        "reference_label": "PHQ-8 >= 10",
        "deployed_threshold": deployed_threshold,
        "baseline_metrics": baseline_metrics,
        "optimized_baseline_threshold": optimized_baseline["threshold"],
        "optimized_baseline_metrics": optimized_baseline["metrics"],
        "most_important_by_leave_one_out": leave_one_rows[:5],
        "best_single_biomarkers": single_rows[:5],
        "group_ablation": group_rows,
        "interpretation": "Negative delta means performance decreased after removing that biomarker or group. A larger negative delta indicates stronger contribution to the screening rule.",
    }
    write_json(output_dir / "biomarker_ablation_summary.json", summary)
    write_csv(output_dir / "leave_one_biomarker_out.csv", leave_one_rows)
    write_csv(output_dir / "single_biomarker_performance.csv", single_rows)
    write_csv(output_dir / "feature_source_group_ablation.csv", group_rows)
    write_markdown(output_dir / "biomarker_ablation_analysis.md", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate how biomarkers affect PHQ-8 screening performance.")
    parser.add_argument("--participant-results", default=str(DEFAULT_PARTICIPANT_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--deployed-threshold", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_ablation(
        participant_results=args.participant_results,
        output_dir=args.output_dir,
        deployed_threshold=args.deployed_threshold,
    )
    print(json.dumps(result, indent=2))
