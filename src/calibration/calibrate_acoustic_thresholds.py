from pathlib import Path
import argparse
import csv
import json
import math
import statistics

from src.graph.knowledge_graph import SEED_GRAPH_PATH, load_graph_document


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_FILES = [
    BASE_DIR / "documentations" / "train_split_Depression_AVEC2017.csv",
    BASE_DIR / "documentations" / "dev_split_Depression_AVEC2017.csv",
    BASE_DIR / "documentations" / "full_test_split.csv",
]


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def read_csv_rows(path):
    path = Path(path)
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:2048]
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","

    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def parse_float(value):
    try:
        if value is None or value == "":
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except ValueError:
        return None


def participant_id_from_path(path, suffix):
    return int(path.name.replace(suffix, ""))


def load_phq_labels(label_files, phq_cutoff):
    labels = {}

    for label_file in label_files:
        label_path = resolve_path(label_file)
        if not label_path.exists():
            continue

        for row in read_csv_rows(label_path):
            participant = row.get("Participant_ID") or row.get("participant_ID")
            score = row.get("PHQ8_Score") or row.get("PHQ_Score")
            binary = row.get("PHQ8_Binary") or row.get("PHQ_Binary")

            if participant is None:
                continue

            participant_id = int(participant)
            score_value = parse_float(score)

            if score_value is not None:
                label = int(score_value >= phq_cutoff)
            elif binary not in (None, ""):
                label = int(float(binary))
            else:
                continue

            labels[participant_id] = {
                "phq_score": score_value,
                "label": label,
            }

    return labels


def tokenize(text):
    return [
        token
        for token in text.replace("'", " ").split()
        if token.strip()
    ]


def extract_transcript_features(dataset_dir):
    features = {}

    for path in resolve_path(dataset_dir).glob("*_TRANSCRIPT.csv"):
        participant_id = participant_id_from_path(path, "_TRANSCRIPT.csv")
        participant_turns = []

        for row in read_csv_rows(path):
            if row.get("speaker") != "Participant":
                continue

            start = parse_float(row.get("start_time"))
            stop = parse_float(row.get("stop_time"))
            if start is None or stop is None or stop <= start:
                continue

            participant_turns.append({
                "start": start,
                "stop": stop,
                "words": len(tokenize(row.get("value", ""))),
            })

        if not participant_turns:
            continue

        speech_duration = sum(
            turn["stop"] - turn["start"]
            for turn in participant_turns
        )
        word_count = sum(turn["words"] for turn in participant_turns)
        start_time = min(turn["start"] for turn in participant_turns)
        stop_time = max(turn["stop"] for turn in participant_turns)
        total_span = max(stop_time - start_time, speech_duration)
        pause_duration = max(total_span - speech_duration, 0.0)

        features.setdefault(participant_id, {}).update({
            "speech_rate": word_count / speech_duration if speech_duration else None,
            "pause_ratio": pause_duration / total_span if total_span else None,
            "participant_speech_duration": speech_duration,
            "participant_word_count": word_count,
        })

    return features


def stream_column_values(path, column_index, positive_only=False):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) <= column_index:
                continue

            value = parse_float(row[column_index])
            if value is None:
                continue
            if positive_only and value <= 0:
                continue

            yield value


def safe_mean(values):
    return statistics.fmean(values) if values else None


def safe_std(values):
    return statistics.pstdev(values) if len(values) > 1 else None


def extract_covarep_features(dataset_dir):
    features = {}

    for path in resolve_path(dataset_dir).glob("*_COVAREP.csv"):
        participant_id = participant_id_from_path(path, "_COVAREP.csv")

        # AVEC COVAREP files are frame-level. Column 0 behaves as F0 in the
        # distributed files: unvoiced frames are 0, voiced frames are positive.
        f0_values = list(stream_column_values(path, 0, positive_only=True))

        # The first few COVAREP columns are voice-source descriptors. They are
        # not universal clinical cutoffs, but can be calibrated as operating
        # thresholds against PHQ-8 labels.
        voice_values = list(stream_column_values(path, 2, positive_only=True))

        features.setdefault(participant_id, {}).update({
            "pitch_mean": safe_mean(f0_values),
            "pitch_std": safe_std(f0_values),
            "covarep_voice_source_mean": safe_mean(voice_values),
            "covarep_voice_source_std": safe_std(voice_values),
        })

    return features


def extract_feature_matrix(matrix_path):
    matrix_path = resolve_path(matrix_path)
    features = {}

    for row in read_csv_rows(matrix_path):
        participant = (
            row.get("participant_id")
            or row.get("Participant_ID")
            or row.get("participant")
            or row.get("Participant")
        )
        if participant is None:
            continue

        participant_id = int(float(participant))
        values = {}

        for key, value in row.items():
            if key in {
                "participant_id",
                "Participant_ID",
                "participant",
                "Participant",
                "phq_score",
                "PHQ8_Score",
                "PHQ_Score",
                "phq8_binary",
                "PHQ8_Binary",
                "PHQ_Binary",
            }:
                continue

            parsed = parse_float(value)
            if parsed is not None:
                values[key] = parsed

        features[participant_id] = values

    return features


def merge_feature_sources(*sources):
    merged = {}
    for source in sources:
        for participant_id, values in source.items():
            clean_values = {
                key: value
                for key, value in values.items()
                if value is not None
            }
            merged.setdefault(participant_id, {}).update(clean_values)
    return merged


def roc_threshold(values, labels, direction):
    pairs = sorted(
        (value, label)
        for value, label in zip(values, labels)
        if value is not None
    )
    positives = sum(label == 1 for _, label in pairs)
    negatives = sum(label == 0 for _, label in pairs)

    if positives == 0 or negatives == 0:
        return None

    best = None
    candidate_thresholds = sorted(set(value for value, _ in pairs))

    for threshold in candidate_thresholds:
        predictions = []
        for value, _ in pairs:
            if direction == "high":
                predictions.append(int(value > threshold))
            elif direction == "low":
                predictions.append(int(value < threshold))
            else:
                raise ValueError(f"Unsupported direction: {direction}")

        tp = sum(
            prediction == 1 and label == 1
            for prediction, (_, label) in zip(predictions, pairs)
        )
        tn = sum(
            prediction == 0 and label == 0
            for prediction, (_, label) in zip(predictions, pairs)
        )
        fp = sum(
            prediction == 1 and label == 0
            for prediction, (_, label) in zip(predictions, pairs)
        )
        fn = sum(
            prediction == 0 and label == 1
            for prediction, (_, label) in zip(predictions, pairs)
        )

        sensitivity = tp / positives
        specificity = tn / negatives
        youden_j = sensitivity + specificity - 1
        balanced_accuracy = (sensitivity + specificity) / 2

        candidate = {
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "youden_j": youden_j,
            "balanced_accuracy": balanced_accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

        if best is None or candidate["youden_j"] > best["youden_j"]:
            best = candidate

    return best


def calibration_feature_name(feature_key, feature_index=None):
    if feature_index is not None and feature_key in {"mfcc_mean", "mfcc_std"}:
        return f"{feature_key}_{feature_index}"
    return feature_key


def graph_biomarker_specs(graph_path, available_features):
    document = load_graph_document(graph_path)
    specs = []
    seen = set()

    for node in document["nodes"]:
        if node.get("type") != "Biomarker":
            continue

        feature_name = calibration_feature_name(
            node.get("feature_key"),
            node.get("feature_index"),
        )

        if feature_name not in available_features:
            continue

        spec_key = (node["id"], feature_name, node["direction"])
        if spec_key in seen:
            continue
        seen.add(spec_key)

        specs.append({
            "node_id": node["id"],
            "label": node["name"],
            "feature": feature_name,
            "direction": node["direction"],
        })

    return specs


def calibrate(features, labels, specs):
    rows = []

    for spec in specs:
        feature_name = spec["feature"]
        values = []
        target_labels = []
        participant_ids = []

        for participant_id, feature_values in features.items():
            if participant_id not in labels:
                continue
            if feature_name not in feature_values:
                continue
            value = feature_values[feature_name]
            if value is None:
                continue
            values.append(value)
            target_labels.append(labels[participant_id]["label"])
            participant_ids.append(participant_id)

        direction = spec["direction"]
        result = roc_threshold(values, target_labels, direction)

        if result is None:
            rows.append({
                "node_id": spec["node_id"],
                "label": spec["label"],
                "feature": feature_name,
                "direction": direction,
                "status": "not_calibrated",
                "reason": "Requires both depressed and non-depressed labeled examples.",
                "n": len(values),
                "positives": sum(target_labels),
                "negatives": len(target_labels) - sum(target_labels),
            })
            continue

        rows.append({
            "node_id": spec["node_id"],
            "label": spec["label"],
            "feature": feature_name,
            "direction": direction,
            "status": "calibrated",
            "n": len(values),
            "positives": sum(target_labels),
            "negatives": len(target_labels) - sum(target_labels),
            "threshold": result["threshold"],
            "sensitivity": result["sensitivity"],
            "specificity": result["specificity"],
            "youden_j": result["youden_j"],
            "balanced_accuracy": result["balanced_accuracy"],
            "tp": result["tp"],
            "tn": result["tn"],
            "fp": result["fp"],
            "fn": result["fn"],
            "participant_ids": participant_ids,
        })

    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "node_id",
        "label",
        "feature",
        "direction",
        "status",
        "reason",
        "n",
        "positives",
        "negatives",
        "threshold",
        "sensitivity",
        "specificity",
        "youden_j",
        "balanced_accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key)
                for key in fieldnames
            })


def write_feature_matrix(path, features, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = sorted({
        feature_name
        for values in features.values()
        for feature_name in values
    })
    fieldnames = [
        "participant_id",
        "phq_score",
        "phq8_binary",
        *feature_names,
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for participant_id in sorted(features):
            if participant_id not in labels:
                continue
            row = {
                "participant_id": participant_id,
                "phq_score": labels[participant_id]["phq_score"],
                "phq8_binary": labels[participant_id]["label"],
            }
            row.update(features[participant_id])
            writer.writerow(row)


def update_graph_thresholds(graph_path, calibration_rows, output_path):
    document = load_graph_document(graph_path)
    calibrated = {
        row["node_id"]: row
        for row in calibration_rows
        if row.get("status") == "calibrated"
    }

    updated = []
    for node in document["nodes"]:
        if node.get("type") == "Biomarker":
            node["calibration_status"] = "not_calibrated"
            node["active_for_threshold_detection"] = False

        if node.get("id") not in calibrated:
            continue

        row = calibrated[node["id"]]
        node["threshold"] = row["threshold"]
        node["direction"] = row["direction"]
        node["threshold_source"] = "DAIC-WOZ PHQ-8 ROC calibration"
        node["calibration_reference_label"] = "PHQ-8 >= 10"
        node["calibration_status"] = "calibrated"
        node["active_for_threshold_detection"] = True
        node["calibration_n"] = row["n"]
        node["calibration_sensitivity"] = row["sensitivity"]
        node["calibration_specificity"] = row["specificity"]
        node["calibration_youden_j"] = row["youden_j"]
        updated.append(node["id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate acoustic operating thresholds against PHQ-8 labels."
    )
    parser.add_argument("--dataset-dir", default="Dataset")
    parser.add_argument(
        "--feature-matrix",
        default=None,
        help=(
            "Optional participant-level CSV containing acoustic features such as "
            "jitter, energy, mfcc_mean_0, or mfcc_std_1."
        ),
    )
    parser.add_argument(
        "--include-covarep",
        action="store_true",
        help="Also scan COVAREP frame files for pitch/voice-source summaries.",
    )
    parser.add_argument(
        "--label-file",
        action="append",
        default=None,
        help="CSV with Participant_ID and PHQ8_Score/PHQ_Score columns. Can be repeated.",
    )
    parser.add_argument("--phq-cutoff", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        default="data/month_02_knowledge_graph/calibration",
    )
    parser.add_argument("--graph", default=str(SEED_GRAPH_PATH))
    parser.add_argument(
        "--calibrated-graph",
        default="data/knowledge_graph/siegen_audio_anxiety_kg.calibrated.jsonld",
    )

    args = parser.parse_args()
    output_dir = resolve_path(args.output_dir)
    label_files = args.label_file or DEFAULT_LABEL_FILES

    labels = load_phq_labels(label_files, args.phq_cutoff)
    if not labels:
        raise ValueError("No PHQ labels found. Provide --label-file with PHQ scores.")

    feature_sources = [extract_transcript_features(args.dataset_dir)]
    if args.include_covarep:
        feature_sources.append(extract_covarep_features(args.dataset_dir))
    if args.feature_matrix:
        feature_sources.append(extract_feature_matrix(args.feature_matrix))

    features = merge_feature_sources(*feature_sources)

    available_features = {
        feature_name
        for values in features.values()
        for feature_name in values
    }
    specs = graph_biomarker_specs(resolve_path(args.graph), available_features)
    rows = calibrate(features, labels, specs)

    write_csv(output_dir / "calibrated_thresholds.csv", rows)
    write_feature_matrix(output_dir / "participant_feature_matrix.csv", features, labels)

    summary = {
        "reference_label": f"PHQ-8 >= {args.phq_cutoff:g}",
        "participants_with_labels": len(labels),
        "participants_with_features": len(features),
        "calibrated_features": [
            row
            for row in rows
            if row.get("status") == "calibrated"
        ],
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    updated_nodes = update_graph_thresholds(
        resolve_path(args.graph),
        rows,
        resolve_path(args.calibrated_graph),
    )
    summary["updated_graph_nodes"] = updated_nodes
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
