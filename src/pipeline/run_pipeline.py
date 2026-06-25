from pathlib import Path
import argparse
import csv
import json

from src.feature_extraction import AudioFeatureExtractor
from src.graph.feature_mapping import assess_features
from src.graph.knowledge_graph import build_graph, terminal_risk_nodes
from src.rag.graph_rag import get_all_explanations
from src.rag.reporter_agent import generate_report


BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_path(input_path: str) -> Path:
    path = Path(input_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def resolve_audio_path(input_path: str) -> Path:
    path = resolve_path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {path}")

    return path


def analyze_audio(audio_file: Path):
    extractor = AudioFeatureExtractor(str(audio_file))
    features = extractor.extract_all()
    findings = assess_features(features)

    graph = build_graph()
    explanations = []

    for finding in findings:
        if finding["matched"]:
            explanations.extend(
                get_all_explanations(
                    graph,
                    finding["node"],
                    terminal_risk_nodes()
                )
            )

    return generate_report(
        explanations=explanations,
        features=features,
        file_name=audio_file.name,
        findings=findings
    )


def find_audio_files(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    return sorted(dataset_dir.glob("*_AUDIO.wav"))


def flatten_report(report):
    features = report["biomarkers"]
    matched_nodes = [
        finding["node"]
        for finding in report["rule_findings"]
        if finding["matched"]
    ]

    return {
        "file": report["file"],
        "duration_seconds": features["duration_seconds"],
        "energy": features["energy"],
        "pitch_mean": features["pitch_mean"],
        "pitch_std": features["pitch_std"],
        "pause_ratio": features["pause_ratio"],
        "jitter": features["jitter"],
        "matched_rules": "; ".join(matched_nodes),
        "explanation_targets": "; ".join(
            explanation["target"]
            for explanation in report["explanation_paths"]
        )
    }


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "duration_seconds",
        "energy",
        "pitch_mean",
        "pitch_std",
        "pause_ratio",
        "jitter",
        "matched_rules",
        "explanation_targets"
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single(audio_path: str, output_path: str | None):
    audio_file = resolve_audio_path(audio_path)
    report = analyze_audio(audio_file)

    if output_path:
        save_json(resolve_path(output_path), report)

    print(json.dumps(report, indent=2))


def run_batch(dataset_dir: str, output_dir: str):
    dataset_path = resolve_path(dataset_dir)
    output_path = resolve_path(output_dir)
    audio_files = find_audio_files(dataset_path)

    if not audio_files:
        raise FileNotFoundError(f"No *_AUDIO.wav files found in: {dataset_path}")

    reports = []
    rows = []

    for audio_file in audio_files:
        report = analyze_audio(audio_file)
        reports.append(report)
        rows.append(flatten_report(report))

    save_json(output_path / "reports.json", reports)
    save_csv(output_path / "features.csv", rows)

    print(f"Processed {len(audio_files)} audio files")
    print(f"Reports saved to {output_path / 'reports.json'}")
    print(f"Feature table saved to {output_path / 'features.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Explainable audio analysis pipeline"
    )

    parser.add_argument(
        "audio_path",
        nargs="?",
        type=str,
        help="Path to one .wav file, relative to the project root or absolute."
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Process all *_AUDIO.wav files in a dataset folder."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the single-file JSON report."
    )
    parser.add_argument(
        "--output-dir",
        default="data/month_01_baseline",
        help="Where to save batch reports and feature CSV."
    )

    args = parser.parse_args()

    if args.dataset_dir:
        run_batch(args.dataset_dir, args.output_dir)
        return

    if not args.audio_path:
        parser.error("Provide an audio_path or use --dataset-dir.")

    run_single(args.audio_path, args.output)


if __name__ == "__main__":
    main()
