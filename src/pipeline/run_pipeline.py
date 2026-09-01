from pathlib import Path
import argparse
import csv
import json

from src.agentic.orchestrator import AgenticReportingEngine
from src.rag.llm_client import LocalLLMConfig


BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_path(input_path: str) -> Path:
    path = Path(input_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def resolve_optional_path(input_path: str | None) -> Path | None:
    if input_path is None:
        return None

    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Optional input file not found: {path}")

    return path


def resolve_audio_path(input_path: str) -> Path:
    path = resolve_path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {path}")

    return path


def analyze_audio(
    audio_file: Path,
    persona: str = "psychologist",
    llm_config: LocalLLMConfig | None = None,
    max_duration_seconds: float | None = None,
    transcript_path: Path | None = None,
    covarep_path: Path | None = None,
    auto_sidecars: bool = True,
    history_path: Path | None = None,
    simulated_history_weeks: int = 0,
):
    engine = AgenticReportingEngine()
    return engine.run(
        audio_file=audio_file,
        persona=persona,
        llm_config=llm_config,
        max_duration_seconds=max_duration_seconds,
        transcript_path=transcript_path,
        covarep_path=covarep_path,
        auto_sidecars=auto_sidecars,
        history_path=history_path,
        simulated_history_weeks=simulated_history_weeks,
    )


def find_audio_files(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    return sorted(dataset_dir.glob("*_AUDIO.wav"))


def flatten_report(report):
    features = report["biomarkers"]
    aggregate = report.get("aggregate_evidence") or {}
    recursive_context = report.get("recursive_context") or {}
    agentic_engine = report.get("agentic_engine") or {}
    matched_nodes = [
        finding["node"]
        for finding in report["rule_findings"]
        if finding["matched"]
    ]

    return {
        "file": report["file"],
        "persona": report.get("persona"),
        "duration_seconds": features.get("duration_seconds"),
        "energy": features.get("energy"),
        "pitch_mean": features.get("pitch_mean"),
        "pitch_std": features.get("pitch_std"),
        "pause_ratio": features.get("pause_ratio"),
        "speech_rate": features.get("speech_rate"),
        "jitter": features.get("jitter"),
        "matched_biomarker_count": aggregate.get("matched_biomarker_count"),
        "available_biomarker_count": aggregate.get("available_biomarker_count"),
        "aggregate_screen_positive": aggregate.get("screen_positive"),
        "evidence_level": aggregate.get("evidence_level"),
        "recursive_context_enabled": recursive_context.get("enabled"),
        "recursive_context_simulated": recursive_context.get("simulated"),
        "agent_flow": " -> ".join(agentic_engine.get("flow", [])),
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
        "persona",
        "duration_seconds",
        "energy",
        "pitch_mean",
        "pitch_std",
        "pause_ratio",
        "speech_rate",
        "jitter",
        "matched_biomarker_count",
        "available_biomarker_count",
        "aggregate_screen_positive",
        "evidence_level",
        "recursive_context_enabled",
        "recursive_context_simulated",
        "agent_flow",
        "matched_rules",
        "explanation_targets"
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_llm_config(args):
    if args.llm_provider == "none":
        return None

    return LocalLLMConfig(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.ollama_url,
        timeout_seconds=args.llm_timeout,
        temperature=args.temperature
    )


def run_single(
    audio_path: str,
    output_path: str | None,
    persona: str,
    llm_config: LocalLLMConfig | None,
    max_duration_seconds: float | None,
    transcript_path: str | None,
    covarep_path: str | None,
    auto_sidecars: bool,
    history_path: str | None,
    simulated_history_weeks: int,
    quiet: bool = False
):
    audio_file = resolve_audio_path(audio_path)
    report = analyze_audio(
        audio_file,
        persona=persona,
        llm_config=llm_config,
        max_duration_seconds=max_duration_seconds,
        transcript_path=resolve_optional_path(transcript_path),
        covarep_path=resolve_optional_path(covarep_path),
        auto_sidecars=auto_sidecars,
        history_path=resolve_optional_path(history_path),
        simulated_history_weeks=simulated_history_weeks,
    )

    if output_path:
        saved_path = resolve_path(output_path)
        save_json(saved_path, report)

    if quiet and output_path:
        print(f"Report saved to {saved_path}")
    else:
        print(json.dumps(report, indent=2))


def run_batch(
    dataset_dir: str,
    output_dir: str,
    persona: str,
    llm_config: LocalLLMConfig | None,
    max_duration_seconds: float | None,
    auto_sidecars: bool,
    history_path: str | None,
    simulated_history_weeks: int,
):
    dataset_path = resolve_path(dataset_dir)
    output_path = resolve_path(output_dir)
    audio_files = find_audio_files(dataset_path)

    if not audio_files:
        raise FileNotFoundError(f"No *_AUDIO.wav files found in: {dataset_path}")

    reports = []
    rows = []

    for audio_file in audio_files:
        report = analyze_audio(
            audio_file,
            persona=persona,
            llm_config=llm_config,
            max_duration_seconds=max_duration_seconds,
            auto_sidecars=auto_sidecars,
            history_path=resolve_optional_path(history_path),
            simulated_history_weeks=simulated_history_weeks,
        )
        reports.append(report)
        rows.append(flatten_report(report))

    save_json(output_path / "reports.json", reports)
    save_csv(output_path / "features.csv", rows)

    print(f"Processed {len(audio_files)} audio files")
    print(f"Reports saved to {output_path / 'reports.json'}")
    print(f"Feature table saved to {output_path / 'features.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Graph-RAG audio reporting pipeline"
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
        default="data/month_04_agentic_reporting_engine",
        help="Where to save batch reports and feature CSV."
    )
    parser.add_argument(
        "--persona",
        choices=["psychologist", "patient"],
        default="psychologist",
        help="Audience used by the Reporter Agent."
    )
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "none"],
        default="ollama",
        help="Local LLM provider for agentic Graph-RAG reporting."
    )
    parser.add_argument(
        "--llm-model",
        default="gemma3",
        help="Local Ollama model name, for example gemma3 or gemma3:1b."
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for the local Ollama server."
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=60,
        help="Seconds to wait for the local LLM response."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for report generation."
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Optional limit for quick demos; omit it for full-audio analysis."
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="Optional DAIC-WOZ transcript CSV for transcript-aligned speech rate and pause ratio."
    )
    parser.add_argument(
        "--covarep",
        default=None,
        help="Optional DAIC-WOZ COVAREP CSV for calibration-aligned pitch and jitter features."
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Optional JSON file containing previous report(s) for recursive context."
    )
    parser.add_argument(
        "--simulate-history-weeks",
        type=int,
        default=0,
        help="Create clearly labelled synthetic previous-week summaries for Month 4 prototype demos."
    )
    parser.add_argument(
        "--no-auto-sidecars",
        action="store_true",
        help="Disable automatic *_TRANSCRIPT.csv and *_COVAREP.csv lookup next to *_AUDIO.wav files."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the saved report path when --output is used."
    )

    args = parser.parse_args()
    llm_config = build_llm_config(args)
    auto_sidecars = not args.no_auto_sidecars

    if args.dataset_dir:
        run_batch(
            args.dataset_dir,
            args.output_dir,
            persona=args.persona,
            llm_config=llm_config,
            max_duration_seconds=args.max_duration_seconds,
            auto_sidecars=auto_sidecars,
            history_path=args.history,
            simulated_history_weeks=args.simulate_history_weeks,
        )
        return

    if not args.audio_path:
        parser.error("Provide an audio_path or use --dataset-dir.")

    run_single(
        args.audio_path,
        args.output,
        persona=args.persona,
        llm_config=llm_config,
        max_duration_seconds=args.max_duration_seconds,
        transcript_path=args.transcript,
        covarep_path=args.covarep,
        auto_sidecars=auto_sidecars,
        history_path=args.history,
        simulated_history_weeks=args.simulate_history_weeks,
        quiet=args.quiet
    )


if __name__ == "__main__":
    main()
