from pathlib import Path
import argparse
import csv
import json

from src.calibration.calibrate_acoustic_thresholds import load_phq_labels
from src.evaluation.metrics import binary_classification_metrics, mean
from src.evaluation.mock_review import build_mock_review_row
from src.evaluation.vector_baseline import compare_vector_baseline
from src.graph.knowledge_graph import build_graph
from src.pipeline.run_pipeline import analyze_audio, find_audio_files, resolve_path
from src.rag.reporter_agent import PERSONA_PROMPT_CHAINS


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_FILES = [
    BASE_DIR / "documentations" / "train_split_Depression_AVEC2017.csv",
    BASE_DIR / "documentations" / "dev_split_Depression_AVEC2017.csv",
    BASE_DIR / "documentations" / "full_test_split.csv",
    BASE_DIR.parent / "Documentations" / "train_split_Depression_AVEC2017.csv",
    BASE_DIR.parent / "Documentations" / "dev_split_Depression_AVEC2017.csv",
    BASE_DIR.parent / "Documentations" / "full_test_split.csv",
]


def participant_id_from_audio(path):
    return int(path.name.replace("_AUDIO.wav", ""))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_participant_result(participant_id, label_info, report):
    aggregate = report.get("aggregate_evidence") or {}
    faithfulness = report.get("faithfulness_check") or {}
    findings = report.get("rule_findings") or []
    explanations = report.get("explanation_paths") or []
    matched = [finding for finding in findings if finding.get("matched")]

    return {
        "participant_id": participant_id,
        "file": report.get("file"),
        "phq8_score": label_info.get("phq_score"),
        "phq8_label": label_info.get("label"),
        "prediction": int(bool(aggregate.get("screen_positive"))),
        "evidence_level": aggregate.get("evidence_level"),
        "matched_biomarker_count": aggregate.get("matched_biomarker_count"),
        "available_biomarker_count": aggregate.get("available_biomarker_count"),
        "aggregate_threshold": aggregate.get("aggregate_match_threshold"),
        "calibration_support_score": aggregate.get("calibration_support_score"),
        "matched_biomarkers": "; ".join(finding.get("label", "") for finding in matched),
        "explanation_path_count": len(explanations),
        "faithfulness_supported": faithfulness.get("supported"),
        "faithfulness_warnings": "; ".join(faithfulness.get("warnings") or []),
    }


def build_comparative_markdown(summary, output_files):
    metrics = summary["classification_metrics"]
    retrieval = summary["retrieval_comparison"]
    qualitative = summary["mock_qualitative_review"]

    return "\n".join([
        "# Month 5 Comparative Analysis",
        "",
        "## Evaluation Scope",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- PHQ-8 clinical reference label: {summary['reference_label']}",
        f"- Participants evaluated: {summary['participants_evaluated']}",
        f"- LLM usage during evaluation: {summary['llm_mode']}",
        "",
        "## Depression Screening Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Sensitivity: {metrics['sensitivity']:.3f}",
        f"- Specificity: {metrics['specificity']:.3f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.3f}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- F1 score: {metrics['f1_score']:.3f}",
        f"- Confusion matrix: TP={metrics['true_positive']}, TN={metrics['true_negative']}, FP={metrics['false_positive']}, FN={metrics['false_negative']}",
        "",
        "## Retrieval Precision",
        "",
        f"- Graph-RAG mean path precision: {retrieval['graph_rag_mean_precision']:.3f}",
        f"- Graph-RAG source-backed path ratio: {retrieval['graph_rag_mean_source_backed_ratio']:.3f}",
        f"- Vector baseline mean precision@k: {retrieval['vector_mean_precision_at_k']:.3f}",
        "",
        "Graph-RAG is evaluated as path retrieval: a retrieved path is relevant when it starts from a matched biomarker node and reaches a clinical target node. The vector baseline retrieves edge chunks using lexical TF-IDF cosine similarity, then checks whether the retrieved chunks are directly connected to the matched biomarker.",
        "",
        "## Mock Qualitative Review",
        "",
        f"- Review type: {qualitative['review_type']}",
        f"- Mean psychologist score: {qualitative['psychologist_mean_score']:.3f}",
        f"- Mean patient score: {qualitative['patient_mean_score']:.3f}",
        "",
        "The qualitative review is a deterministic proxy rubric for thesis development. It should be replaced or complemented by human review in the final study.",
        "",
        "## Output Files",
        "",
        *[f"- {name}: `{path}`" for name, path in output_files.items()],
        "",
    ])


def persona_view(report, persona):
    viewed_report = dict(report)
    viewed_report["persona"] = persona
    viewed_report["persona_prompt_chain"] = PERSONA_PROMPT_CHAINS.get(persona, [])
    return viewed_report


def evaluate(args):
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    label_files = [Path(path) for path in args.label_file] if args.label_file else DEFAULT_LABEL_FILES
    labels = load_phq_labels(label_files, args.phq_cutoff)
    if not labels:
        raise FileNotFoundError(
            "No PHQ-8 labels were loaded. Pass --label-file with the DAIC-WOZ split CSV files."
        )
    graph = build_graph()
    audio_files = [
        path
        for path in find_audio_files(dataset_dir)
        if participant_id_from_audio(path) in labels
    ]

    if args.limit:
        audio_files = audio_files[:args.limit]

    participant_rows = []
    retrieval_rows = []
    review_rows = []
    graph_precisions = []
    graph_source_ratios = []
    vector_precisions = []
    labels_eval = []
    predictions = []

    for index, audio_file in enumerate(audio_files, start=1):
        participant_id = participant_id_from_audio(audio_file)
        label_info = labels[participant_id]
        print(f"[{index}/{len(audio_files)}] Evaluating participant {participant_id}", flush=True)

        report = analyze_audio(
            audio_file=audio_file,
            persona="psychologist",
            llm_config=None,
            max_duration_seconds=args.max_duration_seconds,
            auto_sidecars=True,
            simulated_history_weeks=args.simulate_history_weeks,
        )

        participant_row = flatten_participant_result(participant_id, label_info, report)
        participant_rows.append(participant_row)
        labels_eval.append(int(label_info["label"]))
        predictions.append(int(participant_row["prediction"]))

        comparison = compare_vector_baseline(
            graph,
            report.get("rule_findings") or [],
            report.get("explanation_paths") or [],
            top_k=args.top_k,
        )
        graph_rag = comparison["graph_rag"] or {}
        if graph_rag.get("precision") is not None:
            graph_precisions.append(graph_rag["precision"])
        if graph_rag.get("source_backed_ratio") is not None:
            graph_source_ratios.append(graph_rag["source_backed_ratio"])

        for row in comparison["vector_rows"]:
            row = {
                "participant_id": participant_id,
                "phq8_label": label_info["label"],
                **row,
                "graph_rag_precision": graph_rag.get("precision"),
                "graph_rag_source_backed_ratio": graph_rag.get("source_backed_ratio"),
            }
            retrieval_rows.append(row)
            vector_precisions.append(row["precision_at_k"])

        review_rows.append(
            build_mock_review_row(participant_id, persona_view(report, "psychologist"), "psychologist")
        )
        review_rows.append(
            build_mock_review_row(participant_id, persona_view(report, "patient"), "patient")
        )

    metrics = binary_classification_metrics(labels_eval, predictions)
    psychologist_scores = [
        row["overall_mean"] for row in review_rows
        if row["persona"] == "psychologist"
    ]
    patient_scores = [
        row["overall_mean"] for row in review_rows
        if row["persona"] == "patient"
    ]

    summary = {
        "milestone": "Month 5: Evaluation and User Study",
        "dataset": str(dataset_dir),
        "reference_label": f"PHQ-8 >= {args.phq_cutoff}",
        "participants_evaluated": len(audio_files),
        "llm_mode": "disabled for deterministic batch evaluation",
        "classification_metrics": metrics,
        "retrieval_comparison": {
            "top_k": args.top_k,
            "graph_rag_mean_precision": mean(graph_precisions),
            "graph_rag_mean_source_backed_ratio": mean(graph_source_ratios),
            "vector_mean_precision_at_k": mean(vector_precisions),
            "retrieval_rows": len(retrieval_rows),
        },
        "mock_qualitative_review": {
            "review_type": "mock qualitative rubric",
            "psychologist_mean_score": mean(psychologist_scores),
            "patient_mean_score": mean(patient_scores),
            "rows": len(review_rows),
        },
    }

    output_files = {
        "summary": str(output_dir / "evaluation_summary.json"),
        "participant_results": str(output_dir / "participant_results.csv"),
        "retrieval_comparison": str(output_dir / "retrieval_comparison.csv"),
        "mock_review_scores": str(output_dir / "mock_review_scores.csv"),
        "comparative_analysis": str(output_dir / "comparative_analysis.md"),
    }

    write_json(output_dir / "evaluation_summary.json", summary)
    write_csv(output_dir / "participant_results.csv", participant_rows)
    write_csv(output_dir / "retrieval_comparison.csv", retrieval_rows)
    write_csv(output_dir / "mock_review_scores.csv", review_rows)
    (output_dir / "comparative_analysis.md").write_text(
        build_comparative_markdown(summary, output_files),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Month 5 evaluation saved to {output_dir}")
    return {
        "summary": summary,
        "output_files": output_files,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Month 5 DAIC-WOZ evaluation.")
    parser.add_argument("--dataset-dir", default="Dataset")
    parser.add_argument("--output-dir", default="data/month_05_evaluation")
    parser.add_argument("--phq-cutoff", type=float, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--label-file",
        action="append",
        default=None,
        help="Optional PHQ-8 label CSV path. Repeat this argument for train/dev/test split files.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-duration-seconds", type=float, default=None)
    parser.add_argument("--simulate-history-weeks", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
