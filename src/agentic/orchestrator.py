from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from src.agentic.recursive_context import build_recursive_context, load_history
from src.feature_extraction import AudioFeatureExtractor
from src.graph.feature_mapping import assess_features
from src.graph.knowledge_graph import build_graph, terminal_risk_nodes
from src.rag.evidence_scoring import score_evidence
from src.rag.graph_rag import get_all_explanation_contexts
from src.rag.reporter_agent import generate_report


@dataclass
class AgentTrace:
    stages: list[dict] = field(default_factory=list)

    def add(self, agent, action, status="completed", **details):
        self.stages.append({
            "agent": agent,
            "action": action,
            "status": status,
            "details": details,
        })


def infer_sidecar_path(audio_file: Path, suffix: str) -> Path | None:
    if audio_file.name.endswith("_AUDIO.wav"):
        candidate = audio_file.with_name(audio_file.name.replace("_AUDIO.wav", suffix))
        if candidate.exists():
            return candidate

    return None


class AgenticReportingEngine:
    """Coordinates the Month 4 Analyzer -> Graph-RAG -> Reporter workflow."""

    def __init__(self, graph_path: Path | str | None = None):
        self.graph_path = graph_path

    def run(
        self,
        audio_file: Path,
        persona="psychologist",
        llm_config=None,
        max_duration_seconds=None,
        transcript_path: Path | None = None,
        covarep_path: Path | None = None,
        auto_sidecars=True,
        history_path: Path | None = None,
        simulated_history_weeks=0,
    ):
        started_at = perf_counter()
        trace = AgentTrace()

        if auto_sidecars:
            transcript_path = transcript_path or infer_sidecar_path(audio_file, "_TRANSCRIPT.csv")
            covarep_path = covarep_path or infer_sidecar_path(audio_file, "_COVAREP.csv")

        trace.add(
            "OrchestratorAgent",
            "resolved inputs and sidecar files",
            audio_file=str(audio_file),
            transcript_path=str(transcript_path) if transcript_path else None,
            covarep_path=str(covarep_path) if covarep_path else None,
            persona=persona,
            max_duration_seconds=max_duration_seconds,
        )

        extractor = AudioFeatureExtractor(
            str(audio_file),
            max_duration_seconds=max_duration_seconds,
            transcript_path=str(transcript_path) if transcript_path else None,
            covarep_path=str(covarep_path) if covarep_path else None,
        )
        features = extractor.extract_all()
        features["sidecar_files"] = {
            "transcript": str(transcript_path) if transcript_path else None,
            "covarep": str(covarep_path) if covarep_path else None,
        }
        trace.add(
            "AnalyzerAgent",
            "extracted aligned acoustic biomarkers",
            feature_count=len(features),
            feature_sources=features.get("feature_sources", {}),
        )

        graph = build_graph(self.graph_path)
        findings = assess_features(features, graph)
        matched_findings = [finding for finding in findings if finding["matched"]]
        trace.add(
            "GraphRetrieverAgent",
            "mapped features to calibrated biomarker nodes",
            available_rules=len(findings),
            matched_rules=len(matched_findings),
            matched_nodes=[finding["node"] for finding in matched_findings],
        )

        aggregate_evidence = score_evidence(findings)
        trace.add(
            "EvidenceScorerAgent",
            "computed aggregate screening evidence",
            screen_positive=aggregate_evidence.get("screen_positive"),
            evidence_level=aggregate_evidence.get("evidence_level"),
            matched_biomarker_count=aggregate_evidence.get("matched_biomarker_count"),
            aggregate_match_threshold=aggregate_evidence.get("aggregate_match_threshold"),
        )

        explanations = []
        target_nodes = terminal_risk_nodes(graph)
        for finding in matched_findings:
            explanations.extend(
                get_all_explanation_contexts(
                    graph,
                    finding["node"],
                    target_nodes,
                )
            )
        trace.add(
            "GraphRetrieverAgent",
            "retrieved graph reasoning paths from matched biomarkers",
            explanation_path_count=len(explanations),
            target_nodes=target_nodes,
        )

        history = load_history(history_path) if history_path else []
        recursive_context = build_recursive_context(
            aggregate_evidence,
            history=history,
            simulated_weeks=simulated_history_weeks,
        )
        trace.add(
            "ContextAgent",
            "built recursive previous-week context",
            enabled=recursive_context.get("enabled"),
            simulated=recursive_context.get("simulated"),
            snapshot_count=len(recursive_context.get("snapshots", [])),
        )

        report = generate_report(
            explanations=explanations,
            features=features,
            file_name=audio_file.name,
            findings=findings,
            persona=persona,
            llm_config=llm_config,
            aggregate_evidence=aggregate_evidence,
            recursive_context=recursive_context,
        )
        trace.add(
            "ReporterAgent",
            "generated persona-tailored report and faithfulness check",
            llm_used=report.get("llm_status", {}).get("used"),
            faithfulness_supported=report.get("faithfulness_check", {}).get("supported"),
        )

        report["agentic_engine"] = {
            "name": "Agentic Reporting Engine",
            "version": "2026-09-month-04",
            "flow": [
                "AnalyzerAgent",
                "GraphRetrieverAgent",
                "EvidenceScorerAgent",
                "ContextAgent",
                "ReporterAgent",
            ],
            "elapsed_seconds": round(perf_counter() - started_at, 4),
            "trace": trace.stages,
        }
        return report
