from pathlib import Path
import argparse

from src.feature_extraction import AudioFeatureExtractor
from src.graph.knowledge_graph import build_graph
from src.rag.graph_rag import get_explanation
from src.rag.reporter_agent import generate_report


def resolve_audio_path(input_path: str) -> Path:
   
    BASE_DIR = Path(__file__).resolve().parents[2]
    path = Path(input_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    return path


def main():
   
    parser = argparse.ArgumentParser(description="Audio analysis pipeline")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to audio file (relative to project root or absolute)"
    )

    args = parser.parse_args()

    # Resolve audio file path
    audio_file = resolve_audio_path(args.audio_path)

    # 1. Extract features
    extractor = AudioFeatureExtractor(str(audio_file))
    features = extractor.extract_all()

    # 2. Build knowledge graph
    G = build_graph()

    paths = []
    
    # 3. Rule-based mapping
    if features["jitter"] > 0.015:
        paths.append(
            get_explanation(G, "High Jitter", "Anxiety Risk")
        )

    if features["pause_ratio"] > 0.8:
        paths.append(
            get_explanation(G, "High Pause Ratio", "Cognitive Load")
        )
    # 4. Generate report
    report = generate_report(paths, features, audio_file.name)

    print(report)


if __name__ == "__main__":
    main()