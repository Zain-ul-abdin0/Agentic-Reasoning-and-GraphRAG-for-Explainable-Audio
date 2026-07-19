import argparse
import json
from pathlib import Path

from src.graph.knowledge_graph import (
    build_graph,
    get_biomarker_rules,
    terminal_risk_nodes,
    validate_graph,
)


BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_path(input_path: str) -> Path:
    path = Path(input_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def summarize_graph(graph):
    rules = get_biomarker_rules(graph)

    return {
        "name": graph.graph["name"],
        "version": graph.graph["version"],
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "biomarkers": len(rules),
        "terminal_targets": terminal_risk_nodes(graph),
        "validation_problems": validate_graph(graph),
        "biomarker_labels": [
            rule["label"]
            for rule in rules.values()
        ],
    }


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the lightweight Siegen audio-anxiety knowledge graph."
    )
    parser.add_argument(
        "--graph",
        default=None,
        help="Optional JSON-LD graph path. Defaults to the Month 2 seed graph.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for writing the summary JSON.",
    )

    args = parser.parse_args()
    graph = build_graph(args.graph)
    summary = summarize_graph(graph)

    if args.output:
        save_json(resolve_path(args.output), summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
