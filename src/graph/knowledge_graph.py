from pathlib import Path
import json

import networkx as nx


BASE_DIR = Path(__file__).resolve().parents[2]
SEED_GRAPH_PATH = (
    BASE_DIR / "data" / "knowledge_graph" / "siegen_audio_depression_kg.jsonld"
)
CALIBRATED_GRAPH_PATH = (
    BASE_DIR
    / "data"
    / "knowledge_graph"
    / "siegen_audio_depression_kg.calibrated.jsonld"
)
DEFAULT_GRAPH_PATH = (
    CALIBRATED_GRAPH_PATH
    if CALIBRATED_GRAPH_PATH.exists()
    else SEED_GRAPH_PATH
)


def load_graph_document(path: Path | str | None = None):
    graph_path = Path(path) if path else DEFAULT_GRAPH_PATH

    if not graph_path.is_absolute():
        graph_path = BASE_DIR / graph_path

    return json.loads(graph_path.read_text(encoding="utf-8"))


def build_graph(path: Path | str | None = None):
    document = load_graph_document(path)
    graph = nx.DiGraph(
        id=document.get("id"),
        name=document.get("name"),
        description=document.get("description"),
        version=document.get("version"),
        source_path=str(DEFAULT_GRAPH_PATH if path is None else path)
    )

    for node in document["nodes"]:
        node_id = node["id"]
        attributes = {
            key: value
            for key, value in node.items()
            if key != "id"
        }
        graph.add_node(node_id, **attributes)

    for edge in document["edges"]:
        graph.add_edge(
            edge["from"],
            edge["to"],
            relation=edge.get("relation", "related_to")
        )

    return graph


def validate_graph(graph):
    problems = []

    for node_id, attributes in graph.nodes(data=True):
        if "type" not in attributes:
            problems.append(f"Node {node_id} is missing a type.")
        if "name" not in attributes:
            problems.append(f"Node {node_id} is missing a name.")

    for source, target in graph.edges():
        if source not in graph:
            problems.append(f"Edge source is missing: {source}.")
        if target not in graph:
            problems.append(f"Edge target is missing: {target}.")

    return problems


def get_nodes_by_type(graph, node_type):
    return [
        node_id
        for node_id, attributes in graph.nodes(data=True)
        if attributes.get("type") == node_type
    ]


def get_biomarker_rules(graph=None):
    graph = graph or build_graph()
    rules = {}

    for node_id, attributes in graph.nodes(data=True):
        if attributes.get("type") != "Biomarker":
            continue
        if attributes.get("active_for_threshold_detection") is False:
            continue

        rules[node_id] = {
            "feature_key": attributes["feature_key"],
            "feature_index": attributes.get("feature_index"),
            "threshold": attributes["threshold"],
            "direction": attributes["direction"],
            "node": node_id,
            "label": attributes["name"],
            "summary": attributes["clinical_summary"]
        }

    return rules


def terminal_risk_nodes(graph=None):
    graph = graph or build_graph()
    return (
        get_nodes_by_type(graph, "Risk")
        + get_nodes_by_type(graph, "Intervention")
        + get_nodes_by_type(graph, "ScreeningConstruct")
    )
