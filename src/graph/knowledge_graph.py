import networkx as nx


def build_graph():
    graph = nx.DiGraph()

    edges = [
        ("High Jitter", "Vocal Instability"),
        ("High Pitch Variability", "Vocal Instability"),
        ("Vocal Instability", "Autonomic Arousal"),
        ("Autonomic Arousal", "Anxiety Risk"),
        ("High Pause Ratio", "Speech Disruption"),
        ("Speech Disruption", "Cognitive Load"),
        ("Low Energy", "Reduced Arousal"),
        ("Reduced Arousal", "Depressive Indicator")
    ]

    graph.add_edges_from(edges)
    return graph


def terminal_risk_nodes():
    return [
        "Anxiety Risk",
        "Cognitive Load",
        "Depressive Indicator"
    ]
