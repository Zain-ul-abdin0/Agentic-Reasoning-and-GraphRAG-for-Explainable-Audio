import networkx as nx

def build_graph():
    G = nx.DiGraph()

    # Biomarkers → clinical concepts
    G.add_edge("High Jitter", "Vocal Instability")
    G.add_edge("Vocal Instability", "Autonomic Arousal")
    G.add_edge("Autonomic Arousal", "Anxiety Risk")

    G.add_edge("High Pause Ratio", "Speech Disruption")
    G.add_edge("Speech Disruption", "Cognitive Load")

    G.add_edge("Low Energy", "Reduced Arousal")
    G.add_edge("Reduced Arousal", "Depressive Indicator")
    
    return G