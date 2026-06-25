import networkx as nx


def get_explanation(graph, start_node, end_node):
    try:
        path = nx.shortest_path(graph, start_node, end_node)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_all_explanations(graph, start_node, target_nodes):
    explanations = []

    for target_node in target_nodes:
        path = get_explanation(graph, start_node, target_node)
        if path:
            explanations.append({
                "start": start_node,
                "target": target_node,
                "path": path
            })

    return explanations
