import networkx as nx


def _node_label(graph, node_id):
    return graph.nodes[node_id].get("name", node_id)


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


def describe_path(graph, path):
    steps = []

    for index, node_id in enumerate(path):
        node = graph.nodes[node_id]
        step = {
            "id": node_id,
            "name": _node_label(graph, node_id),
            "type": node.get("type")
        }

        if index < len(path) - 1:
            edge = graph.edges[node_id, path[index + 1]]
            step["relation_to_next"] = edge.get("relation")
            step["weight_to_next"] = edge.get("weight")

        steps.append(step)

    return steps


def get_all_explanation_contexts(graph, start_node, target_nodes):
    contexts = []

    for explanation in get_all_explanations(graph, start_node, target_nodes):
        contexts.append({
            **explanation,
            "path_labels": [
                _node_label(graph, node_id)
                for node_id in explanation["path"]
            ],
            "steps": describe_path(graph, explanation["path"])
        })

    return contexts
