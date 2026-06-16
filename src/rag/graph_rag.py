import networkx as nx

def get_explanation(graph, start_node, end_node):
    try:
        path = nx.shortest_path(graph, start_node, end_node)
        return path
    except:
        return []