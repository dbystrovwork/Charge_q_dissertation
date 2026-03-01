import numpy as np


def add_hub(edges, degree, direction="inward", seed=None):
    """
    Extend a directed graph by adding a hub node.

    Args:
        edges: List of (i, j) tuples for existing directed edges
        degree: Number of edges connecting to the hub
        direction: Edge direction relative to hub:
            - "inward": edges point TO the hub (node -> hub)
            - "outward": edges point FROM the hub (hub -> node)
            - "mixed": half inward, half outward
        seed: Random seed for reproducibility

    Returns:
        new_edges: Extended edge list including the hub edges
        hub_id: Index of the newly created hub node
    """
    rng = np.random.default_rng(seed)

    existing_nodes = set()
    for i, j in edges:
        existing_nodes.add(i)
        existing_nodes.add(j)

    hub_id = max(existing_nodes) + 1
    nodes = rng.choice(list(existing_nodes), size=degree, replace=False)

    new_edges = list(edges)

    if direction == "inward":
        for n in nodes:
            new_edges.append((int(n), hub_id))
    elif direction == "outward":
        for n in nodes:
            new_edges.append((hub_id, int(n)))
    elif direction == "mixed":
        half = degree // 2
        for n in nodes[:half]:
            new_edges.append((int(n), hub_id))
        for n in nodes[half:]:
            new_edges.append((hub_id, int(n)))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return new_edges, hub_id
