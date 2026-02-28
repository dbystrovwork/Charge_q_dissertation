import numpy as np


def add_hub(edges, degree, seed=None):
    """
    Extend a directed graph by adding a hub node that receives edges
    from existing nodes.

    A new node is created with index max(existing nodes) + 1. Then
    `degree` existing nodes are sampled uniformly at random and a
    directed edge from each sampled node to the hub is added.

    Args:
        edges: List of (i, j) tuples for existing directed edges
        degree: Number of edges pointing to the new hub node
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

    sources = rng.choice(list(existing_nodes), size=degree, replace=False)

    new_edges = list(edges)
    for src in sources:
        new_edges.append((int(src), hub_id))

    return new_edges, hub_id
