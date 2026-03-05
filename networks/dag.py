import numpy as np


def random_dag(num_layers, n_per_layer, p, seed=None):
    """
    Random layered directed acyclic graph.

    Nodes are arranged into layers. Edges go only from lower layers
    to higher layers (i.e. layer i -> layer j where j > i), each
    included independently with probability p. This guarantees
    acyclicity by construction.

    Args:
        num_layers: Number of layers
        n_per_layer: Nodes per layer
        p: Probability of a directed edge from a node in layer i
           to a node in layer j (j > i)
        seed: Random seed

    Returns:
        edges: List of (i, j) tuples
        labels: Array of layer labels (0 to num_layers - 1)
    """
    rng = np.random.default_rng(seed)
    num_nodes = num_layers * n_per_layer
    labels = np.repeat(np.arange(num_layers), n_per_layer)
    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if labels[i] < labels[j] and rng.random() < p:
                edges.append((i, j))

    return edges, labels
