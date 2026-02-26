import numpy as np


def sbm(k, n_per_class, p_in, p_out, seed=None):
    """
    Generate an undirected Stochastic Block Model graph.

    Args:
        k: Number of communities.
        n_per_class: Nodes per community.
        p_in: Within-community edge probability.
        p_out: Between-community edge probability.
        seed: Random seed.

    Returns:
        edges: List of (i, j) tuples (one per undirected link, i < j).
        labels: Array of community labels.
    """
    rng = np.random.default_rng(seed)
    num_nodes = k * n_per_class
    labels = np.repeat(np.arange(k), n_per_class)

    edge_set = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = p_in if labels[i] == labels[j] else p_out
            if rng.random() < prob:
                edge_set.add((i, j))

    return list(edge_set), labels
