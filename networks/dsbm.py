import numpy as np


def dsbm_cycle(k, n_per_class, p, s, r, seed=None):
    """
    Generate a Directed Stochastic Block Model with k communities in a cycle.

    Args:
        k: Number of communities
        n_per_class: Nodes per community
        p: Within-community edge probability
        s: Forward edge probability (community i -> i+1)
        r: Random noise edge probability

    Returns:
        edges: List of (i, j) tuples
        labels: Array of community labels
    """
    rng = np.random.default_rng(seed)
    num_nodes = k * n_per_class
    edges = []

    # Assign nodes to communities
    labels = np.repeat(np.arange(k), n_per_class)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue

            ci, cj = labels[i], labels[j]

            if ci == cj:
                # Within community
                prob = p
            elif cj == (ci + 1) % k:
                # Forward edge (i -> i+1 in cycle)
                prob = s
            else:
                # Random noise
                prob = r

            if rng.random() < prob:
                edges.append((i, j))

    return edges, labels
