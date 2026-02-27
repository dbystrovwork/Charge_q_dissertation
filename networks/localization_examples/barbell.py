import numpy as np


def directed_barbell(n_clique, seed=None):
    """
    Directed barbell graph: two symmetric cliques connected by
    a single directed edge from clique 0 to clique 1.

    Within each clique, every pair of nodes is connected by edges
    in both directions (fully symmetric). The two cliques are joined
    by exactly one directed edge: node (n_clique - 1) -> node (n_clique).

    Args:
        n_clique: Number of nodes per clique (total nodes = 2 * n_clique)
        seed: Random seed (unused, included for API consistency)

    Returns:
        edges: List of (i, j) tuples
        labels: Array of community labels (0 or 1)
    """
    edges = []

    # Clique 0: nodes [0, n_clique)
    # Clique 1: nodes [n_clique, 2*n_clique)
    for clique_start in (0, n_clique):
        for i in range(clique_start, clique_start + n_clique):
            for j in range(clique_start, clique_start + n_clique):
                if i != j:
                    edges.append((i, j))

    # Single directed bridge: last node of clique 0 -> first node of clique 1
    edges.append((n_clique - 1, n_clique))

    labels = np.array([0] * n_clique + [1] * n_clique)

    return edges, labels
