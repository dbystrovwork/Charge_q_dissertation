import numpy as np


def two_cycles(k1, k2, p=1, seed=None):
    """
    Two directed cycles connected by a directed path of length p.

    Cycle 1: nodes [0, k1) with edges 0->1->2->...->k1-1->0
    Cycle 2: nodes [k1, k1+k2) with edges k1->k1+1->...->k1+k2-1->k1
    Path: p edges from last node of cycle 1 to first node of cycle 2

    Args:
        k1: Length of first cycle
        k2: Length of second cycle
        p: Length of path (number of edges) connecting the cycles
        seed: Random seed (unused, included for API consistency)

    Returns:
        edges: List of (i, j) tuples
        labels: Array of community labels (0 or 1 for cycles, 2 for path nodes)
    """
    edges = []
    path_start = k1 + k2  # path nodes start after both cycles

    # Cycle 1: nodes [0, k1)
    for i in range(k1):
        edges.append((i, (i + 1) % k1))

    # Cycle 2: nodes [k1, k1 + k2)
    for i in range(k2):
        edges.append((k1 + i, k1 + (i + 1) % k2))

    # Path: k1-1 -> path_node_0 -> ... -> path_node_{p-2} -> k1
    if p == 1:
        edges.append((k1 - 1, k1))
    else:
        edges.append((k1 - 1, path_start))
        for i in range(p - 2):
            edges.append((path_start + i, path_start + i + 1))
        edges.append((path_start + p - 2, k1))

    labels = np.array([0] * k1 + [1] * k2 + [2] * (p - 1))

    return edges, labels
