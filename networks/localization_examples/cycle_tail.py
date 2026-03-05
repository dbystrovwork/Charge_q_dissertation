import numpy as np


def cycle_tail(k, p, seed=None):
    """
    Directed cycle with a directed path (tail) leading out from it.

    Cycle: nodes [0, k) with edges 0->1->2->...->k-1->0
    Tail: p edges from node k-1 outward: k-1 -> t0 -> t1 -> ... -> t_{p-1}

    Args:
        k: Length of the cycle
        p: Length of the tail (number of edges)
        seed: Random seed (unused, included for API consistency)

    Returns:
        edges: List of (i, j) tuples
        labels: Array of community labels (0 for cycle, 1 for tail nodes)
    """
    edges = []
    tail_start = k  # tail nodes start after cycle

    # Cycle: nodes [0, k)
    for i in range(k):
        edges.append((i, (i + 1) % k))

    # Tail: k-1 -> tail_0 -> tail_1 -> ... -> tail_{p-1}
    edges.append((k - 1, tail_start))
    for i in range(p - 1):
        edges.append((tail_start + i, tail_start + i + 1))

    labels = np.array([0] * k + [1] * p)

    return edges, labels
