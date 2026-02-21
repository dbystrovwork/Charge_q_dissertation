import numpy as np


def directed_small_world(n, k, p, seed=None):
    """
    Generate a directed small-world network (directed Watts-Strogatz).

    Args:
        n: Number of nodes
        k: Each node connects to k next nodes in the ring (out-degree before rewiring)
        p: Rewiring probability
        seed: Random seed

    Returns:
        edges: List of (i, j) tuples
        labels: Array of node indices (no community structure)
    """
    rng = np.random.default_rng(seed)
    edges = set()

    # Create directed ring lattice: node i -> i+1, i+2, ..., i+k
    for i in range(n):
        for j in range(1, k + 1):
            target = (i + j) % n
            edges.add((i, target))

    # Rewire edges
    edges_list = list(edges)
    for src, tgt in edges_list:
        if rng.random() < p:
            edges.discard((src, tgt))
            # Pick new target (not self, not existing edge)
            candidates = [v for v in range(n) if v != src and (src, v) not in edges]
            if candidates:
                new_tgt = rng.choice(candidates)
                edges.add((src, new_tgt))
            else:
                edges.add((src, tgt))  # Keep original if no valid target

    return list(edges), np.arange(n)
