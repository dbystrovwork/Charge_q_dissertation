import numpy as np


def barabasi_albert(n, m, seed=None):
    """
    Generate an undirected Barabási-Albert preferential attachment graph.

    Starts with a fully connected seed of m+1 nodes. Each new node
    creates m undirected edges to existing nodes, with probability
    proportional to their degree + 1.

    Each undirected link is stored once as (i, j) with i < j.
    The normal Laplacian symmetrises the adjacency internally.

    Args:
        n: Total number of nodes.
        m: Number of edges each new node creates.
        seed: Random seed.

    Returns:
        edges: List of (i, j) tuples (one per undirected link, i < j).
        labels: None (no community structure).
    """
    rng = np.random.default_rng(seed)

    seed_nodes = m + 1
    degree = np.zeros(n, dtype=int)

    # Seed graph: fully connected clique of m+1 nodes
    edge_set = set()
    for i in range(seed_nodes):
        for j in range(i + 1, seed_nodes):
            edge_set.add((i, j))
            degree[i] += 1
            degree[j] += 1

    for new_node in range(seed_nodes, n):
        existing = np.arange(new_node)
        weights = (degree[:new_node] + 1).astype(float)
        weights /= weights.sum()

        targets = rng.choice(existing, size=m, replace=False, p=weights)
        for t in targets:
            edge_set.add((min(new_node, t), max(new_node, t)))
            degree[new_node] += 1
            degree[t] += 1

    return list(edge_set), None


def directed_barabasi_albert(n, m, seed=None):
    """
    Generate a directed Barabási-Albert preferential attachment graph.

    Starts with a fully connected seed of m+1 nodes. Each new node
    creates m directed edges to existing nodes, with probability
    proportional to their in-degree + 1.

    Args:
        n: Total number of nodes.
        m: Number of directed edges each new node creates.
        seed: Random seed.

    Returns:
        edges: List of (i, j) tuples.
        labels: None (no community structure).
    """
    rng = np.random.default_rng(seed)

    seed_nodes = m + 1
    in_degree = np.zeros(n, dtype=int)

    # Seed graph: fully connected clique of m+1 nodes
    edges = []
    for i in range(seed_nodes):
        for j in range(seed_nodes):
            if i != j:
                edges.append((i, j))
                in_degree[j] += 1

    for new_node in range(seed_nodes, n):
        existing = np.arange(new_node)
        weights = (in_degree[:new_node] + 1).astype(float)
        weights /= weights.sum()

        targets = rng.choice(existing, size=m, replace=False, p=weights)
        for t in targets:
            edges.append((new_node, t))
            in_degree[t] += 1

    return edges, None
