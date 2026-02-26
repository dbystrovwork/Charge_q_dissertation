import numpy as np


def configuration_model(n, gamma, k_min=1, k_max=None, seed=None):
    """
    Generate an undirected graph using the configuration model with power-law degree distribution.

    Degree distribution: P(k) ∝ k^(-gamma) for k in [k_min, k_max].

    Args:
        n: Number of nodes.
        gamma: Power-law exponent (typically 2 < gamma < 3 for scale-free networks).
        k_min: Minimum degree (default 1).
        k_max: Maximum degree (default n-1).
        seed: Random seed.

    Returns:
        edges: List of (i, j) tuples (one per undirected link, i < j).
        labels: None (no community structure).
    """
    rng = np.random.default_rng(seed)

    if k_max is None:
        k_max = n - 1

    # Generate degree sequence from power-law
    degrees = _sample_power_law_degrees(n, gamma, k_min, k_max, rng)

    # Ensure sum of degrees is even
    if degrees.sum() % 2 == 1:
        # Add 1 to a random node's degree
        idx = rng.integers(n)
        degrees[idx] += 1

    # Create stub list: node i appears degree[i] times
    stubs = np.repeat(np.arange(n), degrees)

    # Shuffle and pair stubs
    rng.shuffle(stubs)

    edge_set = set()
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v:  # No self-loops
            edge_set.add((min(u, v), max(u, v)))

    return list(edge_set), None


def _sample_power_law_degrees(n, gamma, k_min, k_max, rng):
    """Sample n degrees from discrete power-law distribution."""
    k_values = np.arange(k_min, k_max + 1)
    probs = k_values.astype(float) ** (-gamma)
    probs /= probs.sum()

    return rng.choice(k_values, size=n, p=probs)


def configuration_model_from_sequence(degree_sequence, seed=None):
    """
    Generate an undirected graph from a given degree sequence.

    Args:
        degree_sequence: Array of degrees for each node.
        seed: Random seed.

    Returns:
        edges: List of (i, j) tuples (one per undirected link, i < j).
        labels: None (no community structure).
    """
    rng = np.random.default_rng(seed)
    degrees = np.array(degree_sequence, dtype=int)
    n = len(degrees)

    # Ensure sum is even
    if degrees.sum() % 2 == 1:
        idx = rng.integers(n)
        degrees[idx] += 1

    stubs = np.repeat(np.arange(n), degrees)
    rng.shuffle(stubs)

    edge_set = set()
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v:
            edge_set.add((min(u, v), max(u, v)))

    return list(edge_set), None
