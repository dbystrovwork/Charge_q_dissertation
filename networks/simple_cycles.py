def directed_cycle(n):
    """
    Generate a directed cycle graph with n nodes.

    Args:
        n: Number of nodes

    Returns:
        edges: List of (i, j) tuples
    """
    return [(i, (i + 1) % n) for i in range(n)]
