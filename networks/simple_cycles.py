def directed_cycle(n):
    """
    Generate a directed cycle graph with n nodes.

    Args:
        n: Number of nodes

    Returns:
        edges: List of (i, j) tuples
    """
    return [(i, (i + 1) % n) for i in range(n)]


def directed_cycle_flipped(n):
    """
    Generate a directed cycle with one edge flipped.

    Args:
        n: Number of nodes
        flip_idx: Index of edge to flip (default: 0, the edge 0->1 becomes 1->0)

    Returns:
        edges: List of (i, j) tuples
    """
    edges = directed_cycle(n)
    edges[0] = (1, 0) 
    return edges


def nested_cycles(sizes):
    edges = directed_cycle(sizes[0])

    for k in sizes[1:]:
        edges.append((k-1, 0))

    return edges


def cycles_from_node(sizes):
    """
    Generate multiple directed cycles all sharing node 0.

    Args:
        sizes: List of cycle sizes, e.g. [5, 3, 4]

    Returns:
        edges: List of (i, j) tuples
        num_nodes: Total number of nodes
    """
    edges = []
    node = 1

    for size in sizes:
        # Cycle: 0 -> node -> node+1 -> ... -> node+size-2 -> 0
        edges.append((0, node))
        for i in range(size - 2):
            edges.append((node + i, node + i + 1))
        edges.append((node + size - 2, 0))
        node += size - 1

    return edges, node

