import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs


def _build_non_backtracking(directed_edges):
    """
    Build the non-backtracking matrix from an ordered list of directed edges.

    B_{(i->j), (j->l)} = 1  if  i != l

    Args:
        directed_edges: List of (src, dst) tuples.

    Returns:
        B: CSR matrix of shape (num_directed, num_directed).
    """
    num_directed = len(directed_edges)

    # Group outgoing edges by source node
    outgoing = {}
    for idx, (src, dst) in enumerate(directed_edges):
        outgoing.setdefault(src, []).append((src, dst, idx))

    # Build B: for each directed edge (i->j), connect to every (j->l) where l != i
    rows = []
    cols = []
    for e_idx, (i, j) in enumerate(directed_edges):
        for src, dst, f_idx in outgoing.get(j, []):
            if dst != i:  # no backtracking: j->l where l != i
                rows.append(e_idx)
                cols.append(f_idx)

    data = np.ones(len(rows), dtype=float)
    B = csr_matrix((data, (rows, cols)), shape=(num_directed, num_directed))

    return B


def non_backtracking_matrix(edges, num_nodes):
    """
    Compute the non-backtracking (Hashimoto) matrix for an undirected graph.

    For an undirected graph with m edges, each edge (i, j) gives two directed
    edges i -> j and j -> i, yielding 2m directed edges in total. The
    non-backtracking matrix B is a (2m x 2m) matrix indexed by directed edges:

        B_{(i->j), (j->l)} = 1  if  i != l

    Args:
        edges: List of (i, j) tuples (undirected edges).
        num_nodes: Number of nodes.

    Returns:
        B: CSR matrix of shape (2m, 2m).
        directed_edges: List of (src, dst) tuples for the 2m directed edges,
            giving the edge ordering used to index B.
    """
    # Build the full set of directed edges
    directed_edges = []
    for i, j in edges:
        directed_edges.append((i, j))
        directed_edges.append((j, i))

    # Remove duplicates while preserving order
    directed_edges = list(dict.fromkeys(directed_edges))

    B = _build_non_backtracking(directed_edges)

    return B, directed_edges


def magnetic_non_backtracking_matrix(edges, num_nodes, q):
    """
    Compute the magnetic non-backtracking matrix B_q = B @ Λ.

    First builds the real non-backtracking matrix B from the symmetrised
    directed edges, then right-multiplies by a diagonal phase matrix Λ
    indexed by directed edges:

        Λ_{e,e} = 1                    if edge e is symmetric (both i->j and j->i exist)
        Λ_{e,e} = exp(i * 2πq)        if edge e is forward-only (i->j exists, j->i does not)
        Λ_{e,e} = exp(-i * 2πq)       if edge e is backward-only (j->i exists, i->j does not)

    Args:
        edges: List of (i, j) tuples (directed edges).
        num_nodes: Number of nodes.
        q: Magnetic potential parameter (charge).

    Returns:
        B_q: CSR matrix of shape (2m, 2m), complex-valued.
        directed_edges: List of (src, dst) tuples for the directed edges.
    """
    # Build the original directed adjacency to classify edges
    row, col = zip(*edges)
    A = csr_matrix((np.ones(len(edges)), (row, col)),
                   shape=(num_nodes, num_nodes), dtype=float)

    # Build the full set of directed edges (symmetrised)
    directed_edges = []
    for i, j in edges:
        directed_edges.append((i, j))
        directed_edges.append((j, i))

    # Remove duplicates while preserving order
    directed_edges = list(dict.fromkeys(directed_edges))

    B = _build_non_backtracking(directed_edges)

    # Build diagonal phase matrix Λ
    phases = np.empty(len(directed_edges), dtype=complex)
    for idx, (i, j) in enumerate(directed_edges):
        fwd = A[i, j]  # does i->j exist in original edges?
        bwd = A[j, i]  # does j->i exist in original edges?
        if fwd and bwd:
            phases[idx] = 1.0
        elif fwd:
            phases[idx] = np.exp(1j * 2 * np.pi * q)
        else:
            phases[idx] = np.exp(-1j * 2 * np.pi * q)

    Lambda = diags(phases)
    B_q = B @ Lambda

    return B_q, directed_edges


def non_backtracking_eig(edges, num_nodes, k):
    """
    Compute the k largest eigenvalues/eigenvectors of the non-backtracking matrix.

    Args:
        edges: List of (i, j) tuples (undirected edges).
        num_nodes: Number of nodes.
        k: Number of eigenvalues to compute.

    Returns:
        eigenvalues: Array of k eigenvalues (complex in general).
        eigenvectors: Array of shape (2m, k).
        directed_edges: The directed edge ordering used to index the matrix.
    """
    B, directed_edges = non_backtracking_matrix(edges, num_nodes)
    eigenvalues, eigenvectors = eigs(B, k=k, which='LM')
    idx = np.argsort(-np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx], directed_edges


def magnetic_non_backtracking_eig(edges, num_nodes, q, k):
    """
    Compute the k largest eigenvalues/eigenvectors of the magnetic
    non-backtracking matrix.

    Args:
        edges: List of (i, j) tuples (directed edges).
        num_nodes: Number of nodes.
        q: Magnetic potential parameter (charge).
        k: Number of eigenvalues to compute.

    Returns:
        eigenvalues: Array of k eigenvalues (complex).
        eigenvectors: Array of shape (2m, k).
        directed_edges: The directed edge ordering used to index the matrix.
    """
    B_q, directed_edges = magnetic_non_backtracking_matrix(edges, num_nodes, q)
    eigenvalues, eigenvectors = eigs(B_q, k=k, which='LM')
    idx = np.argsort(-np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx], directed_edges
