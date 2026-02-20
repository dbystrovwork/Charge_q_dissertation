import numpy as np
from scipy.sparse import csr_matrix, diags


def magnetic_laplacian(edges, num_nodes, q, normalized=False):
    """
    Compute the magnetic Laplacian for a directed graph.

    Args:
        edges: List of (i, j) tuples for directed edges i -> j
        num_nodes: Number of nodes
        q: Magnetic potential parameter
        normalized: If True, return normalized Laplacian

    Returns:
        CSR matrix (complex-valued)
    """
    # Build directed adjacency matrix
    row, col = zip(*edges)
    A = csr_matrix((np.ones(len(edges)), (row, col)),
                   shape=(num_nodes, num_nodes), dtype=float)

    # Symmetric magnitude: A_s = 0.5 * (A + A^T)
    A_s = 0.5 * (A + A.T)

    # Skew phase: Θ = 2πq * (A - A^T)
    Theta = 2 * np.pi * q * (A - A.T)

    # Magnetic adjacency: H = A_s ⊙ exp(iΘ)
    H = A_s.multiply(np.exp(1j * Theta.toarray()))
    H = csr_matrix(H, dtype=complex)

    # Degree matrix
    d_s = np.array(A_s.sum(axis=1)).flatten()

    if normalized:
        # L_N = I - D_s^{-1/2} A_s D_s^{-1/2} ⊙ exp(iΘ)
        d_inv_sqrt = np.where(d_s > 0, 1.0 / np.sqrt(d_s), 0)
        D_inv_sqrt = diags(d_inv_sqrt)
        H_norm = D_inv_sqrt @ A_s @ D_inv_sqrt
        H_norm = H_norm.multiply(np.exp(1j * Theta.toarray()))
        return diags(np.ones(num_nodes), dtype=complex) - csr_matrix(H_norm, dtype=complex)
    else:
        # L_U = D_s - H
        D_s = diags(d_s, dtype=complex)
        return D_s - H
