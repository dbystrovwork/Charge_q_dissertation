import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


def magnetic_adjacency(edges, num_nodes, q):
    """
    Compute the magnetic adjacency matrix for a directed graph.

    H = A_s ⊙ exp(iΘ) where A_s = 0.5*(A + A^T) and Θ = 2πq*(A - A^T).

    Args:
        edges: List of (i, j) tuples for directed edges i -> j
        num_nodes: Number of nodes
        q: Magnetic potential parameter

    Returns:
        CSR matrix (complex-valued) of shape (num_nodes, num_nodes)
    """
    row, col = zip(*edges)
    A = csr_matrix((np.ones(len(edges)), (row, col)),
                   shape=(num_nodes, num_nodes), dtype=float)

    A_s = 0.5 * (A + A.T)
    Theta = 2 * np.pi * q * (A - A.T)

    H = A_s.multiply(np.exp(1j * Theta.toarray()))
    return csr_matrix(H, dtype=complex)


def magnetic_adjacency_eig(edges, num_nodes, q, k):
    """
    Compute the k largest eigenvalues/eigenvectors of the magnetic adjacency matrix.

    Args:
        edges: List of (i, j) tuples for directed edges i -> j
        num_nodes: Number of nodes
        q: Magnetic potential parameter
        k: Number of eigenvalues to compute

    Returns:
        eigenvalues: Array of k largest eigenvalues (real)
        eigenvectors: Array of shape (num_nodes, k)
    """
    H = magnetic_adjacency(edges, num_nodes, q)

    if num_nodes <= 512:
        vals, vecs = np.linalg.eigh(H.toarray())
        return vals[-k:][::-1], vecs[:, -k:][:, ::-1]

    eigenvalues, eigenvectors = eigsh(H, k=k, which='LM')
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


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
    H = magnetic_adjacency(edges, num_nodes, q)

    # Degree from symmetric weights
    d_s = np.array(np.abs(H).sum(axis=1)).flatten()

    if normalized:
        # L_N = I - D_s^{-1/2} H D_s^{-1/2}
        d_inv_sqrt = np.where(d_s > 0, 1.0 / np.sqrt(d_s), 0)
        D_inv_sqrt = diags(d_inv_sqrt)
        H_norm = D_inv_sqrt @ H @ D_inv_sqrt
        return diags(np.ones(num_nodes), dtype=complex) - csr_matrix(H_norm, dtype=complex)
    else:
        # L_U = D_s - H
        D_s = diags(d_s, dtype=complex)
        return D_s - H


def magnetic_laplacian_eig(edges, num_nodes, q, k, normalized=False):
    """
    Compute the first k eigenvalues/eigenvectors of the magnetic Laplacian.

    Args:
        edges: List of (i, j) tuples for directed edges i -> j
        num_nodes: Number of nodes
        q: Magnetic potential parameter
        k: Number of eigenvalues to compute
        normalized: If True, use normalized Laplacian

    Returns:
        eigenvalues: Array of k smallest eigenvalues (real)
        eigenvectors: Array of shape (num_nodes, k)
    """
    L = magnetic_laplacian(edges, num_nodes, q, normalized)

    if num_nodes <= 512:
        vals, vecs = np.linalg.eigh(L.toarray())
        return vals[:k], vecs[:, :k]

    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    return eigenvalues, eigenvectors