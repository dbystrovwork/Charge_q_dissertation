import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


def laplacian(edges, num_nodes, normalized=False):
    """
    Compute the graph Laplacian from an edge list.

    Args:
        edges: List of (i, j) tuples (undirected edges)
        num_nodes: Number of nodes
        normalized: If True, return normalized Laplacian L_N = I - D^{-1/2} A D^{-1/2}

    Returns:
        CSR matrix (real-valued)
    """
    row, col = zip(*edges)
    data = np.ones(len(edges))

    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=float)
    # Symmetrise in case only one direction is provided
    A = A + A.T
    A = (A > 0).astype(float)  # clamp duplicate entries to 1

    d = np.array(A.sum(axis=1)).flatten()

    if normalized:
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0)
        D_inv_sqrt = diags(d_inv_sqrt)
        return diags(np.ones(num_nodes)) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        return diags(d) - A


def laplacian_eig(edges, num_nodes, k, normalized=False):
    """
    Compute the first k eigenvalues/eigenvectors of the graph Laplacian.

    Args:
        edges: List of (i, j) tuples (undirected edges)
        num_nodes: Number of nodes
        k: Number of eigenvalues to compute
        normalized: If True, use normalized Laplacian

    Returns:
        eigenvalues: Array of k smallest eigenvalues
        eigenvectors: Array of shape (num_nodes, k)
    """
    L = laplacian(edges, num_nodes, normalized)

    if num_nodes <= 512:
        vals, vecs = np.linalg.eigh(L.toarray())
        return vals[:k], vecs[:, :k]

    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]
