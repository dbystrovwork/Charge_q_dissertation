import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


def magnetic_bethe_hessian(edges, num_nodes, q, r=None):
    """
    Compute the magnetic Bethe-Hessian matrix using the Ihara formula.

    The Ihara determinant formula gives:
        ζ_G(u)^{-1} = (1 - u²)^{m-n} · det(I - u·T_q + u²·(D_s - I))

    Setting r = 1/u and multiplying through by r² yields the Bethe-Hessian:
        H_q(r) = (r² - 1)·I  -  r·T_q  +  D_s

    where T_q is the magnetic adjacency matrix (Hermitian) constructed
    from the directed edge list and the charge parameter q.

    Args:
        edges: List of (i, j) tuples for directed edges i -> j
        num_nodes: Number of nodes
        q: Magnetic potential parameter (charge)
        r: Regularization parameter. If None, defaults to sqrt(mean degree).

    Returns:
        CSR matrix (complex-valued) of shape (num_nodes, num_nodes)
    """
    # Build directed adjacency matrix
    row, col = zip(*edges)
    A = csr_matrix((np.ones(len(edges)), (row, col)),
                   shape=(num_nodes, num_nodes), dtype=float)

    # Symmetric magnitude: A_s = 0.5 * (A + A^T)
    A_s = 0.5 * (A + A.T)

    # Skew phase: Θ = 2πq * (A - A^T)
    Theta = 2 * np.pi * q * (A - A.T)

    # Magnetic adjacency: T_q = A_s ⊙ exp(iΘ)
    T_q = A_s.multiply(np.exp(1j * Theta.toarray()))
    T_q = csr_matrix(T_q, dtype=complex)

    # Symmetric degree vector
    d_s = np.array(A_s.sum(axis=1)).flatten()

    # Default r = sqrt(mean degree)
    if r is None:
        r = np.sqrt(d_s.mean())

    # Bethe-Hessian via Ihara: H_q(r) = (r² - 1)I - r·T_q + D_s
    I_n = diags(np.ones(num_nodes), dtype=complex)
    D_s = diags(d_s, dtype=complex)
    H = (r**2 - 1) * I_n - r * T_q + D_s

    return H


def magnetic_bethe_hessian_eig(edges, num_nodes, q, k, r=None):
    """
    Compute the k smallest eigenvalues/eigenvectors of the magnetic Bethe-Hessian.

    Args:
        edges: List of (i, j) tuples for directed edges i -> j
        num_nodes: Number of nodes
        q: Magnetic potential parameter (charge)
        k: Number of eigenvalues to compute
        r: Regularization parameter. If None, defaults to sqrt(mean degree).

    Returns:
        eigenvalues: Array of k smallest eigenvalues (real)
        eigenvectors: Array of shape (num_nodes, k)
    """
    H = magnetic_bethe_hessian(edges, num_nodes, q, r)
    eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]