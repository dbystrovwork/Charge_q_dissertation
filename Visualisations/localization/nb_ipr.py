import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from networks.dsbm import generate_graph
from laplacians.bethe_hessian.non_backtracking import (
    magnetic_non_backtracking_eig,
    non_backtracking_eig,
)
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig


def nb_eigenvector_ipr(edges, num_nodes, k, q=None):
    """
    Compute the IPR of each non-backtracking eigenvector.

    IPR = sum_e |ψ(e)|^4  for a normalized eigenvector.
    Fully delocalized => IPR ~ 1/(2m), fully localized => IPR ~ 1.

    Args:
        edges: List of (i, j) tuples for directed edges.
        num_nodes: Number of nodes in the graph.
        k: Number of eigenvectors to compute.
        q: If given, use the magnetic non-backtracking matrix B_q.
            If None, use the standard non-backtracking matrix.

    Returns:
        eigenvalues: Array of k eigenvalues (complex).
        ipr: Array of shape (k,) with the IPR of each eigenvector.
    """
    if q is not None:
        eigenvalues, eigenvectors, _ = magnetic_non_backtracking_eig(
            edges, num_nodes, q, k=k
        )
    else:
        eigenvalues, eigenvectors, _ = non_backtracking_eig(
            edges, num_nodes, k=k
        )

    ipr = np.array([
        np.sum(np.abs(eigenvectors[:, i]) ** 4)
        for i in range(k)
    ])

    return eigenvalues, ipr


def ml_eigenvector_ipr(edges, num_nodes, k, q):
    """
    Compute the IPR of each magnetic Laplacian eigenvector.

    Args:
        edges: List of (i, j) tuples for directed edges.
        num_nodes: Number of nodes in the graph.
        k: Number of eigenvectors to compute.
        q: Magnetic potential parameter.

    Returns:
        eigenvalues: Array of k eigenvalues (real).
        ipr: Array of shape (k,) with the IPR of each eigenvector.
    """
    eigenvalues, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=k, normalized=True
    )

    ipr = np.array([
        np.sum(np.abs(eigenvectors[:, i]) ** 4)
        for i in range(k)
    ])

    return eigenvalues, ipr


def _print_table(title, eigenvalues, ipr, k, complex_eigs=True):
    print(f"\n{title}")
    label = "|λ|" if complex_eigs else "λ"
    print(f"{'Vec':<6} {label:<10} {'IPR':<12} {'1/IPR':<10}")
    print("-" * 38)
    for i in range(k):
        ev = np.abs(eigenvalues[i]) if complex_eigs else eigenvalues[i].real
        print(f"ψ{i+1:<5} {ev:<10.4f} {ipr[i]:<12.6f} {1/ipr[i]:<10.1f}")


GRAPH_TYPE = "dcsbm_cycle"


if __name__ == "__main__":
    edges, _, num_nodes = generate_graph(GRAPH_TYPE, seed=42)

    q = 0.25
    k = 6

    print(f"Graph: {GRAPH_TYPE}   q = {q}   k = {k}")

    nb_evals, nb_ipr = nb_eigenvector_ipr(edges, num_nodes, k=k, q=q)
    _print_table("Non-Backtracking", nb_evals, nb_ipr, k, complex_eigs=True)

    ml_evals, ml_ipr = ml_eigenvector_ipr(edges, num_nodes, k=k, q=q)
    _print_table("Magnetic Laplacian", ml_evals, ml_ipr, k, complex_eigs=False)
