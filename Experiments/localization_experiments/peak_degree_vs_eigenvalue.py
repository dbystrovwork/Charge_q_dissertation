import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.normal_laplacian.laplacian_ops import laplacian_eig


def node_degrees(edges, num_nodes):
    """Compute undirected degree of each node from an edge list."""
    row, col = zip(*edges)
    A = csr_matrix(
        (np.ones(len(edges)), (row, col)),
        shape=(num_nodes, num_nodes), dtype=float,
    )
    A = A + A.T
    A = (A > 0).astype(float)
    return np.array(A.sum(axis=1)).flatten().astype(int)


def peak_degree_vs_eigenvalue(
    graph_type="barabasi_albert",
    k=20,
    seed=42,
    normalized=False,
):
    """
    For each of the first k eigenvectors, find the node with the largest
    |psi(v)|^2 and record its degree. Plot degree vs eigenvalue.

    Args:
        graph_type: Any symmetric graph type registered in generate_graph.
        k: Number of eigenvectors to compute.
        seed: Random seed.
        normalized: Whether to use the normalized Laplacian.

    Returns:
        fig: The matplotlib Figure.
    """
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)

    eigenvalues, eigenvectors = laplacian_eig(
        edges, num_nodes, k=k, normalized=normalized,
    )

    degrees = node_degrees(edges, num_nodes)

    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    peak_nodes = np.argmax(np.abs(eigenvectors) ** 2, axis=0)
    peak_degrees = degrees[peak_nodes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(peak_degrees, eigenvalues, s=30, alpha=0.7, edgecolors="none")

    ax.set_xlabel("Degree of peak node")
    ax.set_ylabel(r"$\lambda$")
    lap_label = "normalised" if normalized else "unnormalised"
    ax.set_title(
        f"Peak-node degree vs eigenvalue — {graph_type} "
        f"({lap_label}, k={k}, N={num_nodes})"
    )
    ax.grid(True)

    plt.tight_layout()
    return fig


GRAPH_TYPE = "barabasi_albert"


if __name__ == "__main__":
    peak_degree_vs_eigenvalue(graph_type=GRAPH_TYPE, k=300, seed=42)
    plt.show()
