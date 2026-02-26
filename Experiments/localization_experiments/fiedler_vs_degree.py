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


def fiedler_magnitude_vs_degree(
    graph_type="barabasi_albert",
    seed=42,
    normalized=False,
):
    """
    Plot |psi_2(v)|^2 (Fiedler vector magnitude squared) vs node degree.

    Args:
        graph_type: Any symmetric graph type registered in generate_graph.
        seed: Random seed.
        normalized: Whether to use the normalized Laplacian.

    Returns:
        fig: The matplotlib Figure.
    """
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)

    eigenvalues, eigenvectors = laplacian_eig(
        edges, num_nodes, k=2, normalized=normalized,
    )

    fiedler = eigenvectors[:, 1]
    mag_sq = np.abs(fiedler) ** 2

    degrees = node_degrees(edges, num_nodes)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(degrees, mag_sq, s=12, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Node degree")
    ax.set_ylabel(r"$|\psi_2(v)|^2$")
    lap_label = "normalised" if normalized else "unnormalised"
    ax.set_title(
        f"Fiedler vector localisation — {graph_type} "
        f"({lap_label}, N={num_nodes})"
    )
    ax.grid(True)

    plt.tight_layout()
    return fig


GRAPH_TYPE = "barabasi_albert"


if __name__ == "__main__":
    fiedler_magnitude_vs_degree(graph_type=GRAPH_TYPE, seed=42)
    plt.show()
