import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.sparse import csr_matrix

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


def eigenvector_heatmap(
    graph_type="barabasi_albert",
    k=None,
    seed=42,
    normalized=False,
):
    """
    Heatmap of eigenvector magnitude, with nodes sorted by degree.

    X-axis = node index sorted by degree (low to high).
    Y-axis = eigenvalue index (ascending eigenvalue, skipping lambda_0).
    Colour = |psi_i(v)|.

    Args:
        graph_type: Any symmetric graph type registered in generate_graph.
        k: Number of eigenvectors to compute (including the trivial one).
        seed: Random seed.
        normalized: Whether to use the normalized Laplacian.

    Returns:
        fig: The matplotlib Figure.
    """
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)

    if k is None:
        k = num_nodes - 1

    eigenvalues, eigenvectors = laplacian_eig(
        edges, num_nodes, k=k, normalized=normalized,
    )

    # Skip the trivial zero eigenvalue
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    degrees = node_degrees(edges, num_nodes)
    degree_order = np.argsort(-degrees)

    mag = np.abs(eigenvectors[degree_order, :]).T

    # Custom colormap: white -> yellow -> orange -> red -> black
    cmap = LinearSegmentedColormap.from_list(
        "localisation",
        ["white", "yellow", "orange", "red", "black"],
    )

    X = np.arange(len(degree_order))
    Y = np.arange(len(eigenvalues))

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.contourf(
        X, Y, mag,
        levels=50,
        cmap=cmap,
    )

    # Node index tick labels
    n_xticks = min(10, len(degree_order))
    xtick_pos = np.linspace(0, len(degree_order) - 1, n_xticks, dtype=int)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_pos)

    # Eigenvalue index tick labels
    n_yticks = min(10, len(eigenvalues))
    ytick_pos = np.linspace(0, len(eigenvalues) - 1, n_yticks, dtype=int)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_pos + 1)

    ax.set_xlabel("Node index (sorted by degree)")
    ax.set_ylabel("Eigenvalue index")
    lap_label = "normalised" if normalized else "unnormalised"
    ax.set_title(
        f"Eigenvector localisation — {graph_type} "
        f"({lap_label}, N={num_nodes})"
    )

    fig.colorbar(im, ax=ax, label=r"$|\psi_i(v)|$", shrink=0.8)
    plt.tight_layout()
    return fig


GRAPH_TYPE = "barabasi_albert"


if __name__ == "__main__":
    eigenvector_heatmap(graph_type=GRAPH_TYPE, seed=42)
    plt.show()
