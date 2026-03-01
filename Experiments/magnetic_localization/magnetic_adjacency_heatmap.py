import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix

from networks.dsbm import generate_graph
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_adjacency_eig


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


def magnetic_adjacency_heatmap(
    graph_type,
    q_values,
    seed=42,
):
    """
    Heatmap of principal eigenvector magnitude vs q and node index.

    X-axis = node index sorted by degree (high to low).
    Y-axis = q values.
    Colour = |psi(v)| for principal eigenvector (largest eigenvalue).

    Args:
        graph_type: Graph type string registered in generate_graph.
        q_values: Array of q values to sweep.
        seed: Random seed.

    Returns:
        fig: The matplotlib Figure.
    """
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)

    degrees = node_degrees(edges, num_nodes)
    degree_order = np.argsort(-degrees)

    # Build heatmap data: rows = q, cols = node (sorted by degree)
    heatmap = np.zeros((len(q_values), num_nodes))

    for i, q in enumerate(q_values):
        _, vecs = magnetic_adjacency_eig(edges, num_nodes, q, k=1)
        heatmap[i, :] = np.abs(vecs[:, 0][degree_order])

    cmap = LinearSegmentedColormap.from_list(
        "localisation",
        ["white", "yellow", "orange", "red", "black"],
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        heatmap,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        extent=[0, num_nodes, q_values[0], q_values[-1]],
    )

    ax.set_xlabel("Node index (sorted by degree)")
    ax.set_ylabel("q")
    ax.set_title(
        f"Magnetic adjacency principal eigenvector — {graph_type}\n"
        f"(N={num_nodes})"
    )

    fig.colorbar(im, ax=ax, label=r"$|\psi(v)|$", shrink=0.8)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    q_vals = np.linspace(0, 0.5, 100)
    magnetic_adjacency_heatmap("dcsbm_cycle", q_vals, seed=42)
    plt.show()
