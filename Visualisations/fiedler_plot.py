import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from graph_plot import plot_graph


def node_degrees(edges, num_nodes):
    row, col = zip(*edges)
    A = csr_matrix(
        (np.ones(len(edges)), (row, col)),
        shape=(num_nodes, num_nodes), dtype=float,
    )
    A = A + A.T
    A = (A > 0).astype(float)
    return np.array(A.sum(axis=1)).flatten().astype(int)


def plot_fiedler_graph(edges, num_nodes, q, labels=None):
    """
    Plot a graph with node sizes proportional to the magnitude of the
    Fiedler vector (second smallest eigenvector) of the magnetic Laplacian.

    Args:
        edges: List of (i, j) tuples for directed edges.
        num_nodes: Number of nodes in the graph.
        q: Magnetic potential parameter.
        labels: Optional array of node labels for colouring.

    Returns:
        The matplotlib Axes used.
    """
    eigenvalues, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=2, normalized=True
    )

    fiedler = eigenvectors[:, 1]
    magnitudes = np.abs(fiedler)

    # Scale to reasonable node sizes
    min_size, max_size = 10, 300
    if magnitudes.max() > magnitudes.min():
        node_sizes = min_size + (max_size - min_size) * (
            (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
        )
    else:
        node_sizes = np.full(num_nodes, (min_size + max_size) / 2)

    ax = plot_graph(edges, num_nodes, labels=labels, node_sizes=node_sizes)
    ax.set_title(f"Fiedler vector magnitude (q={q:.3f}, N={num_nodes})")

    return ax


def plot_fiedler_heatmap(edges, num_nodes, q_values):
    """
    Heatmap of Fiedler vector magnitude.

    Y-axis = node index sorted by degree (high to low).
    X-axis = q values.
    Colour = |Fiedler(v)|.
    """
    degrees = node_degrees(edges, num_nodes)
    degree_order = np.argsort(-degrees)

    # Build heatmap: rows = nodes, cols = q
    heatmap = np.zeros((num_nodes, len(q_values)))

    for i, q in enumerate(q_values):
        _, vecs = magnetic_laplacian_eig(edges, num_nodes, q, k=1, normalized=True)
        heatmap[:, i] = np.abs(vecs[:, 0][degree_order])

    cmap = LinearSegmentedColormap.from_list(
        "localisation", ["white", "yellow", "orange", "red", "black"]
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        heatmap,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        extent=[q_values[0], q_values[-1], num_nodes, 0],
    )

    ax.set_xlabel("q")
    ax.set_ylabel("Node index (sorted by degree)")
    ax.set_title(f"Fiedler vector — N={num_nodes}")
    fig.colorbar(im, ax=ax, label=r"$|\psi_1(v)|$", shrink=0.8)

    plt.tight_layout()
    return fig


GRAPH_TYPE = "two_cycles"  # "directed_barbell", "dsbm_cycle", "cora_ml", "c_elegans", "food_web"
qs = [0.1, 0.15, 0.2, 0.25, 0.3]

if __name__ == "__main__":
    edges, true_labels, num_nodes = generate_graph(GRAPH_TYPE, seed=42)
    plot_fiedler_graph(edges, num_nodes, q=1/4, labels=true_labels)
    plt.show()
