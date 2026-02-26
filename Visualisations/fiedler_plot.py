import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from graph_plot import plot_graph


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


GRAPH_TYPE = "dsbm_cycle"


if __name__ == "__main__":
    edges, true_labels, num_nodes = generate_graph(GRAPH_TYPE, seed=42)
    plot_fiedler_graph(edges, num_nodes, q=0.3, labels=true_labels)
    plt.tight_layout()
    plt.show()
