import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import csr_matrix
import networkx as nx

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


def plot_fiedler_graph(edges, num_nodes, q, labels=None, eig_index=1):
    """
    Plot a graph with node sizes proportional to the magnitude of a
    chosen eigenvector of the magnetic Laplacian.

    Args:
        edges: List of (i, j) tuples for directed edges.
        num_nodes: Number of nodes in the graph.
        q: Magnetic potential parameter.
        labels: Optional array of node labels for colouring.
        eig_index: Index of eigenvector to use (0 = smallest, 1 = Fiedler, etc.)

    Returns:
        The matplotlib Axes used.
    """
    eigenvalues, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=eig_index + 1, normalized=True
    )

    vec = eigenvectors[:, eig_index]
    magnitudes = np.abs(vec)

    # Scale to reasonable node sizes
    min_size, max_size = 10, 300
    if magnitudes.max() > magnitudes.min():
        node_sizes = min_size + (max_size - min_size) * (
            (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
        )
    else:
        node_sizes = np.full(num_nodes, (min_size + max_size) / 2)

    ax = plot_graph(edges, num_nodes, labels=labels, node_sizes=node_sizes)
    ax.set_title(f"Eigenvector {eig_index} magnitude (q={q:.3f}, N={num_nodes})")

    return ax


def plot_fiedler_heatmap(edges, num_nodes, q_values, eig_index=0):
    """
    Heatmap of eigenvector magnitude.

    Y-axis = node index sorted by degree (high to low).
    X-axis = q values.
    Colour = |eigenvector(v)|.

    Args:
        eig_index: Index of eigenvector to use (0 = smallest, 1 = Fiedler, etc.)
    """
    degrees = node_degrees(edges, num_nodes)
    degree_order = np.argsort(-degrees)

    # Build heatmap: rows = nodes, cols = q
    heatmap = np.zeros((num_nodes, len(q_values)))

    for i, q in enumerate(q_values):
        _, vecs = magnetic_laplacian_eig(edges, num_nodes, q, k=eig_index + 1, normalized=True)
        heatmap[:, i] = np.abs(vecs[:, eig_index][degree_order])

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
    ax.set_title(f"Eigenvector {eig_index} — N={num_nodes}")
    fig.colorbar(im, ax=ax, label=rf"$|\psi_{eig_index}(v)|$", shrink=0.8)

    plt.tight_layout()
    return fig


def plot_two_cycles(k1, k2, p, q, eig_index=1, figsize=(10, 5), save_path=None):
    """
    Publication-ready plot of two_cycles graph with eigenvector magnitude.

    Args:
        k1: Length of first cycle
        k2: Length of second cycle
        p: Length of path connecting cycles
        q: Magnetic potential parameter
        eig_index: Eigenvector index to visualize
        figsize: Figure size tuple
        save_path: If provided, save figure to this path

    Returns:
        fig, ax
    """
    from networks.localization_examples.two_cycles import two_cycles

    edges, labels = two_cycles(k1, k2, p)
    num_nodes = k1 + k2 + (p - 1)

    _, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=eig_index + 1, normalized=True
    )
    magnitudes = np.abs(eigenvectors[:, eig_index])
    mag_norm = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-10)

    # Build networkx graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # Layout: cycles as circles, path as line
    r1 = 1.0
    r2 = 1.0 * (k2 / k1) ** 0.5  # scale radius by relative size
    gap = r1 + r2 + 2.5 + 0.3 * (p - 1)  # adjust gap for path length

    pos = {}

    # Cycle 1: clockwise from right (so exit node faces the path)
    for i in range(k1):
        angle = -2 * np.pi * i / k1
        pos[i] = (r1 * np.cos(angle), r1 * np.sin(angle))

    # Cycle 2: clockwise from left (so entry node faces the path)
    for i in range(k2):
        angle = np.pi - 2 * np.pi * i / k2
        pos[k1 + i] = (gap + r2 * np.cos(angle), r2 * np.sin(angle))

    # Path nodes: horizontal line between cycles
    if p > 1:
        x_start = pos[k1 - 1][0] + 0.4
        x_end = pos[k1][0] - 0.4
        for i in range(p - 1):
            t = (i + 1) / p
            pos[k1 + k2 + i] = (x_start + t * (x_end - x_start), 0)

    # Color scheme
    cmap = LinearSegmentedColormap.from_list(
        "eigenvector", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    node_colors = [cmap(m) for m in mag_norm]

    # Node sizes scaled by magnitude
    min_size, max_size = 50, 250
    node_sizes = min_size + (max_size - min_size) * mag_norm

    # Edge colors by type
    edge_colors = []
    for i, j in edges:
        if labels[i] == 0 and labels[j] == 0:
            edge_colors.append("#4393c3")  # blue for cycle 1
        elif labels[i] == 1 and labels[j] == 1:
            edge_colors.append("#2166ac")  # darker blue for cycle 2
        else:
            edge_colors.append("#666666")  # gray for path

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=1.5,
        alpha=0.8,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        node_size=node_sizes,
    )

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#333333",
        linewidths=1.2,
    )

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=magnitudes.min(), vmax=magnitudes.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label(rf"$|\psi_{{{eig_index}}}(v)|$", fontsize=12)

    # Title
    ax.set_title(
        rf"Two cycles ($k_1={k1}$, $k_2={k2}$, $p={p}$) — $q={q:.2f}$, eigenvector {eig_index}",
        fontsize=13, pad=10
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")

    return fig, ax


GRAPH_TYPE = "two_cycles"

if __name__ == "__main__":
    fig, ax = plot_two_cycles(k1=7, k2=11, p=10, q=1/7, eig_index=3)
    plt.show()
