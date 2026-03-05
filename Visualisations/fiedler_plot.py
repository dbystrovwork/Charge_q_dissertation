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


def plot_eigenvector_graph(graph_type, q, eig_index=1, figsize=(8, 6),
                           title=None, save_path=None, seed=42, **overrides):
    """
    Plot any graph with node colour/size proportional to eigenvector magnitude.

    Args:
        graph_type: Any key from generate_graph (e.g. "two_cycles", "cycle_tail", "dcsbm_cycle")
        q: Magnetic potential parameter
        eig_index: Eigenvector index (0 = smallest, 1 = Fiedler, etc.)
        figsize: Figure size tuple
        title: Custom title (auto-generated if None)
        save_path: If provided, save figure to this path
        seed: Random seed for graph generation and layout
        **overrides: Override any graph config parameter

    Returns:
        fig, ax
    """
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed, **overrides)

    _, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=eig_index + 1, normalized=True
    )
    vec = eigenvectors[:, eig_index]
    magnitudes = np.abs(vec)
    phases = np.angle(vec)  # in [-pi, pi]
    mag_norm = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-10)

    # Build networkx graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # Layout
    pos = nx.spring_layout(G, seed=seed, k=1.5 / np.sqrt(num_nodes), iterations=100)

    # Color scheme for magnitude
    mag_cmap = LinearSegmentedColormap.from_list(
        "eigenvector", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    node_colors = [mag_cmap(m) for m in mag_norm]

    # Node sizes scaled by magnitude
    min_size, max_size = 50, 250
    node_sizes = min_size + (max_size - min_size) * mag_norm

    # Edge colors by community label
    edge_colors = []
    palette = ["#4393c3", "#2166ac", "#666666", "#5aae61", "#762a83"]
    for i, j in edges:
        if labels is not None and labels[i] == labels[j]:
            edge_colors.append(palette[int(labels[i]) % len(palette)])
        else:
            edge_colors.append("#999999")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=1.2,
        alpha=0.6,
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        node_size=node_sizes,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#333333",
        linewidths=1.0,
    )

    # Phase labels next to each node
    phase_cmap = plt.cm.hsv
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    for node_id in range(num_nodes):
        x, y = pos[node_id]
        phase = phases[node_id]
        phase_color = phase_cmap(phase_norm(phase))
        ax.annotate(
            rf"${phase / np.pi:+.1f}\pi$",
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=5,
            color=phase_color,
            fontweight="bold",
        )

    # Magnitude colorbar
    sm_mag = ScalarMappable(cmap=mag_cmap, norm=Normalize(vmin=magnitudes.min(), vmax=magnitudes.max()))
    sm_mag.set_array([])
    cbar_mag = fig.colorbar(sm_mag, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar_mag.set_label(rf"$|\psi_{{{eig_index}}}(v)|$", fontsize=11)

    # Phase colorbar
    sm_phase = ScalarMappable(cmap=phase_cmap, norm=phase_norm)
    sm_phase.set_array([])
    cbar_phase = fig.colorbar(sm_phase, ax=ax, shrink=0.5, aspect=20, pad=0.06)
    cbar_phase.set_label(rf"$\arg(\psi_{{{eig_index}}}(v))$", fontsize=11)
    cbar_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar_phase.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    # Title
    if title is None:
        title = rf"{graph_type} — $q={q:.2f}$, eigenvector {eig_index}, $N={num_nodes}$"
    ax.set_title(title, fontsize=13, pad=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_eigenvector_graph("cycle_tail", q=1/10, eig_index=1, k=10, p=3)
    plt.show()
