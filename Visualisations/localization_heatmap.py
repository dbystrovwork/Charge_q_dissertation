import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig


def plot_localization_heatmap(edges, num_nodes, q, k, labels=None):
    """
    Plot a heatmap of eigenvector localization |ψ_i(v)|² on each node.

    Nodes are sorted by community label (if provided) so that
    block structure is visible.

    Args:
        edges: List of (i, j) tuples for directed edges.
        num_nodes: Number of nodes in the graph.
        q: Magnetic potential parameter.
        k: Number of eigenvectors to compute.
        labels: Optional array of node community labels for sorting.

    Returns:
        The matplotlib Figure.
    """
    eigenvalues, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=k, normalized=True
    )

    # |ψ_i(v)|² matrix: (N, k)
    prob = np.abs(eigenvectors) ** 2

    # Sort nodes by label if available
    if labels is not None:
        sort_idx = np.argsort(labels, kind="stable")
        prob = prob[sort_idx]
        sorted_labels = labels[sort_idx]
    else:
        sort_idx = np.arange(num_nodes)

    # Build figure: label strip (if labels) + heatmap
    has_labels = labels is not None
    if has_labels:
        fig = plt.figure(figsize=(max(k * 0.8, 6), 8))
        gs = GridSpec(1, 2, width_ratios=[0.4, k], wspace=0.05, figure=fig)

        # Community label strip
        ax_labels = fig.add_subplot(gs[0])
        ax_labels.imshow(
            sorted_labels[:, np.newaxis],
            aspect="auto",
            cmap="tab10",
            interpolation="nearest",
        )
        ax_labels.set_xticks([])
        ax_labels.set_ylabel("Node (sorted by community)")
        ax_labels.set_title("Class")

        ax_heat = fig.add_subplot(gs[1])
    else:
        fig, ax_heat = plt.subplots(figsize=(max(k * 0.8, 6), 8))

    # Main heatmap
    im = ax_heat.imshow(prob, aspect="auto", cmap="viridis", interpolation="nearest")
    ax_heat.set_xlabel("Eigenvector index")
    ax_heat.set_xticks(range(k))
    ax_heat.set_xticklabels([f"ψ{i+1}" for i in range(k)])
    ax_heat.set_title(f"|ψ(v)|²   (q = {q:.3f})")
    fig.colorbar(im, ax=ax_heat, label="|ψ(v)|²", shrink=0.6)

    if has_labels:
        ax_heat.set_yticks([])
    else:
        ax_heat.set_ylabel("Node")

    # Draw community boundary lines
    if has_labels:
        boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
        for b in boundaries:
            ax_heat.axhline(b, color="white", linewidth=1.0)
            ax_labels.axhline(b, color="white", linewidth=1.0)

    plt.tight_layout()
    return fig


GRAPH_TYPE = "barabasi_albert"


if __name__ == "__main__":
    edges, true_labels, num_nodes = generate_graph(GRAPH_TYPE, seed=42)
    plot_localization_heatmap(edges, num_nodes, q=0, k=6, labels=true_labels)
    plt.show()
