import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph


def plot_graph(edges, num_nodes, labels=None, node_sizes=None, ax=None):
    """
    Plot a directed graph from an edge list.

    Args:
        edges: List of (i, j) tuples for directed edges.
        num_nodes: Number of nodes in the graph.
        labels: Optional array of node labels for colouring.
        node_sizes: Optional array of per-node sizes.
        ax: Optional matplotlib Axes to draw on.

    Returns:
        The matplotlib Axes used.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # Seed initial positions by class so same-label nodes cluster together
    if labels is not None:
        unique = np.unique(labels)
        k = len(unique)
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        centers = {c: np.array([np.cos(a), np.sin(a)]) for c, a in zip(unique, angles)}
        rng = np.random.default_rng(42)
        init_pos = {
            i: centers[labels[i]] + rng.normal(scale=0.1, size=2)
            for i in range(num_nodes)
        }
        pos = nx.spring_layout(G, pos=init_pos, seed=42, iterations=1)
    else:
        pos = nx.spring_layout(G, seed=42)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    draw_kwargs = dict(
        G=G,
        pos=pos,
        ax=ax,
        node_size=20,
        width=0.3,
        arrowsize=4,
        alpha=0.8,
    )

    if labels is not None:
        draw_kwargs["node_color"] = labels
        draw_kwargs["cmap"] = plt.cm.tab10

    if node_sizes is not None:
        draw_kwargs["node_size"] = node_sizes

    nx.draw(
        **draw_kwargs,
    )

    ax.set_title(f"Graph  (N={num_nodes}, E={len(edges)})")

    return ax


GRAPH_TYPE = "directed_barbell"  # "directed_barbell", "dsbm_cycle", "cora_ml", "c_elegans", "food_web"


if __name__ == "__main__":
    edges, true_labels, num_nodes = generate_graph(GRAPH_TYPE, seed=42)
    plot_graph(edges, num_nodes, labels=true_labels)
    plt.tight_layout()
    plt.show()
