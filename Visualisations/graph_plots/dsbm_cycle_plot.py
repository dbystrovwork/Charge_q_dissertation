import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph


def dsbm_cycle_layout(labels, community_radius=1.0, node_radius=0.25):
    """
    Arrange communities in a circle (first at pi/2) with nodes
    within each community also placed in a small circle.

    Args:
        labels: Array of integer community labels.
        community_radius: Radius of the outer circle of community centres.
        node_radius: Radius of the inner circle for nodes within a community.

    Returns:
        pos: dict mapping node index to (x, y) position.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    # Community centres on a circle, starting at pi/2
    community_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + np.pi / 2
    centres = {
        lbl: community_radius * np.array([np.cos(a), np.sin(a)])
        for lbl, a in zip(unique_labels, community_angles)
    }

    pos = {}
    for lbl in unique_labels:
        members = np.where(labels == lbl)[0]
        n = len(members)
        # Nodes in a small circle around their community centre
        node_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx, cy = centres[lbl]
        for idx, theta in zip(members, node_angles):
            pos[idx] = (cx + node_radius * np.cos(theta),
                        cy + node_radius * np.sin(theta))

    return pos


def plot_dsbm_cycle(seed=42, community_radius=1.0, node_radius=0.25, ax=None):
    edges, labels, num_nodes = generate_graph("dsbm_cycle", seed=seed)
    pos = dsbm_cycle_layout(labels, community_radius, node_radius)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Draw edges
    for i, j in edges:
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="grey",
                lw=0.4,
                alpha=0.4,
                shrinkA=3, shrinkB=3,
            ),
        )

    # Draw nodes coloured by community
    unique_labels = np.unique(labels)
    cmap = plt.cm.tab10
    for lbl in unique_labels:
        members = np.where(labels == lbl)[0]
        xs = [pos[m][0] for m in members]
        ys = [pos[m][1] for m in members]
        ax.scatter(xs, ys, s=150, color=cmap(lbl), edgecolors="black",
                   linewidths=0.5, zorder=3, label=f"Community {lbl}")

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"DSBM Cycle  (N={num_nodes}, E={len(edges)})")
    ax.axis("off")

    return ax


if __name__ == "__main__":
    plot_dsbm_cycle()
    plt.tight_layout()
    plt.show()
