import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.bethe_hessian.non_backtracking import (
    non_backtracking_matrix,
    magnetic_non_backtracking_matrix,
)


def compute_spectrum(graph_type, q=None, seed=42):
    """
    Compute the non-backtracking (or magnetic NB) spectrum for a graph.

    Returns:
        eigenvalues, bulk_radius
    """
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)

    if q is not None:
        B, _ = magnetic_non_backtracking_matrix(edges, num_nodes, q)
    else:
        B, _ = non_backtracking_matrix(edges, num_nodes)

    eigenvalues = np.linalg.eig(B.toarray())[0]

    degrees = np.zeros(num_nodes)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    bulk_radius = np.sqrt(degrees.mean())

    return eigenvalues, bulk_radius


graphs = ["dcsbm_cycle", "c_elegans"]
qs = [0.2, 0.2]

if __name__ == "__main__":
    n_graphs = len(graphs)
    fig, axes = plt.subplots(1, n_graphs, figsize=(5 * n_graphs, 5))
    if n_graphs == 1:
        axes = [axes]

    for ax, graph_type, q in zip(axes, graphs, qs):
        eigenvalues, bulk_radius = compute_spectrum(graph_type, q=q, seed=42)

        # Bulk circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(bulk_radius * np.cos(theta), bulk_radius * np.sin(theta),
                'k--', linewidth=1)

        real_part = eigenvalues.real
        imag_part = eigenvalues.imag
        modulus = np.abs(eigenvalues)

        outlier_mask = modulus > bulk_radius + 0.1
        bulk_mask = ~outlier_mask

        ax.scatter(real_part[bulk_mask], imag_part[bulk_mask],
                   s=8, alpha=0.4, c="steelblue")
        ax.scatter(real_part[outlier_mask], imag_part[outlier_mask],
                   s=8, c="red", zorder=5)

        # Square limits clipped to bulk + small margin
        lim = bulk_radius * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.97, f"{graph_type}\nq = {q}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)

    plt.tight_layout()
    plt.show()
