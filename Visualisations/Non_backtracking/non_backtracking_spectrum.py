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


def plot_non_backtracking_spectrum(graph_type="sbm", q=None, seed=42):
    """
    Plot all eigenvalues of the non-backtracking matrix in the complex plane
    for an SBM graph.

    Informative eigenvalues (corresponding to communities) appear as real
    eigenvalues outside the bulk circle of radius sqrt(mean degree).

    Args:
        graph_type: Graph generator name.
        q: If not None, use the magnetic non-backtracking matrix with this charge.
        seed: Random seed.
    """
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)

    if q is not None:
        B, directed_edges = magnetic_non_backtracking_matrix(edges, num_nodes, q)
    else:
        B, directed_edges = non_backtracking_matrix(edges, num_nodes)

    # Dense eig for full spectrum (sparse eigs cannot compute all eigenvalues)
    eigenvalues = np.linalg.eig(B.toarray())[0]

    # Mean degree from the undirected adjacency
    degrees = np.zeros(num_nodes)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    mean_deg = degrees.mean()
    bulk_radius = np.sqrt(mean_deg)

    K = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Bulk circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(bulk_radius * np.cos(theta), bulk_radius * np.sin(theta),
            'k--', linewidth=1, label=f"r = √d̄ = {bulk_radius:.2f}")

    # Colour eigenvalues: real outliers vs bulk
    real_part = eigenvalues.real
    imag_part = eigenvalues.imag
    modulus = np.abs(eigenvalues)

    outlier_mask = modulus > bulk_radius + 0.1
    bulk_mask = ~outlier_mask

    ax.scatter(real_part[bulk_mask], imag_part[bulk_mask],
               s=8, alpha=0.4, c="steelblue", label="Bulk")
    ax.scatter(real_part[outlier_mask], imag_part[outlier_mask],
               s=40, c="red", zorder=5, label=f"Outliers ({outlier_mask.sum()})")

    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    operator = f"Magnetic NB (q={q})" if q is not None else "Non-backtracking"
    ax.set_title(f"{operator} spectrum — {graph_type} (K={K}, n={num_nodes})")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return eigenvalues


if __name__ == "__main__":
    plot_non_backtracking_spectrum(graph_type="c_elegans", q=0.2, seed=42)
