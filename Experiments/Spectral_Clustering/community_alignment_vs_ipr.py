import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.normal_laplacian.laplacian_ops import laplacian_eig


def inverse_participation_ratio(vec):
    """IPR = sum_i |v_i|^4 for normalized vector."""
    return np.sum(np.abs(vec) ** 4)


def community_alignment_score(vec, labels):
    """
    Compute community alignment as within-class variance / total variance.

    Lower values indicate eigenvector aligns with community structure
    (uniform within communities, different between).
    """
    total_var = np.var(vec)
    if total_var == 0:
        return 1.0

    within_var = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 1:
            within_var += np.var(vec[mask]) * mask.sum()
    within_var /= len(vec)

    return within_var / total_var


def community_alignment_vs_ipr(
    graph_type="dsbm_cycle",
    k=50,
    normalized=True,
    seed=42,
):
    """
    Scatter plot of community alignment score vs IPR for each eigenvector.

    Args:
        graph_type: Graph generator name (must have community labels).
        k: Number of eigenvectors to compute.
        normalized: Use normalized Laplacian.
        seed: Random seed.

    Returns:
        fig: matplotlib Figure.
    """
    edges, labels, n = generate_graph(graph_type, seed=seed)

    if labels is None:
        raise ValueError(f"Graph type '{graph_type}' has no community labels")

    eigenvalues, eigenvectors = laplacian_eig(edges, n, k=k, normalized=normalized)

    # Skip trivial zero eigenvalue
    eigenvectors = eigenvectors[:, 1:]
    eigenvalues = eigenvalues[1:]

    iprs = []
    alignments = []

    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        iprs.append(inverse_participation_ratio(vec))
        alignments.append(community_alignment_score(vec, labels))

    iprs = np.array(iprs)
    alignments = np.array(alignments)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(iprs, alignments, c=np.arange(len(iprs)), cmap="viridis", alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Eigenvalue index")

    ax.set_xlabel("IPR")
    ax.set_ylabel("Community Alignment (within-var / total-var)")
    ax.set_xscale("log")

    lap_label = "normalized" if normalized else "unnormalized"
    ax.set_title(f"Community Alignment vs IPR — {graph_type}\n({lap_label}, k={k-1})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


GRAPH_TYPE = "dsbm_cycle"


if __name__ == "__main__":
    community_alignment_vs_ipr(graph_type=GRAPH_TYPE, k=50, normalized=True)
    plt.show()
