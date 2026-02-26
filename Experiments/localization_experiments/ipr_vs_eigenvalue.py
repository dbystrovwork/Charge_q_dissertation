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


def ipr_vs_eigenvalue(
    graph_type="barabasi_albert",
    k=50,
    n_instances=10,
    normalized=False,
    seed=42,
):
    """
    Plot IPR vs eigenvalue index, averaged over multiple graph instances.

    Args:
        graph_type: Graph generator name.
        k: Number of eigenvectors to compute.
        n_instances: Number of graph instances to average over.
        normalized: Use normalized Laplacian.
        seed: Random seed.

    Returns:
        fig: matplotlib Figure.
    """
    rng = np.random.default_rng(seed)

    all_ipr = []
    num_nodes = None

    for _ in range(n_instances):
        instance_seed = rng.integers(0, 2**31)
        edges, _, n = generate_graph(graph_type, seed=instance_seed)
        num_nodes = n

        eigenvalues, eigenvectors = laplacian_eig(
            edges, n, k=k, normalized=normalized
        )

        # Skip trivial zero eigenvalue
        eigenvectors = eigenvectors[:, 1:]

        ipr = np.array([
            inverse_participation_ratio(eigenvectors[:, i])
            for i in range(eigenvectors.shape[1])
        ])
        all_ipr.append(ipr)

    all_ipr = np.array(all_ipr)
    mean_ipr = all_ipr.mean(axis=0)
    std_ipr = all_ipr.std(axis=0)

    eigenvalue_indices = np.arange(1, len(mean_ipr) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eigenvalue_indices, mean_ipr, "b-", linewidth=1.5)

    if n_instances > 1:
        ax.fill_between(
            eigenvalue_indices,
            mean_ipr - std_ipr,
            mean_ipr + std_ipr,
            alpha=0.3,
        )

    ax.axhline(1 / num_nodes, color="red", linestyle="--", label="1/N (delocalized)")
    ax.axhline(1, color="green", linestyle="--", label="1 (fully localized)")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("IPR")
    lap_label = "normalized" if normalized else "unnormalized"
    ax.set_title(
        f"IPR vs Eigenvalue Index — {graph_type}\n"
        f"({lap_label}, k={k-1}, n_instances={n_instances})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


GRAPH_TYPE = "barabasi_albert"


if __name__ == "__main__":
    ipr_vs_eigenvalue(graph_type=GRAPH_TYPE, k=200, n_instances=20)
    plt.show()
