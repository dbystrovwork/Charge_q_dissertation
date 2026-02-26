import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.normal_laplacian.laplacian_ops import laplacian


def adjacency_matrix(edges, num_nodes):
    """Build symmetric adjacency matrix from edge list."""
    row, col = zip(*edges)
    data = np.ones(len(edges))
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=float)
    A = A + A.T
    A = (A > 0).astype(float)
    return A


def inverse_participation_ratio(vec):
    """IPR = sum_i |v_i|^4 for normalized vector."""
    return np.sum(np.abs(vec) ** 4)


def principal_ipr_vs_gamma(
    n=400,
    gamma_range=(2.0, 3.2),
    n_gamma=20,
    n_instances=10,
    use_adjacency=True,
    seed=42,
):
    """
    Plot eigenvector IPR vs power-law exponent gamma for configuration model.

    Args:
        n: Number of nodes.
        gamma_range: (min, max) for gamma sweep.
        n_gamma: Number of gamma values to test.
        n_instances: Number of graph instances per gamma.
        use_adjacency: If True, use adjacency principal eigenvector.
                       If False, use Laplacian Fiedler vector.
        seed: Random seed.

    Returns:
        fig: matplotlib Figure.
    """
    rng = np.random.default_rng(seed)

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
    mean_ipr = []
    std_ipr = []

    for gamma in gammas:
        ipr_values = []

        for _ in range(n_instances):
            instance_seed = rng.integers(0, 2**31)
            edges, _, num_nodes = generate_graph(
                "configuration_model",
                seed=instance_seed,
                n=n,
                gamma=gamma,
                k_min=2,
                k_max=int(n**0.5),
            )

            if use_adjacency:
                A = adjacency_matrix(edges, num_nodes)
                _, eigenvectors = eigsh(A, k=1, which='LM')
                vec = eigenvectors[:, 0]
            else:
                L = laplacian(edges, num_nodes)
                _, eigenvectors = eigsh(L, k=2, which='SM')
                vec = eigenvectors[:, 1]  # Fiedler vector

            ipr_values.append(inverse_participation_ratio(vec))

        mean_ipr.append(np.mean(ipr_values))
        std_ipr.append(np.std(ipr_values))

    mean_ipr = np.array(mean_ipr)
    std_ipr = np.array(std_ipr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gammas, mean_ipr, "b-", linewidth=1.5, marker="o", markersize=4)

    if n_instances > 1:
        ax.fill_between(
            gammas,
            mean_ipr - std_ipr,
            mean_ipr + std_ipr,
            alpha=0.3,
        )

    ax.axhline(1 / n, color="red", linestyle="--", label="1/N (delocalized)")
    ax.axhline(1, color="green", linestyle="--", label="1 (fully localized)")

    vec_label = "Principal (Adjacency)" if use_adjacency else "Fiedler (Laplacian)"
    ax.set_xlabel(r"$\gamma$ (power-law exponent)")
    ax.set_ylabel(f"{vec_label} IPR")
    ax.set_title(
        f"{vec_label} IPR vs γ — Configuration Model\n"
        f"(N={n}, {n_instances} instances)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    principal_ipr_vs_gamma(n=int(1e4), n_gamma=20, n_instances=10, use_adjacency=False)
    plt.show()
