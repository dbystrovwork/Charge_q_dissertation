import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig


def inverse_participation_ratio(vec):
    """
    Compute the Inverse Participation Ratio (IPR) of a vector.

    IPR = sum_i |v_i|^4  (for a normalized vector)

    A fully delocalized vector has IPR ~ 1/N.
    A fully localized vector has IPR ~ 1.

    Args:
        vec: (n,) complex or real vector (assumed normalized)

    Returns:
        Scalar IPR value.
    """
    return np.sum(np.abs(vec) ** 4)


GRAPH_TYPE = "food_web"


def localization_experiment(
    graph_type=GRAPH_TYPE,
    k=6,
    q_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
):
    """
    Measure eigenvector localization (IPR) as a function of q.

    For each q value, computes the first k eigenvectors of the magnetic
    Laplacian and plots the IPR of each eigenvector.

    Args:
        graph_type: Generator or dataset name (e.g. "dsbm_cycle", "cora_ml").
        k: Number of eigenvectors to consider.
        q_values: Array of charge parameter values to sweep.
        n_repeats: Number of graph realisations to average over.
        seed: Random seed.
        plot: Whether to show the plot.

    Returns:
        q_values, mean_ipr, std_ipr
    """
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)

    rng = np.random.default_rng(seed)

    all_ipr = []

    for rep in range(n_repeats):
        rep_seed = rng.integers(0, 2**31)
        edges, true_labels, num_nodes = generate_graph(graph_type, seed=rep_seed)

        ipr_this_rep = []

        for q in q_values:
            eigenvalues, eigenvectors = magnetic_laplacian_eig(
                edges, num_nodes, q, k=k, normalized=True
            )

            ipr_per_vec = np.array([
                inverse_participation_ratio(eigenvectors[:, i])
                for i in range(k)
            ])
            ipr_this_rep.append(ipr_per_vec)

        all_ipr.append(ipr_this_rep)

    all_ipr = np.array(all_ipr)  # (n_repeats, num_q, k)
    mean_ipr = all_ipr.mean(axis=0)  # (num_q, k)
    std_ipr = all_ipr.std(axis=0)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))

        for i in range(k):
            ax.plot(q_values, mean_ipr[:, i], label=f"ψ{i+1}")
            if n_repeats > 1:
                ax.fill_between(
                    q_values,
                    mean_ipr[:, i] - std_ipr[:, i],
                    mean_ipr[:, i] + std_ipr[:, i],
                    alpha=0.2,
                )

        ax.axhline(1 / num_nodes, color="black", linestyle="--", label="1/N")

        ax.set_xlabel("q")
        ax.set_ylabel("IPR")
        ax.set_title(f"Eigenvector Localization (IPR) vs q — {graph_type}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return q_values, mean_ipr, std_ipr


if __name__ == "__main__":
    localization_experiment(n_repeats=1)
