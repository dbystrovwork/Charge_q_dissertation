import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from sklearn.metrics import normalized_mutual_info_score

from networks.dsbm import generate_graph
from magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from magnetic_laplacian.spectral_clustering import spectral_clustering


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


GRAPH_TYPE = "directed_erdos_renyi"


def localization_experiment(
    graph_type=GRAPH_TYPE,
    k=6,
    q_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
    plot_nmi=False,
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
        plot_nmi: If True, also plot spectral clustering NMI (requires labels).

    Returns:
        q_values, mean_ipr, std_ipr, mean_nmi, std_nmi
    """
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)

    rng = np.random.default_rng(seed)

    all_ipr = []
    all_nmi = []

    for rep in range(n_repeats):
        rep_seed = rng.integers(0, 2**31)
        edges, true_labels, num_nodes = generate_graph(graph_type, seed=rep_seed)

        has_labels = true_labels is not None
        compute_nmi = plot_nmi and has_labels
        if compute_nmi:
            K = len(np.unique(true_labels))

        ipr_this_rep = []
        nmi_this_rep = []

        for q in q_values:
            eigenvalues, eigenvectors = magnetic_laplacian_eig(
                edges, num_nodes, q, k=k, normalized=True
            )

            ipr_per_vec = np.array([
                inverse_participation_ratio(eigenvectors[:, i])
                for i in range(k)
            ])
            ipr_this_rep.append(ipr_per_vec)

            if compute_nmi:
                pred_labels = spectral_clustering(eigenvectors, K)
                nmi_this_rep.append(
                    normalized_mutual_info_score(true_labels, pred_labels)
                )

        all_ipr.append(ipr_this_rep)
        if compute_nmi:
            all_nmi.append(nmi_this_rep)

    all_ipr = np.array(all_ipr)  # (n_repeats, num_q, k)
    mean_ipr = all_ipr.mean(axis=0)  # (num_q, k)
    std_ipr = all_ipr.std(axis=0)

    if compute_nmi:
        all_nmi = np.array(all_nmi)
        mean_nmi = all_nmi.mean(axis=0)
        std_nmi = all_nmi.std(axis=0)
    else:
        mean_nmi = std_nmi = None

    if plot:
        n_cols = 1 + int(compute_nmi)
        fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        ax = axes[0]
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

        if compute_nmi:
            ax = axes[1]
            ax.plot(q_values, mean_nmi)
            if n_repeats > 1:
                ax.fill_between(
                    q_values,
                    mean_nmi - std_nmi,
                    mean_nmi + std_nmi,
                    alpha=0.2,
                )
            ax.set_xlabel("q")
            ax.set_ylabel("NMI")
            ax.set_title(f"Spectral Clustering NMI vs q — {graph_type}")
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return q_values, mean_ipr, std_ipr, mean_nmi, std_nmi


if __name__ == "__main__":
    localization_experiment(n_repeats=1)
