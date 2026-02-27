import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from sklearn.metrics import normalized_mutual_info_score

from networks.dsbm import generate_graph
from laplacians.bethe_hessian.magnetic_bethe_hessian import magnetic_bethe_hessian
from laplacians.magnetic_laplacian.spectral_clustering import spectral_clustering


def nmi_vs_r(
    graph_type="dsbm_cycle",
    q=0.25,
    r_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
):
    """
    Sweep the regularization parameter r of the magnetic Bethe-Hessian,
    select eigenvectors whose eigenvalues are negative, and compute NMI
    from spectral clustering on those eigenvectors.

    Args:
        graph_type: Generator or dataset name.
        q: Magnetic potential parameter (charge).
        r_values: Array of r values to sweep.
        n_repeats: Number of graph realisations to average over.
        seed: Random seed.
        plot: Whether to show the plot.

    Returns:
        r_values, mean_nmi, std_nmi, mean_n_neg
    """
    if r_values is None:
        r_values = np.linspace(0.5, 5.0, 50)

    rng = np.random.default_rng(seed)

    all_nmi = []
    all_n_neg = []

    for rep in range(n_repeats):
        rep_seed = rng.integers(0, 2**31)
        edges, true_labels, num_nodes = generate_graph(graph_type, seed=rep_seed)
        K = len(np.unique(true_labels))

        nmi_this_rep = []
        n_neg_this_rep = []

        for r in r_values:
            H = magnetic_bethe_hessian(edges, num_nodes, q, r=r)
            eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())

            # Select eigenvectors with negative eigenvalues
            neg_mask = eigenvalues < 0
            n_neg = neg_mask.sum()
            n_neg_this_rep.append(n_neg)

            if n_neg >= 2:
                neg_vecs = eigenvectors[:, neg_mask]
                pred_labels = spectral_clustering(neg_vecs, K)
                nmi = normalized_mutual_info_score(true_labels, pred_labels)
            else:
                nmi = 0.0

            nmi_this_rep.append(nmi)

        all_nmi.append(nmi_this_rep)
        all_n_neg.append(n_neg_this_rep)

    all_nmi = np.array(all_nmi)
    all_n_neg = np.array(all_n_neg, dtype=float)
    mean_nmi = all_nmi.mean(axis=0)
    std_nmi = all_nmi.std(axis=0)
    mean_n_neg = all_n_neg.mean(axis=0)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(r_values, mean_nmi)
        if n_repeats > 1:
            ax.fill_between(r_values,
                            mean_nmi - std_nmi,
                            mean_nmi + std_nmi,
                            alpha=0.2)
        ax.set_xlabel("r")
        ax.set_ylabel("NMI")
        ax.set_title(f"NMI vs r — Magnetic Bethe-Hessian (q={q}) — {graph_type}")
        ax.grid(True)

        ax = axes[1]
        ax.plot(r_values, mean_n_neg)
        ax.set_xlabel("r")
        ax.set_ylabel("Number of negative eigenvalues")
        ax.set_title(f"Negative eigenvalue count vs r — {graph_type}")
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return r_values, mean_nmi, std_nmi, mean_n_neg


if __name__ == "__main__":
    nmi_vs_r()
