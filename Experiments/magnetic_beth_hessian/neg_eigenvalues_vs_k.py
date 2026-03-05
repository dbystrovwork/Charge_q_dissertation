import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import dcsbm_cycle
from laplacians.bethe_hessian.magnetic_bethe_hessian import magnetic_bethe_hessian


def count_negative_eigenvalues(edges, num_nodes, q):
    """Count eigenvalues < 0 of the magnetic Bethe-Hessian with r = sqrt(avg degree)."""
    avg_degree = len(edges) / num_nodes
    r = np.sqrt(avg_degree)
    H = magnetic_bethe_hessian(edges, num_nodes, q, r=r)
    eigenvalues = np.linalg.eigvalsh(H.toarray())
    return np.sum(eigenvalues < 0)


def run_experiment(k_range, n_per_class, q, n_trials=5,
                   gamma_fwd=5.0, gamma_bwd=2.0, gamma_intra=5.0, seed=0):
    """
    For each k in k_range, generate n_trials DCSBM instances and count
    negative eigenvalues of the magnetic Bethe-Hessian.

    Returns:
        ks: array of true k values (repeated per trial)
        neg_counts: array of negative eigenvalue counts
    """
    rng = np.random.default_rng(seed)
    ks, neg_counts = [], []

    for k in k_range:
        for t in range(n_trials):
            trial_seed = rng.integers(0, 2**31)
            edges, _ = dcsbm_cycle(k, n_per_class, gamma_fwd, gamma_bwd,
                                   gamma_intra, seed=trial_seed)
            num_nodes = k * n_per_class
            n_neg = count_negative_eigenvalues(edges, num_nodes, q)
            ks.append(k)
            neg_counts.append(n_neg)
            print(f"k={k}, trial={t+1}/{n_trials}, neg_eigs={n_neg}")

    return np.array(ks), np.array(neg_counts)


def plot_neg_eigenvalues_vs_k(ks, neg_counts, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(ks, neg_counts, s=30, alpha=0.7, zorder=3, edgecolors="white",
               linewidths=0.5)

    lo, hi = min(ks.min(), neg_counts.min()), max(ks.max(), neg_counts.max())
    margin = 1
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            ls="--", color="black", lw=1, alpha=0.6, label=r"$y = x$")

    ax.set_xlabel("True number of communities $k$")
    ax.set_ylabel("Negative eigenvalues of $H_q(r)$")
    ax.legend(frameon=True)
    ax.set_aspect("equal")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    k_range = range(2, 21)
    q = 0.25
    n_per_class = 100

    ks, neg_counts = run_experiment(k_range, n_per_class, q, n_trials=5)
    plot_neg_eigenvalues_vs_k(ks, neg_counts)
    plt.show()
