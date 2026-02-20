import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import dsbm_cycle
from magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from magnetic_laplacian.spectral_clustering import spectral_clustering
from Metrics.fdr import fisher_discriminant_ratio


def spectral_gap_accuracy_experiment(
    K=5,
    n_per_class=50,
    p=0.2,
    s=0.2,
    r=0.05,
    q_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
    plot_eigenvalues=True,
    plot_fdr=True
):
    """
    Run spectral gap vs clustering accuracy experiment over q values.

    Args:
        plot_eigenvalues: If True, plot eigenvalues. If False, plot spectral gaps.
        plot_fdr: If True, compute and plot FDR heatmap.

    Returns:
        q_values: array of q values tested
        mean_eigs: (num_q, K) mean eigenvalues
        mean_gaps: (num_q, K-1) mean spectral gaps
        std_gaps: (num_q, K-1) std of spectral gaps
        mean_ari: (num_q,) mean ARI scores
        std_ari: (num_q,) std of ARI scores
        mean_fdr: (num_q, K) mean FDR per eigenvector, or None if plot_fdr=False
    """
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)

    num_nodes = K * n_per_class
    rng = np.random.default_rng(seed)

    all_eigs = []
    all_gaps = []
    all_aris = []
    all_fdrs = []

    for rep in range(n_repeats):
        rep_seed = rng.integers(0, 2**31)
        edges, true_labels = dsbm_cycle(K, n_per_class, p, s, r, seed=rep_seed)

        eigs_this_rep = []
        gaps_this_rep = []
        aris_this_rep = []
        fdrs_this_rep = []

        for q in q_values:
            eigenvalues, eigenvectors = magnetic_laplacian_eig(
                edges, num_nodes, q, k=K, normalized=True
            )

            sorted_eigs = np.sort(eigenvalues.real)
            eigs_this_rep.append(sorted_eigs)
            gaps_this_rep.append(np.diff(sorted_eigs))

            if plot_fdr:
                fdr = fisher_discriminant_ratio(eigenvectors.real, true_labels)
                fdrs_this_rep.append(fdr)

            pred_labels = spectral_clustering(eigenvectors, K)
            ari = adjusted_rand_score(true_labels, pred_labels)
            aris_this_rep.append(ari)

        all_eigs.append(eigs_this_rep)
        all_gaps.append(gaps_this_rep)
        all_aris.append(aris_this_rep)
        if plot_fdr:
            all_fdrs.append(fdrs_this_rep)

    all_eigs = np.array(all_eigs)  # (n_repeats, num_q, K)
    all_gaps = np.array(all_gaps)  # (n_repeats, num_q, K-1)
    all_aris = np.array(all_aris)  # (n_repeats, num_q)

    mean_eigs = all_eigs.mean(axis=0)
    mean_gaps = all_gaps.mean(axis=0)
    std_gaps = all_gaps.std(axis=0)
    mean_ari = all_aris.mean(axis=0)
    std_ari = all_aris.std(axis=0)
    mean_fdr = np.array(all_fdrs).mean(axis=0) if plot_fdr else None  # (num_q, K)

    if plot:
        n_cols = 3 if plot_fdr else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))

        if plot_eigenvalues:
            for i in range(K):
                axes[0].plot(q_values, mean_eigs[:, i], label=f'λ{i+1}')
            axes[0].set_ylabel('Eigenvalue')
            axes[0].set_title('Eigenvalues vs q')
        else:
            for i in range(K - 1):
                axes[0].plot(q_values, mean_gaps[:, i], label=f'λ{i+2} - λ{i+1}')
                if n_repeats > 1:
                    axes[0].fill_between(
                        q_values,
                        mean_gaps[:, i] - std_gaps[:, i],
                        mean_gaps[:, i] + std_gaps[:, i],
                        alpha=0.2
                    )
            axes[0].set_ylabel('Spectral Gap')
            axes[0].set_title('Spectral Gaps vs q')

        axes[0].set_xlabel('q')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(q_values, mean_ari)
        if n_repeats > 1:
            axes[1].fill_between(
                q_values,
                mean_ari - std_ari,
                mean_ari + std_ari,
                alpha=0.2
            )
        axes[1].set_xlabel('q')
        axes[1].set_ylabel('ARI')
        axes[1].set_title('Clustering Accuracy vs q')
        axes[1].grid(True)

        if plot_fdr:
            im = axes[2].imshow(
                mean_fdr.T,
                aspect='auto',
                origin='lower',
                extent=[q_values[0], q_values[-1], 0.5, K + 0.5]
            )
            axes[2].set_xlabel('q')
            axes[2].set_ylabel('Eigenvector')
            axes[2].set_title('FDR Heatmap')
            axes[2].set_yticks(range(1, K + 1))
            fig.colorbar(im, ax=axes[2], label='FDR')

        plt.tight_layout()
        plt.show()

    return q_values, mean_eigs, mean_gaps, std_gaps, mean_ari, std_ari, mean_fdr


if __name__ == "__main__":
    spectral_gap_accuracy_experiment(n_repeats=5)
