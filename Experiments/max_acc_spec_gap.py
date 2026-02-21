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


def fubini_study_distance(vecs_a, vecs_b):
    """
    Compute the Fubini-Study distance between corresponding columns of two
    complex eigenvector matrices.

    d_FS(ψ, φ) = arccos( |⟨ψ|φ⟩| )

    Args:
        vecs_a: (n, k) complex eigenvector matrix
        vecs_b: (n, k) complex eigenvector matrix

    Returns:
        (k,) array of Fubini-Study distances, one per eigenvector
    """
    overlaps = np.abs(np.sum(vecs_a.conj() * vecs_b, axis=0))
    # Clip for numerical safety before arccos
    overlaps = np.clip(overlaps, 0.0, 1.0)
    return np.arccos(overlaps)


def spectral_gap_accuracy_experiment(
    K=7,
    n_per_class=50,
    p=0.2,
    s=0.2,
    r=0.05,
    q_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
    plot_eigenvalues=True,
    plot_fdr=True,
    plot_fubini_study=True
):
    """
    Run spectral gap vs clustering accuracy experiment over q values.

    Args:
        plot_eigenvalues: If True, plot eigenvalues. If False, plot spectral gaps.
        plot_fdr: If True, compute and plot FDR heatmap.
        plot_fubini_study: If True, compute and plot Fubini-Study distance between
            eigenvectors at consecutive q values.

    Returns:
        q_values: array of q values tested
        mean_eigs: (num_q, K) mean eigenvalues
        mean_gaps: (num_q, K-1) mean spectral gaps
        std_gaps: (num_q, K-1) std of spectral gaps
        mean_ari: (num_q,) mean ARI scores
        std_ari: (num_q,) std of ARI scores
        mean_fdr: (num_q, K) mean FDR per eigenvector, or None if plot_fdr=False
        mean_fs_dist: (num_q-1, K) mean Fubini-Study distances, or None if plot_fubini_study=False
    """
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)

    num_nodes = K * n_per_class
    rng = np.random.default_rng(seed)

    all_eigs = []
    all_gaps = []
    all_aris = []
    all_fdrs = []
    all_fs_dists = []

    for rep in range(n_repeats):
        rep_seed = rng.integers(0, 2**31)
        edges, true_labels = dsbm_cycle(K, n_per_class, p, s, r, seed=rep_seed)

        eigs_this_rep = []
        gaps_this_rep = []
        aris_this_rep = []
        fdrs_this_rep = []
        fs_dists_this_rep = []
        prev_eigenvectors = None

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

            if plot_fubini_study:
                if prev_eigenvectors is not None:
                    fs_dist = fubini_study_distance(prev_eigenvectors, eigenvectors)
                    fs_dists_this_rep.append(fs_dist)
                prev_eigenvectors = eigenvectors

            pred_labels = spectral_clustering(eigenvectors, K)
            ari = adjusted_rand_score(true_labels, pred_labels)
            aris_this_rep.append(ari)

        all_eigs.append(eigs_this_rep)
        all_gaps.append(gaps_this_rep)
        all_aris.append(aris_this_rep)
        if plot_fdr:
            all_fdrs.append(fdrs_this_rep)
        if plot_fubini_study:
            all_fs_dists.append(fs_dists_this_rep)

    all_eigs = np.array(all_eigs)  # (n_repeats, num_q, K)
    all_gaps = np.array(all_gaps)  # (n_repeats, num_q, K-1)
    all_aris = np.array(all_aris)  # (n_repeats, num_q)

    mean_eigs = all_eigs.mean(axis=0)
    mean_gaps = all_gaps.mean(axis=0)
    std_gaps = all_gaps.std(axis=0)
    mean_ari = all_aris.mean(axis=0)
    std_ari = all_aris.std(axis=0)
    mean_fdr = np.array(all_fdrs).mean(axis=0) if plot_fdr else None  # (num_q, K)
    mean_fs_dist = np.array(all_fs_dists).mean(axis=0) if plot_fubini_study else None  # (num_q-1, K)

    if plot:
        n_cols = 2 + int(plot_fdr) + int(plot_fubini_study)
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

        col = 2
        if plot_fdr:
            im = axes[col].imshow(
                mean_fdr.T,
                aspect='auto',
                origin='lower',
                extent=[q_values[0], q_values[-1], 0.5, K + 0.5]
            )
            axes[col].set_xlabel('q')
            axes[col].set_ylabel('Eigenvector')
            axes[col].set_title('FDR Heatmap')
            axes[col].set_yticks(range(1, K + 1))
            fig.colorbar(im, ax=axes[col], label='FDR')
            col += 1

        if plot_fubini_study:
            q_midpoints = 0.5 * (q_values[:-1] + q_values[1:])
            im = axes[col].imshow(
                mean_fs_dist.T,
                aspect='auto',
                origin='lower',
                extent=[q_midpoints[0], q_midpoints[-1], 0.5, K + 0.5]
            )
            axes[col].set_xlabel('q')
            axes[col].set_ylabel('Eigenvector')
            axes[col].set_title('Fubini-Study Distance')
            axes[col].set_yticks(range(1, K + 1))
            fig.colorbar(im, ax=axes[col], label='$d_{FS}$')
            col += 1

        plt.tight_layout()
        plt.show()

    return q_values, mean_eigs, mean_gaps, std_gaps, mean_ari, std_ari, mean_fdr, mean_fs_dist


if __name__ == "__main__":
    spectral_gap_accuracy_experiment(n_repeats=5)
