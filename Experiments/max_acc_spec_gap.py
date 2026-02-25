import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from laplacians.magnetic_laplacian.spectral_clustering import spectral_clustering
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


GRAPH_TYPE = "nested_dsbm_cycle"  # "dsbm_cycle", "cora_ml", "c_elegans", "food_web"


def spectral_gap_accuracy_experiment(
    graph_type=GRAPH_TYPE,
    k=6,
    q_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
    plot_eigenvalues=True,
    plot_fdr=True,
    plot_fubini_study=True,
    metric="ari"
):
    """
    Run spectral gap vs clustering accuracy experiment over q values.

    Args:
        graph_type: Generator or dataset name (e.g. "dsbm_cycle", "cora_ml",
            "c_elegans", "food_web").
        k: Number of eigenvalues/clusters. Required for datasets without
           labels (c_elegans, food_web). Inferred from labels when available.
        plot_eigenvalues: If True, plot eigenvalues. If False, plot spectral gaps.
        plot_fdr: If True, compute and plot FDR heatmap (requires labels).
        plot_fubini_study: If True, compute and plot Fubini-Study distance between
            eigenvectors at consecutive q values.
        metric: Clustering accuracy metric — "ari" for Adjusted Rand Index
            or "nmi" for Normalized Mutual Information.

    Returns:
        q_values, mean_eigs, mean_gaps, std_gaps, mean_acc, std_acc, mean_fdr, mean_fs_dist
        (mean_acc/std_acc/mean_fdr are None when the dataset has no labels)
    """
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)

    rng = np.random.default_rng(seed)

    all_eigs = []
    all_gaps = []
    all_aris = []
    all_fdrs = []
    all_fs_dists = []

    for rep in range(n_repeats):
        rep_seed = rng.integers(0, 2**31)
        edges, true_labels, num_nodes = generate_graph(graph_type, seed=rep_seed)

        has_labels = true_labels is not None
        if has_labels:
            K = len(np.unique(true_labels))
        elif k is not None:
            K = k
        else:
            raise ValueError(
                f"Graph '{graph_type}' has no labels; pass k= explicitly"
            )

        # Disable label-dependent plots when no ground truth
        compute_ari = has_labels
        compute_fdr = plot_fdr and has_labels

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

            if compute_fdr:
                fdr = fisher_discriminant_ratio(eigenvectors.real, true_labels)
                fdrs_this_rep.append(fdr)

            if plot_fubini_study:
                if prev_eigenvectors is not None:
                    fs_dist = fubini_study_distance(prev_eigenvectors, eigenvectors)
                    fs_dists_this_rep.append(fs_dist)
                prev_eigenvectors = eigenvectors

            if compute_ari:
                pred_labels = spectral_clustering(eigenvectors, K)
                if metric == "nmi":
                    score = normalized_mutual_info_score(true_labels, pred_labels)
                else:
                    score = adjusted_rand_score(true_labels, pred_labels)
                aris_this_rep.append(score)

        all_eigs.append(eigs_this_rep)
        all_gaps.append(gaps_this_rep)
        if compute_ari:
            all_aris.append(aris_this_rep)
        if compute_fdr:
            all_fdrs.append(fdrs_this_rep)
        if plot_fubini_study:
            all_fs_dists.append(fs_dists_this_rep)

    all_eigs = np.array(all_eigs)  # (n_repeats, num_q, K)
    all_gaps = np.array(all_gaps)  # (n_repeats, num_q, K-1)

    mean_eigs = all_eigs.mean(axis=0)
    mean_gaps = all_gaps.mean(axis=0)
    std_gaps = all_gaps.std(axis=0)

    if compute_ari:
        all_aris = np.array(all_aris)
        mean_ari = all_aris.mean(axis=0)
        std_ari = all_aris.std(axis=0)
    else:
        mean_ari = std_ari = None

    mean_fdr = np.array(all_fdrs).mean(axis=0) if compute_fdr else None
    mean_fs_dist = np.array(all_fs_dists).mean(axis=0) if plot_fubini_study else None

    if plot:
        n_cols = 1 + int(compute_ari) + int(compute_fdr) + int(plot_fubini_study)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        col = 0
        if plot_eigenvalues:
            for i in range(K):
                axes[col].plot(q_values, mean_eigs[:, i], label=f'λ{i+1}')
            axes[col].set_ylabel('Eigenvalue')
            axes[col].set_title('Eigenvalues vs q')
        else:
            for i in range(K - 1):
                axes[col].plot(q_values, mean_gaps[:, i], label=f'λ{i+2} - λ{i+1}')
                if n_repeats > 1:
                    axes[col].fill_between(
                        q_values,
                        mean_gaps[:, i] - std_gaps[:, i],
                        mean_gaps[:, i] + std_gaps[:, i],
                        alpha=0.2
                    )
            axes[col].set_ylabel('Spectral Gap')
            axes[col].set_title('Spectral Gaps vs q')

        axes[col].set_xlabel('q')
        axes[col].legend()
        axes[col].grid(True)
        col += 1

        if compute_ari:
            axes[col].plot(q_values, mean_ari)
            if n_repeats > 1:
                axes[col].fill_between(
                    q_values,
                    mean_ari - std_ari,
                    mean_ari + std_ari,
                    alpha=0.2
                )
            metric_label = "NMI" if metric == "nmi" else "ARI"
            axes[col].set_xlabel('q')
            axes[col].set_ylabel(metric_label)
            axes[col].set_title(f'Clustering {metric_label} vs q')
            axes[col].grid(True)
            col += 1

        if compute_fdr:
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
    spectral_gap_accuracy_experiment(n_repeats=5, metric="nmi")
