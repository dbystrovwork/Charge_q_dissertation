import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

from networks.dsbm import generate_graph
from laplacians.magnetic_laplacian.mag_lap_ops import (
    magnetic_laplacian_eig,
    magnetic_adjacency_eig,
)
from laplacians.magnetic_laplacian.spectral_clustering import spectral_clustering
from laplacians.bethe_hessian.magnetic_bethe_hessian import magnetic_bethe_hessian_eig


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


# ── operator registry ───────────────────────────────────────────────
def _ml_eig(edges, num_nodes, q, k):
    return magnetic_laplacian_eig(edges, num_nodes, q, k=k, normalized=True)


def _bh_eig(edges, num_nodes, q, k):
    return magnetic_bethe_hessian_eig(edges, num_nodes, q, k=k)


def _ma_eig(edges, num_nodes, q, k):
    return magnetic_adjacency_eig(edges, num_nodes, q, k=k)


OPERATOR_REGISTRY = {
    "Magnetic Laplacian": _ml_eig,
    "Bethe-Hessian": _bh_eig,
    "Magnetic adjacency": _ma_eig,
}
# ────────────────────────────────────────────────────────────────────


# ── metric registry ─────────────────────────────────────────────────
def _clustering_accuracy(true_labels, pred_labels):
    """Clustering accuracy with optimal label permutation (Hungarian method)."""
    classes = np.unique(true_labels)
    clusters = np.unique(pred_labels)
    n = max(len(classes), len(clusters))
    cost = np.zeros((n, n))
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cost[i, j] = -np.sum((true_labels == c) & (pred_labels == k))
    row_ind, col_ind = linear_sum_assignment(cost)
    return -cost[row_ind, col_ind].sum() / len(true_labels)


METRIC_REGISTRY = {
    "NMI": normalized_mutual_info_score,
    "ARI": adjusted_rand_score,
    "Accuracy": _clustering_accuracy,
}
# ────────────────────────────────────────────────────────────────────


def _sweep_q(eig_fn, edges, num_nodes, true_labels, q_values, k, metric_fn):
    """Run IPR and optional clustering metric sweep over q."""
    K = len(np.unique(true_labels)) if metric_fn is not None else None

    ipr_list = []
    metric_list = []
    for q in q_values:
        eigenvalues, eigenvectors = eig_fn(edges, num_nodes, q, k)

        ipr_per_vec = np.array([
            inverse_participation_ratio(eigenvectors[:, i])
            for i in range(k)
        ])
        ipr_list.append(ipr_per_vec)

        if metric_fn is not None:
            pred_labels = spectral_clustering(eigenvectors, K)
            metric_list.append(metric_fn(true_labels, pred_labels))

    return np.array(ipr_list), np.array(metric_list) if metric_fn is not None else None


def _plot_row(axes, q_values, mean_ipr, std_ipr, mean_metric, std_metric,
              k, n_repeats, num_nodes, operator, graph_type, metric_name):
    """Plot IPR (and optional clustering metric) into a row of axes."""
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
    ax.set_title(f"IPR vs q — {operator} — {graph_type}")
    ax.legend()
    ax.grid(True)

    if metric_name is not None:
        ax = axes[1]
        ax.plot(q_values, mean_metric)
        if n_repeats > 1:
            ax.fill_between(
                q_values,
                mean_metric - std_metric,
                mean_metric + std_metric,
                alpha=0.2,
            )
        ax.set_xlabel("q")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs q — {operator} — {graph_type}")
        ax.grid(True)


def localization_experiment(
    graph_type=None,
    operators=None,
    k=6,
    q_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
    metric=None,
):
    """
    Measure eigenvector localization (IPR) as a function of q.

    For each q value and each operator, computes the first k eigenvectors
    and plots the IPR. Each operator is plotted in its own row.

    Args:
        graph_type: Generator or dataset name (e.g. "dsbm_cycle", "cora_ml").
        operators: List of operator names to compare. Available:
            "Magnetic Laplacian", "Bethe-Hessian", "Magnetic adjacency".
            Defaults to ["Magnetic Laplacian"].
        k: Number of eigenvectors to consider.
        q_values: Array of charge parameter values to sweep.
        n_repeats: Number of graph realisations to average over.
        seed: Random seed.
        plot: Whether to show the plot.
        metric: Clustering metric to plot alongside IPR. Available:
            "NMI", "ARI", "Accuracy". None to skip (default).

    Returns:
        results: Dict mapping operator name to
            (mean_ipr, std_ipr, mean_metric, std_metric).
    """
    if operators is None:
        operators = ["Magnetic Laplacian"]
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)

    metric_fn = METRIC_REGISTRY[metric] if metric is not None else None

    eig_fns = [(name, OPERATOR_REGISTRY[name]) for name in operators]

    rng = np.random.default_rng(seed)

    results = {}
    use_metric = False
    num_nodes = None

    for op_name, eig_fn in eig_fns:
        all_ipr = []
        all_metric = []

        for rep in range(n_repeats):
            rep_seed = rng.integers(0, 2**31)
            edges, true_labels, num_nodes = generate_graph(graph_type, seed=rep_seed)

            has_labels = true_labels is not None
            use_metric = metric_fn is not None and has_labels
            active_metric_fn = metric_fn if use_metric else None

            ipr, met = _sweep_q(eig_fn, edges, num_nodes, true_labels,
                                q_values, k, active_metric_fn)
            all_ipr.append(ipr)
            if use_metric:
                all_metric.append(met)

        all_ipr = np.array(all_ipr)
        mean_ipr = all_ipr.mean(axis=0)
        std_ipr = all_ipr.std(axis=0)

        if use_metric:
            all_metric = np.array(all_metric)
            mean_metric = all_metric.mean(axis=0)
            std_metric = all_metric.std(axis=0)
        else:
            mean_metric = std_metric = None

        results[op_name] = (mean_ipr, std_ipr, mean_metric, std_metric)

    if plot:
        n_rows = len(operators)
        n_cols = 1 + int(use_metric)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(8 * n_cols, 5 * n_rows),
                                 squeeze=False)

        active_metric_name = metric if use_metric else None
        for row, op_name in enumerate(operators):
            mean_ipr, std_ipr, mean_metric, std_metric = results[op_name]
            _plot_row(axes[row], q_values, mean_ipr, std_ipr,
                      mean_metric, std_metric, k, n_repeats, num_nodes,
                      op_name, graph_type, active_metric_name)

        plt.tight_layout()
        plt.show()

    return results


GRAPH_TYPE = "dcsbm_cycle"

OPERATORS = ["Magnetic adjacency", "Bethe-Hessian"]

if __name__ == "__main__":
    localization_experiment(
        graph_type=GRAPH_TYPE,
        operators=OPERATORS,
        n_repeats=1,
        metric="Accuracy",
        k=5
    )
