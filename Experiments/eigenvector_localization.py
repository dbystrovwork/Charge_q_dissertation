import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from sklearn.metrics import normalized_mutual_info_score

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


def _sweep_q(eig_fn, edges, num_nodes, true_labels, q_values, k, compute_nmi):
    """Run IPR and optional NMI sweep over q for a given eigendecomposition function."""
    K = len(np.unique(true_labels)) if compute_nmi else None

    ipr_list = []
    nmi_list = []
    for q in q_values:
        eigenvalues, eigenvectors = eig_fn(edges, num_nodes, q, k)

        ipr_per_vec = np.array([
            inverse_participation_ratio(eigenvectors[:, i])
            for i in range(k)
        ])
        ipr_list.append(ipr_per_vec)

        if compute_nmi:
            pred_labels = spectral_clustering(eigenvectors, K)
            nmi_list.append(
                normalized_mutual_info_score(true_labels, pred_labels)
            )

    return np.array(ipr_list), np.array(nmi_list) if compute_nmi else None


def _plot_row(axes, q_values, mean_ipr, std_ipr, mean_nmi, std_nmi,
              k, n_repeats, num_nodes, operator, graph_type, compute_nmi):
    """Plot IPR (and optional NMI) into a row of axes."""
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
        ax.set_title(f"NMI vs q — {operator} — {graph_type}")
        ax.grid(True)


def localization_experiment(
    graph_type=None,
    operators=None,
    k=6,
    q_values=None,
    n_repeats=1,
    seed=42,
    plot=True,
    plot_nmi=False,
):
    """
    Measure eigenvector localization (IPR) as a function of q.

    For each q value and each operator, computes the first k eigenvectors
    and plots the IPR. Each operator is plotted in its own row.

    Args:
        graph_type: Generator or dataset name (e.g. "dsbm_cycle", "cora_ml").
        operators: List of operator names to compare. Available:
            "Magnetic Laplacian", "Bethe-Hessian", "Hermitian adjacency".
            Defaults to ["Magnetic Laplacian"].
        k: Number of eigenvectors to consider.
        q_values: Array of charge parameter values to sweep.
        n_repeats: Number of graph realisations to average over.
        seed: Random seed.
        plot: Whether to show the plot.
        plot_nmi: If True, also plot spectral clustering NMI (requires labels).

    Returns:
        results: Dict mapping operator name to (mean_ipr, std_ipr, mean_nmi, std_nmi).
    """
    if operators is None:
        operators = ["Magnetic Laplacian"]
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)

    eig_fns = [(name, OPERATOR_REGISTRY[name]) for name in operators]

    rng = np.random.default_rng(seed)

    results = {}
    compute_nmi = False
    num_nodes = None

    for op_name, eig_fn in eig_fns:
        all_ipr = []
        all_nmi = []

        for rep in range(n_repeats):
            rep_seed = rng.integers(0, 2**31)
            edges, true_labels, num_nodes = generate_graph(graph_type, seed=rep_seed)

            has_labels = true_labels is not None
            compute_nmi = plot_nmi and has_labels

            ipr, nmi = _sweep_q(eig_fn, edges, num_nodes, true_labels,
                                q_values, k, compute_nmi)
            all_ipr.append(ipr)
            if compute_nmi:
                all_nmi.append(nmi)

        all_ipr = np.array(all_ipr)
        mean_ipr = all_ipr.mean(axis=0)
        std_ipr = all_ipr.std(axis=0)

        if compute_nmi:
            all_nmi = np.array(all_nmi)
            mean_nmi = all_nmi.mean(axis=0)
            std_nmi = all_nmi.std(axis=0)
        else:
            mean_nmi = std_nmi = None

        results[op_name] = (mean_ipr, std_ipr, mean_nmi, std_nmi)

    if plot:
        n_rows = len(operators)
        n_cols = 1 + int(compute_nmi)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(8 * n_cols, 5 * n_rows),
                                 squeeze=False)

        for row, op_name in enumerate(operators):
            mean_ipr, std_ipr, mean_nmi, std_nmi = results[op_name]
            _plot_row(axes[row], q_values, mean_ipr, std_ipr,
                      mean_nmi, std_nmi, k, n_repeats, num_nodes,
                      op_name, graph_type, compute_nmi)

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
        plot_nmi=True,
    )
