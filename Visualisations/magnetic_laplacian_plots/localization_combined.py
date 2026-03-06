import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use("seaborn-v0_8-paper")

from Experiments.eigenvector_localization import localization_experiment
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from networks.dsbm import generate_graph


def localization_combined(
    graph_type,
    operator="Magnetic Laplacian",
    k=3,
    plot_k=None,
    q_values=None,
    q_heatmap=None,
    n_repeats=1,
    seed=42,
    metric="Accuracy",
):
    """
    Combined figure: IPR vs q and clustering metric vs q (from the
    localization experiment) side-by-side with the eigenvector
    localization heatmap.

    Args:
        graph_type: Generator or dataset name.
        operator: Operator name for IPR / metric sweep.
        k: Number of eigenvectors.
        plot_k: How many eigenvectors to show in the IPR plot (default: k).
        q_values: Array of q values for the sweep.
        q_heatmap: q value used for the heatmap snapshot. Defaults to 0.25.
        n_repeats: Graph realisations to average over.
        seed: Random seed.
        metric: Clustering metric ("NMI", "ARI", "Accuracy") or None.
    """
    if q_values is None:
        q_values = np.linspace(0, 0.5, 50)
    if q_heatmap is None:
        q_heatmap = 0.25
    if plot_k is None:
        plot_k = k

    # Run the localization experiment (no plot — we draw ourselves)
    results = localization_experiment(
        graph_type=graph_type,
        operators=[operator],
        k=k,
        plot_k=plot_k,
        q_values=q_values,
        n_repeats=n_repeats,
        seed=seed,
        plot=False,
        metric=metric,
    )

    mean_ipr, std_ipr, mean_metric, std_metric = results[operator]
    has_metric = mean_metric is not None

    # Generate graph for the heatmap
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)

    # Layout: [IPR] [Metric?] [Heatmap]
    n_cols = 2 + int(has_metric)
    fig = plt.figure(figsize=(6 * n_cols, 5))
    gs = GridSpec(1, n_cols, figure=fig, wspace=0.35)

    # ── IPR vs q ──
    ax_ipr = fig.add_subplot(gs[0])
    for i in range(plot_k):
        ax_ipr.plot(q_values, mean_ipr[:, i], label=f"$\\psi_{i}$")
    ax_ipr.axhline(1 / num_nodes, color="black", linestyle="--", label="1/N")
    ax_ipr.set_xlabel("q")
    ax_ipr.set_ylabel("IPR")
    ax_ipr.set_title(f"IPR vs q — {operator}")
    ax_ipr.legend()
    ax_ipr.grid(True)

    # ── Metric vs q ──
    col = 1
    if has_metric:
        ax_met = fig.add_subplot(gs[1])
        ax_met.plot(q_values, mean_metric)
        if n_repeats > 1:
            ax_met.fill_between(
                q_values,
                mean_metric - std_metric,
                mean_metric + std_metric,
                alpha=0.2,
            )
        ax_met.set_xlabel("q")
        ax_met.set_ylabel(metric)
        ax_met.set_title(f"{metric} vs q — {operator}")
        ax_met.grid(True)
        col = 2

    # ── Heatmap ──
    eigenvalues, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q_heatmap, k=k, normalized=True
    )

    prob = np.abs(eigenvectors) ** 2
    if labels is not None:
        sort_idx = np.argsort(labels, kind="stable")
        prob = prob[sort_idx]
        sorted_labels = labels[sort_idx]

    ax_heat = fig.add_subplot(gs[col])
    im = ax_heat.imshow(prob, aspect="auto", cmap="viridis", interpolation="nearest")
    ax_heat.set_xlabel("Eigenvector index")
    ax_heat.set_xticks(range(k))
    ax_heat.set_xticklabels([f"$\\psi_{i}$" for i in range(k)])
    ax_heat.set_ylabel("Node (sorted by community)")
    ax_heat.set_title(f"|ψ(v)|²   (q = {q_heatmap:.3f})")
    fig.colorbar(im, ax=ax_heat, label="|ψ(v)|²", shrink=0.6)

    if labels is not None:
        boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
        for b in boundaries:
            ax_heat.axhline(b, color="white", linewidth=1.0)

    plt.tight_layout()
    plt.show()


GRAPH_TYPE = "dsbm_cycle"

if __name__ == "__main__":
    localization_combined(
        graph_type=GRAPH_TYPE,
        operator="Magnetic Laplacian",
        k=3,
        plot_k=3,
        n_repeats=1,
        metric="Accuracy",
        q_heatmap=0.25,
    )
