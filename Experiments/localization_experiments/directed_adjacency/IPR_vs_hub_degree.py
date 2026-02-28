import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from networks.extensions.hub import add_hub
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_adjacency


def inverse_participation_ratio(vec):
    """IPR = sum_i |v_i|^4 for normalized vector."""
    return np.sum(np.abs(vec) ** 4)


def ipr_vs_hub_degree(
    graph_type="directed_barabasi_albert",
    q=0.25,
    multiples=np.arange(1, 11),
    n_instances=10,
    use_effective= False,
    seed=42,
):
    """
    Plot IPR of the largest eigenvector of the magnetic adjacency matrix
    as the hub degree increases in multiples of the average degree.

    Args:
        graph_type: Graph type passed to generate_graph
        q: Magnetic potential parameter
        multiples: Array of multipliers applied to d_bar for hub degree
        n_instances: Number of graph instances to average over
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    all_ipr = []
    num_nodes_base = None
    d_bar = None

    for rep in range(n_instances):
        instance_seed = int(rng.integers(0, 2**31))
        edges, _, num_nodes = generate_graph(graph_type, seed=instance_seed)
        num_nodes_base = num_nodes

        # Average degree of the base graph
        d_bar = 2 * len(edges) / num_nodes
        if rep == 0:
            print(f"Graph type: {graph_type}")
            print(f"Nodes: {num_nodes}, Edges: {len(edges)}")
            print(f"Average degree (d_bar): {d_bar:.2f}")

        ipr_for_rep = []
        for m in multiples:
            hub_degree = int(m * d_bar)
            hub_degree = min(hub_degree, num_nodes)

            hub_edges, hub_id = add_hub(edges, hub_degree, seed=instance_seed + int(m))
            n_total = hub_id + 1

            H = magnetic_adjacency(hub_edges, n_total, q)

            if n_total <= 512:
                vals, vecs = np.linalg.eigh(H.toarray())
                # Largest eigenvalue is last
                leading_vec = vecs[:, -1]
            else:
                vals, vecs = eigsh(H, k=1, which='LM')
                leading_vec = vecs[:, 0]

            leading_vec = leading_vec / np.linalg.norm(leading_vec)
            ipr = inverse_participation_ratio(leading_vec)
            ipr_for_rep.append(1.0 / ipr if use_effective else ipr)

        all_ipr.append(ipr_for_rep)

    all_vals = np.array(all_ipr)
    mean_vals = all_vals.mean(axis=0)
    std_vals = all_vals.std(axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(multiples, mean_vals, "b-o", linewidth=1.5, markersize=4)

    if n_instances > 1:
        ax.fill_between(
            multiples,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.2,
        )

    N = num_nodes_base + 1
    if use_effective:
        ax.set_ylabel("1 / IPR (effective support size)")
        metric_label = "Effective support"
    else:
        ax.set_ylabel("IPR")
        metric_label = "IPR"

    ax.set_xlabel(r"Hub degree multiple $m$ ($d_{\mathrm{hub}} = m \cdot \bar{d}$)")
    ax.set_title(
        f"{metric_label} of leading eigenvector (magnetic adjacency) vs hub degree\n"
        f"{graph_type}, q={q}, $\\bar{{d}}$={d_bar:.1f}, n_instances={n_instances}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


GRAPH_TYPE = "dcsbm_cycle"
Q = 0.2

Ms = np.arange(1, 11)

if __name__ == "__main__":
    fig = ipr_vs_hub_degree(
        graph_type=GRAPH_TYPE,
        q=Q,
        multiples=Ms,
        use_effective=True,
        n_instances=1,
    )
    plt.show()
