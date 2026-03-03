import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from Experiments.localization_experiments.eigenvector_heatmap import node_degrees


def ca_hepth_localization(k=25):
    """
    Two-panel figure for the ca-HepTh collaboration network:
      (a) Degree distribution (log-log)
      (b) Eigenvector heatmap: |psi_i(v)| for the top-k adjacency
          eigenvectors, nodes sorted by degree (high → low)
    """
    edges, _, num_nodes = generate_graph("ca_hepth")

    # Build symmetric adjacency matrix
    row, col = zip(*edges)
    A = csr_matrix(
        (np.ones(len(edges)), (row, col)),
        shape=(num_nodes, num_nodes), dtype=float,
    )
    A = A + A.T
    A = (A > 0).astype(float)

    degrees = node_degrees(edges, num_nodes)
    degree_order = np.argsort(-degrees)

    # Top-k eigenvectors by largest eigenvalue
    eigenvalues, eigenvectors = eigsh(A, k=k, which="LM")
    # eigsh returns ascending order; flip to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Reorder nodes by degree (high → low), shape: (k, num_nodes)
    mag = np.abs(eigenvectors[degree_order, :]).T

    # Custom colormap matching eigenvector_heatmap
    cmap = LinearSegmentedColormap.from_list(
        "localisation",
        ["white", "yellow", "orange", "red", "black"],
    )

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Degree distribution
    unique_deg, counts = np.unique(degrees, return_counts=True)
    ax1.scatter(unique_deg, counts, s=10, alpha=0.7, edgecolors="none")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Degree $k$")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Degree distribution — ca-HepTh")
    ax1.grid(True, alpha=0.3)

    # (b) Eigenvector heatmap
    X = np.arange(num_nodes)
    Y = np.arange(k)
    im = ax2.contourf(X, Y, mag, levels=50, cmap=cmap)

    n_xticks = min(10, num_nodes)
    xtick_pos = np.linspace(0, num_nodes - 1, n_xticks, dtype=int)
    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(xtick_pos)

    n_yticks = min(10, k)
    ytick_pos = np.linspace(0, k - 1, n_yticks, dtype=int)
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels(ytick_pos + 1)

    ax2.set_xlabel("Node index (sorted by degree)")
    ax2.set_ylabel("Eigenvector index")
    ax2.set_title("(b) Eigenvector localisation — ca-HepTh")
    fig.colorbar(im, ax=ax2, label=r"$|\psi_i(v)|$", shrink=0.8)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    ca_hepth_localization()
    plt.show()
