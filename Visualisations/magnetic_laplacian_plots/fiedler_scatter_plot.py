import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from networks.dsbm import generate_graph


def plot_eigenvector_scatter(graph_type, q, eigenvector_index=1, seed=42):
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)
    eigenvalues, eigenvectors = magnetic_laplacian_eig(edges, num_nodes, q, k=eigenvector_index + 1, normalized=False)

    vec = eigenvectors[:, eigenvector_index]

    fig, ax = plt.subplots(figsize=(6, 6))

    unique_labels = np.unique(labels)
    cmap = plt.cm.tab10
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(vec[mask].real, vec[mask].imag,
                   s=30, alpha=0.6, color=cmap(i), label=f"Class {lbl}")

    ax.set_xlabel(r"Re($\Phi_0$)")
    ax.set_ylabel(r"Im($\Phi_0$)")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{graph_type}, q={q}, eigenvector {eigenvector_index}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_eigenvector_scatter("dsbm_cycle", q=0.2, eigenvector_index=0)
