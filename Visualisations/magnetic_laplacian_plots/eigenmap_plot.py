import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from networks.dsbm import generate_graph


def plot_eigenmap(graph_type, q, seed=42, ax=None):
    edges, labels, num_nodes = generate_graph(graph_type, seed=seed)
    eigenvalues, eigenvectors = magnetic_laplacian_eig(edges, num_nodes, q, k=2, normalized=False)

    phase_0 = np.angle(eigenvectors[:, 0]) + np.pi
    phase_1 = np.angle(eigenvectors[:, 1]) + np.pi

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    unique_labels = np.unique(labels)
    cmap = plt.cm.tab10
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(phase_0[mask] / np.pi, phase_1[mask] / np.pi,
                   s=50, color=cmap(i), label=f"Class {lbl}")

    ax.set_xlabel(r"arg($\Phi_0$)")
    ax.set_ylabel(r"arg($\Phi_1$)")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_aspect("equal")
    ax.set_title(f"Eigenmap — {graph_type}, q={q}")

    return ax


if __name__ == "__main__":
    plot_eigenmap("dsbm_cycle", q=0.2)
    plt.tight_layout()
    plt.show()
