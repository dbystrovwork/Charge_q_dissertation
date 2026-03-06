import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from Visualisations.graph_plots.dsbm_cycle_plot import plot_dsbm_cycle
from Visualisations.magnetic_laplacian_plots.fiedler_scatter_plot import plot_eigenvector_scatter
from Visualisations.magnetic_laplacian_plots.eigenmap_plot import plot_eigenmap


def dsbm_cycle_combined(q=0.2, eigenvector_index=0, seed=42):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_dsbm_cycle(seed=seed, ax=axes[0])
    plot_eigenvector_scatter("dsbm_cycle", q, eigenvector_index=eigenvector_index, seed=seed, ax=axes[1])
    plot_eigenmap("dsbm_cycle", q, seed=seed, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dsbm_cycle_combined(q=0.2, eigenvector_index=0)
