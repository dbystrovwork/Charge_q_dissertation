import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

plt.style.use("seaborn-v0_8-paper")

from networks.dsbm import generate_graph
from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig


def dag_layout(num_layers, n_per_layer):
    """
    Custom layout for the layered DAG.

    Layer 0 (top row) -> layer 1 -> ... -> layer num_layers-1 (bottom).
    Within each layer, nodes are spread horizontally and centred.

    Returns:
        pos: dict mapping node index -> (x, y)
    """
    pos = {}
    h_spacing = 1.0
    v_spacing = 1.5

    for layer in range(num_layers):
        y = -layer * v_spacing
        x_offset = -(n_per_layer - 1) * h_spacing / 2
        for j in range(n_per_layer):
            node = layer * n_per_layer + j
            pos[node] = (x_offset + j * h_spacing, y)

    return pos


def plot_dag_fiedler(num_layers=5, n_per_layer=40, p=0.05, q=0.25,
                     eig_index=1, ax=None, title=None, seed=42, mag_vmax=None,
                     colorbar=True):
    """
    Fiedler-style eigenvector plot for the random DAG with
    a layered top-to-bottom layout.

    Node colour = eigenvector magnitude.
    Phase annotation on each node.
    """
    edges, labels, num_nodes = generate_graph(
        "random_dag", seed=seed, num_layers=num_layers,
        n_per_layer=n_per_layer, p=p
    )

    _, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=eig_index + 1, normalized=True
    )
    vec = eigenvectors[:, eig_index]
    magnitudes = np.abs(vec)
    phases = np.angle(vec)

    if mag_vmax is None:
        mag_vmax = magnitudes.max()
    mag_norm = magnitudes / (mag_vmax + 1e-10)

    # Layout
    pos = dag_layout(num_layers, n_per_layer)

    # Colourmap for magnitude
    mag_cmap = LinearSegmentedColormap.from_list(
        "eigenvector", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    node_colors = [mag_cmap(m) for m in mag_norm]

    # Node sizes
    min_size, max_size = 80, 350
    node_sizes = min_size + (max_size - min_size) * mag_norm

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    # Add margin so nodes at the boundary aren't clipped
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    margin = 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # Draw edges
    for i, j in edges:
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.annotate(
            "",
            xy=(xj, yj), xytext=(xi, yi),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#666666",
                lw=0.6,
                alpha=0.3,
                shrinkA=4, shrinkB=4,
            ),
        )

    # Draw nodes
    for node_id in range(num_nodes):
        x, y = pos[node_id]
        ax.scatter(
            x, y,
            s=node_sizes[node_id],
            c=[node_colors[node_id]],
            edgecolors="#333333",
            linewidths=1.0,
            zorder=3,
        )

    # Phase annotations (plain black text)
    for node_id in range(num_nodes):
        x, y = pos[node_id]
        phase = phases[node_id]
        ax.annotate(
            rf"${phase / np.pi:+.2f}\pi$",
            xy=(x, y),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=5,
            color="black",
        )

    # Magnitude colorbar
    if colorbar:
        sm_mag = ScalarMappable(
            cmap=mag_cmap,
            norm=Normalize(vmin=0, vmax=mag_vmax),
        )
        sm_mag.set_array([])
        fig = ax.get_figure()
        cbar_mag = fig.colorbar(sm_mag, ax=ax, shrink=0.5, aspect=20, pad=0.02)
        cbar_mag.set_label(rf"$|\psi_{{{eig_index}}}(v)|$", fontsize=11)

    if title is None:
        title = (
            rf"Random DAG — $q={q:.2f}$, eigenvector {eig_index}, "
            rf"layers$={num_layers}$, $n/\mathrm{{layer}}={n_per_layer}$, $p={p}$"
        )
    ax.set_title(title, fontsize=13, pad=10)

    return ax


num_layerss = [3]
n_per_layers = [3]
ps = [1]
qs = [0.4]

if __name__ == "__main__":
    n = len(num_layerss)
    eig_index = 0

    # Pre-compute global max magnitude across all panels
    global_max = 0
    for i in range(n):
        edges, _, num_nodes = generate_graph(
            "random_dag", seed=42, num_layers=num_layerss[i],
            n_per_layer=n_per_layers[i], p=ps[i]
        )
        _, vecs = magnetic_laplacian_eig(edges, num_nodes, qs[i], k=eig_index + 1, normalized=True)
        global_max = max(global_max, np.abs(vecs[:, eig_index]).max())

    fig, axes = plt.subplots(1, n, figsize=(12 * n, 8), squeeze=False)
    for i in range(n):
        plot_dag_fiedler(
            num_layers=num_layerss[i], n_per_layer=n_per_layers[i],
            p=ps[i], q=qs[i], eig_index=eig_index, ax=axes[0, i],
            mag_vmax=global_max, colorbar=False
        )

    # Single shared colorbar
    mag_cmap = LinearSegmentedColormap.from_list(
        "eigenvector", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    sm = ScalarMappable(cmap=mag_cmap, norm=Normalize(vmin=0, vmax=global_max))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0, :].tolist(), shrink=0.5, aspect=20, pad=0.02,
                 label=rf"$|\psi_{{{eig_index}}}(v)|$")
    plt.tight_layout()
    plt.show()
