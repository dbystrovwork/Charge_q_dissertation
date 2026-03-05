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


def cycle_tail_layout(k, p):
    """
    Custom layout for the cycle-tail graph.

    Cycle nodes [0, k) are placed on a unit circle with node k-1
    (the junction node) at theta=0. The cycle is traversed
    counter-clockwise: node 0 is one step CCW from k-1, etc.

    Tail nodes [k, k+p) extend along the positive real axis
    to the right of node k-1.

    Returns:
        pos: dict mapping node index -> (x, y)
    """
    pos = {}
    radius = 1.0
    tail_spacing = 2 * np.pi * radius / k  # match cycle edge length

    # Cycle: node k-1 at theta=0, node k-2 at theta=2pi/k, etc.
    # i.e. node i at theta = (k-1-i) * 2pi/k
    for i in range(k):
        theta = (k - 1 - i) * 2 * np.pi / k
        pos[i] = (radius * np.cos(theta), radius * np.sin(theta))

    # Tail: starting from node k-1 (at theta=0, x=radius),
    # extend rightward along y=0
    x_start = radius + tail_spacing
    for j in range(p):
        pos[k + j] = (x_start + j * tail_spacing, 0.0)

    return pos


def plot_cycle_tail_fiedler(k=10, p=5, q=0.25, eig_index=1,
                            ax=None, title=None, seed=42, mag_vmax=None,
                            colorbar=True):
    """
    Fiedler-style eigenvector plot for the cycle-tail graph with
    a custom layout: cycle as a circle, tail extending rightward.

    Node colour = eigenvector magnitude.
    Phase annotation on each node.
    """
    edges, labels, num_nodes = generate_graph(
        "cycle_tail", seed=seed, k=k, p=p
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
    pos = cycle_tail_layout(k, p)

    # Colourmap for magnitude
    mag_cmap = LinearSegmentedColormap.from_list(
        "eigenvector", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    node_colors = [mag_cmap(m) for m in mag_norm]

    # Node sizes
    min_size, max_size = 80, 350
    node_sizes = min_size + (max_size - min_size) * mag_norm

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Add margin so nodes at the boundary aren't clipped
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    margin = 0.3
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
                lw=1.2,
                connectionstyle="arc3,rad=0.08",
                shrinkA=6, shrinkB=6,
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
            fontsize=6,
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
        cbar_mag = fig.colorbar(sm_mag, ax=ax, orientation="horizontal",
                                shrink=0.5, aspect=30, pad=0.08)
        cbar_mag.set_label(rf"$|\psi_{{{eig_index}}}(v)|$", fontsize=11)

    if title is None:
        title = (
            rf"Cycle-tail — $q={q:.2f}$, eigenvector {eig_index}, "
            rf"$k={k}$, $p={p}$"
        )
    ax.set_title(title, fontsize=13, pad=10)

    return ax


ks = [6, 6]
ps = [5, 5]
qs = [1/4, 1/6]

if __name__ == "__main__":
    n = len(ks)
    eig_index = 0

    # Pre-compute global max magnitude across all panels
    global_max = 0
    for i in range(n):
        edges, _, num_nodes = generate_graph("cycle_tail", seed=42, k=ks[i], p=ps[i])
        _, vecs = magnetic_laplacian_eig(edges, num_nodes, qs[i], k=eig_index + 1, normalized=True)
        global_max = max(global_max, np.abs(vecs[:, eig_index]).max())

    fig, axes = plt.subplots(1, n, figsize=(10 * n, 3.5), squeeze=False)
    for i in range(n):
        plot_cycle_tail_fiedler(k=ks[i], p=ps[i], q=qs[i], eig_index=eig_index,
                                ax=axes[0, i], mag_vmax=global_max, colorbar=False)

    # Single shared colorbar underneath both plots
    mag_cmap = LinearSegmentedColormap.from_list(
        "eigenvector", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    sm = ScalarMappable(cmap=mag_cmap, norm=Normalize(vmin=0, vmax=global_max))
    sm.set_array([])

    fig.subplots_adjust(bottom=0.22)
    cbar_ax = fig.add_axes([0.25, 0.15, 0.5, 0.03])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                 label=rf"$|\psi_{{{eig_index}}}(v)|$")
    plt.show()
