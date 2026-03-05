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


def two_cycles_layout(k1, k2, p):
    """
    Custom layout for the two-cycles graph.

    C1 (nodes [0, k1)): unit circle on the left, with the junction
    node k1-1 at theta=0 (rightmost point, facing the path).

    C2 (nodes [k1, k1+k2)): unit circle on the right, with the
    junction node k1 at theta=pi (leftmost point, facing the path).

    Path nodes are evenly spaced along y=0 between the two junction
    nodes.

    Returns:
        pos: dict mapping node index -> (x, y)
    """
    pos = {}
    radius = 1.0
    gap = max(p * 0.6, 1.0)  # space between circle edges for the path

    # C1 centred at x=0. Junction node k1-1 at theta=0 (x=radius).
    # Cycle traversed CCW: node i at theta = (k1-1-i) * 2pi/k1
    for i in range(k1):
        theta = (k1 - 1 - i) * 2 * np.pi / k1
        pos[i] = (radius * np.cos(theta), radius * np.sin(theta))

    # C2 centred at x = 2*radius + gap. Junction node k1 at theta=pi
    # (leftmost point). Cycle traversed CCW: node k1+i at
    # theta = pi - i * 2pi/k2
    cx2 = 2 * radius + gap
    for i in range(k2):
        theta = np.pi - i * 2 * np.pi / k2
        pos[k1 + i] = (cx2 + radius * np.cos(theta), radius * np.sin(theta))

    # Path nodes between the two junction nodes along y=0
    # Junction of C1: pos[k1-1] = (radius, 0)
    # Junction of C2: pos[k1]   = (cx2 - radius, 0)
    x_left = radius
    x_right = cx2 - radius
    path_start = k1 + k2
    num_path_nodes = p - 1  # p edges means p-1 interior nodes
    for j in range(num_path_nodes):
        t = (j + 1) / p
        pos[path_start + j] = (x_left + t * (x_right - x_left), 0.0)

    return pos


def plot_two_cycles_fiedler(k1=7, k2=11, p=15, q=0.25, eig_index=1,
                            ax=None, title=None, seed=42):
    """
    Fiedler-style eigenvector plot for the two-cycles graph with
    a custom layout: two circles connected by a horizontal path.

    Node colour = eigenvector magnitude.
    Phase annotation on each node.
    """
    edges, labels, num_nodes = generate_graph(
        "two_cycles", seed=seed, k1=k1, k2=k2, p=p
    )

    _, eigenvectors = magnetic_laplacian_eig(
        edges, num_nodes, q, k=eig_index + 1, normalized=True
    )
    vec = eigenvectors[:, eig_index]
    magnitudes = np.abs(vec)
    phases = np.angle(vec)

    mag_norm = (magnitudes - magnitudes.min()) / (
        magnitudes.max() - magnitudes.min() + 1e-10
    )

    # Layout
    pos = two_cycles_layout(k1, k2, p)

    # Colourmap for magnitude
    mag_cmap = LinearSegmentedColormap.from_list(
        "eigenvector", ["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    node_colors = [mag_cmap(m) for m in mag_norm]

    # Node sizes
    min_size, max_size = 80, 350
    node_sizes = min_size + (max_size - min_size) * mag_norm

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))
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
    sm_mag = ScalarMappable(
        cmap=mag_cmap,
        norm=Normalize(vmin=magnitudes.min(), vmax=magnitudes.max()),
    )
    sm_mag.set_array([])
    fig = ax.get_figure()
    cbar_mag = fig.colorbar(sm_mag, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar_mag.set_label(rf"$|\psi_{{{eig_index}}}(v)|$", fontsize=11)

    if title is None:
        title = (
            rf"Two cycles — $q={q:.2f}$, eigenvector {eig_index}, "
            rf"$k_1={k1}$, $k_2={k2}$, $p={p}$"
        )
    ax.set_title(title, fontsize=13, pad=10)

    return ax


k1s = [6, 6]
k2s = [10, 10]
ps = [5, 5]
qs = [1/6, 1/10]

if __name__ == "__main__":
    n = len(k1s)
    fig, axes = plt.subplots(1, n, figsize=(14 * n, 6), squeeze=False)
    for i in range(n):
        plot_two_cycles_fiedler(k1=k1s[i], k2=k2s[i], p=ps[i], q=qs[i],
                                eig_index=1, ax=axes[0, i])
    plt.tight_layout()
    plt.show()
