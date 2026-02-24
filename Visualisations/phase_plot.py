import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from networks.dsbm import generate_graph
from networks.cora_ml import load_cora_ml
from networks.citeseer import load_citeseer

# === Parameters ===
SEED = 42

# === Select graph (labelled only) ===
# Options: "dsbm_cycle", "dsbm_cycle_general", "nested_dsbm_cycle",
#          "directed_small_world", "cora_ml", "citeseer"
GRAPH = "nested_dsbm_cycle"
Q = 0.3
EIGENVECTOR = 1  # 1-indexed

if GRAPH in ("dsbm_cycle", "dsbm_cycle_general", "nested_dsbm_cycle", "directed_small_world"):
    edges, labels, num_nodes = generate_graph(GRAPH, seed=SEED)
elif GRAPH == "cora_ml":
    edges, labels, num_nodes = load_cora_ml()
elif GRAPH == "citeseer":
    edges, labels, num_nodes = load_citeseer()
else:
    raise ValueError(f"Unknown or unlabelled graph: {GRAPH}")


if GRAPH == "nested_dsbm_cycle":
    labels = labels%5

K = len(np.unique(labels))

# === Compute eigenvectors ===
eigenvalues, eigenvectors = magnetic_laplacian_eig(
    edges, num_nodes, Q, k=K, normalized=True
)

# Sort by eigenvalue so eigenvector indices are consistent
order = np.argsort(eigenvalues.real)
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

vec = eigenvectors[:, EIGENVECTOR - 1]
phases = np.angle(vec)

# === Plot ===
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

unique_labels = np.unique(labels)
cmap = plt.cm.get_cmap('tab10', len(unique_labels))

for i, lab in enumerate(unique_labels):
    mask = labels == lab
    ax.scatter(
        phases[mask],
        0.95 * np.ones(mask.sum()),
        c=[cmap(i)],
        label=f'Class {lab}',
        s=15,
        alpha=0.7,
    )

ax.set_ylim(0, 1)
ax.set_yticks([])
ax.spines['polar'].set_visible(False)
ax.set_title(
    f'{GRAPH}  |  q = {Q}  |  eigenvector {EIGENVECTOR}'
    f'  (λ = {eigenvalues[EIGENVECTOR - 1].real:.4f})',
    pad=20,
)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.tight_layout()
plt.show()
