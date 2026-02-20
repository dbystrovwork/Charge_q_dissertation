import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from spectral_vis import plot_eigenvalues_vs_q

from networks.simple_cycles import directed_cycle
from networks.dsbm import dsbm_cycle

# === Parameters ===
N = 7
K = 8
N_PER_CLASS = 50
P, S, R = 0.2, 0.2, 0.05
SEED = 42

# === Generate graph (uncomment one) ===
edges = directed_cycle(N); num_nodes = N; k = 3
# edges, _ = dsbm_cycle(K, N_PER_CLASS, P, S, R, seed=SEED); num_nodes = K * N_PER_CLASS; k = K

# === Compute eigenvalues over q ===
q_values = np.linspace(0, 0.5, 50)
all_eigenvalues = []

for q in q_values:
    eigenvalues, _ = magnetic_laplacian_eig(edges, num_nodes, q, k=k, normalized=True)
    all_eigenvalues.append(np.sort(eigenvalues.real))

all_eigenvalues = np.array(all_eigenvalues)

# === Plot ===
plot_eigenvalues_vs_q(q_values, all_eigenvalues)
plt.show()