import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from magnetic_laplacian.network_generation import dsbm_cycle
from magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from magnetic_laplacian.spectral_clustering import spectral_clustering

# Parameters
SEED = 42
K = 5        # Number of communities
N_PER_CLASS = 50   # Nodes per community
P = 0.2           # Within-community prob
S = 0.2           # Forward edge prob
R = 0.05           # Noise prob

# Generate graph once
edges, true_labels = dsbm_cycle(K, N_PER_CLASS, P, S, R, seed=SEED)
num_nodes = K * N_PER_CLASS

# Sweep over q
q_values = np.linspace(0, 0.5, 50)
all_gaps = []  # Shape will be (len(q_values), K-1)
accuracies = []

for q in q_values:
    eigenvalues, eigenvectors = magnetic_laplacian_eig(edges, num_nodes, q, k=K, normalized=True)

    # All spectral gaps between consecutive eigenvalues
    sorted_eigs = np.sort(eigenvalues.real)
    gaps = np.diff(sorted_eigs)
    all_gaps.append(gaps)

    # Clustering accuracy
    pred_labels = spectral_clustering(eigenvectors, K)
    ari = adjusted_rand_score(true_labels, pred_labels)
    accuracies.append(ari)

all_gaps = np.array(all_gaps)  # (num_q, K-1)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i in range(K - 1):
    axes[0].plot(q_values, all_gaps[:, i], label=f'λ{i+2} - λ{i+1}')
axes[0].set_xlabel('q')
axes[0].set_ylabel('Spectral Gap')
axes[0].set_title('Spectral Gaps vs q')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(q_values, accuracies)
axes[1].set_xlabel('q')
axes[1].set_ylabel('ARI')
axes[1].set_title('Clustering Accuracy vs q')
axes[1].grid(True)

plt.tight_layout()
plt.show()
