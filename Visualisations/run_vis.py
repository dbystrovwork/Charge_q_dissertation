import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

from laplacians.magnetic_laplacian.mag_lap_ops import magnetic_laplacian_eig
from spectral_vis import plot_eigenvalues_vs_q

from networks.simple_cycles import directed_cycle, directed_cycle_flipped, nested_cycles, cycles_from_node
from networks.dsbm import dsbm_cycle, nested_dsbm_cycle, generate_graph
from networks.cora_ml import load_cora_ml
from networks.citeseer import load_citeseer
from networks.c_elegans import load_c_elegans
from networks.food_web import load_food_web

# === Parameters ===
SEED = 42

# === Select graph ===
# Options: "cycles", "dsbm_cycle", "dsbm_cycle_general", "nested_dsbm_cycle",
#          "directed_small_world", "cora_ml", "citeseer", "c_elegans", "food_web"
GRAPH = "food_web"

if GRAPH == "cycles":
    edges, num_nodes = cycles_from_node([3, 6]); k = 3
elif GRAPH in ("dsbm_cycle", "dsbm_cycle_general", "nested_dsbm_cycle", "directed_small_world"):
    edges, labels, num_nodes = generate_graph(GRAPH, seed=SEED)
    k = 3
elif GRAPH == "cora_ml":
    edges, labels, num_nodes = load_cora_ml(); k = len(np.unique(labels))
elif GRAPH == "citeseer":
    edges, labels, num_nodes = load_citeseer(); k = len(np.unique(labels))
elif GRAPH == "c_elegans":
    edges, _, num_nodes = load_c_elegans(); k = 6
elif GRAPH == "food_web":
    edges, _, num_nodes = load_food_web(); k = 6
else:
    raise ValueError(f"Unknown graph: {GRAPH}")

# === Compute eigenvalues over q ===
q_values = np.linspace(0, 0.5, 100)
all_eigenvalues = []

for q in q_values:
    eigenvalues, _ = magnetic_laplacian_eig(edges, num_nodes, q, k=k, normalized=True)
    all_eigenvalues.append(np.sort(eigenvalues.real))

all_eigenvalues = np.array(all_eigenvalues)

# === Plot ===
plot_eigenvalues_vs_q(q_values, all_eigenvalues)
plt.show()