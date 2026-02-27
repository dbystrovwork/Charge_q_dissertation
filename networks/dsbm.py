import json
from pathlib import Path
import numpy as np

from .small_world import directed_small_world
from .barabasi_albert import barabasi_albert, directed_barabasi_albert
from .configuration_model import configuration_model
from .sbm import sbm
from .cora_ml import load_cora_ml
from .citeseer import load_citeseer
from .c_elegans import load_c_elegans
from .food_web import load_food_web


def dsbm_cycle(k, n_per_class, p, eta, seed=None):
    """
    Directed Stochastic Block Model with cyclic structure.

    Each pair of nodes (i, j) is connected with probability p.
    The direction is determined by the matrix F of size KxK where:
      - F(a, b) + F(b, a) = 1
      - F(a, b) = 1 - eta  if (a + 1) mod K == b
      - F(a, b) = 0.5      otherwise

    For a pair of nodes (i, j) in communities a, b:
      - With probability p * F(a, b), edge (i -> j) is added
      - With probability p * F(b, a), edge (j -> i) is added

    Args:
        k: Number of communities
        n_per_class: Nodes per community
        p: Edge probability between any pair of nodes
        eta: Asymmetry parameter (0 = fully forward, 0.5 = undirected, 1 = fully backward)

    Returns:
        edges: List of (i, j) tuples
        labels: Array of community labels
    """
    rng = np.random.default_rng(seed)
    num_nodes = k * n_per_class
    edges = []

    labels = np.repeat(np.arange(k), n_per_class)

    # Build F matrix
    F = 0.5 * np.ones((k, k))
    for a in range(k):
        b = (a + 1) % k
        F[a, b] = 1 - eta
        F[b, a] = eta

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < p:
                ci, cj = labels[i], labels[j]
                # Edge exists; determine direction(s) using F
                if rng.random() < F[ci, cj]:
                    edges.append((i, j))
                else:
                    edges.append((j, i))

    return edges, labels


def nested_dsbm_cycle(c1, c2, n_per_block, p, s_inner, s_outer, r, seed=None):
    """
    Generate a nested Directed SBM: outer cycle of c1 blocks, each containing an inner cycle of c2 sub-blocks.

    Args:
        c1: Number of outer blocks (arranged in cycle)
        c2: Number of sub-blocks per outer block (arranged in inner cycle)
        n_per_block: Nodes per sub-block
        p: Within sub-block edge probability
        s_inner: Forward edge prob between adjacent sub-blocks (inner cycle)
        s_outer: Forward edge prob between adjacent outer blocks
        r: Random noise edge probability

    Returns:
        edges: List of (i, j) tuples
        labels: Array of block labels (0 to c1*c2 - 1)
    """
    rng = np.random.default_rng(seed)
    num_nodes = c1 * c2 * n_per_block
    edges = []

    # Assign block labels: block (i, j) -> label i*c2 + j
    labels = np.repeat(np.arange(c1 * c2), n_per_block)
    outer_labels = labels // c2
    inner_labels = labels % c2

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue

            oi, oj = outer_labels[i], outer_labels[j]
            ii, ij = inner_labels[i], inner_labels[j]

            if oi == oj and ii == ij:
                # Same sub-block
                prob = p
            elif oi == oj and ij == (ii + 1) % c2:
                # Same outer block, forward in inner cycle
                prob = s_inner
            elif oj == (oi + 1) % c1:
                # Forward in outer cycle
                prob = s_outer
            else:
                prob = r

            if rng.random() < prob:
                edges.append((i, j))

    return edges, labels


def directed_erdos_renyi(n, p, seed=None):
    """
    Generate a directed Erdos-Renyi graph.

    Each directed edge (i, j) with i != j is included independently
    with probability p.

    Args:
        n: Number of nodes
        p: Probability of each directed edge

    Returns:
        edges: List of (i, j) tuples
        labels: None (no community structure)
    """
    rng = np.random.default_rng(seed)
    edges = []

    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                edges.append((i, j))

    return edges, None


CONFIG_PATH = Path(__file__).parent / "graph_config.json"

_GENERATORS = {
    "dsbm_cycle": dsbm_cycle,
    # "dsbm_cycle_general": dsbm_cycle_general,  # TODO: restore when reimplemented
    "nested_dsbm_cycle": nested_dsbm_cycle,
    "directed_small_world": directed_small_world,
    "directed_erdos_renyi": directed_erdos_renyi,
    "barabasi_albert": barabasi_albert,
    "directed_barabasi_albert": directed_barabasi_albert,
    "configuration_model": configuration_model,
    "sbm": sbm,
}

_LOADERS = {
    "cora_ml": load_cora_ml,
    "citeseer": load_citeseer,
    "c_elegans": load_c_elegans,
    "food_web": load_food_web,
}


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def generate_graph(graph_type, seed=None, **overrides):
    """
    Generate or load a graph by type.

    Args:
        graph_type: A generator name ("dsbm_cycle",
            "nested_dsbm_cycle", "directed_small_world") or a real-world
            dataset name ("cora_ml", "citeseer", "c_elegans", "food_web").
        seed: Random seed (only used for generators, ignored for datasets)
        **overrides: Override any config parameter (generators only)

    Returns:
        edges, labels, num_nodes
        (labels is None for datasets without ground-truth communities)
    """
    if graph_type in _LOADERS:
        return _LOADERS[graph_type]()

    config = load_config()[graph_type]
    config.update(overrides)
    config["seed"] = seed

    edges, labels = _GENERATORS[graph_type](**config)

    if graph_type == "nested_dsbm_cycle":
        num_nodes = config["c1"] * config["c2"] * config["n_per_block"]
    elif graph_type in ("directed_small_world", "directed_erdos_renyi",
                         "barabasi_albert", "directed_barabasi_albert",
                         "configuration_model"):
        num_nodes = config["n"]
    elif graph_type == "sbm":
        num_nodes = config["k"] * config["n_per_class"]
    else:
        num_nodes = config["k"] * config["n_per_class"]

    return edges, labels, num_nodes
