import json
from pathlib import Path
import numpy as np

from .small_world import directed_small_world
from .barabasi_albert import barabasi_albert, directed_barabasi_albert
from .configuration_model import configuration_model
from .cora_ml import load_cora_ml
from .citeseer import load_citeseer
from .c_elegans import load_c_elegans
from .food_web import load_food_web


def dsbm_cycle_general(k, n_per_class, p, fwd, bwd, r, seed=None):
    """
    General Directed SBM with per-block forward/backward edge probabilities.

    Args:
        k: Number of communities
        n_per_class: Nodes per community
        p: Within-community edge probability
        fwd: List of length k. fwd[i] = prob of edges from block i to block (i+1) % k
        bwd: List of length k. bwd[i] = prob of edges from block i to block (i-1) % k
        r: Random noise edge probability

    Returns:
        edges: List of (i, j) tuples
        labels: Array of community labels
    """
    rng = np.random.default_rng(seed)
    num_nodes = k * n_per_class
    edges = []

    labels = np.repeat(np.arange(k), n_per_class)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue

            ci, cj = labels[i], labels[j]

            if ci == cj:
                prob = p
            elif cj == (ci + 1) % k:
                prob = fwd[ci]
            elif cj == (ci - 1) % k:
                prob = bwd[ci]
            else:
                prob = r

            if rng.random() < prob:
                edges.append((i, j))

    return edges, labels


def dsbm_cycle(k, n_per_class, p, s, r, seed=None):
    """
    Directed SBM with uniform forward probability, backward = noise.

    Wrapper around dsbm_cycle_general.
    """
    return dsbm_cycle_general(
        k, n_per_class, p,
        fwd=[s] * k, bwd=[r] * k,
        r=r, seed=seed
    )


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
    "dsbm_cycle_general": dsbm_cycle_general,
    "nested_dsbm_cycle": nested_dsbm_cycle,
    "directed_small_world": directed_small_world,
    "directed_erdos_renyi": directed_erdos_renyi,
    "barabasi_albert": barabasi_albert,
    "directed_barabasi_albert": directed_barabasi_albert,
    "configuration_model": configuration_model,
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
        graph_type: A generator name ("dsbm_cycle", "dsbm_cycle_general",
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
    else:
        num_nodes = config["k"] * config["n_per_class"]

    return edges, labels, num_nodes
