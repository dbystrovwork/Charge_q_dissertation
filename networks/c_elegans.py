import numpy as np
import requests
import zipfile
import io
import re
from pathlib import Path


def load_c_elegans(root=None):
    """
    Load the C. elegans neural connectome as a directed graph.

    297 neurons connected by directed chemical synapses.

    Source: Watts & Strogatz (1998), from Mark Newman's network data.
    http://www-personal.umich.edu/~mejn/netdata/

    Returns:
        edges: List of (i, j) tuples representing directed synaptic connections
        labels: None (no standard community labels)
        num_nodes: Total number of neurons
    """
    if root is None:
        root = Path(__file__).parent / '_data' / 'c_elegans'
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    gml_path = root / 'celegansneural.gml'

    if not gml_path.exists():
        url = 'http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip'
        print(f"Downloading C. elegans connectome from {url}...")
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(root)

    gml_data = gml_path.read_text()

    # Parse edges from GML (handles duplicate edges in the file)
    raw_edges = set()
    for match in re.finditer(
        r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)', gml_data
    ):
        src, tgt = int(match.group(1)), int(match.group(2))
        raw_edges.add((src, tgt))

    # Re-index to contiguous 0-based IDs
    nodes = sorted(set(n for e in raw_edges for n in e))
    mapping = {n: i for i, n in enumerate(nodes)}
    edges = [(mapping[u], mapping[v]) for u, v in raw_edges]
    num_nodes = len(nodes)

    return edges, None, num_nodes


if __name__ == "__main__":
    edges, labels, num_nodes = load_c_elegans()
    print(f"C. elegans: {num_nodes} nodes, {len(edges)} directed edges")
