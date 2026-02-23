import numpy as np
import requests
import tarfile
import io
from pathlib import Path


def load_food_web(root=None):
    """
    Load the Florida Bay food web (wet season) as a directed graph.

    128 species/functional groups with directed edges representing
    trophic energy flow (predator-prey relationships).

    Source: Ulanowicz et al., via KONECT project.
    http://konect.cc/networks/foodweb-baywet/

    Returns:
        edges: List of (i, j) tuples representing directed trophic links
        labels: None (no standard community labels)
        num_nodes: Total number of species/groups
    """
    if root is None:
        root = Path(__file__).parent / '_data' / 'food_web'
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    edges_file = root / 'edges.tsv'

    if not edges_file.exists():
        url = 'http://konect.cc/files/download.tsv.foodweb-baywet.tar.bz2'
        print(f"Downloading Florida Bay food web from KONECT...")
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()

        with tarfile.open(fileobj=io.BytesIO(r.content), mode='r:bz2') as tar:
            for member in tar.getmembers():
                if 'out.' in member.name:
                    f = tar.extractfile(member)
                    edges_file.write_bytes(f.read())
                    break

    # Parse KONECT TSV edge list
    raw_edges = []
    with open(edges_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('%') or not line:
                continue
            parts = line.split()
            raw_edges.append((int(parts[0]), int(parts[1])))

    # Re-index to contiguous 0-based IDs
    nodes = sorted(set(n for e in raw_edges for n in e))
    mapping = {n: i for i, n in enumerate(nodes)}
    edges = [(mapping[u], mapping[v]) for u, v in raw_edges]
    num_nodes = len(nodes)

    return edges, None, num_nodes


if __name__ == "__main__":
    edges, labels, num_nodes = load_food_web()
    print(f"Florida Bay food web: {num_nodes} nodes, {len(edges)} directed edges")
