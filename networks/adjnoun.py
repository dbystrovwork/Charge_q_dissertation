import numpy as np
import requests
import zipfile
import io
from pathlib import Path


def load_adjnoun(root=None):
    """
    Load the adjective-noun adjacency network from David Copperfield.

    112 words (adjectives and nouns) with edges representing
    adjacent word pairs in the novel.

    Source: M.E.J. Newman, via Netzschleuder.
    https://networks.skewed.de/net/adjnoun

    Returns:
        edges: List of (i, j) tuples representing word adjacencies
        labels: Array of word types (0 = adjective, 1 = noun)
        num_nodes: Total number of words
    """
    if root is None:
        root = Path(__file__).parent / '_data' / 'adjnoun'
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    edges_file = root / 'edges.csv'
    nodes_file = root / 'nodes.csv'

    if not edges_file.exists() or not nodes_file.exists():
        url = 'https://networks.skewed.de/net/adjnoun/files/adjnoun.csv.zip'
        print("Downloading adjnoun network from Netzschleuder...")
        r = requests.get(url)
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                (root / name).write_bytes(z.read(name))

    # Parse edge list
    edges = []
    with open(edges_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            edges.append((int(parts[0]), int(parts[1])))

    # Parse node labels (0 = adjective, 1 = noun)
    labels = {}
    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            labels[int(parts[0])] = int(parts[2])

    num_nodes = len(labels)
    label_array = np.array([labels[i] for i in range(num_nodes)])

    return edges, label_array, num_nodes


if __name__ == "__main__":
    edges, labels, num_nodes = load_adjnoun()
    n_adj = np.sum(labels == 0)
    n_noun = np.sum(labels == 1)
    print(f"Adjnoun: {num_nodes} nodes ({n_adj} adj, {n_noun} noun), {len(edges)} edges")
