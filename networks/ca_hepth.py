from pathlib import Path


def load_ca_hepth(root=None):
    """
    Load the ca-HepTh collaboration network as an undirected graph.

    9877 authors with edges representing co-authorship on papers
    in the HEP-TH section of arXiv (Jan 1993 – Apr 2003).

    Source: SNAP, https://snap.stanford.edu/data/ca-HepTh.html

    Returns:
        edges: List of (i, j) tuples representing co-authorship links
        labels: None (no ground-truth communities)
        num_nodes: Total number of authors
    """
    if root is None:
        root = Path(__file__).parent / '_data'
    root = Path(root)

    edges_file = root / 'ca-HepTh.txt'

    if not edges_file.exists():
        raise FileNotFoundError(
            f"ca-HepTh dataset not found at {edges_file}. "
            "Download from https://snap.stanford.edu/data/ca-HepTh.txt.gz"
        )

    raw_edges = []
    with open(edges_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
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
    edges, labels, num_nodes = load_ca_hepth()
    print(f"ca-HepTh: {num_nodes} nodes, {len(edges)} edges")
