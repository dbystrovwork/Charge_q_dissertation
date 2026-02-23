import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path


def load_citeseer(root=None):
    """
    Load the CiteSeer citation network as a directed graph.

    4230 scientific publications classified into 6 classes.
    Edges represent directed citations (paper i cites paper j).

    Source: Bojchevski & Günnemann (2018), via torch_geometric download.

    Returns:
        edges: List of (i, j) tuples representing directed citation links
        labels: numpy array of node class labels
        num_nodes: Total number of nodes
    """
    if root is None:
        root = Path(__file__).parent / '_data'
    root = Path(root)

    npz_path = root / 'CiteSeer' / 'raw' / 'citeseer.npz'

    if not npz_path.exists():
        from torch_geometric.datasets import CitationFull
        CitationFull(root=str(root), name='CiteSeer')

    data = np.load(str(npz_path), allow_pickle=True)
    adj = csr_matrix(
        (data['adj_data'], data['adj_indices'], data['adj_indptr']),
        shape=data['adj_shape']
    )

    rows, cols = adj.nonzero()
    edges = list(zip(rows.tolist(), cols.tolist()))
    labels = data['labels']
    num_nodes = adj.shape[0]

    return edges, labels, num_nodes


if __name__ == "__main__":
    edges, labels, num_nodes = load_citeseer()
    print(f"CiteSeer: {num_nodes} nodes, {len(edges)} directed edges, "
          f"{len(np.unique(labels))} classes")
