from sklearn.cluster import KMeans
import numpy as np


def spectral_clustering(eigvecs, num_clusters):
    """
    Cluster nodes using eigenvectors of the magnetic Laplacian.

    Args:
        eigvecs: Complex eigenvector matrix (num_nodes, k)
        num_clusters: Number of clusters

    Returns:
        labels: Cluster assignment for each node
    """
    # Stack real and imaginary parts
    features = np.hstack([eigvecs.real, eigvecs.imag])

    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    return kmeans.fit_predict(features)
