import numpy as np


def fisher_discriminant_ratio(eigenvectors, labels):
    """
    Compute Fisher Discriminant Ratio for each eigenvector.

    FDR = between-class variance / within-class variance

    Args:
        eigenvectors: (n_samples, n_components) array
        labels: (n_samples,) class labels

    Returns:
        fdr: (n_components,) FDR for each eigenvector
    """
    n_samples, n_components = eigenvectors.shape
    classes = np.unique(labels)

    global_mean = eigenvectors.mean(axis=0)

    between_class = np.zeros(n_components)
    within_class = np.zeros(n_components)

    for c in classes:
        mask = labels == c
        n_c = mask.sum()
        class_data = eigenvectors[mask]
        class_mean = class_data.mean(axis=0)

        # Between-class: n_k * (μ_k - μ)²
        between_class += n_c * (class_mean - global_mean) ** 2

        # Within-class: Σ (x - μ_k)²
        within_class += ((class_data - class_mean) ** 2).sum(axis=0)

    # Avoid division by zero
    within_class = np.maximum(within_class, 1e-10)

    return between_class / within_class
