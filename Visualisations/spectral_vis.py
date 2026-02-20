import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")


def plot_eigenvalues_vs_q(q_values, eigenvalues, title="Eigenvalues vs q"):
    """
    Plot eigenvalues as a function of q.

    Args:
        q_values: Array of q values
        eigenvalues: Array of shape (len(q_values), k) - eigenvalues for each q
        title: Plot title
    """
    eigenvalues = np.array(eigenvalues)
    k = eigenvalues.shape[1]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(k):
        ax.plot(q_values, eigenvalues[:, i], label=f'λ{i+1}')

    ax.set_xlabel('q')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig, ax