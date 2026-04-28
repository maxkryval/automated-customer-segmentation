
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def save_segment_size_plot(clustered_df: pd.DataFrame, path: Path):
    counts = clustered_df["cluster"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Segment Size Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def save_pca_cluster_plot(X, labels, path: Path, random_state: int, max_points: int = 5000):
    if len(X) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X_plot = X[idx]
        labels_plot = labels[idx]
    else:
        X_plot = X
        labels_plot = labels
    reduced = PCA(n_components=2, random_state=random_state).fit_transform(X_plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_plot, s=8, alpha=0.7)
    plt.title("PCA Cluster Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()
