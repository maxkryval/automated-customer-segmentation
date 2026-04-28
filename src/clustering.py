import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def sample_for_metrics(X, labels, max_sample_size: int, random_state: int):
    if len(X) <= max_sample_size:
        return X, labels
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_sample_size, replace=False)
    return X[idx], labels[idx]


def calculate_clustering_metrics(X, labels, max_metric_sample_size: int, random_state: int):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return None
    X_eval, labels_eval = sample_for_metrics(X, labels, max_metric_sample_size, random_state)
    if len(np.unique(labels_eval)) < 2 or len(np.unique(labels_eval)) >= len(labels_eval):
        return None
    return {
        "silhouette": float(silhouette_score(X_eval, labels_eval)),
        "davies_bouldin": float(davies_bouldin_score(X_eval, labels_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(X_eval, labels_eval)),
    }


def combined_score(metrics: dict) -> float:
    return float(
        0.45 * metrics["silhouette"]
        + 0.30 * (1 / (1 + metrics["davies_bouldin"]))
        + 0.25 * np.log1p(metrics["calinski_harabasz"])
    )


def run_multiple_clustering_algorithms(
    X,
    cluster_range: range,
    max_metric_sample_size: int,
    random_state: int,
):
    candidates = []

    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        metrics = calculate_clustering_metrics(X, labels, max_metric_sample_size, random_state)
        if metrics:
            candidates.append({
                "algorithm": "KMeans",
                "k": int(k),
                "labels": labels,
                "metrics": metrics,
                "combined_score": combined_score(metrics),
            })

    for k in cluster_range:
        model = GaussianMixture(n_components=k, random_state=random_state)
        labels = model.fit_predict(X)
        metrics = calculate_clustering_metrics(X, labels, max_metric_sample_size, random_state)
        if metrics:
            candidates.append({
                "algorithm": "GMM",
                "k": int(k),
                "labels": labels,
                "metrics": metrics,
                "combined_score": combined_score(metrics),
            })

    if not candidates:
        raise ValueError("No valid clustering candidates produced.")

    best = max(candidates, key=lambda item: item["metrics"]["silhouette"])
    metrics_df = pd.DataFrame([
        {
            "algorithm": c["algorithm"],
            "k": c["k"],
            "silhouette": c["metrics"]["silhouette"],
            "davies_bouldin": c["metrics"]["davies_bouldin"],
            "calinski_harabasz": c["metrics"]["calinski_harabasz"],
            "combined_score": c["combined_score"],
        }
        for c in candidates
    ]).sort_values("silhouette", ascending=False).reset_index(drop=True)

    return best, metrics_df
