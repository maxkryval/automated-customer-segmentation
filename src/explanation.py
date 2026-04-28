
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_segment_numeric_descriptions(clustered_df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    global_means = clustered_df[numeric_cols].mean(numeric_only=True)
    rows = []
    for cluster in sorted(clustered_df["cluster"].unique()):
        segment = clustered_df[clustered_df["cluster"] == cluster]
        segment_means = segment[numeric_cols].mean(numeric_only=True)
        for feature in numeric_cols:
            seg = segment_means[feature]
            glob = global_means[feature]
            diff = seg - glob
            rel = None if glob == 0 else diff / abs(glob)
            rows.append({
                "cluster": int(cluster),
                "feature": feature,
                "segment_mean": float(seg),
                "global_mean": float(glob),
                "absolute_difference": float(diff),
                "relative_difference": None if rel is None else float(rel),
            })
    return pd.DataFrame(rows)


def compute_shap_values_for_segments(X, labels, transformed_feature_names: list[str], shap_sample_size: int, random_state: int):
    import shap

    sample_size = min(len(X), shap_sample_size)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=sample_size, replace=False)
    X_sample = X[idx]
    y_sample = labels[idx]

    surrogate = RandomForestClassifier(n_estimators=120, max_depth=8, n_jobs=-1, random_state=random_state)
    surrogate.fit(X_sample, y_sample)
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X_sample)

    rows = []
    unique_clusters = sorted(np.unique(y_sample))

    if isinstance(shap_values, list):
        for class_index, cluster in enumerate(unique_clusters):
            values = np.abs(shap_values[class_index]).mean(axis=0)
            for feature, value in zip(transformed_feature_names, values):
                rows.append({"cluster": int(cluster), "feature": feature, "mean_abs_shap_value": float(value)})
    elif len(shap_values.shape) == 3:
        for class_index, cluster in enumerate(unique_clusters):
            values = np.abs(shap_values[:, :, class_index]).mean(axis=0)
            for feature, value in zip(transformed_feature_names, values):
                rows.append({"cluster": int(cluster), "feature": feature, "mean_abs_shap_value": float(value)})
    else:
        values = np.abs(shap_values).mean(axis=0)
        for feature, value in zip(transformed_feature_names, values):
            rows.append({"cluster": -1, "feature": feature, "mean_abs_shap_value": float(value)})

    return pd.DataFrame(rows), {
        "surrogate_model": "RandomForestClassifier",
        "sample_size": int(sample_size),
        "n_features": int(len(transformed_feature_names)),
    }


def build_segment_summary_objects(
    clustered_df: pd.DataFrame,
    numeric_description_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    demographic_cols: list[str] | None = None,
):
    summaries = []
    for cluster in sorted(clustered_df["cluster"].unique()):
        segment = clustered_df[clustered_df["cluster"] == cluster]

        demographic_cols = demographic_cols or []
        demographic_summary = {}

        for col in demographic_cols:
            if col not in segment.columns:
                continue

            if pd.api.types.is_numeric_dtype(segment[col]):
                demographic_summary[col] = {
                    "segment_mean": float(segment[col].mean()),
                    "global_mean": float(clustered_df[col].mean()),
                }
            else:
                segment_dist = segment[col].value_counts(normalize=True).head(5)
                global_dist = clustered_df[col].value_counts(normalize=True).head(5)

                demographic_summary[col] = {
                    "segment_distribution": {
                        str(k): float(v) for k, v in segment_dist.items()
                    },
                    "global_distribution": {
                        str(k): float(v) for k, v in global_dist.items()
                    },
                }

        if not numeric_description_df.empty:
            segment_numeric = numeric_description_df[numeric_description_df["cluster"] == cluster].copy()
            segment_numeric["abs_relative_difference"] = segment_numeric["relative_difference"].abs()
            top_numeric = segment_numeric.sort_values("abs_relative_difference", ascending=False).head(8).drop(columns=["abs_relative_difference"]).to_dict(orient="records")
        else:
            top_numeric = []

        if not shap_df.empty:
            segment_shap = shap_df[shap_df["cluster"] == cluster].copy()
            top_shap = segment_shap.sort_values("mean_abs_shap_value", ascending=False).head(8).to_dict(orient="records")
        else:
            top_shap = []

        summaries.append({
            "cluster": int(cluster),
            "size": int(len(segment)),
            "share": float(len(segment) / len(clustered_df)),
            "top_numeric_differences": top_numeric,
            "top_shap_features": top_shap[:3],
            "demographic_summary": demographic_summary,
        })
    
    return summaries
