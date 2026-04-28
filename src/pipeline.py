from pathlib import Path

import pandas as pd

from .io_utils import make_request_id, save_dataframe, save_json, validate_inputs
from .feature_selection import select_features
from .preprocessing import prepare_customer_matrix
from .clustering import run_multiple_clustering_algorithms
from .explanation import (
    get_segment_numeric_descriptions,
    compute_shap_values_for_segments,
    build_segment_summary_objects,
)
from .llm_recommendation import generate_segment_recommendations
from .visualization import save_segment_size_plot, save_pca_cluster_plot


class SegmentationSkip(Exception):
    def __init__(self, reason: str, message: str | None = None):
        self.reason = reason
        super().__init__(message or reason)


def run_segmentation_pipeline(
    campaign_prompt: str,
    customers_df: pd.DataFrame,
    feature_store_df: pd.DataFrame,
    output_dir: str | Path,
    retrieval_method: str,
    embedding_model: str | None,
    selection_strategy: str,
    similarity_threshold: float | None,
    top_k: int | None,
    correlation_threshold: float,
    cluster_min: int,
    cluster_max: int,
    max_metric_sample_size: int,
    shap_sample_size: int,
    random_state: int,
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    generate_recommendations: bool = False,
    generate_explanations: bool = True,
    save_artifacts: bool = True,
):
    validate_inputs(customers_df, feature_store_df)

    output_dir = Path(output_dir)
    request_id = make_request_id()
    request_dir = output_dir / request_id
    if save_artifacts:
        request_dir.mkdir(parents=True, exist_ok=True)

    selection = select_features(
        campaign_prompt=campaign_prompt,
        feature_store_df=feature_store_df,
        customers_df=customers_df,
        retrieval_method=retrieval_method,
        embedding_model=embedding_model,
        selection_strategy=selection_strategy,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        correlation_threshold=correlation_threshold,
    )

    selected_before = selection.selected_before_corr["feature_name"].tolist()
    selected_after = selection.selected_after_corr["feature_name"].tolist()

    if len(selected_before) == 0:
        raise SegmentationSkip("no_features_selected")
    if len(selected_after) == 0 and len(selected_before) > 0:
        raise SegmentationSkip("all_features_removed_by_correlation")
    if len(selected_after) < 2:
        raise SegmentationSkip("no_features_selected", "Fewer than two features selected after filtering.")

    prepared = prepare_customer_matrix(customers_df=customers_df, selected_features=selected_after)

    best_model, metrics_df = run_multiple_clustering_algorithms(
        X=prepared["X"],
        cluster_range=range(cluster_min, cluster_max + 1),
        max_metric_sample_size=max_metric_sample_size,
        random_state=random_state,
    )

    clustered_df = prepared["working_df"].copy()
    clustered_df["cluster"] = best_model["labels"]

    numeric_description_df = pd.DataFrame()
    shap_df = pd.DataFrame()
    shap_metadata = {}
    segment_summary_objects = []
    recommendations = []

    if generate_explanations:
        numeric_description_df = get_segment_numeric_descriptions(clustered_df, prepared["numeric_cols"])
        shap_df, shap_metadata = compute_shap_values_for_segments(
            X=prepared["X"],
            labels=best_model["labels"],
            transformed_feature_names=prepared["transformed_feature_names"],
            shap_sample_size=shap_sample_size,
            random_state=random_state,
        )
        segment_summary_objects = build_segment_summary_objects(
            clustered_df=clustered_df,
            numeric_description_df=numeric_description_df,
            shap_df=shap_df,
            demographic_cols=["age", "sex"],
        )
        if generate_recommendations:
            recommendations = generate_segment_recommendations(
                campaign_prompt=campaign_prompt,
                segment_summaries=segment_summary_objects,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )

    metadata = {
        "request_id": request_id,
        "campaign_prompt": campaign_prompt,
        "selected_features_before_corr": selected_before,
        "selected_features_after_corr": selected_after,
        "removed_by_correlation": selection.removed_by_correlation,
        "best_model": {
            "algorithm": best_model["algorithm"],
            "k": best_model["k"],
            "combined_score": best_model["combined_score"],
            "metrics": best_model["metrics"],
        },
        "parameters": {
            "retrieval_method": retrieval_method,
            "embedding_model": embedding_model,
            "selection_strategy": selection_strategy,
            "similarity_threshold": similarity_threshold,
            "top_k": top_k,
            "correlation_threshold": correlation_threshold,
            "cluster_min": cluster_min,
            "cluster_max": cluster_max,
            "max_metric_sample_size": max_metric_sample_size,
            "shap_sample_size": shap_sample_size,
            "random_state": random_state,
        },
        "shap_metadata": shap_metadata,
        "n_users": int(len(clustered_df)),
    }

    if save_artifacts:
        save_dataframe(request_dir / "ranked_features.csv", selection.ranked_features)
        save_dataframe(request_dir / "selected_features_before_corr.csv", selection.selected_before_corr)
        save_dataframe(request_dir / "selected_features_after_corr.csv", selection.selected_after_corr)
        save_dataframe(request_dir / "clustering_metrics.csv", metrics_df)
        save_dataframe(request_dir / "clustered_users.csv", clustered_df)
        if not numeric_description_df.empty:
            save_dataframe(request_dir / "segment_numeric_descriptions.csv", numeric_description_df)
        if not shap_df.empty:
            save_dataframe(request_dir / "segment_shap_values.csv", shap_df)
        save_json(request_dir / "segment_summary_objects.json", segment_summary_objects)
        save_json(request_dir / "segment_recommendations.json", recommendations)
        save_json(request_dir / "run_metadata.json", metadata)
        save_segment_size_plot(clustered_df, request_dir / "segment_size_distribution.png")
        save_pca_cluster_plot(prepared["X"], best_model["labels"], request_dir / "pca_cluster_visualization.png", random_state)

    return {
        "request_id": request_id,
        "request_dir": request_dir,
        "metadata": metadata,
        "ranked_features": selection.ranked_features,
        "selected_features_before_corr": selection.selected_before_corr,
        "selected_features_after_corr": selection.selected_after_corr,
        "removed_by_correlation": selection.removed_by_correlation,
        "clustering_candidates": metrics_df,
        "clustered_users": clustered_df,
        "numeric_descriptions": numeric_description_df,
        "shap_values": shap_df,
        "segment_summary_objects": segment_summary_objects,
        "recommendations": recommendations,
    }
