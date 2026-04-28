from __future__ import annotations
import json
import os
import sys
import pandas as pd

from pathlib import Path
from typing import Any
from src.pipeline import run_segmentation_pipeline, SegmentationSkip


OPENAI_API_KEY = ""
CAMPAIGN_ID = "campaign_1" # folder where campaign result will be stored
CAMPAIGN_PROMPT = ("I need users who are likely interested in expensive fish dishes")

CUSTOMERS_PATH = "data/customers.csv"
FEATURES_PATH = "data/feature_store.csv"
OUTPUT_DIR = "outputs/single_campaign_examples"

# Clustering setup.
CLUSTER_MIN = 3
CLUSTER_MAX = 7
CORRELATION_THRESHOLD = 0.90
RANDOM_STATE = 42

GENERATE_EXPLANATIONS = True
GENERATE_RECOMMENDATIONS = True
OPENAI_CHAT_MODEL = "gpt-4o-mini"

MAX_METRIC_SAMPLE_SIZE = 5000
SHAP_SAMPLE_SIZE = 1500

CONFIGS = [
    {
        "label": "best_openai_large_threshold_0_40",
        "retrieval_method": "embedding",
        "embedding_model": "text-embedding-3-large",
        "selection_strategy": "threshold",
        "similarity_threshold": 0.40,
        "top_k": None,
    },
    {
        "label": "poor_noisy_minilm_threshold_0_20",
        "retrieval_method": "embedding",
        "embedding_model": "all-MiniLM-L6-v2",
        "selection_strategy": "threshold",
        "similarity_threshold": 0.20,
        "top_k": None,
    }
]


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def ensure_project_imports() -> None:
    current = Path(__file__).resolve().parent
    if str(current) not in sys.path:
        sys.path.insert(0, str(current))


def safe_slug(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
        .lower()
    )


def summarize_selected_features(selected_df: pd.DataFrame) -> list[dict[str, Any]]:
    if selected_df is None or selected_df.empty:
        return []

    keep_cols = [
        "feature_name",
        "feature_description",
        "feature_group",
        "similarity_score",
        "retrieval_method",
        "embedding_model",
    ]
    existing = [c for c in keep_cols if c in selected_df.columns]
    rows = selected_df[existing].copy()

    if "similarity_score" in rows.columns:
        rows["similarity_score"] = rows["similarity_score"].round(4)

    return rows.to_dict(orient="records")


def summarize_clustering_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    sort_cols = []
    if "combined_score" in df.columns:
        sort_cols.append("combined_score")
    elif "silhouette" in df.columns:
        sort_cols.append("silhouette")

    out = df.copy()
    if sort_cols:
        out = out.sort_values(sort_cols[0], ascending=False)

    keep_cols = [
        "algorithm",
        "k",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "combined_score",
    ]
    existing = [c for c in keep_cols if c in out.columns]
    out = out[existing].head(10).copy()

    for col in ["silhouette", "davies_bouldin", "calinski_harabasz", "combined_score"]:
        if col in out.columns:
            out[col] = out[col].round(4)

    return out.to_dict(orient="records")


def summarize_segments(segment_summary_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []

    for segment in segment_summary_objects or []:
        compact.append({
            "cluster": segment.get("cluster"),
            "size": segment.get("size"),
            "share_percent": round(100 * float(segment.get("share", 0)), 2),
            "top_numeric_differences": segment.get("top_numeric_differences", [])[:5],
            "top_shap_features": segment.get("top_shap_features", [])[:3],
            "demographic_summary": segment.get("demographic_summary", {}),
        })

    return compact


def build_markdown_report(
    campaign_id: str,
    campaign_prompt: str,
    run_summaries: list[dict[str, Any]],
) -> str:
    lines = []

    lines.append(f"# Single Campaign Segmentation Comparison")
    lines.append("")
    lines.append(f"**Campaign ID:** `{campaign_id}`")
    lines.append("")
    lines.append(f"**Campaign prompt:** {campaign_prompt}")
    lines.append("")

    for run in run_summaries:
        lines.append("---")
        lines.append("")
        lines.append(f"## {run['label']}")
        lines.append("")

        if run["status"] != "completed":
            lines.append(f"**Status:** `{run['status']}`")
            lines.append(f"**Reason:** `{run.get('reason', '')}`")
            lines.append("")
            continue

        meta = run["metadata"]
        best = meta["best_model"]

        lines.append("### Configuration")
        lines.append("")
        params = meta["parameters"]
        lines.append(f"- Retrieval method: `{params['retrieval_method']}`")
        lines.append(f"- Embedding model: `{params['embedding_model']}`")
        lines.append(f"- Selection strategy: `{params['selection_strategy']}`")
        lines.append(f"- Similarity threshold: `{params['similarity_threshold']}`")
        lines.append(f"- Top-k: `{params['top_k']}`")
        lines.append(f"- Correlation threshold: `{params['correlation_threshold']}`")
        lines.append("")

        lines.append("### Selected Features")
        lines.append("")
        lines.append(f"- Before correlation removal: `{len(meta['selected_features_before_corr'])}`")
        lines.append(f"- After correlation removal: `{len(meta['selected_features_after_corr'])}`")
        lines.append(f"- Removed by correlation: `{len(meta['removed_by_correlation'])}`")
        lines.append("")

        for item in run["selected_features_after_corr"]:
            score = item.get("similarity_score")
            score_text = "" if score is None else f" — similarity: `{score}`"
            desc = item.get("feature_description", "")
            lines.append(f"- `{item.get('feature_name')}`{score_text}: {desc}")
        lines.append("")

        lines.append("### Best Clustering")
        lines.append("")
        lines.append(f"- Algorithm: `{best['algorithm']}`")
        lines.append(f"- k: `{best['k']}`")
        metrics = best.get("metrics", {})
        if metrics:
            lines.append(f"- Silhouette: `{metrics.get('silhouette')}`")
            lines.append(f"- Davies-Bouldin: `{metrics.get('davies_bouldin')}`")
            lines.append(f"- Calinski-Harabasz: `{metrics.get('calinski_harabasz')}`")
        lines.append("")

        lines.append("### Segment Summaries")
        lines.append("")
        for segment in run["segment_summaries"]:
            lines.append(
                f"- Cluster `{segment['cluster']}`: "
                f"{segment['share_percent']}% of users, size `{segment['size']}`"
            )
            top_shap = segment.get("top_shap_features", [])
            if top_shap:
                shap_str = ", ".join(
                    f"`{x.get('feature')}` ({round(float(x.get('mean_abs_shap_value', 0)), 4)})"
                    for x in top_shap
                )
                lines.append(f"  - Top SHAP features: {shap_str}")
        lines.append("")

        if run.get("recommendations"):
            lines.append("### LLM Recommendations")
            lines.append("")
            for rec in run["recommendations"]:
                cluster = rec.get("cluster")
                decision = rec.get("targeting_decision")
                score = rec.get("priority_score")
                final = rec.get("final_recommendation", "")
                reasoning = rec.get("reasoning", "")
                lines.append(f"- Cluster `{cluster}` — `{decision}`, priority `{score}`")
                lines.append(f"  - Reasoning: {reasoning}")
                lines.append(f"  - Recommendation: {final}")
            lines.append("")

        lines.append("### Artifact Folder")
        lines.append("")
        lines.append(f"`{run['request_dir']}`")
        lines.append("")

    return "\n".join(lines)


def run_one_config(
    cfg: dict[str, Any],
    campaign_id: str,
    campaign_prompt: str,
    customers_df: pd.DataFrame,
    feature_store_df: pd.DataFrame,
    output_dir: Path,
):

    label = cfg["label"]
    run_output_dir = output_dir / safe_slug(campaign_id) / safe_slug(label)

    try:
        result = run_segmentation_pipeline(
            campaign_prompt=campaign_prompt,
            customers_df=customers_df,
            feature_store_df=feature_store_df,
            output_dir=run_output_dir,
            retrieval_method=cfg["retrieval_method"],
            embedding_model=cfg.get("embedding_model"),
            selection_strategy=cfg["selection_strategy"],
            similarity_threshold=cfg.get("similarity_threshold"),
            top_k=cfg.get("top_k"),
            correlation_threshold=CORRELATION_THRESHOLD,
            cluster_min=CLUSTER_MIN,
            cluster_max=CLUSTER_MAX,
            max_metric_sample_size=MAX_METRIC_SAMPLE_SIZE,
            shap_sample_size=SHAP_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
            openai_api_key=OPENAI_API_KEY,
            openai_model=OPENAI_CHAT_MODEL,
            generate_recommendations=GENERATE_RECOMMENDATIONS,
            generate_explanations=GENERATE_EXPLANATIONS,
            save_artifacts=True,
        )

        return {
            "label": label,
            "status": "completed",
            "request_dir": str(result["request_dir"]),
            "metadata": result["metadata"],
            "selected_features_before_corr": summarize_selected_features(result["selected_features_before_corr"]),
            "selected_features_after_corr": summarize_selected_features(result["selected_features_after_corr"]),
            "removed_by_correlation": result["removed_by_correlation"],
            "clustering_candidates": summarize_clustering_candidates(result["clustering_candidates"]),
            "segment_summaries": summarize_segments(result["segment_summary_objects"]),
            "recommendations": result["recommendations"],
        }

    except SegmentationSkip as exc:
        return {
            "label": label,
            "status": "skipped",
            "reason": exc.reason,
            "message": str(exc),
            "request_dir": str(run_output_dir),
        }

    except Exception as exc:
        return {
            "label": label,
            "status": "failed",
            "reason": type(exc).__name__,
            "message": str(exc),
            "request_dir": str(run_output_dir),
        }


def main() -> None:
    ensure_project_imports()

    if OPENAI_API_KEY:
        # Existing embedding code reads this variable. This keeps the key in this file only.
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    customers_df = pd.read_csv(CUSTOMERS_PATH)
    feature_store_df = pd.read_csv(FEATURES_PATH)

    run_summaries = []
    configs = list(CONFIGS.values()) if isinstance(CONFIGS, dict) else list(CONFIGS)

    for cfg in configs:
        print(f"Running: {cfg['label']}")
        summary = run_one_config(
            cfg=cfg,
            campaign_id=CAMPAIGN_ID,
            campaign_prompt=CAMPAIGN_PROMPT,
            customers_df=customers_df,
            feature_store_df=feature_store_df,
            output_dir=output_dir,
        )
        run_summaries.append(summary)

        status = summary["status"]
        if status == "completed":
            selected = summary["metadata"]["selected_features_after_corr"]
            best = summary["metadata"]["best_model"]
            print(
                f"  completed | features={len(selected)} | "
                f"algorithm={best['algorithm']} | k={best['k']} | "
                f"silhouette={best['metrics'].get('silhouette')}"
            )
        else:
            print(f"  {status} | reason={summary.get('reason')} | message={summary.get('message', '')}")

    comparison_dir = output_dir / safe_slug(CAMPAIGN_ID)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    json_path = comparison_dir / "comparison_summary.json"
    md_path = comparison_dir / "comparison_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "campaign_id": CAMPAIGN_ID,
                "campaign_prompt": CAMPAIGN_PROMPT,
                "runs": run_summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    md_report = build_markdown_report(
        campaign_id=CAMPAIGN_ID,
        campaign_prompt=CAMPAIGN_PROMPT,
        run_summaries=run_summaries,
    )
    md_path.write_text(md_report, encoding="utf-8")

    print("")
    print("Done.")
    print(f"Comparison JSON: {json_path}")
    print(f"Comparison report: {md_path}")


if __name__ == "__main__":
    main()
