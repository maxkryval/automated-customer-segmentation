
# Automated Customer Segmentation — 25 Campaign Optimization Version

This repository contains a bachelor thesis demo pipeline for automated customer segmentation from natural language campaign descriptions.

## What it does

```text
campaign description
→ TF-IDF feature selection
→ optional correlation filtering
→ clustering model selection
→ SHAP-based explanation
→ optional LLM segment relevance scoring
```

## Setup and run

```bash
bash run.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python generate_mock_data.py
python run_experiments.py
```

## Experiment design

The experiment runs 25 campaigns split into 3 groups:

1. `marketing_manager_request` — realistic marketing-manager style requests.
2. `strict_metric_request` — strict requests targeting a specific metric.
3. `broad_low_signal_request` — broad requests expected to degrade or fail.

The experiment compares four optimization configurations:

1. no correlation filtering + TF-IDF threshold > 0
2. no correlation filtering + TF-IDF threshold > 0.05
3. correlation filtering + TF-IDF threshold > 0
4. correlation filtering + TF-IDF threshold > 0.05

This supports the thesis requirement related to model/parameter optimization and rational selection of analytical configurations.

## Main outputs

```text
outputs/optimization_experiment_.../
├── optimization_config_summary.csv
├── feature_selection_evaluation.csv
├── clustering_metrics_summary.csv
├── all_selected_features.csv
├── positive_feature_counts.csv
├── skipped_campaigns.csv
├── primary_shap_features.csv
├── run_index.csv
├── figures/
│   ├── fig_mean_f1_by_config.png
│   ├── fig_mean_silhouette_by_config.png
│   └── fig_algorithm_k_distribution.png
└── config/campaign/run folders
```

## Optional LLM recommendations

By default, `GENERATE_RECOMMENDATIONS = False` in `run_experiments.py` to avoid many LLM calls.

To enable it:
1. Set `OPENAI_API_KEY`
2. Set `GENERATE_RECOMMENDATIONS = True`

## Single campaign run

```bash
python run_pipeline.py \
  --campaign "We want to promote premium dining occasions with beer-related products to users who are likely to spend more." \
  --exclude_correlated_features \
  --relevance_threshold 0.0
```
