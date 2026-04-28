import os
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TEXT_COLS = ["feature_name", "feature_description", "feature_group"]
RANKING_COLS = [
    "feature_name",
    "feature_description",
    "feature_group",
    "retrieval_method",
    "embedding_model",
    "similarity_score",
]


@dataclass(frozen=True)
class SelectionResult:
    ranked_features: pd.DataFrame
    selected_before_corr: pd.DataFrame
    selected_after_corr: pd.DataFrame
    removed_by_correlation: list[dict]


def build_feature_text(row: pd.Series) -> str:
    return " ".join(str(row.get(col, "")) for col in TEXT_COLS).lower().strip()


def _available_feature_store(feature_store_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    available = set(customers_df.columns) - {"PER_ID"}
    fs = feature_store_df.copy()
    fs = fs[fs["feature_name"].isin(available)].copy()
    if fs.empty:
        raise ValueError("No feature store features match customer dataset columns.")
    fs["feature_text"] = fs.apply(build_feature_text, axis=1)
    return fs


def rank_features_by_tfidf(
    campaign_prompt: str,
    feature_store_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    embedding_model: str | None = None,
) -> pd.DataFrame:
    fs = _available_feature_store(feature_store_df, customers_df)
    query = campaign_prompt.lower().strip()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95)
    matrix = vectorizer.fit_transform(fs["feature_text"].tolist() + [query])
    fs["similarity_score"] = cosine_similarity(matrix[:-1], matrix[-1]).flatten()
    fs["retrieval_method"] = "tfidf"
    fs["embedding_model"] = None
    fs["tfidf_similarity"] = fs["similarity_score"]
    return fs.sort_values("similarity_score", ascending=False).reset_index(drop=True)


def _openai_embed_texts(texts: list[str], model: str) -> np.ndarray:
    api_key = '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'
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is required for embedding model {model}. "
            "Set it in the environment, or run only tfidf/all-MiniLM configs."
        )
    from openai import OpenAI

    client = OpenAI(api_key='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')
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=float)


def _sentence_transformer_embed_texts(texts: list[str], model: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for all-MiniLM-L6-v2. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    encoder = SentenceTransformer(model)
    return np.asarray(encoder.encode(texts, normalize_embeddings=True), dtype=float)


def _hashing_embedding_fallback(texts: list[str], n_features: int = 384) -> np.ndarray:
    """
    Deterministic offline fallback for quick smoke tests only.
    It is not used unless ALLOW_HASHING_EMBEDDING_FALLBACK=1.
    """
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2")
    return vectorizer.transform(texts).toarray()


def embed_texts(texts: list[str], model: str) -> np.ndarray:
    allow_fallback = os.getenv("ALLOW_HASHING_EMBEDDING_FALLBACK", "0") == "1"
    try:
        if model == "all-MiniLM-L6-v2":
            return _sentence_transformer_embed_texts(texts, model)
        if model in {"text-embedding-3-small", "text-embedding-3-large"}:
            return _openai_embed_texts(texts, model)
        raise ValueError(f"Unsupported embedding model: {model}")
    except Exception:
        if allow_fallback:
            return _hashing_embedding_fallback(texts)
        raise


def rank_features_by_embedding(
    campaign_prompt: str,
    feature_store_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    embedding_model: str,
) -> pd.DataFrame:
    fs = _available_feature_store(feature_store_df, customers_df)
    texts = fs["feature_text"].tolist() + [campaign_prompt.lower().strip()]
    vectors = embed_texts(texts, embedding_model)
    feature_vectors = vectors[:-1]
    query_vector = vectors[-1].reshape(1, -1)
    fs["similarity_score"] = cosine_similarity(feature_vectors, query_vector).flatten()
    fs["retrieval_method"] = "embedding"
    fs["embedding_model"] = embedding_model
    fs["tfidf_similarity"] = np.nan
    return fs.sort_values("similarity_score", ascending=False).reset_index(drop=True)


def rank_features(
    campaign_prompt: str,
    feature_store_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    retrieval_method: str,
    embedding_model: str | None = None,
) -> pd.DataFrame:
    if retrieval_method == "tfidf":
        return rank_features_by_tfidf(campaign_prompt, feature_store_df, customers_df)
    if retrieval_method == "embedding":
        if not embedding_model:
            raise ValueError("embedding_model is required when retrieval_method='embedding'.")
        return rank_features_by_embedding(campaign_prompt, feature_store_df, customers_df, embedding_model)
    raise ValueError(f"Unsupported retrieval_method: {retrieval_method}")


def apply_selection_strategy(
    ranked_features_df: pd.DataFrame,
    selection_strategy: str,
    similarity_threshold: float | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    if selection_strategy == "threshold":
        if similarity_threshold is None:
            raise ValueError("similarity_threshold is required for threshold selection.")
        return ranked_features_df[ranked_features_df["similarity_score"] > similarity_threshold].copy()
    if selection_strategy == "top_k":
        if top_k is None:
            raise ValueError("top_k is required for top_k selection.")
        return ranked_features_df.head(int(top_k)).copy()
    raise ValueError(f"Unsupported selection_strategy: {selection_strategy}")


def remove_highly_correlated_features(
    selected_before_corr_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    correlation_threshold: float = 0.90,
) -> tuple[pd.DataFrame, list[dict]]:
    if selected_before_corr_df.empty:
        return selected_before_corr_df.copy(), []

    selected_rows = []
    selected_names: list[str] = []
    removed: list[dict] = []
    numeric_df = customers_df.select_dtypes(include=[np.number])

    for _, row in selected_before_corr_df.iterrows():
        feature = row["feature_name"]
        if feature not in customers_df.columns:
            removed.append({
                "feature_name": feature,
                "removed_due_to": None,
                "correlation": None,
                "reason": "missing_from_customer_dataset",
            })
            continue

        if feature not in numeric_df.columns:
            selected_rows.append(row)
            selected_names.append(feature)
            continue

        numeric_selected = [f for f in selected_names if f in numeric_df.columns]
        if not numeric_selected:
            selected_rows.append(row)
            selected_names.append(feature)
            continue

        corr_values = numeric_df[numeric_selected + [feature]].corr()[feature].drop(feature).abs()
        max_corr = corr_values.max() if not corr_values.empty else np.nan
        if pd.isna(max_corr) or max_corr <= correlation_threshold:
            selected_rows.append(row)
            selected_names.append(feature)
        else:
            removed_due_to = corr_values.idxmax()
            removed.append({
                "feature_name": feature,
                "removed_due_to": removed_due_to,
                "correlation": float(max_corr),
                "reason": "high_pairwise_absolute_correlation",
            })

    if not selected_rows:
        return selected_before_corr_df.iloc[0:0].copy(), removed

    return pd.DataFrame(selected_rows).reset_index(drop=True), removed


def select_features(
    campaign_prompt: str,
    feature_store_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    retrieval_method: str = "tfidf",
    embedding_model: str | None = None,
    selection_strategy: str = "threshold",
    similarity_threshold: float | None = 0.05,
    top_k: int | None = None,
    correlation_threshold: float = 0.90,
) -> SelectionResult:
    ranked = rank_features(
        campaign_prompt=campaign_prompt,
        feature_store_df=feature_store_df,
        customers_df=customers_df,
        retrieval_method=retrieval_method,
        embedding_model=embedding_model,
    )

    selected_before = apply_selection_strategy(
        ranked_features_df=ranked,
        selection_strategy=selection_strategy,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
    ).reset_index(drop=True)

    selected_after, removed = remove_highly_correlated_features(
        selected_before_corr_df=selected_before,
        customers_df=customers_df,
        correlation_threshold=correlation_threshold,
    )

    return SelectionResult(
        ranked_features=ranked.reset_index(drop=True),
        selected_before_corr=selected_before.reset_index(drop=True),
        selected_after_corr=selected_after.reset_index(drop=True),
        removed_by_correlation=removed,
    )
