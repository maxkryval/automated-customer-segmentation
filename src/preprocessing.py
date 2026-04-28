
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def prepare_customer_matrix(customers_df: pd.DataFrame, selected_features: list[str]):
    used_features = [f for f in selected_features if f in customers_df.columns]
    if len(used_features) < 2:
        raise ValueError(f"At least 2 features are required for clustering. Got: {used_features}")

    DEMOGRAPHIC_COLUMNS = ["age", "sex"]

    passthrough_cols = [
        col for col in DEMOGRAPHIC_COLUMNS
        if col in customers_df.columns and col not in used_features
    ]

    working_df = customers_df[["PER_ID"] + used_features + passthrough_cols].copy()
    numeric_cols = working_df[used_features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in used_features if c not in numeric_cols]

    for col in numeric_cols:
        working_df[col] = working_df[col].fillna(working_df[col].median())
    for col in categorical_cols:
        working_df[col] = working_df[col].fillna("unknown").astype(str)

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X = preprocessor.fit_transform(working_df[used_features])
    X = MinMaxScaler().fit_transform(X)

    transformed_feature_names = []
    if numeric_cols:
        transformed_feature_names.extend(numeric_cols)
    if categorical_cols:
        encoder = preprocessor.named_transformers_["categorical"]
        transformed_feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())

    return {
        "working_df": working_df,
        "X": X,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "transformed_feature_names": transformed_feature_names,
    }
