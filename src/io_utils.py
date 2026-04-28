
import json
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd


def make_request_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def validate_inputs(customers_df: pd.DataFrame, feature_store_df: pd.DataFrame) -> None:
    if "PER_ID" not in customers_df.columns:
        raise ValueError("Customer dataset must contain PER_ID.")
    required = {"feature_name", "feature_description", "feature_group"}
    missing = required - set(feature_store_df.columns)
    if missing:
        raise ValueError(f"Feature store is missing required columns: {missing}")
