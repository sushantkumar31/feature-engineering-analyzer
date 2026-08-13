"""Categorical encoding helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from fea.config import HIGH_CARDINALITY_THRESHOLD, EncodingMethod


def get_high_cardinality_columns(
    df: pd.DataFrame,
    columns: list[str],
    threshold: int = HIGH_CARDINALITY_THRESHOLD,
) -> list[str]:
    """Return categorical columns with more unique values than ``threshold``."""
    return [c for c in columns if df[c].nunique() > threshold]


def build_categorical_encoder(method: EncodingMethod):
    """Build a scikit-learn transformer for the chosen categorical encoding.

    ``OrdinalEncoder`` is used for label encoding (one ordinal code per column)
    and ``OneHotEncoder`` for one-hot encoding. ``handle_unknown`` is set so
    unseen categories seen at prediction time do not crash the model.
    """
    if method == EncodingMethod.ONE_HOT:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if method == EncodingMethod.LABEL:
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unknown encoding method: {method!r}")
