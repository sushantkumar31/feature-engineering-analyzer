"""Data preprocessing: missing-value imputation, outlier handling and scaling.

All preprocessing is exposed either as plain functions (for row-level, non
pipeline-safe operations such as outlier row removal) or as scikit-learn
transformers so they can be composed into a single reproducible
:class:`sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from fea.config import (
    IQR_MULTIPLIER,
    OUTLIER_Z_THRESHOLD,
    MissingValueStrategy,
    OutlierDetection,
    OutlierTreatment,
    ScalingMethod,
)
from fea.encoding import build_categorical_encoder

__all__ = [
    "detect_outliers",
    "apply_outlier_treatment",
    "impute_missing",
    "drop_missing_rows",
    "build_imputer",
    "build_scaler",
    "build_preprocessing_pipeline",
]


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every row that contains at least one missing value."""
    cleaned = df.dropna().reset_index(drop=True)
    if cleaned.shape[0] == 0:
        raise ValueError(
            "This strategy removed every row (every row had a missing value somewhere). "
            "Try Mean/Median + Mode instead of dropping rows."
        )
    return cleaned


def impute_missing(
    df: pd.DataFrame,
    strategy: MissingValueStrategy,
) -> pd.DataFrame:
    """Impute missing values column-by-column.

    Numeric columns are filled with the mean or median (depending on
    ``strategy``); categorical columns with the most frequent value (mode).
    Columns that are entirely null are dropped, as no imputation is possible.

    Returns a copy of ``df`` with missing values filled in.
    """
    result = df.copy()

    if strategy == MissingValueStrategy.DROP_ROWS:
        return drop_missing_rows(result)

    numeric_cols = result.select_dtypes(include="number").columns
    categorical_cols = result.select_dtypes(exclude="number").columns

    fill_func = (
        (lambda col: col.mean())
        if strategy == MissingValueStrategy.MEAN_MODE
        else (lambda col: col.median())
    )

    for col in numeric_cols:
        if result[col].isnull().any():
            result[col] = result[col].fillna(fill_func(result[col]))

    for col in categorical_cols:
        if result[col].isnull().any():
            mode_value = result[col].mode(dropna=True)
            if len(mode_value) > 0:
                result[col] = result[col].fillna(mode_value[0])
            else:
                result = result.dropna(subset=[col])

    return result.reset_index(drop=True)


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: OutlierDetection,
) -> pd.DataFrame:
    """Return a boolean mask (``False`` for normal rows) of outliers.

    The mask has the same index and columns as ``df`` (only the requested
    ``columns`` are populated; others are ``False``).
    """
    mask = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        series = df[col]
        if method == OutlierDetection.IQR:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
            mask[col] = (series < lower) | (series > upper)
        else:  # Z-score
            mean, std = series.mean(), series.std()
            if std > 0:
                z_scores = np.abs((series - mean) / std)
                mask[col] = z_scores > OUTLIER_Z_THRESHOLD

    return mask


def apply_outlier_treatment(
    df: pd.DataFrame,
    columns: list[str],
    treatment: OutlierTreatment,
    outlier_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """Apply the chosen outlier treatment in place of a new frame.

    Parameters
    ----------
    df:
        The working dataframe.
    columns:
        Numeric feature columns to consider for treatment.
    treatment:
        Which treatment to apply.
    outlier_mask:
        Boolean mask from :func:`detect_outliers`. Required when ``treatment``
        is ``REMOVE_ROWS``.

    Returns
    -------
    ``(transformed_df, description)`` where ``description`` summarises what was
    done and is suitable for display in the UI.
    """
    result = df.copy()

    if treatment == OutlierTreatment.NONE:
        return result, "No treatment applied — outliers left as-is."

    if treatment == OutlierTreatment.REMOVE_ROWS:
        if outlier_mask is None:
            raise ValueError("outlier_mask is required for REMOVE_ROWS treatment.")
        n_removed = int(outlier_mask.any(axis=1).sum())
        result = result[~outlier_mask.any(axis=1)].reset_index(drop=True)
        return result, f"Removed {n_removed} row(s)."

    if treatment == OutlierTreatment.CAP:
        for col in columns:
            q1, q3 = result[col].quantile(0.25), result[col].quantile(0.75)
            iqr = q3 - q1
            result[col] = result[col].clip(q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr)
        return result, "Capped outlier values to IQR bounds."

    if treatment == OutlierTreatment.WINSORIZE:
        for col in columns:
            lower, upper = result[col].quantile(0.05), result[col].quantile(0.95)
            result[col] = result[col].clip(lower, upper)
        return result, "Winsorized values below 5th / above 95th percentile."

    raise ValueError(f"Unknown treatment: {treatment!r}")


def build_imputer(strategy: MissingValueStrategy) -> SimpleImputer:
    """Build a numeric :class:`SimpleImputer` for the chosen strategy."""
    numeric_strategy = "mean" if strategy == MissingValueStrategy.MEAN_MODE else "median"
    return SimpleImputer(strategy=numeric_strategy)


def build_scaler(method: ScalingMethod):
    """Return a fitted-on-fit scikit-learn scaler for the chosen method."""
    if method == ScalingMethod.STANDARD:
        return StandardScaler()
    if method == ScalingMethod.MINMAX:
        return MinMaxScaler()
    if method == ScalingMethod.ROBUST:
        return RobustScaler()
    raise ValueError(f"Unknown scaling method: {method!r}")


def build_preprocessing_pipeline(
    config,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Assemble the engineered preprocessing pipeline.

    The pipeline imputes missing values, one-hot encodes categorical columns
    and scales numeric columns. It must be fit on the *training* split only to
    avoid data leakage.
    """
    numeric_transformer = Pipeline(steps=[("imputer", build_imputer(config.missing_strategy))])
    if config.scale:
        numeric_transformer.steps.append(("scaler", build_scaler(config.scaling_method)))

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_categorical_encoder(config.encoding_method)),
        ]
    )

    steps = [
        ("numeric", numeric_transformer, numeric_cols),
        ("categorical", categorical_transformer, categorical_cols),
    ]

    column_transformer = ColumnTransformer(transformers=steps, remainder="drop")
    return Pipeline(steps=[("preprocess", column_transformer)])
