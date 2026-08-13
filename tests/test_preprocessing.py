"""Tests for fea.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from fea.config import MissingValueStrategy, OutlierDetection, OutlierTreatment, ScalingMethod
from fea.preprocessing import (
    apply_outlier_treatment,
    build_preprocessing_pipeline,
    detect_outliers,
    impute_missing,
)


def test_impute_missing_mean_mode(binary_df):
    out = impute_missing(binary_df, MissingValueStrategy.MEAN_MODE)
    assert out.isnull().sum().sum() == 0
    # NaN at row 0 was filled with the column mean
    assert np.isclose(out.loc[0, "feature_a"], binary_df["feature_a"].mean())


def test_impute_missing_median_mode(binary_df):
    out = impute_missing(binary_df, MissingValueStrategy.MEDIAN_MODE)
    assert out.isnull().sum().sum() == 0
    assert np.isclose(out.loc[0, "feature_a"], binary_df["feature_a"].median())


def test_drop_rows_strategy(binary_df):
    out = impute_missing(binary_df, MissingValueStrategy.DROP_ROWS)
    assert out.shape[0] < binary_df.shape[0]
    assert out.isnull().sum().sum() == 0


def test_drop_rows_all_rows_raises():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
    with pytest.raises(ValueError):
        impute_missing(df, MissingValueStrategy.DROP_ROWS)


def test_detect_outliers_iqr():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    mask = detect_outliers(df, ["a"], OutlierDetection.IQR)
    assert mask["a"].sum() == 1  # only 100 is an outlier


def test_detect_outliers_zscore_constant():
    df = pd.DataFrame({"a": [5, 5, 5, 5]})
    mask = detect_outliers(df, ["a"], OutlierDetection.ZSCORE)
    assert mask["a"].sum() == 0  # zero std -> no outliers flagged


def test_outlier_treatment_cap():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    out, desc = apply_outlier_treatment(df, ["a"], OutlierTreatment.CAP)
    assert out["a"].max() < 100
    assert desc


def test_outlier_treatment_remove_requires_mask():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    with pytest.raises(ValueError):
        apply_outlier_treatment(df, ["a"], OutlierTreatment.REMOVE_ROWS)


def test_outlier_treatment_remove_rows():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    mask = detect_outliers(df, ["a"], OutlierDetection.IQR)
    out, _ = apply_outlier_treatment(df, ["a"], OutlierTreatment.REMOVE_ROWS, outlier_mask=mask)
    assert out.shape[0] == 4


def test_outlier_treatment_none_unchanged():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    out, _ = apply_outlier_treatment(df, ["a"], OutlierTreatment.NONE)
    assert out.equals(df)


def test_build_preprocessing_pipeline_end_to_end(titanic_df):
    from fea.config import EncodingMethod, PipelineConfig

    cfg = PipelineConfig(
        missing_strategy=MissingValueStrategy.MEAN_MODE,
        scaling_method=ScalingMethod.STANDARD,
        encoding_method=EncodingMethod.ONE_HOT,
    )
    numeric_cols = ["PassengerId", "Age", "Fare"]
    categorical_cols = ["Sex", "Embarked"]
    pipeline = build_preprocessing_pipeline(cfg, numeric_cols, categorical_cols)

    X = titanic_df.drop(columns=["Survived"])
    out = pipeline.fit_transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == titanic_df.shape[0]
    assert np.all(np.isfinite(out))  # no NaNs after imputation
