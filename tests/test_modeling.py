"""Tests for fea.modeling and fea.pipeline."""

import numpy as np
import pandas as pd
import pytest

from fea.config import (
    EncodingMethod,
    FeatureSelectionMethod,
    MissingValueStrategy,
    PipelineConfig,
    ScalingMethod,
)
from fea.modeling import compare_models, evaluate_model, feature_importance, split_data
from fea.pipeline import build_full_pipeline, engineer_features


def _prepare_engineered():
    rng = np.random.RandomState(1)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200),
            "c": rng.choice(["p", "q", "r"], size=200),
            "target": rng.choice([0, 1], size=200),
        }
    )
    cfg = PipelineConfig(
        missing_strategy=MissingValueStrategy.MEAN_MODE,
        scaling_method=ScalingMethod.STANDARD,
        encoding_method=EncodingMethod.ONE_HOT,
    )
    result = engineer_features(df, "target", cfg)
    return result.df, "target", cfg


def test_engineer_features_no_missing(titanic_df):
    cfg = PipelineConfig(
        missing_strategy=MissingValueStrategy.MEAN_MODE,
        encoding_method=EncodingMethod.ONE_HOT,
    )
    result = engineer_features(titanic_df, "Survived", cfg)
    assert result.df.isnull().sum().sum() == 0
    assert result.missing_after == 0
    assert "Survived" in result.df.columns
    assert result.df["Survived"].dtype in ("int8", "int16", "int32", "int64")


def test_engineer_features_one_hot_increases_columns(titanic_df):
    cfg = PipelineConfig(
        missing_strategy=MissingValueStrategy.MEAN_MODE,
        encoding_method=EncodingMethod.ONE_HOT,
    )
    result = engineer_features(titanic_df, "Survived", cfg)
    assert result.features_after > result.features_before  # one-hot expands columns


def test_split_data_stratified():
    df, target, cfg = _prepare_engineered()
    X, X_test, y, y_test = split_data(df.drop(columns=[target]), df[target], cfg)
    assert len(X) + len(X_test) == len(df)
    assert set(y.unique()) <= set(df[target].unique())


def test_evaluate_model_returns_metrics():
    df, target, cfg = _prepare_engineered()
    from fea.config import build_model

    X, X_test, y, y_test = split_data(df.drop(columns=[target]), df[target], cfg)
    result = evaluate_model(
        build_model("Random Forest"),
        X,
        y,
        X_test,
        y_test,
        feature_names=list(X.columns),
        config=cfg,
    )
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert result.confusion_matrix.shape == (2, 2)
    assert result.cv_mean is not None


def test_feature_importance_ranks():
    df, target, cfg = _prepare_engineered()
    from fea.config import build_model

    X, X_test, y, y_test = split_data(df.drop(columns=[target]), df[target], cfg)
    model = build_model("Decision Tree")
    model.fit(X, y)
    imp = feature_importance(model, list(X.columns))
    assert len(imp) == len(X.columns)
    assert imp["Importance"].is_monotonic_decreasing


def test_compare_models_returns_ranked_frame():
    df, target, cfg = _prepare_engineered()
    X, X_test, y, y_test = split_data(df.drop(columns=[target]), df[target], cfg)
    comp = compare_models(["Logistic Regression", "Decision Tree"], X, y, X_test, y_test, avg="binary")
    assert list(comp.columns) == ["Model", "Accuracy", "F1 Score"]
    assert comp["Accuracy"].is_monotonic_decreasing


def test_build_full_pipeline_predicts_raw_frame(titanic_df):
    cfg = PipelineConfig(
        missing_strategy=MissingValueStrategy.MEAN_MODE,
        encoding_method=EncodingMethod.ONE_HOT,
    )
    numeric_cols = ["PassengerId", "Age", "Fare"]
    categorical_cols = ["Sex", "Embarked"]
    X = titanic_df.drop(columns=["Survived"])
    y = titanic_df["Survived"]
    pipe = build_full_pipeline(X, y, cfg, numeric_cols, categorical_cols, "Logistic Regression")
    preds = pipe.predict(X)
    assert preds.shape == (titanic_df.shape[0],)
    assert set(np.unique(preds)) <= {0, 1}


@pytest.mark.parametrize(
    "fs_method",
    [
        FeatureSelectionMethod.SELECT_K_BEST,
        FeatureSelectionMethod.CORRELATION,
        FeatureSelectionMethod.RFE,
    ],
)
def test_build_model_pipeline_feature_selection_names_align(titanic_df, fs_method):
    """Feature names after selection must match the classifier's input width."""
    from fea.pipeline import build_model_pipeline, final_feature_names

    cfg = PipelineConfig(
        missing_strategy=MissingValueStrategy.MEAN_MODE,
        encoding_method=EncodingMethod.ONE_HOT,
        feature_selection=fs_method,
        k_best=2,
        n_features_rfe=2,
    )
    numeric_cols = ["PassengerId", "Age", "Fare"]
    categorical_cols = ["Sex", "Embarked"]
    X = titanic_df.drop(columns=["Survived"])
    y = titanic_df["Survived"]

    pipe = build_model_pipeline(cfg, numeric_cols, categorical_cols, "Random Forest")
    pipe.fit(X, y)
    names = final_feature_names(pipe, X)
    classifier = pipe.named_steps["classifier"]
    assert len(names) == classifier.n_features_in_


def test_cross_validate_works_with_pipeline():
    """Regression: CV must clone pipelines (with preprocess steps) correctly."""
    import numpy as np

    from fea.modeling import cross_validate
    from fea.pipeline import build_model_pipeline

    rng = np.random.RandomState(7)
    raw = pd.DataFrame(
        {
            "age": rng.normal(size=120),
            "fare": rng.exponential(size=120),
            "sex": rng.choice(["m", "f"], size=120),
            "survived": rng.choice([0, 1], size=120),
        }
    )
    cfg = PipelineConfig(
        missing_strategy=MissingValueStrategy.MEAN_MODE,
        encoding_method=EncodingMethod.ONE_HOT,
    )
    numeric_cols = ["age", "fare"]
    categorical_cols = ["sex"]
    X = raw.drop(columns=["survived"])
    y = raw["survived"]

    pipe = build_model_pipeline(cfg, numeric_cols, categorical_cols, "Logistic Regression")
    mean, std, scores = cross_validate(pipe, X, y, cfg)
    assert mean is not None
    assert 0.0 <= mean <= 1.0
    assert std is not None
    assert len(scores) == 5
