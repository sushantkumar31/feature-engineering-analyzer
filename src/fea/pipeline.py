"""High-level pipeline orchestration.

This module wires the preprocessing pipeline, feature selection and model
together into a single reproducible :class:`sklearn.pipeline.Pipeline` that is
fit on the training split only (avoiding data leakage) and can be exported as a
self-contained artefact for predicting on new data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.pipeline import Pipeline

from fea.config import MissingValueStrategy, PipelineConfig, build_model
from fea.features import build_selector, select_features
from fea.preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)


@dataclass
class EngineeringResult:
    """Output of transforming a raw dataframe into an engineered one."""

    df: pd.DataFrame
    missing_before: int
    missing_after: int
    rows_before: int
    rows_after: int
    features_before: int
    features_after: int
    selection_description: str = ""
    notes: list[str] = field(default_factory=list)


def split_features_target(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Return ``(numeric_feature_cols, categorical_feature_cols)`` excluding target."""
    numeric = [c for c in df.select_dtypes(include="number").columns if c != target]
    categorical = [c for c in df.select_dtypes(exclude="number").columns if c != target]
    return numeric, categorical


def engineer_features(df: pd.DataFrame, target: str, config: PipelineConfig) -> EngineeringResult:
    """Run the full engineering pipeline over ``df`` (impute, encode, scale, select).

    The transformations applied here are computed in-memory for interactive
    exploration. For reproducible deployment use :func:`build_full_pipeline`,
    which fits the same transforms on the training split.
    """
    missing_before = int(df.isnull().sum().sum())
    rows_before = df.shape[0]

    engineered = df.copy()
    numeric_cols, categorical_cols = split_features_target(engineered, target)
    features_before = len(numeric_cols) + len(categorical_cols)

    pipeline = build_preprocessing_pipeline(config, numeric_cols, categorical_cols)
    engineered = pipeline.fit_transform(engineered.drop(columns=[target]))
    engineered = pd.DataFrame(engineered, columns=pipeline.get_feature_names_out())
    engineered[target] = df[target].astype("category").cat.codes

    missing_after = int(engineered.isnull().sum().sum())

    result = EngineeringResult(
        df=engineered,
        missing_before=missing_before,
        missing_after=missing_after,
        rows_before=rows_before,
        rows_after=engineered.shape[0],
        features_before=features_before,
        features_after=engineered.shape[1] - 1,
    )

    if config.feature_selection.value != "None (keep all features)":
        selected, description = select_features(
            engineered,
            target,
            config.feature_selection,
            config,
            build_model=lambda: build_model("Random Forest"),
        )
        engineered = engineered[selected + [target]].reset_index(drop=True)
        result.df = engineered
        result.features_after = len(selected)
        result.selection_description = description

    logger.info("Engineering complete: %s -> %s", (rows_before, features_before), engineered.shape)
    return result


def build_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PipelineConfig,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_name: str,
) -> Pipeline:
    """Build and fit a full ``preprocess -> model`` pipeline on the training split.

    Returns a fitted pipeline whose ``predict``/``predict_proba`` accept a raw
    dataframe (same columns as the training data) and return predictions,
    making it directly serialisable and deployable.
    """
    preprocess = build_preprocessing_pipeline(config, numeric_cols, categorical_cols)
    full_pipeline = Pipeline(steps=[("preprocess", preprocess), ("classifier", build_model(model_name))])
    full_pipeline.fit(X_train, y_train)
    return full_pipeline


def build_model_pipeline(
    config: PipelineConfig,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_name: str,
) -> Pipeline:
    """Compose an un-fitted ``preprocess -> [selector] -> classifier`` pipeline.

    Feature selection is included only when configured and (for RFE) wrapped
    with the requested model. The returned pipeline is fit on the training
    split only, so all statistics (imputation, scaling, selection) are computed
    without leaking test information.
    """
    preprocess = build_preprocessing_pipeline(config, numeric_cols, categorical_cols)
    steps = [("preprocess", preprocess)]

    if config.feature_selection.value != "None (keep all features)":
        selector = build_selector(config.feature_selection, config)
        if selector is None:  # RFE path needs an estimator
            from sklearn.feature_selection import RFE

            estimator = build_model(model_name)
            if not hasattr(estimator, "coef_") and not hasattr(estimator, "feature_importances_"):
                from sklearn.tree import DecisionTreeClassifier

                estimator = DecisionTreeClassifier(random_state=config.random_state)
            selector = RFE(estimator=estimator, n_features_to_select=config.n_features_rfe)
        steps.append(("select", selector))

    steps.append(("classifier", build_model(model_name)))
    return Pipeline(steps=steps)


def build_baseline_pipeline(
    numeric_cols: list[str],
    missing_strategy: MissingValueStrategy,
    model_name: str,
) -> Pipeline:
    """Compose the naive baseline: numeric-only, imputed, un-scaled, un-encoded.

    This mirrors the "raw / no engineering" baseline while still avoiding data
    leakage by fitting the imputer on the training split only.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    numeric_strategy = "mean" if missing_strategy == MissingValueStrategy.MEAN_MODE else "median"
    preprocess = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy=numeric_strategy), numeric_cols)],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", build_model(model_name))])


def final_feature_names(pipeline: Pipeline, X_train: pd.DataFrame) -> list[str]:
    """Return the feature names seen by the classifier step of a fitted pipeline.

    Handles the case where an optional feature-selection step sits between
    preprocessing and the classifier, so importance values align with columns.
    """
    preprocess = pipeline.named_steps["preprocess"]
    names = list(preprocess.get_feature_names_out())

    if "select" in pipeline.named_steps:
        selector = pipeline.named_steps["select"]
        if hasattr(selector, "get_feature_names_out"):
            names = list(selector.get_feature_names_out(names))
        elif hasattr(selector, "support_"):
            names = [n for n, keep in zip(names, selector.support_, strict=False) if keep]

    return names


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted pipeline's preprocessing (and optional selection) to ``X``.

    Returns a numeric dataframe aligned with the classifier's input, suitable
    for cross-model comparison on the *same* engineered features.
    """
    preprocess = pipeline.named_steps["preprocess"]
    Xt = preprocess.transform(X)
    names = final_feature_names(pipeline, X)
    out = pd.DataFrame(Xt, columns=names)

    if "select" in pipeline.named_steps:
        selector = pipeline.named_steps["select"]
        out = selector.transform(out)
    return out
