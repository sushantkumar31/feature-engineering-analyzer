"""Feature selection: variance, correlation, statistical and model-based methods."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)

from fea.config import FeatureSelectionMethod, PipelineConfig

logger = logging.getLogger(__name__)


class CorrelationThreshold(BaseEstimator, TransformerMixin):
    """Drop features that are highly correlated with any other feature.

    Keeps the first feature of each correlated pair (or group), mirroring a
    common collinearity-removal heuristic. Exposed as a scikit-learn
    transformer so it can be composed inside a :class:`sklearn.pipeline.Pipeline`.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.feature_names_in_ = X.columns.tolist()
        self.support_ = np.array([c not in drop for c in self.feature_names_in_])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            drop = [c for c, keep in zip(self.feature_names_in_, self.support_, strict=False) if not keep]
            return X.drop(columns=drop, errors="ignore")
        # ndarray path (e.g. output of a ColumnTransformer)
        return X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else self.feature_names_in_
        return [n for n, keep in zip(names, self.support_, strict=False) if keep]


def build_selector(method: FeatureSelectionMethod, config: PipelineConfig):
    """Return a fitted-on-fit feature-selection transformer for the chosen method.

    ``None`` is returned (meaning "no selector") when the method is ``NONE`` so
    a caller can conditionally add it to a pipeline.
    """
    if method == FeatureSelectionMethod.NONE:
        return None
    if method == FeatureSelectionMethod.VARIANCE:
        return VarianceThreshold(threshold=config.variance_threshold)
    if method == FeatureSelectionMethod.CORRELATION:
        return CorrelationThreshold(threshold=config.correlation_threshold)
    if method == FeatureSelectionMethod.SELECT_K_BEST:
        return SelectKBest(score_func=f_classif, k=config.k_best)
    if method == FeatureSelectionMethod.MUTUAL_INFO:
        return SelectKBest(score_func=mutual_info_classif, k=config.k_best)
    if method == FeatureSelectionMethod.RFE:
        return None  # RFE needs an estimator and is handled separately
    raise ValueError(f"Unknown feature selection method: {method!r}")


def detect_constant_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Return feature columns (excluding ``exclude``) that have no variation."""
    cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    return [c for c in cols if df[c].nunique() <= 1]


def select_features(
    df: pd.DataFrame,
    target: str,
    method: FeatureSelectionMethod,
    config: PipelineConfig,
    build_model: callable,
) -> tuple[list[str], str]:
    """Select a subset of feature columns using the chosen method.

    Parameters
    ----------
    df:
        Dataframe with numeric features and the encoded target column.
    target:
        Name of the target column (must be numeric).
    method:
        The feature selection strategy to apply.
    config:
        Pipeline configuration holding thresholds / ``k``.
    build_model:
        Callable returning an unfitted classifier (used by RFE).

    Returns
    -------
    ``(selected_columns, description)``.
    """
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target]

    if len(feature_cols) < 2:
        return feature_cols, "Not enough numeric features to run feature selection."

    if method == FeatureSelectionMethod.NONE:
        return feature_cols, "All features kept."

    X = df[feature_cols]
    y = df[target]
    description: str = method.value

    if method == FeatureSelectionMethod.VARIANCE:
        threshold = config.variance_threshold
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X)
        selected_cols = list(X.columns[vt.get_support()])
        description = f"Variance Threshold (>= {threshold:g})"

    elif method == FeatureSelectionMethod.CORRELATION:
        threshold = config.correlation_threshold
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        selected_cols = [c for c in feature_cols if c not in to_drop]
        description = f"Correlation-based removal (threshold {threshold:g}); dropped {len(to_drop)}."

    elif method == FeatureSelectionMethod.SELECT_K_BEST:
        k = min(config.k_best, len(feature_cols))
        skb = SelectKBest(score_func=f_classif, k=k)
        skb.fit(X, y)
        selected_cols = list(X.columns[skb.get_support()])
        description = f"SelectKBest (ANOVA F-test), k={k}."

    elif method == FeatureSelectionMethod.MUTUAL_INFO:
        k = min(config.k_best, len(feature_cols))
        skb = SelectKBest(score_func=mutual_info_classif, k=k)
        skb.fit(X, y)
        selected_cols = list(X.columns[skb.get_support()])
        description = f"Mutual Information, k={k}."

    elif method == FeatureSelectionMethod.RFE:
        n = min(config.n_features_rfe, len(feature_cols))
        estimator = build_model()
        if not hasattr(estimator, "coef_") and not hasattr(estimator, "feature_importances_"):
            from sklearn.tree import DecisionTreeClassifier

            estimator = DecisionTreeClassifier(random_state=config.random_state)
        rfe = RFE(estimator=estimator, n_features_to_select=n)
        rfe.fit(X, y)
        selected_cols = list(X.columns[rfe.support_])
        description = f"RFE, {n} features."

    else:
        raise ValueError(f"Unknown feature selection method: {method!r}")

    logger.info(
        "Feature selection '%s' kept %d/%d columns", method.value, len(selected_cols), len(feature_cols)
    )
    return selected_cols, description
