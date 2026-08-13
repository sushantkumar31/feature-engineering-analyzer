"""Model training and evaluation helpers.

Everything here operates on fully-preprocessed, numeric data (encoded target,
scaled/encoded features) so the model layer stays focused and testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from fea.config import PipelineConfig, build_model

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Bundle of a trained model and its evaluation metrics."""

    model: object
    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    confusion_matrix: np.ndarray
    report: str
    cv_mean: float | None
    cv_std: float | None
    cv_scores: list[float] | None
    feature_names: list[str]


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split when the class distribution permits it."""
    y = pd.Series(y).reset_index(drop=True)
    stratify = y if (y.nunique() < 20 and y.value_counts().min() >= 2) else None
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )


def _average_method(n_classes: int) -> str:
    return "binary" if n_classes == 2 else "weighted"


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    config: PipelineConfig,
) -> ModelResult:
    """Fit ``model`` on the training split and compute metrics on the test split."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    n_classes = y_test.nunique()
    avg = _average_method(n_classes)

    precision = precision_score(y_test, preds, average=avg, zero_division=0)
    recall = recall_score(y_test, preds, average=avg, zero_division=0)
    f1 = f1_score(y_test, preds, average=avg, zero_division=0)

    roc_auc = None
    if n_classes == 2 and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ROC-AUC could not be computed: %s", exc)

    cv_mean, cv_std, cv_scores = cross_validate(model, X_train, y_train, config)

    return ModelResult(
        model=model,
        accuracy=float(accuracy_score(y_test, preds)),
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(y_test, preds),
        report=classification_report(y_test, preds, zero_division=0),
        cv_mean=cv_mean,
        cv_std=cv_std,
        cv_scores=cv_scores,
        feature_names=feature_names,
    )


def cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
    cv: int = 5,
) -> tuple[float | None, float | None, list[float] | None]:
    """Run k-fold cross-validation returning ``(mean, std, scores)`` or ``None``s.

    Uses :func:`sklearn.model_selection.cross_val_score`, which clones the
    estimator per fold (so pipelines — including nested ``preprocess`` steps —
    are handled correctly).
    """
    if X.shape[0] < cv or y.nunique() < 2:
        return None, None, None
    try:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.random_state)
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        return float(scores.mean()), float(scores.std()), [float(s) for s in scores]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Cross-validation failed: %s", exc)
        return None, None, None


def compare_models(
    model_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    avg: str,
) -> pd.DataFrame:
    """Train every model on the same data and return a ranked accuracy/F1 table."""
    rows = []
    for name in model_names:
        try:
            m = build_model(name)
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, preds),
                    "F1 Score": f1_score(y_test, preds, average=avg, zero_division=0),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Model '%s' failed: %s", name, exc)
            rows.append({"Model": name, "Accuracy": None, "F1 Score": None})
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False, na_position="last")


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return a ranked importance/coefficient table for the fitted model."""
    if hasattr(model, "feature_importances_"):
        return (
            pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        return (
            pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
            .sort_values("Coefficient", key=abs, ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame()
