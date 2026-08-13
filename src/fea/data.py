"""Data loading, validation and summarisation helpers.

These functions are deliberately free of any Streamlit dependency so they can
be unit-tested in isolation and reused outside the UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import BinaryIO

import pandas as pd

from fea.config import (
    MAX_CLASSES,
    MAX_ID_UNIQUE_RATIO,
    MIN_CLASS_RATIO_WARNING,
    MIN_ROWS_FOR_MODEL,
    SMALL_DATASET_ROWS,
    ProblemType,
)

logger = logging.getLogger(__name__)


@dataclass
class TargetValidation:
    """Result of validating a column as a classification target."""

    is_valid: bool = False
    problem_type: ProblemType | None = None
    n_classes: int | None = None
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.messages.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def load_dataframe(stream: BinaryIO) -> pd.DataFrame:
    """Read a CSV from a file-like object into a :class:`pandas.DataFrame`."""
    df = pd.read_csv(stream)
    if df.empty:
        raise ValueError("The uploaded file contains no rows.")
    if df.shape[1] < 2:
        raise ValueError("The uploaded file contains only one column; at least two are required.")
    logger.info("Loaded dataframe with shape %s", df.shape)
    return df


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column summary table (dtype, missing count, uniqueness)."""
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str).values,
            "Missing": df.isnull().sum().values,
            "Unique values": [df[c].nunique() for c in df.columns],
        }
    )


def validate_target(df: pd.DataFrame, target: str) -> TargetValidation:
    """Validate that ``target`` is usable as a classification label column.

    Returns a :class:`TargetValidation` describing whether training is viable,
    plus any errors/warnings that should be surfaced to the user.
    """
    result = TargetValidation()

    if target not in df.columns:
        result.add_error(f"Target column '{target}' does not exist in the dataset.")
        return result

    series = df[target]
    n_unique = series.nunique()
    n_missing = series.isnull().sum()
    n_rows = df.shape[0]

    if n_unique <= 1:
        result.add_error("This column has only one unique value — can't train a model on a constant target.")
        return result

    if n_unique > MAX_CLASSES and n_unique > MAX_ID_UNIQUE_RATIO * n_rows:
        result.add_error(
            f"This column has {n_unique} unique values out of {n_rows} rows — "
            f"it looks like an ID column, not a label. Pick a different target."
        )
        return result

    if pd.api.types.is_numeric_dtype(series) and n_unique > MAX_CLASSES:
        result.add_error(
            f"This column has {n_unique} unique numeric values — it looks like a continuous "
            f"(regression) target. This app only supports classification targets with a small "
            f"number of distinct classes (like 0/1 or Win/Loss)."
        )
        return result

    result.problem_type = ProblemType.CLASSIFICATION
    result.n_classes = n_unique
    result.is_valid = True

    if n_missing > 0.5 * n_rows:
        result.add_warning(f"Target is {n_missing / n_rows:.0%} missing — results may be unreliable.")
    if n_rows < SMALL_DATASET_ROWS:
        result.add_warning(
            f"Small dataset ({n_rows} rows) — accuracy numbers may be unstable across splits."
        )
    if n_unique > 1 and (series.value_counts().min() / series.value_counts().max()) < MIN_CLASS_RATIO_WARNING:
        result.add_warning("Class imbalance detected — accuracy alone can be misleading; check F1/Recall.")

    return result


def check_minimum_viability(df: pd.DataFrame) -> tuple[bool, str | None]:
    """Return ``(ok, error_message)`` for whether a model can reasonably be trained."""
    if df.shape[0] < MIN_ROWS_FOR_MODEL:
        return False, f"Not enough rows ({df.shape[0]}) to train a reliable model."
    if df.shape[1] < 2:
        return False, "Not enough columns to train a model."
    return True, None


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows, returning ``(cleaned_df, number_removed)``."""
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        cleaned = df.drop_duplicates().reset_index(drop=True)
        logger.info("Removed %d duplicate rows", n_duplicates)
        return cleaned, n_duplicates
    return df, 0
