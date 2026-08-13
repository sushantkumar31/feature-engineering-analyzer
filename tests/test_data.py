"""Tests for fea.data."""

import pandas as pd
import pytest

from fea.config import ProblemType
from fea.data import (
    check_minimum_viability,
    load_dataframe,
    remove_duplicates,
    summarize_dataframe,
    validate_target,
)


def test_validate_target_valid_classification(binary_df):
    result = validate_target(binary_df, "target")
    assert result.is_valid is True
    assert result.problem_type == ProblemType.CLASSIFICATION
    assert result.n_classes == 2


def test_validate_target_constant_fails():
    df = pd.DataFrame({"a": [1, 2, 3], "t": [0, 0, 0]})
    assert validate_target(df, "t").is_valid is False


def test_validate_target_id_column_fails():
    df = pd.DataFrame({"id": range(1000), "t": [0, 1] * 500})
    assert validate_target(df, "id").is_valid is False


def test_validate_target_continuous_numeric_fails():
    df = pd.DataFrame({"x": range(100), "t": [float(i) for i in range(100)]})
    assert validate_target(df, "t").is_valid is False


def test_validate_target_missing_column():
    assert validate_target(pd.DataFrame({"a": [1]}), "missing").is_valid is False


def test_remove_duplicates(binary_df):
    cleaned, removed = remove_duplicates(binary_df)
    assert removed == 2
    assert cleaned.shape[0] == binary_df.shape[0] - 2


def test_remove_duplicates_none():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cleaned, removed = remove_duplicates(df)
    assert removed == 0
    assert cleaned.equals(df)


def test_summarize_dataframe(binary_df):
    summary = summarize_dataframe(binary_df)
    assert set(summary["Column"]) == set(binary_df.columns)
    assert "Missing" in summary.columns


def test_check_minimum_viability():
    ok, _ = check_minimum_viability(pd.DataFrame({"a": range(50), "b": range(50)}))
    assert ok is True
    bad, msg = check_minimum_viability(pd.DataFrame({"a": [1, 2]}))
    assert bad is False
    assert msg is not None


def test_load_dataframe_rejects_single_column(tmp_path):
    csv = tmp_path / "one.csv"
    csv.write_text("col\n1\n2\n3\n")
    with open(csv, "rb") as fh, pytest.raises(ValueError):
        load_dataframe(fh)
