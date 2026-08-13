"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_df() -> pd.DataFrame:
    """A small classification dataset with missing values and a categorical column."""
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(size=n),
            "category": rng.choice(["x", "y", "z"], size=n),
            "target": rng.choice([0, 1], size=n, p=[0.4, 0.6]),
        }
    )
    df.loc[0, "feature_a"] = np.nan
    df.loc[1, "category"] = np.nan
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # duplicate rows
    return df


@pytest.fixture
def titanic_df() -> pd.DataFrame:
    """A tiny hand-built Titanic-like frame used for end-to-end pipeline tests."""
    return pd.DataFrame(
        {
            "PassengerId": range(1, 9),
            "Age": [22, 38, np.nan, 35, np.nan, 54, 2, 27],
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 51.86, 21.07, 11.5],
            "Sex": ["male", "female", "female", "female", "male", "male", "male", "female"],
            "Embarked": ["S", "C", "S", "S", "S", "C", "S", np.nan],
            "Survived": [0, 1, 1, 1, 0, 0, 0, 1],
        }
    )
