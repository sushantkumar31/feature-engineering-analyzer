"""Tests for fea.features and fea.encoding."""

import numpy as np
import pandas as pd

from fea.config import EncodingMethod, FeatureSelectionMethod, PipelineConfig
from fea.encoding import build_categorical_encoder, get_high_cardinality_columns
from fea.features import detect_constant_columns, select_features


def test_detect_constant_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "target": [0, 1, 0]})
    assert detect_constant_columns(df, {"target"}) == ["a"]


def test_get_high_cardinality_columns():
    df = pd.DataFrame({"a": list(range(32)), "b": list("xyzw") * 8})
    assert get_high_cardinality_columns(df, ["a", "b"], threshold=15) == ["a"]


def test_build_categorical_encoder_one_hot():
    enc = build_categorical_encoder(EncodingMethod.ONE_HOT)
    out = enc.fit_transform(pd.DataFrame({"c": ["a", "b", "a"]}))
    assert out.shape == (3, 2)


def test_build_categorical_encoder_label():
    enc = build_categorical_encoder(EncodingMethod.LABEL)
    out = enc.fit_transform(pd.DataFrame({"c": ["a", "b", "a"]}))
    assert out.shape == (3, 1)


def test_select_features_none_keeps_all():
    df = pd.DataFrame(
        {
            "a": np.random.RandomState(0).normal(size=50),
            "b": np.random.RandomState(0).normal(size=50),
            "target": np.random.RandomState(0).choice([0, 1], size=50),
        }
    )
    cols, desc = select_features(
        df, "target", FeatureSelectionMethod.NONE, PipelineConfig(), build_model=lambda: None
    )
    assert set(cols) == {"a", "b"}
    assert desc


def test_select_features_correlation_removes_redundant():
    rng = np.random.RandomState(0)
    a = rng.normal(size=50)
    b = a * 2 + rng.normal(scale=0.001, size=50)  # near-perfect duplicate of a
    df = pd.DataFrame({"a": a, "b": b, "target": rng.choice([0, 1], size=50)})
    cfg = PipelineConfig(correlation_threshold=0.9)
    cols, _ = select_features(df, "target", FeatureSelectionMethod.CORRELATION, cfg, build_model=lambda: None)
    assert len(cols) == 1
