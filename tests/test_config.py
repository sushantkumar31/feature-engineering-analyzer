"""Tests for fea.config."""

import pytest

from fea.config import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model_all_registered_models(name):
    model = build_model(name)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("Not A Model")


def test_all_model_names_unique():
    assert len(set(MODEL_NAMES)) == len(MODEL_NAMES)
