"""Tests for fea.visualization."""

import numpy as np
import pandas as pd

from fea.visualization import confusion_matrix_figure, correlation_heatmap


def test_correlation_heatmap_returns_figure():
    df = pd.DataFrame(np.random.RandomState(0).normal(size=(50, 4)), columns=list("abcd"))
    fig = correlation_heatmap(df.corr())
    assert fig.get_axes()  # has at least one axes


def test_confusion_matrix_figure_returns_figure():
    cm = pd.DataFrame([[10, 2], [3, 8]], index=[0, 1], columns=[0, 1])
    fig = confusion_matrix_figure(cm)
    assert fig.get_axes()
