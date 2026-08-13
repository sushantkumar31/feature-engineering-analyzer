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


def test_figure_text_color_follows_theme():
    df = pd.DataFrame(np.random.RandomState(0).normal(size=(50, 4)), columns=list("abcd"))
    for text_color in ("#fafafa", "#31333f"):
        fig = correlation_heatmap(df.corr(), text_color=text_color)
        ticks = fig.axes[0].get_xticklabels() + fig.axes[0].get_yticklabels()
        assert ticks, "expected tick labels"
        assert all(t.get_color() == text_color for t in ticks), text_color


def test_annotation_text_contrasts_with_cell():
    df = pd.DataFrame(
        [[1.0, 0.0], [0.0, -1.0]], index=["a", "b"], columns=["a", "b"]
    )
    fig = correlation_heatmap(df)
    ax = fig.axes[0]
    colors = {t.get_color() for t in ax.texts}
    assert colors, "expected annotation text"
    # cells span both light (centre) and dark (extreme) regions, so both
    # black-ish and white annotations must be used for contrast.
    assert len(colors) == 2
