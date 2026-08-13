"""Matplotlib/Seaborn figure builders.

Functions return ``matplotlib.figure.Figure`` objects so the caller (Streamlit)
is free to render them. Keeping figure construction here makes it testable and
keeps the UI layer thin.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for figure-only use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Transparent figure backgrounds so charts blend with both light and dark
# application themes. Text colors are applied per-call so figures match the
# theme they are rendered into.
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["savefig.facecolor"] = "none"

ANNOT_TEXT_DARK = "#ffffff"
ANNOT_TEXT_LIGHT = "#111214"


def _style_axes(ax: plt.Axes) -> None:
    """Make an axes background transparent and neutralise its spines."""
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def _style_figure_text(fig: plt.Figure, text_color: str) -> None:
    """Colour ticks, labels and titles (and any colorbar) to match the theme."""
    for ax in fig.axes:
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        if ax.title is not None:
            ax.title.set_color(text_color)
        if ax.collections:
            colorbar = ax.collections[0].colorbar
            if colorbar is not None:
                colorbar.ax.tick_params(colors=text_color)


def _color_annotations(ax: plt.Axes) -> None:
    """Colour each heatmap annotation for contrast against its own cell.

    Seaborn defaults to white annotation text, which vanishes on light cells
    (e.g. the centre of the ``coolwarm`` colormap). This picks black or white
    per cell based on the cell's background luminance.
    """
    quad = ax.collections[0]
    if quad is None or quad.get_array() is None:
        return
    cmap = quad.get_cmap()
    norm = plt.Normalize(*quad.get_clim())
    data = np.asarray(quad.get_array())
    for text in ax.texts:
        x, y = text.get_position()
        row, col = int(np.floor(y)), int(np.floor(x))
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            value = data[row, col]
            if value is np.ma.masked or np.isnan(value):
                continue
            r, g, b, *_ = cmap(norm(float(value)))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text.set_color(ANNOT_TEXT_LIGHT if luminance > 0.55 else ANNOT_TEXT_DARK)


def correlation_heatmap(corr: pd.DataFrame, text_color: str = "#31333f") -> plt.Figure:
    """Heatmap of a correlation matrix."""
    fig_size = max(6, min(0.4 * corr.shape[1], 20))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot=corr.shape[1] <= 12,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    _style_axes(ax)
    ax.set_title("Correlation Heatmap (Engineered Data)")
    _color_annotations(ax)
    _style_figure_text(fig, text_color)
    fig.tight_layout()
    return fig


def confusion_matrix_figure(cm: pd.DataFrame, text_color: str = "#31333f") -> plt.Figure:
    """Heatmap of a confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    _style_axes(ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _color_annotations(ax)
    _style_figure_text(fig, text_color)
    fig.tight_layout()
    return fig
