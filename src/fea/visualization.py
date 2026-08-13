"""Matplotlib/Seaborn figure builders.

Functions return ``matplotlib.figure.Figure`` objects so the caller (Streamlit)
is free to render them. Keeping figure construction here makes it testable and
keeps the UI layer thin.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for figure-only use
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Transparent figure backgrounds so charts blend with both light and dark
# application themes.
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["savefig.facecolor"] = "none"


def _style_axes(ax: plt.Axes) -> None:
    """Make an axes background transparent and apply neutral text colouring."""
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def correlation_heatmap(corr: pd.DataFrame) -> plt.Figure:
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
    fig.tight_layout()
    return fig


def confusion_matrix_figure(cm: pd.DataFrame) -> plt.Figure:
    """Heatmap of a confusion matrix."""
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    _style_axes(ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig
