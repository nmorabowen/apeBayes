"""Convergence-diagnostic visualizations: R-hat bar, ESS bar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .helpers import ensure_dir, savefig
from .style import HALF_WIDTH, PALETTE, fs

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def plot_rhat_bar(
    rhat_df: pd.DataFrame,
    *,
    top_n: int | None = 10,
    threshold: float = 1.01,
    figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "rhat_bar.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    r"""Horizontal bar chart of R-hat (max per parameter group).

    Parameters
    ----------
    rhat_df : DataFrame
        Output of ``rhat_table(aggregate=True)`` with columns
        ``parameter, r_hat_max``.
    top_n : int or None
        Show only the *top_n* worst parameters.  ``None`` shows all.
    threshold : float
        R-hat threshold (dashed line).
    """
    out_dir = ensure_dir(out_dir)
    df = rhat_df.sort_values("r_hat_max", ascending=False).copy()
    if top_n is not None:
        df = df.head(top_n)

    df = df.sort_values("r_hat_max", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    x = df["r_hat_max"].to_numpy(dtype=float)
    colors = np.where(x > threshold, PALETTE[3], PALETTE[0])  # amber = bad, navy = good

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, x, color=colors, alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(threshold, color="0.3", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"].astype(str).values, fontsize=fs(-1))
    ax.set_xlabel(r"$\hat{R}$")
    xmax = max(float(np.nanmax(x)), threshold) + 0.005
    ax.set_xlim(0.99, xmax)
    ax.set_title("R-hat diagnostics")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


def plot_ess_bar(
    ess_df: pd.DataFrame,
    *,
    kind: str = "bulk",
    top_n: int | None = 10,
    threshold: float = 400.0,
    figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    r"""Horizontal bar chart of ESS (min per parameter group).

    Parameters
    ----------
    ess_df : DataFrame
        Output of ``ess_table(aggregate=True)``.
    kind : {"bulk", "tail"}
        Which ESS column to plot.
    """
    out_dir = ensure_dir(out_dir)
    col = f"ess_{kind}_min"
    if col not in ess_df.columns:
        raise KeyError(f"Column '{col}' not found.  Available: {list(ess_df.columns)}")

    df = ess_df.sort_values(col, ascending=True).copy()
    if top_n is not None:
        df = df.head(top_n)
    df = df.sort_values(col, ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    x = df[col].to_numpy(dtype=float)
    colors = np.where(x < threshold, PALETTE[3], PALETTE[0])  # amber = bad, navy = good

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, x, color=colors, alpha=0.85, edgecolor="white", lw=0.3)
    ax.axvline(threshold, color="0.3", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"].astype(str).values, fontsize=fs(-1))
    ax.set_xlabel(f"ESS {kind}")
    ax.set_xlim(0, max(float(np.nanmax(x)), threshold) * 1.1)
    ax.set_title(f"ESS {kind} diagnostics")

    fname = filename or f"ess_{kind}_bar.pdf"
    savefig(fig, out_dir, fname, prefix=prefix)
    return fig, ax
