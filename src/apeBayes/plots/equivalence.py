"""Equivalence visualizations: probability matrices, dendrograms, α-sweeps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage, set_link_color_palette
from scipy.spatial.distance import squareform

from .helpers import case_color, case_colors_for_labels, ensure_dir, order_config_labels, savefig
from .style import CMAP_SEQ, FULL_WIDTH, HALF_WIDTH, PALETTE, fs

if TYPE_CHECKING:
    from pathlib import Path


def plot_equivalence_matrix(
    labels: list[str],
    prob_matrix: np.ndarray,
    *,
    alpha: float = 0.5,
    annot: bool = True,
    fmt: str = ".2f",
    figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
    cmap: str = CMAP_SEQ,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "equivalence_matrix.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of pairwise P(|Δμ| < α·σ_run)."""
    out_dir = ensure_dir(out_dir)

    df = pd.DataFrame(prob_matrix, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df, annot=annot, fmt=fmt, vmin=0, vmax=1,
        cmap=cmap, ax=ax, square=True,
        cbar_kws={"label": rf"$P(|\Delta\mu| < {alpha}\,\sigma_{{\mathrm{{run}}}})$"},
    )
    ax.set_title(rf"Epistemic equivalence ($\alpha = {alpha}$)")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


def plot_equivalence_dendrogram(
    labels: list[str],
    prob_matrix: np.ndarray,
    *,
    alpha: float = 0.5,
    method: str = "average",
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "equivalence_dendrogram.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Dendrogram from epistemic distance matrix D = 1 − P_equiv."""
    out_dir = ensure_dir(out_dir)

    D = np.clip(1.0 - prob_matrix, 0, 1)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)

    set_link_color_palette(PALETTE[:6])

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        Z, labels=labels, ax=ax,
        leaf_rotation=45, leaf_font_size=7,
        above_threshold_color="0.5",
    )
    ax.set_ylabel(rf"Epistemic distance ($1 - P_{{\mathrm{{equiv}}}},\;\alpha={alpha}$)")
    ax.set_title("Epistemic clustering")

    set_link_color_palette(None)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


def plot_equivalence_sweep(
    sweep_df: pd.DataFrame,
    *,
    label_col: str = "Config",
    figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "equivalence_sweep.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of P_equiv vs α for each configuration.

    ``sweep_df`` must have columns: Config, alpha, P_equiv.
    """
    out_dir = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=figsize)

    configs = sweep_df[label_col].unique()
    for cfg in configs:
        sub = sweep_df[sweep_df[label_col] == cfg].sort_values("alpha")
        ax.plot(sub["alpha"], sub["P_equiv"],
                label=cfg, color=case_color(cfg), lw=1.4)

    ax.set_xlabel(r"Threshold $\alpha$")
    ax.set_ylabel(r"$P(|\Delta\mu| < \alpha\,\sigma_{\mathrm{run}})$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=fs(-2), ncol=2)
    ax.set_title("Equivalence probability sweep")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


def plot_equivalence_matrix_with_dendrogram(
    labels: list[str],
    prob_matrix: np.ndarray,
    *,
    alpha: float = 0.5,
    method: str = "average",
    cluster_order: bool = True,
    annot: bool = True,
    fmt: str = ".2f",
    figsize: tuple[float, float] = (FULL_WIDTH, 5.0),
    cmap: str = CMAP_SEQ,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "equivalence_matrix_dendro.pdf",
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """Equivalence heatmap with dendrogram overlay (v7 style).

    Returns (fig, axes_array, reorder_indices).
    """
    out_dir = ensure_dir(out_dir)

    D = np.clip(1.0 - prob_matrix, 0, 1)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)

    if cluster_order:
        order = leaves_list(Z)
    else:
        order = np.arange(len(labels))

    reordered_labels = [labels[i] for i in order]
    reordered_P = prob_matrix[np.ix_(order, order)]

    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 3]},
        constrained_layout=True,
    )

    # Left panel: dendrogram
    set_link_color_palette(PALETTE[:6])
    dendrogram(
        Z, orientation="left", labels=[labels[i] for i in range(len(labels))],
        ax=axes[0], leaf_font_size=7, above_threshold_color="0.5",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel(r"$1 - P_{\mathrm{equiv}}$", fontsize=fs(-1))
    set_link_color_palette(None)

    # Right panel: heatmap
    df_heat = pd.DataFrame(reordered_P, index=reordered_labels, columns=reordered_labels)
    n_labels = len(reordered_labels)
    _annot_size = max(4, 7 - (n_labels - 8) * 0.3)  # scale down for large matrices
    sns.heatmap(
        df_heat, annot=annot, fmt=fmt, vmin=0, vmax=1,
        cmap=cmap, ax=axes[1], square=True,
        annot_kws={"size": _annot_size},
        cbar_kws={"label": rf"$P(|\Delta\mu| < {alpha}\,\sigma_{{\mathrm{{run}}}})$",
                   "shrink": 0.7},
    )
    axes[1].set_title(rf"Epistemic equivalence ($\alpha={alpha}$)")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes, order


def plot_equivalence_matrix_comparison(
    labels: list[str],
    prob_matrix: np.ndarray,
    *,
    alpha: float = 0.4,
    orders: tuple[str, ...] | list[str] = ("similarity", "tier", "case"),
    method: str = "average",
    annot: bool = False,
    fmt: str = ".2f",
    figsize: tuple[float, float] | None = None,
    cmap: str = CMAP_SEQ,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "equivalence_matrix_comparison.pdf",
) -> tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]:
    r"""Side-by-side equivalence heatmaps under different label orderings.

    Produces one panel per ordering in *orders*, sharing a single
    colourbar on the rightmost panel.  Supported orderings:

    ``"similarity"``
        Hierarchical clustering (UPGMA by default) reorders rows/columns
        so that epistemically close configs are adjacent.
    ``"tier"``
        Deterministic tier-major sort (1A, 1B, …, 2A, 2B, …).
    ``"case"``
        Deterministic case-major sort (1A, 2A, 3A, …, 1B, 2B, …).

    Parameters
    ----------
    labels : list[str]
        Configuration labels aligned with *prob_matrix* rows/columns.
    prob_matrix : ndarray, shape (N, N)
        Pairwise P(|Δμ_i − Δμ_j| < α·σ_denom).
    alpha : float
        Equivalence radius (used only in titles/colourbar label).
    orders : sequence of str
        Which orderings to show (one subplot each).
    method : str
        Linkage method for ``"similarity"`` ordering.
    annot, fmt : bool, str
        Passed to ``sns.heatmap``.
    figsize : tuple, optional
        If None, defaults to ``(FULL_WIDTH, FULL_WIDTH / len(orders))``.
    cmap : str
        Colourmap name.

    Returns
    -------
    fig, axes, P_dfs : Figure, ndarray of Axes, dict[str, DataFrame]
        ``P_dfs`` maps each ordering key to its reordered DataFrame.
    """
    out_dir = ensure_dir(out_dir)
    n_panels = len(orders)
    if figsize is None:
        figsize = (FULL_WIDTH, FULL_WIDTH / n_panels + 0.6)

    cbar_label = rf"$P(|\Delta\mu| \leq {alpha:g}\,\sigma_{{\mathrm{{run}}}})$"

    def _reorder(order_key: str):
        """Return (reordered_labels, reordered_P)."""
        labs = list(labels)
        P = np.asarray(prob_matrix, dtype=float).copy()

        if order_key == "similarity":
            D = np.clip(1.0 - P, 0, 1)
            np.fill_diagonal(D, 0.0)
            D = (D + D.T) / 2.0
            condensed = squareform(D, checks=False)
            Z = linkage(condensed, method=method)
            idx = leaves_list(Z)
        elif order_key in {"tier", "case"}:
            sorted_labs = order_config_labels(labs, by=order_key)
            pos = {lab: i for i, lab in enumerate(labs)}
            idx = [pos[lab] for lab in sorted_labs]
        else:
            raise ValueError(
                f"Unknown order_key {order_key!r}; expected "
                "'similarity', 'tier', or 'case'."
            )

        P_out = P[np.ix_(idx, idx)]
        labs_out = [labs[i] for i in idx]
        return labs_out, P_out

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize,
        constrained_layout=True, squeeze=False,
    )
    axes_row = axes[0]

    P_dfs: dict[str, pd.DataFrame] = {}
    for j, ok in enumerate(orders):
        labs_j, P_j = _reorder(ok)
        df_j = pd.DataFrame(P_j, index=labs_j, columns=labs_j)
        P_dfs[ok] = df_j

        is_last = j == n_panels - 1
        n = len(labs_j)
        _annot_size = max(4, 7 - (n - 8) * 0.3)

        sns.heatmap(
            df_j,
            vmin=0.0, vmax=1.0,
            cmap=cmap, square=True,
            annot=annot, fmt=fmt,
            annot_kws={"size": _annot_size} if annot else {},
            cbar=is_last,
            cbar_kws={"label": cbar_label, "shrink": 0.8} if is_last else {},
            ax=axes_row[j],
        )
        axes_row[j].set_xticks(np.arange(n) + 0.5)
        axes_row[j].set_xticklabels(labs_j, rotation=90)
        axes_row[j].set_yticks(np.arange(n) + 0.5)
        axes_row[j].set_yticklabels(labs_j)
        axes_row[j].set_aspect("equal", adjustable="box")
        axes_row[j].set_title(
            rf"$\alpha={alpha:g}$, order={ok}",
        )

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes_row, P_dfs


def plot_equivalence_bars_plus_sweep(
    equiv_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    *,
    alpha: float = 0.5,
    prob_col: str = "P_equiv",
    label_col: str = "Config",
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "equivalence_bars_sweep.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot combined bar chart and sweep lines for equivalence.

    Parameters
    ----------
    equiv_df : DataFrame
        Output of ``equivalence_probability_table(alpha=…)`` with columns
        ``Config, P_equiv``.
    sweep_df : DataFrame
        Output of ``equivalence_probability_sweep()`` with columns
        ``Config, alpha, P_equiv``.
    """
    out_dir = ensure_dir(out_dir)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.5]},
    )

    # ── Left: horizontal bars sorted by P_equiv (highest on top) ──
    df_bar = equiv_df.copy()
    df_bar = df_bar.sort_values(prob_col, ascending=True).reset_index(drop=True)
    labels_sorted = df_bar[label_col].astype(str).tolist()
    vals = df_bar[prob_col].to_numpy(dtype=float)
    y = np.arange(len(labels_sorted))
    colors = case_colors_for_labels(labels_sorted)

    ax = axes[0]
    ax.barh(y, vals, color=colors, edgecolor="white", lw=0.4, alpha=0.85,
            height=0.7)

    # Annotate with P value + beta
    beta_col = "beta_med" if "beta_med" in df_bar.columns else None
    for i in range(len(labels_sorted)):
        p_val = vals[i]
        annot = f"{p_val:.2f}"
        if beta_col is not None:
            b_val = df_bar.iloc[i][beta_col]
            annot += rf"  ($\beta$~{b_val:+.2f})"
        # Place text just past the bar end, or inside if bar is long
        x_text = min(p_val + 0.02, 0.98)
        ha = "left" if p_val < 0.7 else "right"
        if p_val >= 0.7:
            x_text = p_val - 0.02
        ax.text(x_text, y[i], annot, va="center", ha=ha, fontsize=fs(-2))

    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel(rf"$P(|\Delta\mu| < {alpha}\,\sigma_{{\mathrm{{run}}}})$")
    ax.set_xlim(0, 1.0)
    ax.set_title(rf"Equivalence at $\alpha = {alpha}$")

    # ── Right: sweep lines ──
    ax2 = axes[1]
    configs = order_config_labels(sweep_df[label_col].unique().tolist())
    for cfg in configs:
        sub = sweep_df[sweep_df[label_col] == cfg].sort_values("alpha")
        ax2.plot(sub["alpha"], sub["P_equiv"],
                 label=cfg, color=case_color(cfg), lw=1.2)
    ax2.axvline(alpha, color="0.3", lw=0.8, ls=":", zorder=0)
    ax2.set_xlabel(r"Threshold $\alpha$")
    ax2.set_ylabel(r"$P(|\Delta\mu| < \alpha\,\sigma_{\mathrm{run}})$")
    ax2.set_ylim(-0.02, 1.0)
    ax2.legend(ncol=3, loc="lower right")
    ax2.set_title("Equivalence probability sweep")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes
