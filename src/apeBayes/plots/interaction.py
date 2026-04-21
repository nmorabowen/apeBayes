"""
Station × Rupture interaction plots (v8+, v9+).

Three kinds of plots:

1. plot_interaction_heatmap(gamma_sr_draws, station_labels, run_labels)
   — the shared posterior-median γ_sr pattern, one heatmap.

2. plot_interaction_by_case(gamma_sr_draws, xi_case_draws, case_labels,
                            station_labels, run_labels)
   — a panel of per-Case effective interactions ξ_case · γ_sr.  This is
   the v9 hero figure: it shows the bandwidth of the interaction in
   physical units, one tile per Case.

3. plot_interaction_forest(gamma_sr_draws, station_run_labels)
   — an 18-cell (for 3×6) forest plot sorted by |median|, with 90% CIs.
   Reveals whether the interaction is concentrated or smeared.

All functions return (fig, ax/axes) and optionally save to ``out_dir``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .helpers import ensure_dir, savefig, symmetric_bounds
from .style import CMAP_DIV, FULL_WIDTH, HALF_WIDTH, PALETTE, fs

if TYPE_CHECKING:
    from pathlib import Path

# ── Utilities ───────────────────────────────────────────────────────────

def _gamma_matrix(
    gamma_sr_draws: np.ndarray,
    n_stations: int,
    n_runs: int,
    reduce: str = "median",
) -> np.ndarray:
    """Reshape flattened γ_sr draws to a (n_stations, n_runs) matrix.

    ``gamma_sr_draws`` has shape (S, n_stations * n_runs) ordered as
    ``station_idx * n_runs + run_idx`` (this is the order used by
    RandomSlopesInteractionModel).
    """
    if reduce == "median":
        vec = np.median(gamma_sr_draws, axis=0)
    elif reduce == "mean":
        vec = np.mean(gamma_sr_draws, axis=0)
    else:
        raise ValueError(f"Unknown reduce={reduce!r}")
    return vec.reshape(n_stations, n_runs)


# ── 1. Shared γ_sr heatmap ──────────────────────────────────────────────

def plot_interaction_heatmap(
    gamma_sr_draws: np.ndarray,
    station_labels: list[str],
    run_labels: list[str],
    *,
    annotate: bool = True,
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "interaction_heatmap.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Single heatmap of the shared station×rupture interaction pattern.

    Parameters
    ----------
    gamma_sr_draws : (S, n_stations * n_runs)
        Posterior draws of γ_sr.
    station_labels, run_labels : list[str]
        Row and column labels for the matrix.
    annotate : bool
        Write the median value inside each cell.
    """
    out_dir = ensure_dir(out_dir)
    n_st = len(station_labels)
    n_rk = len(run_labels)
    mat = _gamma_matrix(gamma_sr_draws, n_st, n_rk)

    vmin, vmax = symmetric_bounds(mat)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(mat, cmap=CMAP_DIV, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(n_rk))
    ax.set_xticklabels(run_labels, fontsize=fs())
    ax.set_yticks(np.arange(n_st))
    ax.set_yticklabels(station_labels, fontsize=fs())
    ax.set_xlabel("Rupture")
    ax.set_ylabel("Station")
    ax.set_title(r"$\gamma_{sr}$: station$\times$rupture interaction (posterior median)")

    if annotate:
        thr = 0.6 * max(abs(vmin), abs(vmax))
        for i in range(n_st):
            for j in range(n_rk):
                col = "white" if abs(mat[i, j]) > thr else "black"
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center",
                        fontsize=fs(-1), color=col)

    cbar = fig.colorbar(im, ax=ax, label=r"$\gamma_{sr}$  (log EDP)", shrink=0.85)
    cbar.ax.tick_params(labelsize=7)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── 2. Per-Case ξ·γ panel (the v9 hero figure) ──────────────────────────

def plot_interaction_by_case(
    gamma_sr_draws: np.ndarray,
    case_labels: list[str],
    station_labels: list[str],
    run_labels: list[str],
    *,
    xi_case_draws: np.ndarray | None = None,
    annotate: bool = True,
    share_scale: bool = True,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "interaction_by_case.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Panel of per-Case effective station×rupture interactions.

    Each subplot shows ξ_case[c] · γ_sr (posterior median) for one Case.
    All panels share the same colour scale so that the *relative*
    bandwidth across Cases is visually obvious.

    Parameters
    ----------
    gamma_sr_draws : (S, n_stations * n_runs)
        Shared interaction draws.
    case_labels : list[str]
        Case labels (e.g. ["A", "B", "C", "D"]).
    station_labels, run_labels : list[str]
        Axis labels.
    xi_case_draws : (S, n_cases) or None
        Per-Case loadings.  If None, all panels equal γ_sr (v8 behaviour).
    annotate : bool
        Write values inside cells.
    share_scale : bool
        Common colour scale across panels.  Default True so that the
        visual size of the pattern is comparable.
    ncols : int, optional
        Columns in the subplot grid.  Defaults to ``len(case_labels)``
        for 4 Cases → 1 row.
    """
    out_dir = ensure_dir(out_dir)
    n_st = len(station_labels)
    n_rk = len(run_labels)
    n_cases = len(case_labels)

    # Median of the shared interaction matrix
    gamma_med_mat = _gamma_matrix(gamma_sr_draws, n_st, n_rk)

    # Per-Case loadings
    xi_med = np.ones(n_cases) if xi_case_draws is None else np.median(xi_case_draws, axis=0)

    panels = [xi_med[c] * gamma_med_mat for c in range(n_cases)]

    if share_scale:
        stacked = np.stack(panels, axis=0)
        vmin, vmax = symmetric_bounds(stacked)
    else:
        vmin = vmax = None

    if ncols is None:
        ncols = n_cases
    nrows = int(np.ceil(n_cases / ncols))
    if figsize is None:
        figsize = (FULL_WIDTH, 3.0 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True,
        squeeze=False,
    )

    im = None
    for c, lbl in enumerate(case_labels):
        r = c // ncols
        cc = c % ncols
        ax = axes[r, cc]
        mat = panels[c]

        if not share_scale:
            vmin, vmax = symmetric_bounds(mat)
        im = ax.imshow(mat, cmap=CMAP_DIV, vmin=vmin, vmax=vmax, aspect="auto")

        xi_val = xi_med[c]
        suff = "" if xi_case_draws is None else rf"  ($\xi={xi_val:.2f}$)"
        ax.set_title(f"Case {lbl}{suff}", fontsize=fs(+1))

        ax.set_xticks(np.arange(n_rk))
        ax.set_yticks(np.arange(n_st))
        # Only label edges of the grid
        if r == nrows - 1:
            ax.set_xticklabels(run_labels, fontsize=fs(-1))
            ax.set_xlabel("Rupture")
        else:
            ax.set_xticklabels([])
        if cc == 0:
            ax.set_yticklabels(station_labels, fontsize=fs(-1))
            ax.set_ylabel("Station")
        else:
            ax.set_yticklabels([])

        if annotate:
            thr = 0.6 * max(abs(vmin), abs(vmax))
            for i in range(n_st):
                for j in range(n_rk):
                    col = "white" if abs(mat[i, j]) > thr else "black"
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                            fontsize=fs(-2), color=col)

    # Hide extra axes if the grid is not fully filled
    for c in range(n_cases, nrows * ncols):
        axes[c // ncols, c % ncols].axis("off")

    # One shared colorbar
    if im is not None and share_scale:
        cbar = fig.colorbar(
            im, ax=axes.ravel().tolist(),
            label=(r"$\xi_{\mathrm{case}} \cdot \gamma_{sr}$"
                   if xi_case_draws is not None else r"$\gamma_{sr}$"),
            shrink=0.85, pad=0.02,
        )
        cbar.ax.tick_params(labelsize=7)

    title = ("Effective station"
             r"$\times$rupture interaction per Case"
             + ("  (v9: $\\xi$-loaded)" if xi_case_draws is not None else ""))
    fig.suptitle(title, fontsize=fs(+2), fontweight="bold")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes


# ── 3. γ_sr forest plot ─────────────────────────────────────────────────

def plot_interaction_forest(
    gamma_sr_draws: np.ndarray,
    station_run_labels: list[str],
    *,
    ci: tuple[float, float] = (0.05, 0.95),
    sort: bool = True,
    max_cells: int | None = None,
    figsize: tuple[float, float] = (HALF_WIDTH, 4.3),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "interaction_forest.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Forest plot of γ_sr cells with credible intervals.

    Each row is one (station, rupture) cell.  Points are posterior
    medians; horizontal bars are ``ci`` credible intervals.  Cells
    whose CI excludes zero are drawn in a highlight colour — those are
    the interaction cells the data identifies as reliably non-zero.

    Parameters
    ----------
    gamma_sr_draws : (S, n_inter)
        Posterior draws of γ_sr.
    station_run_labels : list[str]
        Labels for each cell (e.g. "Sta01|RSN146").
    ci : tuple
        Credible interval quantiles.
    sort : bool
        Sort cells by |median| descending.  Default True.
    max_cells : int, optional
        Keep only the top-N cells by |median| (after sorting).
    """
    out_dir = ensure_dir(out_dir)

    med = np.median(gamma_sr_draws, axis=0)
    lo = np.quantile(gamma_sr_draws, ci[0], axis=0)
    hi = np.quantile(gamma_sr_draws, ci[1], axis=0)

    labels = np.asarray(station_run_labels, dtype=object)

    if sort:
        order = np.argsort(-np.abs(med))
        med = med[order]
        lo = lo[order]
        hi = hi[order]
        labels = labels[order]

    if max_cells is not None:
        med = med[:max_cells]
        lo = lo[:max_cells]
        hi = hi[:max_cells]
        labels = labels[:max_cells]

    n = len(med)
    y = np.arange(n)

    # Highlight cells whose CI excludes zero
    excludes_zero = (lo > 0) | (hi < 0)
    colors = np.where(excludes_zero, PALETTE[0], PALETTE[5])  # navy (sig) vs taupe (ns)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # CI bars
    for i in range(n):
        ax.plot([lo[i], hi[i]], [y[i], y[i]],
                color=colors[i], lw=1.6, solid_capstyle="round")
    # Medians
    ax.scatter(med, y, c=colors, s=24, zorder=5,
               edgecolors="white", linewidth=0.5)

    # Zero reference
    ax.axvline(0, color="0.3", lw=0.8, ls=":", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs(-2))
    ax.invert_yaxis()
    ax.set_xlabel(r"$\gamma_{sr}$  (log EDP)")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5, zorder=0)
    q_lo, q_hi = int(ci[0] * 100), int(ci[1] * 100)
    ax.set_title(
        rf"Station$\times$rupture interaction — median + {q_hi - q_lo}% CI"
    )

    # Legend for the CI colour
    n_excl = int(excludes_zero.sum())
    ax.plot([], [], color=PALETTE[0], lw=2,
            label=f"CI excludes 0  (n={n_excl})")
    ax.plot([], [], color=PALETTE[5], lw=2,
            label=f"CI includes 0  (n={n - n_excl})")
    ax.legend(fontsize=fs(-1), loc="lower right")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax
